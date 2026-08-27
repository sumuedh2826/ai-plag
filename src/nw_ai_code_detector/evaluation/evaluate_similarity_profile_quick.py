from __future__ import annotations

import csv
import json
import time
from collections import defaultdict
from dataclasses import dataclass
from hashlib import sha256
from itertools import combinations
from pathlib import Path
from collections.abc import Mapping, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from nw_ai_code_detector.config import (
    AI_SOLUTIONS_DIR,
    EVAL_SCORES_PATH,
    OUTPUTS_DIR,
    SELECTED_500_PATH,
)
from nw_ai_code_detector.constants import (
    EvaluationMode,
    PERSONA_ORDER,
    SELECTION_RANDOM_SEED,
)
from nw_ai_code_detector.data_load import load_dataset
from nw_ai_code_detector.embedder import l2_normalize
from nw_ai_code_detector.evaluation.evaluate_centroid_all_humans import (
    EXPECTED_REFERENCE_COUNT,
    LABEL_AI,
    LABEL_HUMAN,
    ROLE_HELD_OUT,
    ROLE_HUMAN,
    ROLE_REFERENCE,
    ExperimentRecord,
    _assert_same_cluster,
    CachedSubsetSelection,
    _cluster_key_for_record,
    _collect_records,
    _held_out_reference_match_keys,
    _load_cached_vectors,
    _pair_token,
    _prepare_scoring_records,
    _validate_vectors,
)
from nw_ai_code_detector.evaluation.experiment_metrics import (
    GROUPED_FOLD_COUNT,
    LabeledScore,
    conservative_threshold,
    method_metrics_bundle,
    method_metrics_to_dict,
    question_folds,
)
from nw_ai_code_detector.generate_ai_refs import _limit_questions, _load_selected_questions
from nw_ai_code_detector.index import ClusterKey

EXPERIMENT_NAME = "similarity_profile_quick"
FIXED_SEED = SELECTION_RANDOM_SEED
EXPECTED_PAIRWISE_COUNT = 15
EXPECTED_MODEL_COUNT = 3
MAX_RUNTIME_SECONDS = 600
MATERIAL_FPR_LIMIT = 0.015
PYTHON_RECALL_GAIN_MIN = 0.05
CPP_RECALL_DROP_MAX = 0.03
PERSONA_DROP_MAX = 0.03
MACRO_DROP_MAX = 0.02
SCORES_PATH = OUTPUTS_DIR / "similarity_profile_quick_scores.json"
SUMMARY_PATH = OUTPUTS_DIR / "similarity_profile_quick_summary.csv"
REPORT_PATH = OUTPUTS_DIR / "similarity_profile_quick_report.md"
STANDALONE_METHODS = (
    "nn",
    "topk_mean",
    "centroid",
    "model_consensus_mean",
    "model_consensus_median",
    "model_consensus_min",
    "all_reference_mean",
)
CLASSIFIER_METHOD = "grouped_oof_logreg"
PERSONAS = ("production_review", "pair_programming")
FORBIDDEN_FEATURE_NAMES = frozenset(
    {
        "question_id",
        "source",
        "persona",
        "label",
        "content_hash",
        "user_id",
        "difficulty",
        "exact_match",
        "language",
    }
)
CLASSIFIER_FEATURE_NAMES = (
    "max_similarity",
    "second_similarity",
    "third_similarity",
    "top2_mean",
    "top3_mean",
    "all_reference_mean",
    "median_similarity",
    "minimum_similarity",
    "similarity_std",
    "similarity_range",
    "max_second_gap",
    "max_mean_gap",
    "centroid_similarity",
    "cluster_pairwise_mean",
    "cluster_pairwise_std",
    "cluster_pairwise_min",
    "cluster_pairwise_max",
    "query_mean_minus_cluster_mean",
    "query_max_minus_cluster_mean",
    "query_std_minus_cluster_std",
    "model_consensus_mean",
    "model_consensus_median",
    "model_consensus_min",
    "model_consensus_std",
    "model_consensus_range",
)


@dataclass(frozen=True)
class ReferenceCluster:
    key: ClusterKey
    vectors: np.ndarray
    models: tuple[str, ...]
    sources: tuple[str, ...]
    hashes: tuple[str, ...]
    centroid: np.ndarray
    pairwise: tuple[float, ...]


@dataclass(frozen=True)
class ProfiledSubmission:
    question_id: str
    language: str
    label: str
    source: str
    content_hash: str
    features: dict[str, float]


@dataclass(frozen=True)
class ClusterLookup:
    record: ExperimentRecord
    cluster_key: ClusterKey
    clusters: Mapping[ClusterKey, ReferenceCluster]


@dataclass(frozen=True)
class OofFitTrace:
    train_question_ids: tuple[str, ...]
    test_question_ids: tuple[str, ...]
    train_row_count: int
    test_row_count: int
    coefficients: tuple[float, ...]


@dataclass(frozen=True)
class OofClassifierResult:
    scores: tuple[float, ...]
    traces: tuple[OofFitTrace, ...]
    mean_coefficients: dict[str, float]


@dataclass(frozen=True)
class ClassifierRunInput:
    rows: tuple[ProfiledSubmission, ...]
    feature_names: tuple[str, ...]
    fold_map: Mapping[str, int]


@dataclass(frozen=True)
class PersonaHitInput:
    positives: tuple[ProfiledSubmission, ...]
    method: str
    items: tuple[LabeledScore, ...]
    fold_map: Mapping[str, int]
    target_fpr: float


@dataclass(frozen=True)
class FoldThresholdInput:
    items: tuple[LabeledScore, ...]
    fold_map: Mapping[str, int]
    fold_id: int
    language: str
    target_fpr: float


@dataclass(frozen=True)
class ExperimentOutput:
    subset: CachedSubsetSelection
    records: tuple[ExperimentRecord, ...]
    profiled: tuple[ProfiledSubmission, ...]
    standalone: Mapping[str, object]
    classifier: Mapping[str, object]
    personas: Mapping[str, object]
    model_issue: str | None
    match_keys: tuple[tuple[str, str, str], ...]
    eval_hash_before: str | None
    feature_names: tuple[str, ...]


def resolve_cluster(lookup: ClusterLookup) -> ReferenceCluster:
    _assert_same_cluster(lookup.record, lookup.cluster_key)
    cluster = lookup.clusters.get(lookup.cluster_key)
    if cluster is None:
        raise RuntimeError(
            "Missing same-question same-language cluster for "
            f"{lookup.record.question_id}:{lookup.record.language}"
        )
    _assert_same_cluster(lookup.record, cluster.key)
    return cluster


def query_cluster_similarities(
    query_vector: np.ndarray,
    cluster: ReferenceCluster,
) -> np.ndarray:
    _require_finite(query_vector, "query_vector")
    _require_finite(cluster.vectors, "cluster.vectors")
    if cluster.vectors.shape[0] != EXPECTED_REFERENCE_COUNT:
        raise RuntimeError(
            f"Cluster {cluster.key.question_id}:{cluster.key.language} "
            f"has {cluster.vectors.shape[0]} references, expected "
            f"{EXPECTED_REFERENCE_COUNT}"
        )
    similarities = cluster.vectors @ query_vector
    _require_finite(similarities, "similarities")
    return np.asarray(similarities, dtype=np.float64)


def sorted_profile_features(similarities: Sequence[float]) -> dict[str, float]:
    values = np.sort(np.asarray(similarities, dtype=np.float64))[::-1]
    if values.shape[0] != EXPECTED_REFERENCE_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_REFERENCE_COUNT} similarities, got {values.shape[0]}"
        )
    _require_finite(values, "sorted_similarities")
    top2_mean = float(np.mean(values[:2]))
    top3_mean = float(np.mean(values[:3]))
    all_mean = float(np.mean(values))
    std = float(np.std(values))
    features = {
        "max_similarity": float(values[0]),
        "second_similarity": float(values[1]),
        "third_similarity": float(values[2]),
        "top2_mean": top2_mean,
        "top3_mean": top3_mean,
        "all_reference_mean": all_mean,
        "median_similarity": float(np.median(values)),
        "minimum_similarity": float(values[5]),
        "similarity_std": std,
        "similarity_range": float(values[0] - values[5]),
        "max_second_gap": float(values[0] - values[1]),
        "max_mean_gap": float(values[0] - all_mean),
    }
    _require_finite_mapping(features, "profile")
    return features


def cluster_pairwise_features(vectors: np.ndarray) -> dict[str, float]:
    pairs = pairwise_reference_similarities(vectors)
    features = {
        "cluster_pairwise_mean": float(np.mean(pairs)),
        "cluster_pairwise_std": float(np.std(pairs)),
        "cluster_pairwise_min": float(np.min(pairs)),
        "cluster_pairwise_max": float(np.max(pairs)),
    }
    _require_finite_mapping(features, "pairwise")
    return features


def pairwise_reference_similarities(vectors: np.ndarray) -> np.ndarray:
    if vectors.shape[0] != EXPECTED_REFERENCE_COUNT:
        raise RuntimeError("Pairwise compactness requires six reference vectors")
    _require_finite(vectors, "pairwise_vectors")
    pairs = [
        float(vectors[left] @ vectors[right])
        for left, right in combinations(range(EXPECTED_REFERENCE_COUNT), 2)
    ]
    if len(pairs) != EXPECTED_PAIRWISE_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_PAIRWISE_COUNT} unique pairs, got {len(pairs)}"
        )
    array = np.asarray(pairs, dtype=np.float64)
    _require_finite(array, "pairwise_similarities")
    return array


def model_consensus_features(
    similarities: Sequence[float],
    models: Sequence[str],
) -> dict[str, float]:
    maxima = model_max_similarities(similarities, models)
    values = np.asarray(sorted(maxima.values()), dtype=np.float64)
    if values.shape[0] != EXPECTED_MODEL_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_MODEL_COUNT} generating-model maxima, "
            f"got {values.shape[0]} from {sorted(maxima)}"
        )
    features = {
        "model_consensus_mean": float(np.mean(values)),
        "model_consensus_median": float(np.median(values)),
        "model_consensus_min": float(np.min(values)),
        "model_consensus_std": float(np.std(values)),
        "model_consensus_range": float(np.max(values) - np.min(values)),
    }
    _require_finite_mapping(features, "consensus")
    return features


def model_max_similarities(
    similarities: Sequence[float],
    models: Sequence[str],
) -> dict[str, float]:
    if len(similarities) != len(models):
        raise RuntimeError("Similarity and model lists must align")
    grouped: dict[str, list[float]] = defaultdict(list)
    for similarity, model in zip(similarities, models):
        if not model:
            raise RuntimeError("Reference model metadata is missing")
        grouped[str(model)].append(float(similarity))
    return {model: float(max(scores)) for model, scores in grouped.items()}


def centroid_vector(vectors: np.ndarray) -> np.ndarray:
    mean_vector = np.mean(vectors.astype(np.float64), axis=0)
    normalized = np.asarray(l2_normalize(mean_vector), dtype=np.float64)
    _require_finite(normalized, "centroid")
    return normalized


def adaptive_features(
    profile: Mapping[str, float],
    compactness: Mapping[str, float],
) -> dict[str, float]:
    features = {
        "query_mean_minus_cluster_mean": (
            profile["all_reference_mean"] - compactness["cluster_pairwise_mean"]
        ),
        "query_max_minus_cluster_mean": (
            profile["max_similarity"] - compactness["cluster_pairwise_mean"]
        ),
        "query_std_minus_cluster_std": (
            profile["similarity_std"] - compactness["cluster_pairwise_std"]
        ),
    }
    _require_finite_mapping(features, "adaptive")
    return features


def build_classifier() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=FIXED_SEED,
                ),
            ),
        ]
    )


def grouped_oof_classifier(run: ClassifierRunInput) -> OofClassifierResult:
    _assert_classifier_features(run.feature_names)
    scores = [0.0] * len(run.rows)
    traces: list[OofFitTrace] = []
    coefficient_rows: list[np.ndarray] = []
    predicted_indexes: list[int] = []
    for fold_id in range(GROUPED_FOLD_COUNT):
        train_idx = _fold_indexes(run.rows, run.fold_map, fold_id, False)
        test_idx = _fold_indexes(run.rows, run.fold_map, fold_id, True)
        if not train_idx or not test_idx:
            continue
        _assert_fold_question_split(run.rows, train_idx, test_idx)
        pipeline = build_classifier()
        train_x = _feature_matrix(run.rows, train_idx, run.feature_names)
        train_y = _label_vector(run.rows, train_idx)
        test_x = _feature_matrix(run.rows, test_idx, run.feature_names)
        pipeline.fit(train_x, train_y)
        predicted = pipeline.predict_proba(test_x)[:, 1]
        for local, row_index in enumerate(test_idx):
            scores[row_index] = float(predicted[local])
            predicted_indexes.append(row_index)
        coefficients = np.asarray(pipeline.named_steps["model"].coef_[0], dtype=np.float64)
        coefficient_rows.append(coefficients)
        traces.append(
            _fold_trace(run.rows, train_idx, test_idx, coefficients)
        )
    if len(predicted_indexes) != len(run.rows):
        raise RuntimeError("Every evaluation row must receive an out-of-fold score")
    mean_coefficients = _mean_coefficients(run.feature_names, coefficient_rows)
    return OofClassifierResult(tuple(scores), tuple(traces), mean_coefficients)


def exclude_exact_match_positives(
    rows: Sequence[ProfiledSubmission],
    match_keys: set[tuple[str, str, str]],
) -> list[ProfiledSubmission]:
    return [
        row
        for row in rows
        if row.label != LABEL_AI
        or (row.question_id, row.language, row.content_hash) not in match_keys
    ]


def same_cluster_exact_match_keys(
    records: Sequence[ExperimentRecord],
) -> set[tuple[str, str, str]]:
    return _held_out_reference_match_keys(records)


def main() -> int:
    total_started = time.perf_counter()
    eval_hash_before = _file_sha256(EVAL_SCORES_PATH)
    print("Loading cached embeddings")
    loading_started = time.perf_counter()
    dataset = load_dataset()
    questions = _limit_questions(_load_selected_questions(SELECTED_500_PATH), None)
    records = _collect_records(dataset, questions)
    scoring_records, subset = _prepare_scoring_records(
        EvaluationMode.CACHED_HUMANS_ONLY,
        records,
    )
    if scoring_records is None or subset is None:
        raise RuntimeError("Cached-human subset is required for this experiment")
    vectors = _load_cached_vectors(scoring_records)
    _validate_vectors(scoring_records, vectors)
    model_map, model_issue = load_reference_model_map(scoring_records)
    loading_elapsed = time.perf_counter() - loading_started
    _check_runtime(total_started, "loading")
    print("Extracting similarity-profile features")
    feature_started = time.perf_counter()
    clusters = build_reference_clusters(scoring_records, vectors, model_map)
    profiled = profile_submissions(
        scoring_records,
        vectors,
        clusters,
        model_issue is None,
    )
    feature_elapsed = time.perf_counter() - feature_started
    _check_runtime(total_started, "feature extraction")
    print("Calculating standalone score metrics")
    metrics_started = time.perf_counter()
    match_keys = same_cluster_exact_match_keys(scoring_records)
    feature_names = _active_feature_names(model_issue is None)
    standalone = _standalone_metrics(profiled, match_keys)
    personas = _persona_metrics(profiled, match_keys)
    metrics_elapsed = time.perf_counter() - metrics_started
    _check_runtime(total_started, "standalone metrics")
    print("Training grouped-OOF classifier")
    train_started = time.perf_counter()
    classifier = _classifier_metrics(profiled, feature_names, match_keys)
    train_elapsed = time.perf_counter() - train_started
    _check_runtime(total_started, "grouped model training")
    print("Writing report")
    report_started = time.perf_counter()
    timings = {
        "loading_seconds": loading_elapsed,
        "feature_extraction_seconds": feature_elapsed,
        "standalone_score_metrics_seconds": metrics_elapsed,
        "grouped_model_training_seconds": train_elapsed,
        "report_writing_seconds": 0.0,
        "total_seconds": 0.0,
    }
    payload = _results_payload(
        ExperimentOutput(
            subset=subset,
            records=tuple(scoring_records),
            profiled=tuple(profiled),
            standalone=standalone,
            classifier=classifier,
            personas=personas,
            model_issue=model_issue,
            match_keys=tuple(sorted(match_keys)),
            eval_hash_before=eval_hash_before,
            feature_names=tuple(feature_names),
        )
    )
    report_elapsed = time.perf_counter() - report_started
    timings["report_writing_seconds"] = report_elapsed
    timings["total_seconds"] = time.perf_counter() - total_started
    payload["timings"] = timings
    payload["eval_scores_sha256_after"] = _file_sha256(EVAL_SCORES_PATH)
    _write_outputs(payload)
    _print_timings(timings)
    if payload["eval_scores_sha256_before"] != payload["eval_scores_sha256_after"]:
        raise RuntimeError("outputs/eval_scores.json changed during the experiment")
    return 0


def load_reference_model_map(
    records: Sequence[ExperimentRecord],
) -> tuple[dict[tuple[str, str, str], str], str | None]:
    wanted = {
        (record.question_id, record.language, record.source)
        for record in records
        if record.role == ROLE_REFERENCE
    }
    loaded: dict[tuple[str, str, str], str] = {}
    missing_field = 0
    for path in sorted(AI_SOLUTIONS_DIR.rglob("*.json")):
        if path.name.startswith("_"):
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        key = (
            str(payload.get("qid") or ""),
            str(payload.get("language") or ""),
            str(payload.get("persona") or ""),
        )
        if key not in wanted:
            continue
        model = payload.get("model")
        if not isinstance(model, str) or not model.strip():
            missing_field += 1
            continue
        loaded[key] = model.strip()
    return _validate_model_map(wanted, loaded, missing_field)


def build_reference_clusters(
    records: Sequence[ExperimentRecord],
    vectors: Sequence[Sequence[float]],
    model_map: Mapping[tuple[str, str, str], str],
) -> dict[ClusterKey, ReferenceCluster]:
    grouped: dict[ClusterKey, list[tuple[ExperimentRecord, np.ndarray]]] = defaultdict(list)
    for record, vector in zip(records, vectors):
        if record.role != ROLE_REFERENCE:
            continue
        key = _cluster_key_for_record(record)
        _assert_same_cluster(record, key)
        grouped[key].append((record, np.asarray(vector, dtype=np.float64)))
    clusters: dict[ClusterKey, ReferenceCluster] = {}
    for key, items in grouped.items():
        if len(items) != EXPECTED_REFERENCE_COUNT:
            raise RuntimeError(
                f"Cluster {key.question_id}:{key.language} has {len(items)} refs"
            )
        matrix = np.vstack([item[1] for item in items])
        models = tuple(
            model_map.get((item[0].question_id, item[0].language, item[0].source), "")
            for item in items
        )
        compactness = pairwise_reference_similarities(matrix)
        clusters[key] = ReferenceCluster(
            key=key,
            vectors=matrix,
            models=models,
            sources=tuple(item[0].source for item in items),
            hashes=tuple(item[0].content_hash for item in items),
            centroid=centroid_vector(matrix),
            pairwise=tuple(float(value) for value in compactness),
        )
    return clusters


def profile_submissions(
    records: Sequence[ExperimentRecord],
    vectors: Sequence[Sequence[float]],
    clusters: Mapping[ClusterKey, ReferenceCluster],
    include_consensus: bool,
) -> list[ProfiledSubmission]:
    rows: list[ProfiledSubmission] = []
    for record, vector in zip(records, vectors):
        if record.role not in {ROLE_HELD_OUT, ROLE_HUMAN}:
            continue
        cluster = resolve_cluster(
            ClusterLookup(record, _cluster_key_for_record(record), clusters)
        )
        query = np.asarray(vector, dtype=np.float64)
        similarities = query_cluster_similarities(query, cluster)
        features = _submission_features(
            similarities,
            cluster,
            query,
            include_consensus,
        )
        label = LABEL_AI if record.role == ROLE_HELD_OUT else LABEL_HUMAN
        rows.append(
            ProfiledSubmission(
                question_id=record.question_id,
                language=record.language,
                label=label,
                source=record.source,
                content_hash=record.content_hash,
                features=features,
            )
        )
    return rows


def _submission_features(
    similarities: np.ndarray,
    cluster: ReferenceCluster,
    query: np.ndarray,
    include_consensus: bool,
) -> dict[str, float]:
    profile = sorted_profile_features(similarities)
    compactness = cluster_pairwise_features(cluster.vectors)
    centroid_similarity = float(cluster.centroid @ query)
    _require_finite(np.asarray([centroid_similarity]), "centroid_similarity")
    features = {
        **profile,
        **compactness,
        **adaptive_features(profile, compactness),
        "centroid_similarity": centroid_similarity,
    }
    if include_consensus:
        features.update(model_consensus_features(similarities, cluster.models))
    _require_finite_mapping(features, "submission_features")
    return features


def _validate_model_map(
    wanted: set[tuple[str, str, str]],
    loaded: Mapping[tuple[str, str, str], str],
    missing_field: int,
) -> tuple[dict[tuple[str, str, str], str], str | None]:
    if missing_field:
        return {}, "AI-reference JSON is missing an explicit model field"
    if set(loaded) != wanted:
        return {}, "Explicit model metadata is incomplete for the reference cluster"
    unique_models = {model for model in loaded.values()}
    if len(unique_models) != EXPECTED_MODEL_COUNT:
        return (
            dict(loaded),
            f"Expected {EXPECTED_MODEL_COUNT} generating models, found {len(unique_models)}",
        )
    counts = defaultdict(int)
    for model in loaded.values():
        counts[model] += 1
    return dict(loaded), None


def _standalone_metrics(
    rows: Sequence[ProfiledSubmission],
    match_keys: set[tuple[str, str, str]],
) -> dict[str, dict[str, dict[str, object]]]:
    result: dict[str, dict[str, dict[str, object]]] = {}
    for variant, subset in _positive_variants(rows, match_keys).items():
        result[variant] = {}
        for language in ("CPP", "PYTHON"):
            language_rows = [row for row in subset if row.language == language]
            result[variant][language] = {
                method: method_metrics_to_dict(
                    method_metrics_bundle(
                        method,
                        _labeled_for_method(language_rows, method),
                        FIXED_SEED,
                    )
                )
                for method in STANDALONE_METHODS
                if _method_available(language_rows, method)
            }
    return result


def _classifier_metrics(
    rows: Sequence[ProfiledSubmission],
    feature_names: tuple[str, ...],
    match_keys: set[tuple[str, str, str]],
) -> dict[str, object]:
    fold_map = question_folds({row.question_id for row in rows}, FIXED_SEED)
    all_result = _classifier_for_rows(tuple(rows), feature_names, fold_map)
    excluded_rows = exclude_exact_match_positives(rows, match_keys)
    excluded_result = _classifier_for_rows(
        tuple(excluded_rows),
        feature_names,
        question_folds({row.question_id for row in excluded_rows}, FIXED_SEED),
    )
    return {
        "all_held_out": all_result,
        "exclude_exact_reference_match": excluded_result,
        "feature_names": list(feature_names),
    }


def _classifier_for_rows(
    rows: tuple[ProfiledSubmission, ...],
    feature_names: tuple[str, ...],
    fold_map: Mapping[str, int],
) -> dict[str, object]:
    by_language: dict[str, dict[str, object]] = {}
    combined: list[ProfiledSubmission] = []
    for language in ("CPP", "PYTHON"):
        language_rows = tuple(row for row in rows if row.language == language)
        oof = grouped_oof_classifier(
            ClassifierRunInput(language_rows, feature_names, fold_map)
        )
        scored = [
            _row_with_score(row, score)
            for row, score in zip(language_rows, oof.scores)
        ]
        bundle = method_metrics_bundle(
            CLASSIFIER_METHOD,
            _labeled_scores(scored, CLASSIFIER_METHOD),
            FIXED_SEED,
        )
        by_language[language] = {
            "metrics": method_metrics_to_dict(bundle),
            "mean_coefficients": oof.mean_coefficients,
            "n_oof_predictions": len(oof.scores),
            "n_fit_traces": len(oof.traces),
            "personas": _persona_from_oof(scored),
        }
        combined.extend(scored)
    by_language["personas_all_languages"] = _persona_from_oof(combined)
    return by_language


def _persona_metrics(
    rows: Sequence[ProfiledSubmission],
    match_keys: set[tuple[str, str, str]],
) -> dict[str, dict[str, dict[str, object]]]:
    result: dict[str, dict[str, dict[str, object]]] = {}
    for variant, subset in _positive_variants(rows, match_keys).items():
        result[variant] = {}
        for method in STANDALONE_METHODS:
            if not _method_available(subset, method):
                continue
            result[variant][method] = {
                persona: _persona_oof(subset, method, persona)
                for persona in PERSONAS
            }
    return result


def _persona_oof(
    rows: Sequence[ProfiledSubmission],
    method: str,
    persona: str,
) -> dict[str, float | int | None]:
    items = _labeled_for_method(rows, method)
    fold_map = question_folds({item.question_id for item in items}, FIXED_SEED)
    positives = [
        row for row in rows if row.label == LABEL_AI and row.source == persona
    ]
    recall_1 = _persona_hits(
        PersonaHitInput(tuple(positives), method, tuple(items), fold_map, 0.01)
    )
    recall_5 = _persona_hits(
        PersonaHitInput(tuple(positives), method, tuple(items), fold_map, 0.05)
    )
    scores = [_method_score(row, method) for row in positives]
    return {
        "n_positives": len(positives),
        "mean": float(np.mean(scores)) if scores else None,
        "grouped_oof_recall_at_1pct_fpr": recall_1,
        "grouped_oof_recall_at_5pct_fpr": recall_5,
        "threshold_source": "language-specific training-fold human negatives",
    }


def _persona_from_oof(rows: Sequence[ProfiledSubmission]) -> dict[str, object]:
    return {
        persona: _persona_oof(rows, CLASSIFIER_METHOD, persona)
        for persona in PERSONAS
    }


def _persona_hits(payload: PersonaHitInput) -> float | None:
    if not payload.positives:
        return None
    hits = 0
    total = 0
    for row in payload.positives:
        fold_id = payload.fold_map[row.question_id]
        threshold = _language_fold_threshold(
            FoldThresholdInput(
                payload.items,
                payload.fold_map,
                fold_id,
                row.language,
                payload.target_fpr,
            )
        )
        if threshold is None:
            continue
        total += 1
        if _method_score(row, payload.method) >= threshold:
            hits += 1
    if total == 0:
        return None
    return hits / total


def _language_fold_threshold(payload: FoldThresholdInput) -> float | None:
    train_neg = [
        item.score
        for item in payload.items
        if item.label == LABEL_HUMAN
        and item.language == payload.language
        and payload.fold_map[item.question_id] != payload.fold_id
    ]
    if not train_neg:
        return None
    return conservative_threshold(train_neg, payload.target_fpr)


def _positive_variants(
    rows: Sequence[ProfiledSubmission],
    match_keys: set[tuple[str, str, str]],
) -> dict[str, list[ProfiledSubmission]]:
    return {
        "all_held_out": list(rows),
        "exclude_exact_reference_match": exclude_exact_match_positives(
            rows,
            match_keys,
        ),
    }


def _labeled_for_method(
    rows: Sequence[ProfiledSubmission],
    method: str,
) -> list[LabeledScore]:
    return [
        LabeledScore(
            question_id=row.question_id,
            pair_token=_pair_token(row.question_id, row.language),
            language=row.language,
            label=row.label,
            score=_method_score(row, method),
        )
        for row in rows
    ]


def _labeled_scores(
    rows: Sequence[ProfiledSubmission],
    method: str,
) -> list[LabeledScore]:
    return _labeled_for_method(rows, method)


def _method_score(row: ProfiledSubmission, method: str) -> float:
    mapping = {
        "nn": "max_similarity",
        "topk_mean": "top3_mean",
        "centroid": "centroid_similarity",
        "model_consensus_mean": "model_consensus_mean",
        "model_consensus_median": "model_consensus_median",
        "model_consensus_min": "model_consensus_min",
        "all_reference_mean": "all_reference_mean",
        CLASSIFIER_METHOD: "classifier_oof",
    }
    key = mapping[method]
    return float(row.features[key])


def _method_available(rows: Sequence[ProfiledSubmission], method: str) -> bool:
    if not rows:
        return False
    try:
        _method_score(rows[0], method)
    except KeyError:
        return False
    return True


def _row_with_score(row: ProfiledSubmission, score: float) -> ProfiledSubmission:
    features = dict(row.features)
    features["classifier_oof"] = float(score)
    return ProfiledSubmission(
        question_id=row.question_id,
        language=row.language,
        label=row.label,
        source=row.source,
        content_hash=row.content_hash,
        features=features,
    )


def _fold_indexes(
    rows: Sequence[ProfiledSubmission],
    fold_map: Mapping[str, int],
    fold_id: int,
    is_test: bool,
) -> list[int]:
    indexes: list[int] = []
    for index, row in enumerate(rows):
        in_fold = fold_map[row.question_id] == fold_id
        if in_fold == is_test:
            indexes.append(index)
    return indexes


def _assert_fold_question_split(
    rows: Sequence[ProfiledSubmission],
    train_idx: Sequence[int],
    test_idx: Sequence[int],
) -> None:
    train_q = {rows[index].question_id for index in train_idx}
    test_q = {rows[index].question_id for index in test_idx}
    overlap = train_q & test_q
    if overlap:
        raise RuntimeError(f"Grouped fold split a question: {sorted(overlap)[:3]}")


def _feature_matrix(
    rows: Sequence[ProfiledSubmission],
    indexes: Sequence[int],
    feature_names: Sequence[str],
) -> np.ndarray:
    matrix = np.asarray(
        [[rows[index].features[name] for name in feature_names] for index in indexes],
        dtype=np.float64,
    )
    _require_finite(matrix, "feature_matrix")
    return matrix


def _label_vector(
    rows: Sequence[ProfiledSubmission],
    indexes: Sequence[int],
) -> np.ndarray:
    return np.asarray(
        [1 if rows[index].label == LABEL_AI else 0 for index in indexes],
        dtype=np.int32,
    )


def _fold_trace(
    rows: Sequence[ProfiledSubmission],
    train_idx: Sequence[int],
    test_idx: Sequence[int],
    coefficients: np.ndarray,
) -> OofFitTrace:
    train_q = tuple(sorted({rows[index].question_id for index in train_idx}))
    test_q = tuple(sorted({rows[index].question_id for index in test_idx}))
    return OofFitTrace(
        train_question_ids=train_q,
        test_question_ids=test_q,
        train_row_count=len(train_idx),
        test_row_count=len(test_idx),
        coefficients=tuple(float(value) for value in coefficients),
    )


def _mean_coefficients(
    feature_names: Sequence[str],
    coefficient_rows: Sequence[np.ndarray],
) -> dict[str, float]:
    stacked = np.vstack(coefficient_rows)
    means = np.mean(stacked, axis=0)
    return {
        name: float(value)
        for name, value in zip(feature_names, means)
    }


def _assert_classifier_features(feature_names: Sequence[str]) -> None:
    forbidden = FORBIDDEN_FEATURE_NAMES.intersection(feature_names)
    if forbidden:
        raise RuntimeError(f"Classifier features include identifiers: {forbidden}")


def _active_feature_names(include_consensus: bool) -> tuple[str, ...]:
    if include_consensus:
        return CLASSIFIER_FEATURE_NAMES
    return tuple(
        name
        for name in CLASSIFIER_FEATURE_NAMES
        if not name.startswith("model_consensus_")
    )


def _results_payload(output: ExperimentOutput) -> dict[str, object]:
    population = _population_payload(output)
    verdict = decide_verdict(output.standalone, output.classifier, output.personas)
    return {
        "experiment": EXPERIMENT_NAME,
        "evaluation_mode": "cached_humans_only",
        "is_production": False,
        "is_final_all_human_evaluation": False,
        "bootstrap_iterations": 0,
        "reversible": True,
        "network_calls": False,
        "embeddings_generated": False,
        "cross_model_status": "enabled" if output.model_issue is None else "disabled",
        "cross_model_issue": output.model_issue,
        "feature_names": list(output.feature_names),
        "forbidden_feature_names": sorted(FORBIDDEN_FEATURE_NAMES),
        "population": population,
        "standalone": output.standalone,
        "classifier": output.classifier,
        "personas": output.personas,
        "verdict": verdict,
        "eval_scores_sha256_before": output.eval_hash_before,
        "reference_personas": [persona.value for persona in PERSONA_ORDER],
    }


def _population_payload(output: ExperimentOutput) -> dict[str, object]:
    records = output.records
    subset = output.subset
    humans = [record for record in records if record.role == ROLE_HUMAN]
    held_out = [record for record in records if record.role == ROLE_HELD_OUT]
    refs = [record for record in records if record.role == ROLE_REFERENCE]
    cpp_humans = sum(1 for record in humans if record.language == "CPP")
    py_humans = sum(1 for record in humans if record.language == "PYTHON")
    cpp_ai = sum(1 for record in held_out if record.language == "CPP")
    py_ai = sum(1 for record in held_out if record.language == "PYTHON")
    cpp_pairs = sum(1 for _qid, language in subset.eligible_pairs if language == "CPP")
    py_pairs = sum(1 for _qid, language in subset.eligible_pairs if language == "PYTHON")
    return {
        "eligible_pairs": len(subset.eligible_pairs),
        "eligible_pairs_cpp": cpp_pairs,
        "eligible_pairs_python": py_pairs,
        "pairs_excluded_no_cached_human": len(subset.pairs_excluded_no_cached_human),
        "cached_humans": len(humans),
        "cached_humans_cpp": cpp_humans,
        "cached_humans_python": py_humans,
        "held_out_ai": len(held_out),
        "held_out_ai_cpp": cpp_ai,
        "held_out_ai_python": py_ai,
        "ai_references": len(refs),
        "same_cluster_exact_matches": len(output.match_keys),
        "missing_humans_skipped": subset.missing_humans_skipped,
    }


def decide_verdict(
    standalone: Mapping[str, object],
    classifier: Mapping[str, object],
    personas: Mapping[str, object],
) -> dict[str, object]:
    excluded = standalone["exclude_exact_reference_match"]
    all_held = standalone["all_held_out"]
    nn_py = excluded["PYTHON"]["nn"]
    nn_cpp = all_held["CPP"]["nn"]
    clf_ex = classifier["exclude_exact_reference_match"]
    clf_all = classifier["all_held_out"]
    clf_py = clf_ex["PYTHON"]["metrics"]
    clf_cpp = clf_all["CPP"]["metrics"]
    py_gain = _recall_delta(clf_py, nn_py, "1%")
    cpp_gain = _recall_delta(clf_cpp, nn_cpp, "1%")
    py_fpr = _oof_fpr(clf_py, "1%")
    py_auroc_gain = _optional_delta(clf_py.get("pooled_auroc"), nn_py.get("pooled_auroc"))
    py_r1_gain_all = _recall_delta(clf_py, nn_py, "1%")
    macro_gain = _optional_delta(clf_py.get("macro_auroc"), nn_py.get("macro_auroc"))
    persona_nn = _persona_value(personas, "all_held_out", "nn", "production_review")
    persona_clf = clf_all["personas_all_languages"]["production_review"][
        "grouped_oof_recall_at_1pct_fpr"
    ]
    promising = _is_promising(py_gain, py_fpr, cpp_gain, persona_clf, persona_nn, macro_gain)
    if promising:
        label = "promising"
    elif py_auroc_gain is not None and py_auroc_gain > 0 and (py_r1_gain_all or 0) <= 0:
        label = "No improvement over NN"
    elif py_gain is not None and 0 < py_gain < PYTHON_RECALL_GAIN_MIN:
        label = "Small directional improvement, insufficient for adoption"
    elif py_gain is not None and py_gain <= 0:
        label = "No improvement over NN"
    else:
        label = "Cached-subset result is inconclusive"
    if py_auroc_gain is not None and py_auroc_gain > 0 and (py_gain or 0) < 0.01:
        deployment = "reject_for_deployment_auroc_without_low_fpr_recall"
    else:
        deployment = "not_adopted" if label != "promising" else "not_production_do_not_integrate"
    return {
        "label": label,
        "deployment_note": deployment,
        "python_oof_recall_1pct_gain_vs_nn": py_gain,
        "python_oof_achieved_fpr_1pct": py_fpr,
        "cpp_oof_recall_1pct_gain_vs_nn": cpp_gain,
        "python_macro_auroc_gain_vs_nn": macro_gain,
        "production_review_recall_1pct_classifier": persona_clf,
        "production_review_recall_1pct_nn": persona_nn,
        "rules": {
            "python_recall_gain_min": PYTHON_RECALL_GAIN_MIN,
            "python_fpr_limit": MATERIAL_FPR_LIMIT,
            "cpp_recall_drop_max": CPP_RECALL_DROP_MAX,
            "persona_drop_max": PERSONA_DROP_MAX,
            "macro_drop_max": MACRO_DROP_MAX,
        },
    }


def _is_promising(
    py_gain: float | None,
    py_fpr: float | None,
    cpp_gain: float | None,
    persona_clf: float | None,
    persona_nn: float | None,
    macro_gain: float | None,
) -> bool:
    if py_gain is None or py_fpr is None or cpp_gain is None:
        return False
    if py_gain < PYTHON_RECALL_GAIN_MIN:
        return False
    if py_fpr > MATERIAL_FPR_LIMIT:
        return False
    if cpp_gain < -CPP_RECALL_DROP_MAX:
        return False
    if persona_clf is not None and persona_nn is not None:
        if persona_clf < persona_nn - PERSONA_DROP_MAX:
            return False
    if macro_gain is not None and macro_gain < -MACRO_DROP_MAX:
        return False
    return True


def _recall_delta(
    left: Mapping[str, object],
    right: Mapping[str, object],
    point: str,
) -> float | None:
    left_value = _oof_recall(left, point)
    right_value = _oof_recall(right, point)
    return _optional_delta(left_value, right_value)


def _oof_recall(metrics: Mapping[str, object], point: str) -> float | None:
    grouped = metrics.get("grouped_out_of_fold")
    if not isinstance(grouped, dict):
        return None
    payload = grouped.get(point)
    if not isinstance(payload, dict):
        return None
    value = payload.get("recall")
    return float(value) if isinstance(value, (int, float)) else None


def _oof_fpr(metrics: Mapping[str, object], point: str) -> float | None:
    grouped = metrics.get("grouped_out_of_fold")
    if not isinstance(grouped, dict):
        return None
    payload = grouped.get(point)
    if not isinstance(payload, dict):
        return None
    value = payload.get("achieved_fpr")
    return float(value) if isinstance(value, (int, float)) else None


def _optional_delta(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return float(left) - float(right)


def _persona_value(
    personas: Mapping[str, object],
    variant: str,
    method: str,
    persona: str,
) -> float | None:
    variant_payload = personas.get(variant)
    if not isinstance(variant_payload, dict):
        return None
    method_payload = variant_payload.get(method)
    if not isinstance(method_payload, dict):
        return None
    persona_payload = method_payload.get(persona)
    if not isinstance(persona_payload, dict):
        return None
    value = persona_payload.get("grouped_oof_recall_at_1pct_fpr")
    return float(value) if isinstance(value, (int, float)) else None


def _write_outputs(payload: Mapping[str, object]) -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    SCORES_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_summary_csv(payload, SUMMARY_PATH)
    REPORT_PATH.write_text(_render_report(payload), encoding="utf-8")
    print(f"Wrote {SCORES_PATH}")
    print(f"Wrote {SUMMARY_PATH}")
    print(f"Wrote {REPORT_PATH}")


def _write_summary_csv(payload: Mapping[str, object], path: Path) -> None:
    fieldnames = [
        "variant",
        "language",
        "method",
        "pooled_auroc",
        "macro_auroc",
        "oof_recall_1pct",
        "oof_fpr_1pct",
        "oof_recall_5pct",
        "oof_fpr_5pct",
        "positive_mean",
        "negative_mean",
        "mean_gap",
        "delta_auroc_vs_nn",
        "delta_macro_auroc_vs_nn",
        "delta_oof_recall_1pct_vs_nn",
        "delta_oof_recall_5pct_vs_nn",
    ]
    rows = _summary_rows(payload)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _summary_rows(payload: Mapping[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    standalone = payload["standalone"]
    for variant, languages in standalone.items():
        for language, methods in languages.items():
            nn = methods.get("nn", {})
            for method, metrics in methods.items():
                rows.append(_summary_row(variant, language, method, metrics, nn))
            clf_body = payload["classifier"][variant].get(language)
            if not isinstance(clf_body, dict) or "metrics" not in clf_body:
                continue
            rows.append(
                _summary_row(variant, language, CLASSIFIER_METHOD, clf_body["metrics"], nn)
            )
    return rows


def _summary_row(
    variant: str,
    language: str,
    method: str,
    metrics: Mapping[str, object],
    nn: Mapping[str, object],
) -> dict[str, object]:
    return {
        "variant": variant,
        "language": language,
        "method": method,
        "pooled_auroc": metrics.get("pooled_auroc"),
        "macro_auroc": metrics.get("macro_auroc"),
        "oof_recall_1pct": _oof_recall(metrics, "1%"),
        "oof_fpr_1pct": _oof_fpr(metrics, "1%"),
        "oof_recall_5pct": _oof_recall(metrics, "5%"),
        "oof_fpr_5pct": _oof_fpr(metrics, "5%"),
        "positive_mean": (metrics.get("positives") or {}).get("mean")
        if isinstance(metrics.get("positives"), dict)
        else None,
        "negative_mean": (metrics.get("negatives") or {}).get("mean")
        if isinstance(metrics.get("negatives"), dict)
        else None,
        "mean_gap": metrics.get("mean_gap"),
        "delta_auroc_vs_nn": _optional_delta(
            metrics.get("pooled_auroc") if isinstance(metrics.get("pooled_auroc"), float) else None,
            nn.get("pooled_auroc") if isinstance(nn.get("pooled_auroc"), float) else None,
        ),
        "delta_macro_auroc_vs_nn": _optional_delta(
            metrics.get("macro_auroc") if isinstance(metrics.get("macro_auroc"), float) else None,
            nn.get("macro_auroc") if isinstance(nn.get("macro_auroc"), float) else None,
        ),
        "delta_oof_recall_1pct_vs_nn": _recall_delta(metrics, nn, "1%"),
        "delta_oof_recall_5pct_vs_nn": _recall_delta(metrics, nn, "5%"),
    }


def _render_report(payload: Mapping[str, object]) -> str:
    population = payload["population"]
    verdict = payload["verdict"]
    timings = payload["timings"]
    lines = [
        "# Similarity-profile quick experiment (Phase B.2)",
        "",
        "Reversible cached-human evaluation. Production scoring was not modified.",
        "",
        f"- experiment: `{payload['experiment']}`",
        f"- evaluation_mode: `{payload['evaluation_mode']}`",
        f"- is_production: `{payload['is_production']}`",
        f"- is_final_all_human_evaluation: `{payload['is_final_all_human_evaluation']}`",
        f"- bootstrap_iterations: `{payload['bootstrap_iterations']}`",
        f"- reversible: `{payload['reversible']}`",
        f"- network_calls: `{payload['network_calls']}`",
        f"- embeddings_generated: `{payload['embeddings_generated']}`",
        f"- cross_model_status: `{payload['cross_model_status']}`",
        "",
        "## Population",
        "",
        f"- eligible pairs: {population['eligible_pairs']} "
        f"(CPP {population['eligible_pairs_cpp']}, "
        f"PYTHON {population['eligible_pairs_python']})",
        f"- pairs excluded (no cached human): {population['pairs_excluded_no_cached_human']}",
        f"- cached humans: {population['cached_humans']} "
        f"(CPP {population['cached_humans_cpp']}, PYTHON {population['cached_humans_python']})",
        f"- held-out AI: {population['held_out_ai']} "
        f"(CPP {population['held_out_ai_cpp']}, PYTHON {population['held_out_ai_python']})",
        f"- AI references: {population['ai_references']}",
        f"- same-cluster exact matches: {population['same_cluster_exact_matches']}",
        f"- uncached humans skipped: {population['missing_humans_skipped']}",
        "",
        "## Timings",
        "",
    ]
    for key, value in timings.items():
        lines.append(f"- {key}: {value:.3f}")
    if payload["cross_model_issue"]:
        lines.extend(
            [
                "",
                "## Cross-model metadata",
                "",
                f"Cross-model agreement was disabled: {payload['cross_model_issue']}",
            ]
        )
    lines.extend(["", "## Standalone methods", ""])
    lines.extend(_markdown_table(payload, "all_held_out", standalone_only=True))
    lines.extend(["", "## Grouped-OOF classifier", ""])
    lines.extend(_markdown_table(payload, "all_held_out", classifier_only=True))
    lines.extend(["", "## Exact-match-excluded comparison", ""])
    lines.extend(_markdown_table(payload, "exclude_exact_reference_match"))
    lines.extend(["", "## production_review", ""])
    lines.extend(_persona_lines(payload))
    lines.extend(["", "## Feature coefficients (mean across folds)", ""])
    lines.extend(_coefficient_lines(payload))
    lines.extend(
        [
            "",
            "## Verdict",
            "",
            f"**{verdict['label']}**",
            "",
            f"- Python exact-match-excluded OOF Recall@1% minus NN: "
            f"{_fmt(verdict['python_oof_recall_1pct_gain_vs_nn'])}",
            f"- Python achieved FPR@1%: {_fmt(verdict['python_oof_achieved_fpr_1pct'])}",
            f"- CPP OOF Recall@1% minus NN: {_fmt(verdict['cpp_oof_recall_1pct_gain_vs_nn'])}",
            f"- Python macro AUROC minus NN: {_fmt(verdict['python_macro_auroc_gain_vs_nn'])}",
            f"- production_review Recall@1% NN: {_fmt(verdict['production_review_recall_1pct_nn'])}",
            f"- production_review Recall@1% classifier: "
            f"{_fmt(verdict['production_review_recall_1pct_classifier'])}",
            f"- deployment note: {verdict['deployment_note']}",
            "",
            "Production `score_item` and `outputs/eval_scores.json` were not modified.",
            "",
        ]
    )
    return "\n".join(lines)


def _markdown_table(
    payload: Mapping[str, object],
    variant: str,
    standalone_only: bool = False,
    classifier_only: bool = False,
) -> list[str]:
    header = (
        "| language | method | AUROC | macro AUROC | OOF R@1% | FPR@1% | "
        "OOF R@5% | FPR@5% | pos mean | neg mean | gap | "
        "dAUROC vs NN | dMacro vs NN | dR@1% vs NN | dR@5% vs NN |"
    )
    sep = "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    lines = [header, sep]
    for row in _summary_rows(payload):
        if row["variant"] != variant:
            continue
        method = str(row["method"])
        if standalone_only and method == CLASSIFIER_METHOD:
            continue
        if classifier_only and method != CLASSIFIER_METHOD:
            continue
        lines.append(
            "| {language} | {method} | {pooled_auroc} | {macro_auroc} | "
            "{oof_recall_1pct} | {oof_fpr_1pct} | {oof_recall_5pct} | "
            "{oof_fpr_5pct} | {positive_mean} | {negative_mean} | {mean_gap} | "
            "{delta_auroc_vs_nn} | {delta_macro_auroc_vs_nn} | "
            "{delta_oof_recall_1pct_vs_nn} | {delta_oof_recall_5pct_vs_nn} |".format(
                **{key: _fmt(value) if key != "language" and key != "method" else value
                   for key, value in row.items()
                   if key != "variant"}
            )
        )
    return lines


def _persona_lines(payload: Mapping[str, object]) -> list[str]:
    lines = [
        "| variant | method | persona | n | OOF R@1% | OOF R@5% |",
        "|---|---|---|---:|---:|---:|",
    ]
    personas = payload["personas"]
    for variant, methods in personas.items():
        for method, persona_map in methods.items():
            for persona, stats in persona_map.items():
                lines.append(
                    f"| {variant} | {method} | {persona} | {stats['n_positives']} | "
                    f"{_fmt(stats['grouped_oof_recall_at_1pct_fpr'])} | "
                    f"{_fmt(stats['grouped_oof_recall_at_5pct_fpr'])} |"
                )
    clf = payload["classifier"]
    for variant, languages in clf.items():
        if variant not in {"all_held_out", "exclude_exact_reference_match"}:
            continue
        for language, body in languages.items():
            if language == "personas_all_languages":
                continue
            for persona, stats in body["personas"].items():
                lines.append(
                    f"| {variant} | {CLASSIFIER_METHOD}/{language} | {persona} | "
                    f"{stats['n_positives']} | "
                    f"{_fmt(stats['grouped_oof_recall_at_1pct_fpr'])} | "
                    f"{_fmt(stats['grouped_oof_recall_at_5pct_fpr'])} |"
                )
    return lines


def _coefficient_lines(payload: Mapping[str, object]) -> list[str]:
    lines = ["| language | feature | mean coefficient |", "|---|---|---:|"]
    all_held = payload["classifier"]["all_held_out"]
    for language, body in all_held.items():
        if language == "personas_all_languages":
            continue
        coeffs = sorted(
            body["mean_coefficients"].items(),
            key=lambda item: abs(item[1]),
            reverse=True,
        )
        for name, value in coeffs:
            lines.append(f"| {language} | {name} | {value:.6f} |")
    return lines


def _fmt(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    return sha256(path.read_bytes()).hexdigest()


def _print_timings(timings: Mapping[str, float]) -> None:
    print("Timing")
    for key, value in timings.items():
        print(f"  {key}: {value:.3f}s")


def _check_runtime(started: float, phase: str) -> None:
    elapsed = time.perf_counter() - started
    if elapsed > MAX_RUNTIME_SECONDS:
        raise RuntimeError(
            f"Experiment exceeded {MAX_RUNTIME_SECONDS}s during {phase} "
            f"({elapsed:.1f}s). Stopping instead of continuing."
        )


def _require_finite(array: np.ndarray, name: str) -> None:
    if not np.isfinite(array).all():
        raise RuntimeError(f"Non-finite values in {name}")


def _require_finite_mapping(values: Mapping[str, float], name: str) -> None:
    array = np.asarray(list(values.values()), dtype=np.float64)
    _require_finite(array, name)


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import csv
import json
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from collections.abc import Mapping, Sequence

import numpy as np

from nw_ai_code_detector.config import (
    AI_REFERENCE_EVALUATION_DIR,
    AI_SOLUTIONS_DIR,
    EMBEDDING_CACHE_DIR,
    MODEL_DATASET_DIR,
    OUTPUTS_DIR,
)
from nw_ai_code_detector.constants import (
    CANONICALITY_EMBEDDING_DIMENSION,
    CANONICALITY_SPLIT_SEED,
    DatasetSplit,
)
from nw_ai_code_detector.evaluation.experiment_metrics import (
    conservative_threshold,
    macro_auroc_by_question,
    pooled_auroc,
    score_statistics,
)
from nw_ai_code_detector.index import ClusterKey

REFERENCE_INDEX_DIR = OUTPUTS_DIR / "reference_index"
EXPECTED_REFERENCE_COUNT = 6
BOOTSTRAP_ITERATIONS = 1000
BOOTSTRAP_SEED = 500
TARGET_FPRS = (0.01, 0.05)
METHOD_MAX = "ai_nn_max"
METHOD_TOP3 = "ai_top3_mean"
METHODS = (METHOD_MAX, METHOD_TOP3)


@dataclass(frozen=True)
class ManifestRow:
    record_id: str
    question_id: str
    language: str
    split: str
    label: int
    source: str
    generator: str | None
    persona: str | None
    stripped_hash: str
    embedding_cache_key: str
    duplicate_group: str
    sample_weight: float


@dataclass(frozen=True)
class ReferenceCluster:
    key: ClusterKey
    vectors: np.ndarray
    distinct_vectors: np.ndarray
    stripped_hashes: tuple[str, ...]


@dataclass(frozen=True)
class ScoredRow:
    record: ManifestRow
    ai_nn_max: float
    ai_top3_mean: float
    logical_top3_mean: float
    exact_match_to_reference: bool


@dataclass(frozen=True)
class ThresholdSet:
    method: str
    language: str
    target_fpr: float
    threshold: float
    train_achieved_fpr: float


@dataclass(frozen=True)
class MetricScope:
    split: str
    language: str
    generator: str | None
    persona: str | None
    exact_match_excluded: bool


@dataclass(frozen=True)
class BootstrapTarget:
    name: str
    language: str
    target_fpr: float | None
    generator: str | None = None
    persona: str | None = None
    exact_match_excluded: bool = False


@dataclass(frozen=True)
class EvaluationResult:
    scored: Sequence[ScoredRow]
    metric_rows: Sequence[ScoredRow]
    clusters: Mapping[ClusterKey, ReferenceCluster]
    thresholds: Sequence[ThresholdSet]
    metrics: Mapping[str, object]
    bootstrap: Mapping[str, object]
    recommendation: str


@dataclass(frozen=True)
class OperatingPointInput:
    rows: Sequence[ScoredRow]
    thresholds: Sequence[ThresholdSet]
    method: str
    target_fpr: float
    language: str


def main() -> int:
    started = time.perf_counter()
    before = snapshot_protected_artifacts()
    clusters = load_reference_clusters()
    rows = load_evaluation_rows()
    scored = score_rows(rows, clusters)
    metric_rows = _deduplicate_metric_rows(scored)
    thresholds = derive_training_thresholds(metric_rows)
    metrics = evaluate_scopes(
        metric_rows,
        thresholds,
        (DatasetSplit.TRAIN, DatasetSplit.VALIDATION),
    )
    bootstrap = bootstrap_validation_deltas(metric_rows, thresholds)
    recommendation = recommend_method(metrics, bootstrap)
    metrics.update(
        evaluate_scopes(
            metric_rows,
            thresholds,
            (DatasetSplit.INTERNAL_TEST,),
        )
    )
    result = EvaluationResult(
        scored=scored,
        metric_rows=metric_rows,
        clusters=clusters,
        thresholds=thresholds,
        metrics=metrics,
        bootstrap=bootstrap,
        recommendation=recommendation,
    )
    payload = build_payload(
        result,
        time.perf_counter() - started,
    )
    write_outputs(payload)
    after = snapshot_protected_artifacts()
    if before != after:
        raise RuntimeError("Question split or mixed-v1 artifacts changed")
    print(f"Recommended AI-reference score: {recommendation}")
    print(f"Wrote evaluation to {AI_REFERENCE_EVALUATION_DIR}")
    return 0


def load_reference_clusters() -> dict[ClusterKey, ReferenceCluster]:
    metadata = _load_reference_metadata()
    manifest = json.loads(
        (REFERENCE_INDEX_DIR / "manifest.json").read_text(encoding="utf-8")
    )
    cluster_manifest = manifest.get("clusters")
    if not isinstance(cluster_manifest, dict) or len(cluster_manifest) != 1000:
        raise RuntimeError("Mixed-v1 manifest must contain 1000 clusters")
    clusters: dict[ClusterKey, ReferenceCluster] = {}
    for token, item in sorted(cluster_manifest.items()):
        if not isinstance(item, dict):
            raise RuntimeError(f"Invalid mixed-v1 manifest entry: {token}")
        question_id = str(item["question_id"])
        language = str(item["language"])
        key = ClusterKey(question_id, language)
        vector_path = REFERENCE_INDEX_DIR / f"{token.replace(':', '__')}.npy"
        vectors = np.asarray(np.load(vector_path), dtype=np.float32)
        hashes = metadata.get(key)
        if hashes is None:
            raise RuntimeError(f"Missing mixed-v1 metadata for {token}")
        _validate_reference_cluster(key, vectors, hashes)
        distinct_indexes = _first_distinct_indexes(hashes)
        distinct = np.asarray(vectors[distinct_indexes], dtype=np.float32)
        clusters[key] = ReferenceCluster(key, vectors, distinct, hashes)
    return clusters


def load_evaluation_rows() -> list[ManifestRow]:
    rows: list[ManifestRow] = []
    for split in DatasetSplit:
        path = MODEL_DATASET_DIR / f"{split.value}_manifest.jsonl"
        rows.extend(_load_manifest(path))
    if any(item.source == "mixed_v1_reference" for item in rows):
        raise RuntimeError("Mixed-v1 references cannot be labeled rows")
    invalid_gpt = [
        item
        for item in rows
        if item.source == "gpt_heavy_extra"
        and item.split != DatasetSplit.TRAIN.value
    ]
    if invalid_gpt:
        raise RuntimeError("GPT-heavy augmentation leaked outside training")
    return rows


def resolve_reference_cluster(
    row: ManifestRow,
    clusters: Mapping[ClusterKey, ReferenceCluster],
) -> ReferenceCluster:
    key = ClusterKey(row.question_id, row.language)
    cluster = clusters.get(key)
    if cluster is None:
        raise RuntimeError(f"Missing exact reference cluster for {key.token}")
    if cluster.key.question_id != row.question_id:
        raise RuntimeError("Cross-question reference routing")
    if cluster.key.language != row.language:
        raise RuntimeError("Cross-language reference routing")
    return cluster


def score_candidate(
    query: Sequence[float],
    cluster: ReferenceCluster,
) -> tuple[float, float, float]:
    query_vector = np.asarray(query, dtype=np.float32)
    _validate_query_vector(query_vector)
    logical = np.asarray(cluster.vectors @ query_vector, dtype=np.float64)
    distinct = np.asarray(cluster.distinct_vectors @ query_vector, dtype=np.float64)
    if not np.isfinite(logical).all() or not np.isfinite(distinct).all():
        raise RuntimeError("Reference similarities must be finite")
    maximum = float(np.max(logical))
    top_count = min(3, len(distinct))
    top3 = float(np.mean(np.sort(distinct)[-top_count:]))
    logical_top3 = float(np.mean(np.sort(logical)[-3:]))
    return maximum, top3, logical_top3


def score_rows(
    rows: Sequence[ManifestRow],
    clusters: Mapping[ClusterKey, ReferenceCluster],
) -> list[ScoredRow]:
    scored: list[ScoredRow] = []
    for row in rows:
        vector = _load_cached_vector(row.embedding_cache_key)
        cluster = resolve_reference_cluster(row, clusters)
        maximum, top3, logical_top3 = score_candidate(vector, cluster)
        scored.append(
            ScoredRow(
                record=row,
                ai_nn_max=maximum,
                ai_top3_mean=top3,
                logical_top3_mean=logical_top3,
                exact_match_to_reference=row.stripped_hash in cluster.stripped_hashes,
            )
        )
    return scored


def derive_training_thresholds(
    rows: Sequence[ScoredRow],
) -> list[ThresholdSet]:
    train_humans = [
        item
        for item in rows
        if item.record.split == DatasetSplit.TRAIN.value
        and item.record.source == "human"
    ]
    thresholds: list[ThresholdSet] = []
    for method in METHODS:
        for language in ("CPP", "PYTHON"):
            negatives = [
                _method_score(item, method)
                for item in train_humans
                if item.record.language == language
            ]
            if not negatives:
                raise RuntimeError(f"No training humans for {language}")
            for target_fpr in TARGET_FPRS:
                threshold = conservative_threshold(negatives, target_fpr)
                achieved = float(np.mean(np.asarray(negatives) >= threshold))
                thresholds.append(
                    ThresholdSet(
                        method,
                        language,
                        target_fpr,
                        threshold,
                        achieved,
                    )
                )
    return thresholds


def evaluate_scopes(
    rows: Sequence[ScoredRow],
    thresholds: Sequence[ThresholdSet],
    splits: Sequence[DatasetSplit],
) -> dict[str, object]:
    payload: dict[str, object] = {}
    for split in splits:
        split_rows = [item for item in rows if item.record.split == split.value]
        split_metrics: dict[str, object] = {}
        for language in ("CPP", "PYTHON", "COMBINED"):
            scope = MetricScope(split.value, language, None, None, False)
            split_metrics[language] = _scope_metrics(split_rows, thresholds, scope)
        split_metrics["generators"] = {
            generator: _scope_metrics(
                split_rows,
                thresholds,
                MetricScope(split.value, "CPP", generator, None, False),
            )
            for generator in sorted(
                {
                    item.record.generator
                    for item in split_rows
                    if item.record.label == 1 and item.record.generator
                }
            )
        }
        split_metrics["personas"] = {
            persona: _scope_metrics(
                split_rows,
                thresholds,
                MetricScope(split.value, "CPP", None, persona, False),
            )
            for persona in sorted(
                {
                    item.record.persona
                    for item in split_rows
                    if item.record.label == 1 and item.record.persona
                }
            )
        }
        split_metrics["exact_match_excluded_CPP"] = _scope_metrics(
            split_rows,
            thresholds,
            MetricScope(split.value, "CPP", None, None, True),
        )
        payload[split.value] = split_metrics
    if DatasetSplit.TRAIN in splits:
        payload["human_reference_exact_matches"] = _hard_negative_report(rows)
        payload["gpt_heavy_train_distribution"] = _gpt_distribution(rows)
    return payload


def bootstrap_validation_deltas(
    rows: Sequence[ScoredRow],
    thresholds: Sequence[ThresholdSet],
) -> dict[str, dict[str, float | int | None]]:
    validation = [
        item
        for item in rows
        if item.record.split == DatasetSplit.VALIDATION.value
    ]
    targets = (
        BootstrapTarget("CPP Recall@1%", "CPP", 0.01),
        BootstrapTarget("CPP Recall@5%", "CPP", 0.05),
        BootstrapTarget("CPP AUROC", "CPP", None),
        BootstrapTarget("Python Recall@1%", "PYTHON", 0.01),
        BootstrapTarget("GPT-held-out CPP Recall@1%", "CPP", 0.01, "openai/gpt-5.5"),
        BootstrapTarget(
            "Gemini-held-out CPP Recall@1%",
            "CPP",
            0.01,
            "google/gemini-3.7-flash",
        ),
        BootstrapTarget(
            "DeepSeek-held-out CPP Recall@1%",
            "CPP",
            0.01,
            "deepseek/deepseek-v4-pro",
        ),
        BootstrapTarget(
            "production_review CPP Recall@1%",
            "CPP",
            0.01,
            persona="production_review",
        ),
        BootstrapTarget(
            "exact-match-excluded CPP Recall@1%",
            "CPP",
            0.01,
            exact_match_excluded=True,
        ),
    )
    question_groups: dict[str, list[ScoredRow]] = defaultdict(list)
    for item in validation:
        question_groups[item.record.question_id].append(item)
    question_ids = tuple(sorted(question_groups))
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    deltas = {target.name: [] for target in targets}
    for _iteration in range(BOOTSTRAP_ITERATIONS):
        sampled_ids = rng.choice(question_ids, size=len(question_ids), replace=True)
        sample = [
            item
            for question_id in sampled_ids
            for item in question_groups[str(question_id)]
        ]
        for target in targets:
            delta = _bootstrap_delta(sample, thresholds, target)
            if delta is not None:
                deltas[target.name].append(delta)
    return {
        target.name: _confidence_interval(deltas[target.name])
        for target in targets
    }


def recommend_method(
    metrics: Mapping[str, object],
    bootstrap: Mapping[str, Mapping[str, float | int | None]],
) -> str:
    validation = metrics[DatasetSplit.VALIDATION.value]
    if not isinstance(validation, dict):
        return METHOD_MAX
    exact = validation["exact_match_excluded_CPP"]
    cpp = validation["CPP"]
    personas = validation["personas"]
    generators = validation["generators"]
    if not all(isinstance(item, dict) for item in (exact, cpp, personas, generators)):
        return METHOD_MAX
    exact_gain = _recall_gain(exact, "1%")
    cpp_gain = _recall_gain(cpp, "1%")
    production_gain = _recall_gain(personas.get("production_review", {}), "1%")
    generator_gains = [
        _recall_gain(generators.get(name, {}), "1%")
        for name in ("google/gemini-3.7-flash", "deepseek/deepseek-v4-pro")
    ]
    ci_low = bootstrap["exact-match-excluded CPP Recall@1%"]["low"]
    top3_fpr = _metric_value(exact, METHOD_TOP3, "1%", "achieved_fpr")
    if not isinstance(ci_low, float) or ci_low <= 0:
        return METHOD_MAX
    if exact_gain <= 0.01 or cpp_gain < 0 or production_gain < -0.02:
        return METHOD_MAX
    if any(gain < -0.03 for gain in generator_gains):
        return METHOD_MAX
    if top3_fpr is None or top3_fpr > 0.011:
        return METHOD_MAX
    return METHOD_TOP3


def build_payload(
    result: EvaluationResult,
    runtime_seconds: float,
) -> dict[str, object]:
    changed = sum(
        not np.isclose(item.ai_top3_mean, item.logical_top3_mean)
        for item in result.metric_rows
    )
    largest_change = max(
        (
            abs(item.ai_top3_mean - item.logical_top3_mean)
            for item in result.metric_rows
        ),
        default=0.0,
    )
    sparse = [
        key.token
        for key, cluster in result.clusters.items()
        if len(cluster.distinct_vectors) < 3
    ]
    return {
        "experiment": "ai_reference_scores_v2",
        "rows": {
            "logical_labeled": len(result.scored),
            "metric_rows_after_duplicate_control": len(result.metric_rows),
            "by_split": dict(
                Counter(item.record.split for item in result.metric_rows)
            ),
        },
        "reference_audit": {
            "clusters": len(result.clusters),
            "logical_vectors": sum(
                len(item.vectors) for item in result.clusters.values()
            ),
            "distinct_pair_hash_vectors": sum(
                len(item.distinct_vectors)
                for item in result.clusters.values()
            ),
            "clusters_with_fewer_than_three_distinct": sparse,
            "rows_where_logical_top3_differs_from_distinct_top3": changed,
            "maximum_logical_vs_distinct_top3_change": largest_change,
            "max_nn_changes_from_deduplication": 0,
        },
        "thresholds_from_training_humans_only": [
            asdict(item) for item in result.thresholds
        ],
        "metrics": result.metrics,
        "bootstrap": {
            "iterations": BOOTSTRAP_ITERATIONS,
            "seed": BOOTSTRAP_SEED,
            "population": "validation question-clustered",
            "top3_mean_minus_max_nn": result.bootstrap,
        },
        "recommendation": result.recommendation,
        "runtime_seconds": runtime_seconds,
    }


def write_outputs(payload: Mapping[str, object]) -> None:
    AI_REFERENCE_EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    _write_json(AI_REFERENCE_EVALUATION_DIR / "scores.json", payload)
    _write_summary_csv(payload)
    (AI_REFERENCE_EVALUATION_DIR / "report.md").write_text(
        _report_markdown(payload),
        encoding="utf-8",
    )


def snapshot_protected_artifacts() -> dict[str, str]:
    split = MODEL_DATASET_DIR / "question_split.json"
    return {
        "question_split": _file_sha256(split),
        "mixed_index": _directory_sha256(REFERENCE_INDEX_DIR),
        "mixed_solutions": _directory_sha256(AI_SOLUTIONS_DIR),
    }


def _load_reference_metadata() -> dict[ClusterKey, tuple[str, ...]]:
    grouped: dict[ClusterKey, list[str]] = defaultdict(list)
    for path in sorted(AI_SOLUTIONS_DIR.rglob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        text = str(payload.get("stripped_code") or "")
        key = ClusterKey(str(payload.get("qid") or ""), str(payload.get("language") or ""))
        grouped[key].append(_text_hash(text))
    return {key: tuple(values) for key, values in grouped.items()}


def _validate_reference_cluster(
    key: ClusterKey,
    vectors: np.ndarray,
    hashes: Sequence[str],
) -> None:
    if vectors.shape != (EXPECTED_REFERENCE_COUNT, CANONICALITY_EMBEDDING_DIMENSION):
        raise RuntimeError(f"Invalid reference vector shape for {key.token}: {vectors.shape}")
    if len(hashes) != EXPECTED_REFERENCE_COUNT:
        raise RuntimeError(f"Invalid reference metadata count for {key.token}")
    if vectors.dtype != np.float32 or not np.isfinite(vectors).all():
        raise RuntimeError(f"Invalid reference vectors for {key.token}")
    norms = np.linalg.norm(vectors, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-3):
        raise RuntimeError(f"Reference vectors are not normalized for {key.token}")


def _first_distinct_indexes(hashes: Sequence[str]) -> list[int]:
    seen: set[str] = set()
    indexes: list[int] = []
    for index, stripped_hash in enumerate(hashes):
        if stripped_hash in seen:
            continue
        seen.add(stripped_hash)
        indexes.append(index)
    return indexes


def _load_manifest(path: Path) -> list[ManifestRow]:
    return [
        ManifestRow(**json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _load_cached_vector(cache_key: str) -> np.ndarray:
    path = EMBEDDING_CACHE_DIR / f"{cache_key}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    vector = np.asarray(payload.get("vector"), dtype=np.float32)
    _validate_query_vector(vector)
    return vector


def _validate_query_vector(vector: np.ndarray) -> None:
    if vector.shape != (CANONICALITY_EMBEDDING_DIMENSION,):
        raise RuntimeError(f"Expected 1024-dimensional query, got {vector.shape}")
    if vector.dtype != np.float32 or not np.isfinite(vector).all():
        raise RuntimeError("Query vector must be finite float32")
    if not np.isclose(np.linalg.norm(vector), 1.0, atol=1e-3):
        raise RuntimeError("Query vector must be unit normalized")


def _deduplicate_metric_rows(rows: Sequence[ScoredRow]) -> list[ScoredRow]:
    seen: set[tuple[str, str, str, str]] = set()
    selected: list[ScoredRow] = []
    for item in rows:
        key = (
            item.record.source,
            item.record.question_id,
            item.record.language,
            item.record.stripped_hash,
        )
        if key in seen:
            continue
        seen.add(key)
        selected.append(item)
    return selected


def _scope_metrics(
    rows: Sequence[ScoredRow],
    thresholds: Sequence[ThresholdSet],
    scope: MetricScope,
) -> dict[str, object]:
    scoped = _filter_scope(rows, scope)
    result: dict[str, object] = {}
    for method in METHODS:
        positives = [
            _method_score(item, method) for item in scoped if item.record.label == 1
        ]
        negatives = [
            _method_score(item, method) for item in scoped if item.record.label == 0
        ]
        labeled = _labeled_scores(scoped, method)
        macro, macro_count = macro_auroc_by_question(labeled)
        result[method] = {
            "population": len(scoped),
            "n_positives": len(positives),
            "n_negatives": len(negatives),
            "auroc": pooled_auroc(positives, negatives),
            "macro_auroc": macro,
            "macro_question_language_pairs": macro_count,
            "positive_mean": score_statistics(positives).mean,
            "negative_mean": score_statistics(negatives).mean,
            "mean_gap": _mean_gap(positives, negatives),
            "operating_points": {
                f"{int(target * 100)}%": _fixed_operating_point(
                    OperatingPointInput(
                        scoped,
                        thresholds,
                        method,
                        target,
                        scope.language,
                    )
                )
                for target in TARGET_FPRS
            },
        }
    return result


def _filter_scope(
    rows: Sequence[ScoredRow],
    scope: MetricScope,
) -> list[ScoredRow]:
    negatives = [item for item in rows if item.record.label == 0]
    positives = [item for item in rows if item.record.label == 1]
    if scope.language != "COMBINED":
        negatives = [
            item for item in negatives if item.record.language == scope.language
        ]
        positives = [
            item for item in positives if item.record.language == scope.language
        ]
    if scope.generator:
        positives = [
            item for item in positives if item.record.generator == scope.generator
        ]
    if scope.persona:
        positives = [
            item for item in positives if item.record.persona == scope.persona
        ]
    if scope.exact_match_excluded:
        positives = [
            item for item in positives if not item.exact_match_to_reference
        ]
    return [*negatives, *positives]


def _fixed_operating_point(
    item: OperatingPointInput,
) -> dict[str, object]:
    per_language = {
        threshold.language: threshold
        for threshold in item.thresholds
        if threshold.method == item.method
        and threshold.target_fpr == item.target_fpr
    }
    positive_hits: list[bool] = []
    negative_hits: list[bool] = []
    for row in item.rows:
        threshold = per_language[row.record.language].threshold
        hit = _method_score(row, item.method) >= threshold
        if row.record.label == 1:
            positive_hits.append(hit)
        else:
            negative_hits.append(hit)
    threshold_payload: object
    if item.language == "COMBINED":
        threshold_payload = {
            key: value.threshold for key, value in per_language.items()
        }
    else:
        threshold_payload = per_language[item.language].threshold
    return {
        "threshold": threshold_payload,
        "recall": float(np.mean(positive_hits)) if positive_hits else None,
        "achieved_fpr": float(np.mean(negative_hits)) if negative_hits else None,
    }


def _labeled_scores(rows: Sequence[ScoredRow], method: str) -> list[object]:
    from nw_ai_code_detector.evaluation.experiment_metrics import LabeledScore

    return [
        LabeledScore(
            question_id=item.record.question_id,
            pair_token=f"{item.record.question_id}:{item.record.language}",
            language=item.record.language,
            label="ai" if item.record.label == 1 else "human",
            score=_method_score(item, method),
        )
        for item in rows
    ]


def _method_score(row: ScoredRow, method: str) -> float:
    if method == METHOD_MAX:
        return row.ai_nn_max
    if method == METHOD_TOP3:
        return row.ai_top3_mean
    raise ValueError(f"Unknown method: {method}")


def _mean_gap(positives: Sequence[float], negatives: Sequence[float]) -> float | None:
    if not positives or not negatives:
        return None
    return float(np.mean(positives) - np.mean(negatives))


def _hard_negative_report(rows: Sequence[ScoredRow]) -> dict[str, object]:
    exact = [
        item
        for item in rows
        if item.record.label == 0 and item.exact_match_to_reference
    ]
    return {
        "logical_count": len(exact),
        "distinct_pair_hash_count": len(
            {
                (
                    item.record.question_id,
                    item.record.language,
                    item.record.stripped_hash,
                )
                for item in exact
            }
        ),
        "kept_in_metrics": True,
    }


def _gpt_distribution(rows: Sequence[ScoredRow]) -> dict[str, object]:
    gpt = [item for item in rows if item.record.source == "gpt_heavy_extra"]
    return {
        method: asdict(score_statistics([_method_score(item, method) for item in gpt]))
        for method in METHODS
    }


def _bootstrap_delta(
    rows: Sequence[ScoredRow],
    thresholds: Sequence[ThresholdSet],
    target: BootstrapTarget,
) -> float | None:
    scope = MetricScope(
        DatasetSplit.VALIDATION.value,
        target.language,
        target.generator,
        target.persona,
        target.exact_match_excluded,
    )
    scoped = _filter_scope(rows, scope)
    if target.target_fpr is None:
        top3 = _bootstrap_auroc(scoped, METHOD_TOP3)
        maximum = _bootstrap_auroc(scoped, METHOD_MAX)
    else:
        top3 = _bootstrap_recall(
            scoped,
            thresholds,
            METHOD_TOP3,
            target.target_fpr,
        )
        maximum = _bootstrap_recall(
            scoped,
            thresholds,
            METHOD_MAX,
            target.target_fpr,
        )
    if top3 is None or maximum is None:
        return None
    return top3 - maximum


def _bootstrap_recall(
    rows: Sequence[ScoredRow],
    thresholds: Sequence[ThresholdSet],
    method: str,
    target_fpr: float,
) -> float | None:
    threshold_by_language = {
        item.language: item.threshold
        for item in thresholds
        if item.method == method and item.target_fpr == target_fpr
    }
    positives = [item for item in rows if item.record.label == 1]
    if not positives:
        return None
    hits = [
        _method_score(item, method) >= threshold_by_language[item.record.language]
        for item in positives
    ]
    return float(np.mean(hits))


def _bootstrap_auroc(rows: Sequence[ScoredRow], method: str) -> float | None:
    positives = [
        _method_score(item, method) for item in rows if item.record.label == 1
    ]
    negatives = [
        _method_score(item, method) for item in rows if item.record.label == 0
    ]
    return pooled_auroc(positives, negatives)


def _confidence_interval(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {"samples": 0, "mean": None, "low": None, "high": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "samples": len(array),
        "mean": float(np.mean(array)),
        "low": float(np.percentile(array, 2.5)),
        "high": float(np.percentile(array, 97.5)),
    }


def _recall_gain(metrics: object, point: str) -> float:
    if not isinstance(metrics, dict):
        return float("-inf")
    top3 = _metric_value(metrics, METHOD_TOP3, point, "recall")
    maximum = _metric_value(metrics, METHOD_MAX, point, "recall")
    if top3 is None or maximum is None:
        return float("-inf")
    return top3 - maximum


def _metric_value(
    metrics: Mapping[str, object],
    method: str,
    point: str,
    field: str,
) -> float | None:
    block = metrics.get(method)
    if not isinstance(block, dict):
        return None
    operating = block.get("operating_points")
    if not isinstance(operating, dict):
        return None
    point_block = operating.get(point)
    if not isinstance(point_block, dict):
        return None
    value = point_block.get(field)
    return float(value) if isinstance(value, (float, int)) else None


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_summary_csv(payload: Mapping[str, object]) -> None:
    rows: list[dict[str, object]] = []
    metrics = payload["metrics"]
    if isinstance(metrics, dict):
        for split in DatasetSplit:
            split_metrics = metrics.get(split.value)
            if not isinstance(split_metrics, dict):
                continue
            for scope_name in ("CPP", "PYTHON", "COMBINED", "exact_match_excluded_CPP"):
                scope = split_metrics.get(scope_name)
                if not isinstance(scope, dict):
                    continue
                rows.extend(_summary_rows(split.value, scope_name, scope))
    path = AI_REFERENCE_EVALUATION_DIR / "summary.csv"
    with path.open("w", encoding="utf-8", newline="") as stream:
        fieldnames = (
            "split",
            "scope",
            "method",
            "auroc",
            "macro_auroc",
            "recall_1pct",
            "fpr_1pct",
            "recall_5pct",
            "fpr_5pct",
            "positive_mean",
            "negative_mean",
            "mean_gap",
        )
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _summary_rows(
    split: str,
    scope: str,
    metrics: Mapping[str, object],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for method in METHODS:
        item = metrics.get(method)
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "split": split,
                "scope": scope,
                "method": method,
                "auroc": item.get("auroc"),
                "macro_auroc": item.get("macro_auroc"),
                "recall_1pct": _metric_value(metrics, method, "1%", "recall"),
                "fpr_1pct": _metric_value(metrics, method, "1%", "achieved_fpr"),
                "recall_5pct": _metric_value(metrics, method, "5%", "recall"),
                "fpr_5pct": _metric_value(metrics, method, "5%", "achieved_fpr"),
                "positive_mean": item.get("positive_mean"),
                "negative_mean": item.get("negative_mean"),
                "mean_gap": item.get("mean_gap"),
            }
        )
    return rows


def _report_markdown(payload: Mapping[str, object]) -> str:
    return "\n".join(
        [
            "# AI-reference scores v2",
            "",
            f"Recommendation: `{payload['recommendation']}`.",
            "",
            "Compared maximum mixed-v1 similarity with the mean of the top three "
            "distinct mixed-v1 reference hashes. Thresholds were derived only from "
            "training-split humans. GPT-heavy rows were training augmentation only.",
            "",
            f"Reference audit: {payload['reference_audit']}",
            "",
            f"Bootstrap: {payload['bootstrap']}",
            "",
        ]
    )


def _text_hash(text: str) -> str:
    from hashlib import sha256

    return sha256(text.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    from hashlib import sha256

    return sha256(path.read_bytes()).hexdigest()


def _directory_sha256(directory: Path) -> str:
    from hashlib import sha256

    digest = sha256()
    for path in sorted(item for item in directory.rglob("*") if item.is_file()):
        digest.update(path.relative_to(directory).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())

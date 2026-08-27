from __future__ import annotations

import csv
import json
import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from collections.abc import Mapping, Sequence

import numpy as np

from nw_ai_code_detector.config import (
    CENTROID_CACHED_HUMANS_SCORES_PATH,
    EVAL_SCORES_PATH,
    OUTPUTS_DIR,
    SELECTED_500_PATH,
)
from nw_ai_code_detector.constants import EvaluationMode, SELECTION_RANDOM_SEED
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
    _cluster_key_for_record,
    _collect_records,
    _held_out_reference_match_keys,
    _load_cached_vectors,
    _pair_token,
    _prepare_scoring_records,
    _validate_vectors,
)
from nw_ai_code_detector.evaluation.evaluate_similarity_profile_quick import (
    centroid_vector,
)
from nw_ai_code_detector.evaluation.experiment_metrics import (
    LabeledScore,
    conservative_threshold,
    method_metrics_bundle,
    method_metrics_to_dict,
    question_folds,
)
from nw_ai_code_detector.generate_ai_refs import _limit_questions, _load_selected_questions
from nw_ai_code_detector.index import ClusterKey

EXPERIMENT_NAME = "ai_human_margin_quick"
FIXED_SEED = SELECTION_RANDOM_SEED
MIN_DISTINCT_HUMANS = 2
DISTANCE_FLOOR = 1e-6
MAX_RUNTIME_SECONDS = 600
CPP_RECALL_GAIN_MIN = 0.05
MATERIAL_FPR_LIMIT = 0.015
MATERIAL_R5_DROP = 0.03
MACRO_DROP_MAX = 0.01
AFFINITY_TOLERANCE = 1e-3
PHASE_B2_SCORES_PATH = OUTPUTS_DIR / "similarity_profile_quick_scores.json"
SCORES_PATH = OUTPUTS_DIR / "ai_human_margin_quick_scores.json"
SUMMARY_PATH = OUTPUTS_DIR / "ai_human_margin_quick_summary.csv"
REPORT_PATH = OUTPUTS_DIR / "ai_human_margin_quick_report.md"
PERSONAS = ("production_review", "pair_programming")
METHODS = (
    "matched_population_nn",
    "ai_nn_minus_human_nn",
    "ai_top3_minus_human_nn",
    "ai_centroid_minus_human_centroid",
    "log_distance_ratio",
    "rank_relative",
)
PRIMARY_MARGIN = "ai_nn_minus_human_nn"


@dataclass(frozen=True)
class VectorItem:
    record: ExperimentRecord
    vector: np.ndarray


@dataclass(frozen=True)
class AiBank:
    key: ClusterKey
    vectors: np.ndarray
    hashes: tuple[str, ...]
    centroid: np.ndarray


@dataclass(frozen=True)
class HumanFolds:
    key: ClusterKey
    fold_0: tuple[VectorItem, ...]
    fold_1: tuple[VectorItem, ...]
    duplicates_removed: int


@dataclass(frozen=True)
class PairLookup:
    record: ExperimentRecord
    cluster_key: ClusterKey
    banks: Mapping[ClusterKey, object]


@dataclass(frozen=True)
class AiAffinity:
    nn: float
    top3_mean: float
    centroid: float


@dataclass(frozen=True)
class HumanAffinity:
    nn: float
    mean: float
    centroid: float


@dataclass(frozen=True)
class ContrastScores:
    nn_margin: float
    top3_margin: float
    centroid_contrast: float
    log_distance_ratio: float
    rank_relative: float


@dataclass(frozen=True)
class ScoredRow:
    question_id: str
    language: str
    label: str
    source: str
    content_hash: str
    human_reference_count: int
    ai_affinity: AiAffinity
    human_affinity: HumanAffinity
    contrast: ContrastScores


@dataclass(frozen=True)
class MarginCoverage:
    eligible_pairs: tuple[tuple[str, str], ...]
    excluded_few_humans: tuple[tuple[str, str], ...]
    duplicates_removed: int
    human_count_distribution: dict[str, int]
    skipped_no_human_refs: int


@dataclass(frozen=True)
class MarginPopulation:
    items: tuple[VectorItem, ...]
    ai_banks: dict[ClusterKey, AiBank]
    human_folds: dict[ClusterKey, HumanFolds]
    coverage: MarginCoverage


def resolve_ai_bank(lookup: PairLookup) -> AiBank:
    _assert_same_cluster(lookup.record, lookup.cluster_key)
    bank = lookup.banks.get(lookup.cluster_key)
    if not isinstance(bank, AiBank):
        raise RuntimeError(
            "Missing same-question same-language AI bank for "
            f"{lookup.record.question_id}:{lookup.record.language}"
        )
    _assert_same_cluster(lookup.record, bank.key)
    return bank


def resolve_human_folds(lookup: PairLookup) -> HumanFolds:
    _assert_same_cluster(lookup.record, lookup.cluster_key)
    folds = lookup.banks.get(lookup.cluster_key)
    if not isinstance(folds, HumanFolds):
        raise RuntimeError(
            "Missing same-question same-language human folds for "
            f"{lookup.record.question_id}:{lookup.record.language}"
        )
    _assert_same_cluster(lookup.record, folds.key)
    return folds


def split_human_folds(items: Sequence[VectorItem]) -> HumanFolds | None:
    if not items:
        return None
    key = _cluster_key_for_record(items[0].record)
    for item in items:
        _assert_same_cluster(item.record, key)
    unique, duplicates = _deduplicated_humans(items)
    if len(unique) < MIN_DISTINCT_HUMANS:
        return None
    fold_0 = tuple(unique[index] for index in range(0, len(unique), 2))
    fold_1 = tuple(unique[index] for index in range(1, len(unique), 2))
    if not fold_0 or not fold_1:
        raise RuntimeError(
            f"Both human folds must be non-empty for {key.question_id}:{key.language}"
        )
    return HumanFolds(key, fold_0, fold_1, duplicates)


def opposite_human_refs(
    folds: HumanFolds,
    eval_item: VectorItem,
    eval_fold: int,
) -> tuple[VectorItem, ...]:
    _assert_same_cluster(eval_item.record, folds.key)
    pool = folds.fold_0 if eval_fold == 1 else folds.fold_1
    refs = tuple(
        item
        for item in pool
        if item.record.content_hash != eval_item.record.content_hash
        and id(item.record) != id(eval_item.record)
    )
    return refs


def ai_affinity(query: np.ndarray, bank: AiBank) -> AiAffinity:
    _require_finite(query, "ai_query")
    _require_finite(bank.vectors, "ai_bank")
    if bank.vectors.shape[0] != EXPECTED_REFERENCE_COUNT:
        raise RuntimeError("AI affinity requires six same-pair references")
    similarities = np.asarray(bank.vectors @ query, dtype=np.float64)
    ranked = np.sort(similarities)[::-1]
    affinity = AiAffinity(
        nn=float(ranked[0]),
        top3_mean=float(np.mean(ranked[:3])),
        centroid=float(bank.centroid @ query),
    )
    _require_finite(
        np.asarray([affinity.nn, affinity.top3_mean, affinity.centroid]),
        "ai_affinity",
    )
    return affinity


def human_affinity(query: np.ndarray, refs: Sequence[VectorItem]) -> HumanAffinity:
    if not refs:
        raise RuntimeError("Human affinity requires at least one opposite-fold reference")
    matrix = np.vstack([item.vector for item in refs])
    _require_finite(query, "human_query")
    _require_finite(matrix, "human_refs")
    similarities = np.asarray(matrix @ query, dtype=np.float64)
    centroid = centroid_vector(matrix)
    affinity = HumanAffinity(
        nn=float(np.max(similarities)),
        mean=float(np.mean(similarities)),
        centroid=float(centroid @ query),
    )
    _require_finite(
        np.asarray([affinity.nn, affinity.mean, affinity.centroid]),
        "human_affinity",
    )
    return affinity


def average_human_affinity(
    first: HumanAffinity,
    second: HumanAffinity,
) -> HumanAffinity:
    affinity = HumanAffinity(
        nn=(first.nn + second.nn) / 2.0,
        mean=(first.mean + second.mean) / 2.0,
        centroid=(first.centroid + second.centroid) / 2.0,
    )
    return affinity


def contrast_scores(ai: AiAffinity, human: HumanAffinity) -> ContrastScores:
    ai_distance = max(1.0 - ai.nn, DISTANCE_FLOOR)
    human_distance = max(1.0 - human.nn, DISTANCE_FLOOR)
    relative_share = ai_distance / (ai_distance + human_distance)
    scores = ContrastScores(
        nn_margin=ai.nn - human.nn,
        top3_margin=ai.top3_mean - human.nn,
        centroid_contrast=ai.centroid - human.centroid,
        log_distance_ratio=math.log(human_distance / ai_distance),
        rank_relative=1.0 - relative_share,
    )
    _require_finite(
        np.asarray(
            [
                scores.nn_margin,
                scores.top3_margin,
                scores.centroid_contrast,
                scores.log_distance_ratio,
                scores.rank_relative,
            ]
        ),
        "contrast",
    )
    return scores


def main() -> int:
    started = time.perf_counter()
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
        raise RuntimeError("Cached-human subset is required")
    vectors = _load_cached_vectors(scoring_records)
    _validate_vectors(scoring_records, vectors)
    loading_elapsed = time.perf_counter() - loading_started
    _check_runtime(started, "loading")
    print("Building pair-isolated AI and human banks")
    bank_started = time.perf_counter()
    population = build_margin_population(scoring_records, vectors)
    bank_elapsed = time.perf_counter() - bank_started
    _check_runtime(started, "bank construction")
    print("Scoring relative AI/human margins")
    score_started = time.perf_counter()
    scored, skipped = score_population(population)
    population = MarginPopulation(
        items=population.items,
        ai_banks=population.ai_banks,
        human_folds=population.human_folds,
        coverage=MarginCoverage(
            eligible_pairs=population.coverage.eligible_pairs,
            excluded_few_humans=population.coverage.excluded_few_humans,
            duplicates_removed=population.coverage.duplicates_removed,
            human_count_distribution=population.coverage.human_count_distribution,
            skipped_no_human_refs=skipped,
        ),
    )
    score_elapsed = time.perf_counter() - score_started
    _check_runtime(started, "scoring")
    print("Calculating metrics")
    metrics_started = time.perf_counter()
    payload = build_results(population, scored, subset, eval_hash_before)
    metrics_elapsed = time.perf_counter() - metrics_started
    _check_runtime(started, "metrics")
    print("Writing report")
    write_started = time.perf_counter()
    timings = {
        "loading_seconds": loading_elapsed,
        "bank_construction_seconds": bank_elapsed,
        "scoring_seconds": score_elapsed,
        "metrics_seconds": metrics_elapsed,
        "report_writing_seconds": 0.0,
        "total_seconds": 0.0,
    }
    payload["timings"] = timings
    payload["eval_scores_sha256_after"] = _file_sha256(EVAL_SCORES_PATH)
    _write_outputs(payload)
    timings["report_writing_seconds"] = time.perf_counter() - write_started
    timings["total_seconds"] = time.perf_counter() - started
    payload["timings"] = timings
    _write_outputs(payload)
    _print_timings(timings)
    if payload["eval_scores_sha256_before"] != payload["eval_scores_sha256_after"]:
        raise RuntimeError("outputs/eval_scores.json changed")
    return 0


def build_margin_population(
    records: Sequence[ExperimentRecord],
    vectors: Sequence[Sequence[float]],
) -> MarginPopulation:
    items = [
        VectorItem(record, np.asarray(vector, dtype=np.float64))
        for record, vector in zip(records, vectors)
    ]
    grouped: dict[tuple[str, str], list[VectorItem]] = defaultdict(list)
    for item in items:
        grouped[(item.record.question_id, item.record.language)].append(item)
    ai_banks: dict[ClusterKey, AiBank] = {}
    human_folds: dict[ClusterKey, HumanFolds] = {}
    eligible: list[tuple[str, str]] = []
    excluded: list[tuple[str, str]] = []
    kept: list[VectorItem] = []
    duplicates = 0
    counts: list[int] = []
    for pair, group in sorted(grouped.items()):
        built = _build_pair_banks(group)
        if built is None:
            excluded.append(pair)
            continue
        bank, folds = built
        ai_banks[bank.key] = bank
        human_folds[folds.key] = folds
        eligible.append(pair)
        duplicates += folds.duplicates_removed
        counts.append(len(folds.fold_0) + len(folds.fold_1))
        kept.extend(group)
    distribution = Counter(str(count) for count in counts)
    coverage = MarginCoverage(
        eligible_pairs=tuple(eligible),
        excluded_few_humans=tuple(excluded),
        duplicates_removed=duplicates,
        human_count_distribution=dict(sorted(distribution.items())),
        skipped_no_human_refs=0,
    )
    return MarginPopulation(tuple(kept), ai_banks, human_folds, coverage)


def score_population(
    population: MarginPopulation,
) -> tuple[list[ScoredRow], int]:
    rows: list[ScoredRow] = []
    skipped = 0
    items_by_pair: dict[ClusterKey, list[VectorItem]] = defaultdict(list)
    for item in population.items:
        key = _cluster_key_for_record(item.record)
        _assert_same_cluster(item.record, key)
        items_by_pair[key].append(item)
    for pair in population.coverage.eligible_pairs:
        key = ClusterKey(pair[0], pair[1])
        ai_bank = population.ai_banks[key]
        folds = population.human_folds[key]
        human_count = len(folds.fold_0) + len(folds.fold_1)
        human_rows, human_skipped = _score_humans(
            items_by_pair[key],
            ai_bank,
            folds,
            human_count,
        )
        skipped += human_skipped
        rows.extend(human_rows)
        rows.extend(
            _score_held_out(items_by_pair[key], ai_bank, folds, human_count)
        )
    _assert_unique_eval_rows(rows)
    return rows, skipped


def build_results(
    population: MarginPopulation,
    scored: Sequence[ScoredRow],
    subset: object,
    eval_hash_before: str | None,
) -> dict[str, object]:
    records = tuple(item.record for item in population.items)
    match_keys = _held_out_reference_match_keys(records)
    human_match_keys = _held_out_human_match_keys(records)
    variants = {
        "all_held_out": list(scored),
        "exclude_exact_reference_match": [
            row
            for row in scored
            if row.label != LABEL_AI
            or (row.question_id, row.language, row.content_hash) not in match_keys
        ],
    }
    metrics = {
        variant: _language_metrics(rows)
        for variant, rows in variants.items()
    }
    personas = {
        variant: _persona_metrics(rows)
        for variant, rows in variants.items()
    }
    slices = _human_count_slices(variants["exclude_exact_reference_match"])
    diagnostics = _diagnostics(scored, population)
    affinity_check = _phase_b_affinity_check(scored)
    original_nn = _original_cached_nn()
    verdict = decide_verdict(metrics, personas, slices)
    return {
        "experiment": EXPERIMENT_NAME,
        "evaluation_mode": "cached_humans_only",
        "is_production": False,
        "is_final_all_human_evaluation": False,
        "bootstrap_iterations": 0,
        "reversible": True,
        "network_calls": False,
        "embeddings_generated": False,
        "population": _population_payload(population, scored, subset),
        "same_cluster_ai_reference_matches": len(match_keys),
        "same_cluster_held_out_human_matches": len(human_match_keys),
        "metrics": metrics,
        "personas": personas,
        "human_reference_slices": slices,
        "diagnostics": diagnostics,
        "phase_b_affinity_check": affinity_check,
        "original_cached_nn": original_nn,
        "verdict": verdict,
        "eval_scores_sha256_before": eval_hash_before,
    }


def _build_pair_banks(
    group: Sequence[VectorItem],
) -> tuple[AiBank, HumanFolds] | None:
    refs = [item for item in group if item.record.role == ROLE_REFERENCE]
    humans = [item for item in group if item.record.role == ROLE_HUMAN]
    held_out = [item for item in group if item.record.role == ROLE_HELD_OUT]
    if len(refs) != EXPECTED_REFERENCE_COUNT or not held_out:
        return None
    key = _cluster_key_for_record(refs[0].record)
    for item in group:
        _assert_same_cluster(item.record, key)
    matrix = np.vstack([item.vector for item in refs])
    bank = AiBank(
        key=key,
        vectors=matrix,
        hashes=tuple(item.record.content_hash for item in refs),
        centroid=centroid_vector(matrix),
    )
    folds = split_human_folds(humans)
    if folds is None:
        return None
    return bank, folds


def _deduplicated_humans(
    items: Sequence[VectorItem],
) -> tuple[list[VectorItem], int]:
    ordered = sorted(items, key=_human_sort_key)
    unique: list[VectorItem] = []
    seen: set[str] = set()
    duplicates = 0
    for item in ordered:
        if item.record.content_hash in seen:
            duplicates += 1
            continue
        seen.add(item.record.content_hash)
        unique.append(item)
    return unique, duplicates


def _human_sort_key(item: VectorItem) -> tuple[str, str, str]:
    user_id = item.record.user_id or ""
    return (item.record.content_hash, item.record.source, user_id)


def _score_humans(
    items: Sequence[VectorItem],
    ai_bank: AiBank,
    folds: HumanFolds,
    human_count: int,
) -> tuple[list[ScoredRow], int]:
    fold_ids = _human_fold_ids(folds)
    rows: list[ScoredRow] = []
    skipped = 0
    seen: set[str] = set()
    for item in folds.fold_0 + folds.fold_1:
        if item.record.content_hash in seen:
            continue
        seen.add(item.record.content_hash)
        eval_fold = fold_ids[item.record.content_hash]
        refs = opposite_human_refs(folds, item, eval_fold)
        if not refs:
            skipped += 1
            continue
        rows.append(
            _scored_row(item, LABEL_HUMAN, ai_bank, human_affinity(item.vector, refs), human_count)
        )
    return rows, skipped


def _score_held_out(
    items: Sequence[VectorItem],
    ai_bank: AiBank,
    folds: HumanFolds,
    human_count: int,
) -> list[ScoredRow]:
    rows: list[ScoredRow] = []
    seen: set[tuple[str, str, str]] = set()
    for item in items:
        if item.record.role != ROLE_HELD_OUT:
            continue
        identity = (item.record.question_id, item.record.language, item.record.content_hash)
        if identity in seen:
            continue
        seen.add(identity)
        first = human_affinity(item.vector, folds.fold_0)
        second = human_affinity(item.vector, folds.fold_1)
        averaged = average_human_affinity(first, second)
        rows.append(_scored_row(item, LABEL_AI, ai_bank, averaged, human_count))
    return rows


def _human_fold_ids(folds: HumanFolds) -> dict[str, int]:
    mapping = {item.record.content_hash: 0 for item in folds.fold_0}
    mapping.update({item.record.content_hash: 1 for item in folds.fold_1})
    return mapping


def _scored_row(
    item: VectorItem,
    label: str,
    ai_bank: AiBank,
    human: HumanAffinity,
    human_count: int,
) -> ScoredRow:
    _assert_same_cluster(item.record, ai_bank.key)
    ai = ai_affinity(item.vector, ai_bank)
    contrast = contrast_scores(ai, human)
    return ScoredRow(
        question_id=item.record.question_id,
        language=item.record.language,
        label=label,
        source=item.record.source,
        content_hash=item.record.content_hash,
        human_reference_count=human_count,
        ai_affinity=ai,
        human_affinity=human,
        contrast=contrast,
    )


def _assert_unique_eval_rows(rows: Sequence[ScoredRow]) -> None:
    human_keys = [
        (row.question_id, row.language, row.content_hash)
        for row in rows
        if row.label == LABEL_HUMAN
    ]
    ai_keys = [
        (row.question_id, row.language, row.source, row.content_hash)
        for row in rows
        if row.label == LABEL_AI
    ]
    if len(human_keys) != len(set(human_keys)):
        raise RuntimeError("A human evaluation row was emitted more than once")
    if len(ai_keys) != len(set(ai_keys)):
        raise RuntimeError("A held-out AI row was emitted more than once")


def _language_metrics(
    rows: Sequence[ScoredRow],
) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for language in ("CPP", "PYTHON"):
        language_rows = [row for row in rows if row.language == language]
        result[language] = {
            "n_pairs": len({(row.question_id, row.language) for row in language_rows}),
            "methods": {
                method: method_metrics_to_dict(
                    method_metrics_bundle(
                        method,
                        _labeled(language_rows, method),
                        FIXED_SEED,
                    )
                )
                for method in METHODS
            },
        }
    return result


def _labeled(rows: Sequence[ScoredRow], method: str) -> list[LabeledScore]:
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


def _method_score(row: ScoredRow, method: str) -> float:
    mapping = {
        "matched_population_nn": row.ai_affinity.nn,
        "ai_nn_minus_human_nn": row.contrast.nn_margin,
        "ai_top3_minus_human_nn": row.contrast.top3_margin,
        "ai_centroid_minus_human_centroid": row.contrast.centroid_contrast,
        "log_distance_ratio": row.contrast.log_distance_ratio,
        "rank_relative": row.contrast.rank_relative,
    }
    return float(mapping[method])


def _persona_metrics(rows: Sequence[ScoredRow]) -> dict[str, dict[str, object]]:
    return {
        method: {
            persona: _persona_oof(rows, method, persona)
            for persona in PERSONAS
        }
        for method in METHODS
    }


def _persona_oof(
    rows: Sequence[ScoredRow],
    method: str,
    persona: str,
) -> dict[str, float | int | None]:
    items = _labeled(rows, method)
    fold_map = question_folds({item.question_id for item in items}, FIXED_SEED)
    positives = [row for row in rows if row.label == LABEL_AI and row.source == persona]
    return {
        "n_positives": len(positives),
        "grouped_oof_recall_at_1pct_fpr": _persona_rate(positives, method, items, fold_map, 0.01),
        "grouped_oof_recall_at_5pct_fpr": _persona_rate(positives, method, items, fold_map, 0.05),
    }


def _persona_rate(
    positives: Sequence[ScoredRow],
    method: str,
    items: Sequence[LabeledScore],
    fold_map: Mapping[str, int],
    target_fpr: float,
) -> float | None:
    if not positives:
        return None
    hits = 0
    total = 0
    for row in positives:
        train_neg = [
            item.score
            for item in items
            if item.label == LABEL_HUMAN
            and item.language == row.language
            and fold_map[item.question_id] != fold_map[row.question_id]
        ]
        if not train_neg:
            continue
        threshold = conservative_threshold(train_neg, target_fpr)
        total += 1
        if _method_score(row, method) >= threshold:
            hits += 1
    if total == 0:
        return None
    return hits / total


def _human_count_slices(rows: Sequence[ScoredRow]) -> dict[str, object]:
    buckets = {
        "2": [row for row in rows if row.human_reference_count == 2],
        "3+": [row for row in rows if row.human_reference_count >= 3],
    }
    return {
        name: {
            "n_rows": len(bucket),
            "n_pairs": len({(row.question_id, row.language) for row in bucket}),
            "languages": _language_metrics(bucket) if bucket else {},
        }
        for name, bucket in buckets.items()
    }


def _diagnostics(
    rows: Sequence[ScoredRow],
    population: MarginPopulation,
) -> dict[str, object]:
    humans = [row for row in rows if row.label == LABEL_HUMAN]
    ais = [row for row in rows if row.label == LABEL_AI]
    human_to_human = [row.human_affinity.nn for row in humans]
    ai_to_human = [row.human_affinity.nn for row in ais]
    ai_to_ai = [row.ai_affinity.nn for row in ais]
    human_closer_to_ai = sum(1 for row in humans if row.ai_affinity.nn > row.human_affinity.nn)
    ai_closer_to_human = sum(1 for row in ais if row.human_affinity.nn > row.ai_affinity.nn)
    pair_margins = _pair_mean_margins(ais)
    strongest = pair_margins[:5]
    weakest = list(reversed(pair_margins[-5:]))
    return {
        "human_to_human_nn": _value_stats(human_to_human),
        "ai_to_human_nn": _value_stats(ai_to_human),
        "ai_to_ai_nn": _value_stats(ai_to_ai),
        "humans_closer_to_ai_than_opposite_humans": {
            "count": human_closer_to_ai,
            "rate": human_closer_to_ai / len(humans) if humans else None,
        },
        "held_out_ai_closer_to_humans_than_ai_refs": {
            "count": ai_closer_to_human,
            "rate": ai_closer_to_human / len(ais) if ais else None,
        },
        "strongest_pairs": strongest,
        "weakest_pairs": weakest,
        "human_count_distribution": population.coverage.human_count_distribution,
        "eligible_human_rows": len(humans),
        "eligible_ai_rows": len(ais),
    }


def _pair_mean_margins(ais: Sequence[ScoredRow]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in ais:
        grouped[(row.question_id, row.language)].append(row.contrast.nn_margin)
    summaries = [
        {
            "question_id": question_id,
            "language": language,
            "mean_nn_margin": float(np.mean(values)),
            "n_ai": len(values),
        }
        for (question_id, language), values in grouped.items()
    ]
    return sorted(summaries, key=lambda item: item["mean_nn_margin"], reverse=True)


def _value_stats(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "p10": None, "p90": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p10": float(np.quantile(array, 0.10)),
        "p90": float(np.quantile(array, 0.90)),
    }


def _phase_b_affinity_check(rows: Sequence[ScoredRow]) -> dict[str, object]:
    path = CENTROID_CACHED_HUMANS_SCORES_PATH
    if not path.is_file():
        return {"status": "skipped_missing_phase_b_scores"}
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload.get("items")
    if not isinstance(items, list):
        return {"status": "skipped_no_items"}
    lookup = {
        (item["question_id"], item["language"], item["label"], item["source"], item["content_hash"]): item
        for item in items
        if isinstance(item, dict)
    }
    diffs = []
    matched = 0
    for row in rows:
        key = (row.question_id, row.language, row.label, row.source, row.content_hash)
        item = lookup.get(key)
        if item is None:
            continue
        matched += 1
        diffs.append(abs(row.ai_affinity.nn - float(item["nn_score"])))
        diffs.append(abs(row.ai_affinity.top3_mean - float(item["topk_mean_score"])))
        diffs.append(abs(row.ai_affinity.centroid - float(item["centroid_score"])))
    max_diff = max(diffs) if diffs else None
    return {
        "status": "ok" if max_diff is not None and max_diff <= AFFINITY_TOLERANCE else "mismatch",
        "matched_rows": matched,
        "max_abs_diff": max_diff,
        "tolerance": AFFINITY_TOLERANCE,
    }


def _original_cached_nn() -> dict[str, object] | None:
    if not PHASE_B2_SCORES_PATH.is_file():
        return None
    payload = json.loads(PHASE_B2_SCORES_PATH.read_text(encoding="utf-8"))
    standalone = payload.get("standalone", {})
    all_held = standalone.get("all_held_out", {})
    return {
        "note": "Phase B.2 NN on the 988-pair cached-human population; context only.",
        "CPP": (all_held.get("CPP") or {}).get("nn"),
        "PYTHON": (all_held.get("PYTHON") or {}).get("nn"),
    }


def _held_out_human_match_keys(
    records: Sequence[ExperimentRecord],
) -> set[tuple[str, str, str]]:
    human_keys = {
        (record.question_id, record.language, record.content_hash)
        for record in records
        if record.role == ROLE_HUMAN
    }
    return {
        (record.question_id, record.language, record.content_hash)
        for record in records
        if record.role == ROLE_HELD_OUT
        and (record.question_id, record.language, record.content_hash) in human_keys
    }


def _population_payload(
    population: MarginPopulation,
    scored: Sequence[ScoredRow],
    subset: object,
) -> dict[str, object]:
    humans = [row for row in scored if row.label == LABEL_HUMAN]
    ais = [row for row in scored if row.label == LABEL_AI]
    cpp_pairs = sum(
        1 for _qid, language in population.coverage.eligible_pairs if language == "CPP"
    )
    py_pairs = sum(
        1 for _qid, language in population.coverage.eligible_pairs if language == "PYTHON"
    )
    cached_skip = getattr(subset, "pairs_excluded_no_cached_human", ())
    return {
        "eligible_pairs": len(population.coverage.eligible_pairs),
        "eligible_pairs_cpp": cpp_pairs,
        "eligible_pairs_python": py_pairs,
        "pairs_excluded_fewer_than_two_distinct_humans": len(
            population.coverage.excluded_few_humans
        ),
        "pairs_excluded_no_cached_human_phase_b1": len(cached_skip),
        "cached_humans_used": len(humans),
        "cached_humans_cpp": sum(1 for row in humans if row.language == "CPP"),
        "cached_humans_python": sum(1 for row in humans if row.language == "PYTHON"),
        "held_out_ai_used": len(ais),
        "held_out_ai_cpp": sum(1 for row in ais if row.language == "CPP"),
        "held_out_ai_python": sum(1 for row in ais if row.language == "PYTHON"),
        "human_duplicates_removed_before_split": population.coverage.duplicates_removed,
        "human_count_distribution": population.coverage.human_count_distribution,
        "human_eval_rows_skipped_no_refs": population.coverage.skipped_no_human_refs,
        "matched_eval_rows": len(scored),
    }


def decide_verdict(
    metrics: Mapping[str, object],
    personas: Mapping[str, object],
    slices: Mapping[str, object],
) -> dict[str, object]:
    excluded = metrics["exclude_exact_reference_match"]["CPP"]["methods"]
    nn = excluded["matched_population_nn"]
    best_name, best = _best_cpp_method(excluded)
    r1_gain = _recall_delta(best, nn, "1%")
    r5_gain = _recall_delta(best, nn, "5%")
    fpr = _oof_fpr(best, "1%")
    macro_gain = _optional_delta(best.get("macro_auroc"), nn.get("macro_auroc"))
    persona_nn = _persona_value(personas, "exclude_exact_reference_match", "matched_population_nn")
    persona_best = _persona_value(personas, "exclude_exact_reference_match", best_name)
    two_vs_more = _slice_gain(slices)
    promising = _promising_cpp(r1_gain, fpr, r5_gain, macro_gain, persona_best, persona_nn, two_vs_more)
    more_helps = two_vs_more is not None and two_vs_more > 0
    justified = promising or more_helps
    if promising:
        label = "promising for C++"
    else:
        label = (
            "Human-reference contrast did not improve canonicality sufficiently. "
            "Do not embed remaining humans solely for this approach. "
            "Proceed to structural AST features."
        )
    return {
        "label": label,
        "best_cpp_method": best_name,
        "cpp_oof_recall_1pct_gain_vs_nn": r1_gain,
        "cpp_oof_recall_5pct_gain_vs_nn": r5_gain,
        "cpp_oof_achieved_fpr_1pct": fpr,
        "cpp_macro_auroc_gain_vs_nn": macro_gain,
        "production_review_nn": persona_nn,
        "production_review_best": persona_best,
        "human_count_3plus_minus_2_recall_1pct": two_vs_more,
        "justified_embed_remaining_humans": justified,
        "promising_for_cpp": promising,
    }


def _best_cpp_method(
    methods: Mapping[str, Mapping[str, object]],
) -> tuple[str, Mapping[str, object]]:
    ranked = []
    for name, metrics in methods.items():
        if name == "matched_population_nn":
            continue
        recall = _oof_recall(metrics, "1%")
        ranked.append((recall if recall is not None else -1.0, name, metrics))
    ranked.sort(reverse=True)
    return ranked[0][1], ranked[0][2]


def _promising_cpp(
    r1_gain: float | None,
    fpr: float | None,
    r5_gain: float | None,
    macro_gain: float | None,
    persona_best: float | None,
    persona_nn: float | None,
    two_vs_more: float | None,
) -> bool:
    if r1_gain is None or fpr is None or r5_gain is None or macro_gain is None:
        return False
    if r1_gain < CPP_RECALL_GAIN_MIN:
        return False
    if fpr > MATERIAL_FPR_LIMIT:
        return False
    if r5_gain < -MATERIAL_R5_DROP:
        return False
    if macro_gain < -MACRO_DROP_MAX:
        return False
    if persona_best is not None and persona_nn is not None and persona_best < persona_nn:
        return False
    if two_vs_more is not None and two_vs_more < 0 and r1_gain > 0:
        return False
    return True


def _slice_gain(slices: Mapping[str, object]) -> float | None:
    two = slices.get("2")
    more = slices.get("3+")
    if not isinstance(two, dict) or not isinstance(more, dict):
        return None
    if more.get("n_pairs", 0) == 0 or two.get("n_pairs", 0) == 0:
        return None
    two_metrics = ((two.get("languages") or {}).get("CPP") or {}).get("methods") or {}
    more_metrics = ((more.get("languages") or {}).get("CPP") or {}).get("methods") or {}
    two_r = _oof_recall(two_metrics.get(PRIMARY_MARGIN) or {}, "1%")
    more_r = _oof_recall(more_metrics.get(PRIMARY_MARGIN) or {}, "1%")
    return _optional_delta(more_r, two_r)


def _recall_delta(
    left: Mapping[str, object],
    right: Mapping[str, object],
    point: str,
) -> float | None:
    return _optional_delta(_oof_recall(left, point), _oof_recall(right, point))


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
) -> float | None:
    variant_payload = personas.get(variant)
    if not isinstance(variant_payload, dict):
        return None
    method_payload = variant_payload.get(method)
    if not isinstance(method_payload, dict):
        return None
    persona = method_payload.get("production_review")
    if not isinstance(persona, dict):
        return None
    value = persona.get("grouped_oof_recall_at_1pct_fpr")
    return float(value) if isinstance(value, (int, float)) else None


def _write_outputs(payload: Mapping[str, object]) -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    SCORES_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_summary(payload)
    REPORT_PATH.write_text(_render_report(payload), encoding="utf-8")
    print(f"Wrote {SCORES_PATH}")
    print(f"Wrote {SUMMARY_PATH}")
    print(f"Wrote {REPORT_PATH}")


def _write_summary(payload: Mapping[str, object]) -> None:
    fieldnames = [
        "variant",
        "language",
        "method",
        "n_positives",
        "n_negatives",
        "n_pairs",
        "pooled_auroc",
        "macro_auroc",
        "oof_recall_1pct",
        "oof_fpr_1pct",
        "oof_recall_5pct",
        "oof_fpr_5pct",
        "positive_mean",
        "negative_mean",
        "mean_gap",
        "median_gap",
        "delta_auroc_vs_nn",
        "delta_macro_vs_nn",
        "delta_r1_vs_nn",
        "delta_r5_vs_nn",
    ]
    with SUMMARY_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(_summary_rows(payload))


def _summary_rows(payload: Mapping[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    metrics = payload["metrics"]
    for variant, languages in metrics.items():
        for language, body in languages.items():
            methods = body["methods"]
            nn = methods["matched_population_nn"]
            for method, values in methods.items():
                rows.append(
                    {
                        "variant": variant,
                        "language": language,
                        "method": method,
                        "n_positives": values.get("n_positives"),
                        "n_negatives": values.get("n_negatives"),
                        "n_pairs": body.get("n_pairs"),
                        "pooled_auroc": values.get("pooled_auroc"),
                        "macro_auroc": values.get("macro_auroc"),
                        "oof_recall_1pct": _oof_recall(values, "1%"),
                        "oof_fpr_1pct": _oof_fpr(values, "1%"),
                        "oof_recall_5pct": _oof_recall(values, "5%"),
                        "oof_fpr_5pct": _oof_fpr(values, "5%"),
                        "positive_mean": (values.get("positives") or {}).get("mean")
                        if isinstance(values.get("positives"), dict)
                        else None,
                        "negative_mean": (values.get("negatives") or {}).get("mean")
                        if isinstance(values.get("negatives"), dict)
                        else None,
                        "mean_gap": values.get("mean_gap"),
                        "median_gap": values.get("median_gap"),
                        "delta_auroc_vs_nn": _optional_delta(
                            values.get("pooled_auroc")
                            if isinstance(values.get("pooled_auroc"), float)
                            else None,
                            nn.get("pooled_auroc")
                            if isinstance(nn.get("pooled_auroc"), float)
                            else None,
                        ),
                        "delta_macro_vs_nn": _optional_delta(
                            values.get("macro_auroc")
                            if isinstance(values.get("macro_auroc"), float)
                            else None,
                            nn.get("macro_auroc")
                            if isinstance(nn.get("macro_auroc"), float)
                            else None,
                        ),
                        "delta_r1_vs_nn": _recall_delta(values, nn, "1%"),
                        "delta_r5_vs_nn": _recall_delta(values, nn, "5%"),
                    }
                )
    return rows


def _render_report(payload: Mapping[str, object]) -> str:
    population = payload["population"]
    verdict = payload["verdict"]
    timings = payload.get("timings") or {}
    lines = [
        "# AI-versus-human relative similarity (Phase B.3)",
        "",
        "Reversible cached-human experiment. Production scoring was not modified.",
        "",
        f"- experiment: `{payload['experiment']}`",
        f"- evaluation_mode: `{payload['evaluation_mode']}`",
        f"- bootstrap_iterations: `{payload['bootstrap_iterations']}`",
        f"- reversible: `{payload['reversible']}`",
        f"- network_calls: `{payload['network_calls']}`",
        "",
        "## Population",
        "",
        f"- eligible pairs: {population['eligible_pairs']} "
        f"(CPP {population['eligible_pairs_cpp']}, PYTHON {population['eligible_pairs_python']})",
        f"- excluded for fewer than two distinct humans: "
        f"{population['pairs_excluded_fewer_than_two_distinct_humans']}",
        f"- cached humans used: {population['cached_humans_used']} "
        f"(CPP {population['cached_humans_cpp']}, PYTHON {population['cached_humans_python']})",
        f"- held-out AI used: {population['held_out_ai_used']} "
        f"(CPP {population['held_out_ai_cpp']}, PYTHON {population['held_out_ai_python']})",
        f"- duplicates removed before split: {population['human_duplicates_removed_before_split']}",
        f"- human-count distribution: {population['human_count_distribution']}",
        f"- same-cluster AI-reference exact matches: {payload['same_cluster_ai_reference_matches']}",
        f"- same-cluster held-out/human exact matches: {payload['same_cluster_held_out_human_matches']}",
        "",
        "## Timings",
        "",
    ]
    for key, value in timings.items():
        lines.append(f"- {key}: {float(value):.3f}")
    lines.extend(["", "## CPP comparison (matched population)", ""])
    lines.extend(_table(payload, "all_held_out", "CPP"))
    lines.extend(["", "## Python comparison (matched population)", ""])
    lines.extend(_table(payload, "all_held_out", "PYTHON"))
    lines.extend(["", "## Exact-match-excluded", ""])
    lines.extend(_table(payload, "exclude_exact_reference_match", "CPP"))
    lines.extend(_table(payload, "exclude_exact_reference_match", "PYTHON"))
    lines.extend(["", "## production_review", ""])
    lines.extend(_persona_lines(payload))
    lines.extend(["", "## Human-reference-count slices", ""])
    slices = payload["human_reference_slices"]
    for name, body in slices.items():
        lines.append(
            f"- {name} humans: {body['n_pairs']} pairs, {body['n_rows']} rows"
        )
    lines.extend(["", "## Diagnostics", ""])
    diag = payload["diagnostics"]
    lines.append(f"- human-to-human NN: {diag['human_to_human_nn']}")
    lines.append(f"- AI-to-human NN: {diag['ai_to_human_nn']}")
    lines.append(f"- AI-to-AI NN: {diag['ai_to_ai_nn']}")
    lines.append(
        f"- humans closer to AI than opposite-fold humans: "
        f"{diag['humans_closer_to_ai_than_opposite_humans']}"
    )
    lines.append(
        f"- held-out AI closer to humans than AI refs: "
        f"{diag['held_out_ai_closer_to_humans_than_ai_refs']}"
    )
    lines.append(f"- strongest pairs: {diag['strongest_pairs']}")
    lines.append(f"- weakest pairs: {diag['weakest_pairs']}")
    lines.extend(
        [
            "",
            "## Verdict",
            "",
            f"**{verdict['label']}**",
            "",
            f"- best C++ method: `{verdict['best_cpp_method']}`",
            f"- C++ OOF R@1% minus matched NN: {_fmt(verdict['cpp_oof_recall_1pct_gain_vs_nn'])}",
            f"- C++ OOF R@5% minus matched NN: {_fmt(verdict['cpp_oof_recall_5pct_gain_vs_nn'])}",
            f"- C++ achieved FPR@1%: {_fmt(verdict['cpp_oof_achieved_fpr_1pct'])}",
            f"- C++ macro AUROC minus NN: {_fmt(verdict['cpp_macro_auroc_gain_vs_nn'])}",
            f"- production_review NN: {_fmt(verdict['production_review_nn'])}",
            f"- production_review best: {_fmt(verdict['production_review_best'])}",
            f"- 3+ minus 2-human R@1%: {_fmt(verdict['human_count_3plus_minus_2_recall_1pct'])}",
            f"- justified embed remaining humans: {verdict['justified_embed_remaining_humans']}",
            "",
            "Production scoring was not modified.",
            "",
        ]
    )
    return "\n".join(lines)


def _table(payload: Mapping[str, object], variant: str, language: str) -> list[str]:
    lines = [
        f"### {language} `{variant}`",
        "",
        "| method | n+ | n- | pairs | AUROC | macro | OOF R@1% | FPR@1% | OOF R@5% | FPR@5% | dR@1% vs NN | dAUROC vs NN |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in _summary_rows(payload):
        if row["variant"] != variant or row["language"] != language:
            continue
        lines.append(
            f"| {row['method']} | {row['n_positives']} | {row['n_negatives']} | "
            f"{row['n_pairs']} | {_fmt(row['pooled_auroc'])} | {_fmt(row['macro_auroc'])} | "
            f"{_fmt(row['oof_recall_1pct'])} | {_fmt(row['oof_fpr_1pct'])} | "
            f"{_fmt(row['oof_recall_5pct'])} | {_fmt(row['oof_fpr_5pct'])} | "
            f"{_fmt(row['delta_r1_vs_nn'])} | {_fmt(row['delta_auroc_vs_nn'])} |"
        )
    return lines


def _persona_lines(payload: Mapping[str, object]) -> list[str]:
    lines = [
        "| variant | method | persona | n | OOF R@1% | OOF R@5% |",
        "|---|---|---|---:|---:|---:|",
    ]
    for variant, methods in payload["personas"].items():
        for method, personas in methods.items():
            for persona, stats in personas.items():
                lines.append(
                    f"| {variant} | {method} | {persona} | {stats['n_positives']} | "
                    f"{_fmt(stats['grouped_oof_recall_at_1pct_fpr'])} | "
                    f"{_fmt(stats['grouped_oof_recall_at_5pct_fpr'])} |"
                )
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
            f"Experiment exceeded {MAX_RUNTIME_SECONDS}s during {phase} ({elapsed:.1f}s)"
        )


def _require_finite(array: np.ndarray, name: str) -> None:
    if not np.isfinite(array).all():
        raise RuntimeError(f"Non-finite values in {name}")


if __name__ == "__main__":
    raise SystemExit(main())

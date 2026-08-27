from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from collections.abc import Mapping, Sequence

import numpy as np

from nw_ai_code_detector.config import (
    EVAL_SCORES_PATH,
    OUTPUTS_DIR,
    SELECTED_500_PATH,
)
from nw_ai_code_detector.constants import EvaluationMode, SELECTION_RANDOM_SEED
from nw_ai_code_detector.data_load import load_dataset
from nw_ai_code_detector.evaluation.evaluate_ai_human_margin_quick import (
    SCORES_PATH as PHASE_B3_SCORES_PATH,
    ScoredRow,
    build_margin_population,
    resolve_ai_bank,
    resolve_human_folds,
    score_population,
    PairLookup,
)
from nw_ai_code_detector.evaluation.evaluate_centroid_all_humans import (
    LABEL_AI,
    LABEL_HUMAN,
    _assert_same_cluster,
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
    MethodMetrics,
    OperatingPoint,
    conservative_threshold,
    macro_auroc_by_question,
    method_metrics_to_dict,
    pooled_auroc,
    question_folds,
    score_statistics,
)
from nw_ai_code_detector.generate_ai_refs import _limit_questions, _load_selected_questions

EXPERIMENT_NAME = "percentile_and_gate_quick"
FIXED_SEED = SELECTION_RANDOM_SEED
MAX_RUNTIME_SECONDS = 600
CPP_RECALL_1_MIN = 0.224
MATERIAL_FPR_LIMIT = 0.015
MATERIAL_R5_DROP = 0.03
MACRO_DROP_MAX = 0.01
REUSE_TOLERANCE = 1e-12
SCORES_PATH = OUTPUTS_DIR / "percentile_and_gate_quick_scores.json"
SUMMARY_PATH = OUTPUTS_DIR / "percentile_and_gate_quick_summary.csv"
REPORT_PATH = OUTPUTS_DIR / "percentile_and_gate_quick_report.md"
PERSONAS = ("production_review", "pair_programming")
METHODS = (
    "matched_population_nn",
    "matched_population_log_distance_ratio",
    "percentile_and_gate",
)
AND_METHOD = "percentile_and_gate"
NN_METHOD = "matched_population_nn"
RATIO_METHOD = "matched_population_log_distance_ratio"


@dataclass(frozen=True)
class GateSignals:
    ai_nn: float
    log_distance_ratio: float


@dataclass(frozen=True)
class GateRow:
    question_id: str
    language: str
    label: str
    source: str
    content_hash: str
    human_reference_count: int
    exact_match: bool
    signals: GateSignals


@dataclass(frozen=True)
class OofRow:
    row: GateRow
    fold_id: int
    nn_percentile: float
    ratio_percentile: float
    and_score: float
    nn_flag_1: bool
    nn_flag_5: bool
    ratio_flag_1: bool
    ratio_flag_5: bool
    and_flag_1: bool
    and_flag_5: bool


@dataclass(frozen=True)
class TrainDistributions:
    language: str
    fold_id: int
    nn_sorted: np.ndarray
    ratio_sorted: np.ndarray
    train_question_ids: tuple[str, ...]


def human_percentile(sorted_train: np.ndarray, value: float) -> float:
    if sorted_train.size == 0:
        raise RuntimeError("Percentile requires training-human scores")
    rank = np.searchsorted(sorted_train, value, side="right")
    percentile = float(rank / sorted_train.size)
    if percentile < 0.0 or percentile > 1.0:
        raise RuntimeError(f"Percentile {percentile} is outside [0, 1]")
    return percentile


def and_score(nn_percentile: float, ratio_percentile: float) -> float:
    score = min(nn_percentile, ratio_percentile)
    if score < 0.0 or score > 1.0:
        raise RuntimeError(f"AND score {score} is outside [0, 1]")
    return score


def gate_rows_from_scored(
    scored: Sequence[ScoredRow],
    match_keys: set[tuple[str, str, str]],
) -> list[GateRow]:
    rows = []
    for item in scored:
        signals = GateSignals(item.ai_affinity.nn, item.contrast.log_distance_ratio)
        _assert_reused_signals(item, signals)
        exact_match = (item.question_id, item.language, item.content_hash) in match_keys
        rows.append(
            GateRow(
                question_id=item.question_id,
                language=item.language,
                label=item.label,
                source=item.source,
                content_hash=item.content_hash,
                human_reference_count=item.human_reference_count,
                exact_match=exact_match,
                signals=signals,
            )
        )
    return rows


def assign_oof_rows(
    rows: Sequence[GateRow],
    fold_map: Mapping[str, int],
) -> list[OofRow]:
    assigned: list[OofRow] = []
    for language in ("CPP", "PYTHON"):
        language_rows = [row for row in rows if row.language == language]
        assigned.extend(_oof_for_language(language_rows, fold_map, language))
    _assert_one_human_each(assigned)
    return assigned


def main() -> int:
    started = time.perf_counter()
    eval_hash_before = _file_sha256(EVAL_SCORES_PATH)
    print("Loading Phase B.3 cached population and scores")
    load_started = time.perf_counter()
    scored, population, match_keys, subset = _load_b3_rows()
    verify = _verify_b3_reuse(scored)
    load_elapsed = time.perf_counter() - load_started
    _check_runtime(started, "loading")
    print("Building grouped-OOF percentile AND scores")
    oof_started = time.perf_counter()
    rows = gate_rows_from_scored(scored, match_keys)
    _assert_identical_population(scored, rows)
    fold_map = question_folds({row.question_id for row in rows}, FIXED_SEED)
    _assert_grouped_folds(rows, fold_map)
    oof_rows = assign_oof_rows(rows, fold_map)
    oof_elapsed = time.perf_counter() - oof_started
    _check_runtime(started, "oof assignment")
    print("Calculating metrics")
    metrics_started = time.perf_counter()
    payload = _build_payload(
        population,
        subset,
        rows,
        oof_rows,
        fold_map,
        match_keys,
        verify,
        eval_hash_before,
    )
    metrics_elapsed = time.perf_counter() - metrics_started
    print("Writing report")
    write_started = time.perf_counter()
    timings = {
        "loading_seconds": load_elapsed,
        "oof_assignment_seconds": oof_elapsed,
        "metrics_seconds": metrics_elapsed,
        "report_writing_seconds": 0.0,
        "total_seconds": 0.0,
    }
    payload["timings"] = timings
    payload["eval_scores_sha256_after"] = _file_sha256(EVAL_SCORES_PATH)
    timings["report_writing_seconds"] = time.perf_counter() - write_started
    timings["total_seconds"] = time.perf_counter() - started
    payload["timings"] = timings
    _write_outputs(payload)
    _print_timings(timings)
    if payload["eval_scores_sha256_before"] != payload["eval_scores_sha256_after"]:
        raise RuntimeError("outputs/eval_scores.json changed")
    return 0


def _load_b3_rows() -> tuple[list[ScoredRow], object, set[tuple[str, str, str]], object]:
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
    population = build_margin_population(scoring_records, vectors)
    _assert_pair_isolation(population)
    scored, skipped = score_population(population)
    if skipped:
        raise RuntimeError(f"Unexpected skipped human rows: {skipped}")
    match_keys = _held_out_reference_match_keys(
        tuple(item.record for item in population.items)
    )
    return scored, population, match_keys, subset


def _assert_pair_isolation(population: object) -> None:
    for item in population.items:
        key = _cluster_key_for_record(item.record)
        _assert_same_cluster(item.record, key)
        resolve_ai_bank(PairLookup(item.record, key, population.ai_banks))
        resolve_human_folds(PairLookup(item.record, key, population.human_folds))


def _assert_reused_signals(item: ScoredRow, signals: GateSignals) -> None:
    if item.ai_affinity.nn != signals.ai_nn:
        raise RuntimeError("ai_nn changed while copying Phase B.3 scores")
    if item.contrast.log_distance_ratio != signals.log_distance_ratio:
        raise RuntimeError("log_distance_ratio changed while copying Phase B.3 scores")


def _assert_identical_population(
    scored: Sequence[ScoredRow],
    rows: Sequence[GateRow],
) -> None:
    left = {
        (item.question_id, item.language, item.label, item.source, item.content_hash)
        for item in scored
    }
    right = {
        (item.question_id, item.language, item.label, item.source, item.content_hash)
        for item in rows
    }
    if left != right:
        raise RuntimeError("AND-gate rows do not match the Phase B.3 population")


def _assert_grouped_folds(
    rows: Sequence[GateRow],
    fold_map: Mapping[str, int],
) -> None:
    by_question: dict[str, set[int]] = {}
    for row in rows:
        by_question.setdefault(row.question_id, set()).add(fold_map[row.question_id])
    split = [qid for qid, folds in by_question.items() if len(folds) != 1]
    if split:
        raise RuntimeError(f"Grouped fold split a question: {split[:3]}")


def _assert_one_human_each(rows: Sequence[OofRow]) -> None:
    humans = [
        (item.row.question_id, item.row.language, item.row.content_hash)
        for item in rows
        if item.row.label == LABEL_HUMAN
    ]
    if len(humans) != len(set(humans)):
        raise RuntimeError("A human evaluation row was counted more than once")


def _oof_for_language(
    rows: Sequence[GateRow],
    fold_map: Mapping[str, int],
    language: str,
) -> list[OofRow]:
    assigned: list[OofRow] = []
    for fold_id in range(GROUPED_FOLD_COUNT):
        train, held = _split_fold(rows, fold_map, fold_id)
        train_humans = [row for row in train if row.label == LABEL_HUMAN]
        if not train_humans or not held:
            continue
        _assert_no_held_out_leak(train_humans, held)
        distributions = _train_distributions(train_humans, language, fold_id)
        thresholds = _fold_thresholds(train_humans, distributions)
        assigned.extend(_transform_held_out(held, distributions, thresholds, fold_id))
    if len(assigned) != len(rows):
        raise RuntimeError("Every row must receive an out-of-fold AND score")
    return assigned


def _split_fold(
    rows: Sequence[GateRow],
    fold_map: Mapping[str, int],
    fold_id: int,
) -> tuple[list[GateRow], list[GateRow]]:
    train = [row for row in rows if fold_map[row.question_id] != fold_id]
    held = [row for row in rows if fold_map[row.question_id] == fold_id]
    return train, held


def _assert_no_held_out_leak(
    train_humans: Sequence[GateRow],
    held: Sequence[GateRow],
) -> None:
    train_q = {row.question_id for row in train_humans}
    held_q = {row.question_id for row in held}
    overlap = train_q & held_q
    if overlap:
        raise RuntimeError(f"Held-out questions leaked into training: {sorted(overlap)[:3]}")


def _train_distributions(
    train_humans: Sequence[GateRow],
    language: str,
    fold_id: int,
) -> TrainDistributions:
    if any(row.language != language for row in train_humans):
        raise RuntimeError("Training humans must be language-specific")
    if any(row.label != LABEL_HUMAN for row in train_humans):
        raise RuntimeError("Percentiles must not use AI labels")
    nn_sorted = np.sort(np.asarray([row.signals.ai_nn for row in train_humans], dtype=np.float64))
    ratio_sorted = np.sort(
        np.asarray([row.signals.log_distance_ratio for row in train_humans], dtype=np.float64)
    )
    return TrainDistributions(
        language=language,
        fold_id=fold_id,
        nn_sorted=nn_sorted,
        ratio_sorted=ratio_sorted,
        train_question_ids=tuple(sorted({row.question_id for row in train_humans})),
    )


def _fold_thresholds(
    train_humans: Sequence[GateRow],
    distributions: TrainDistributions,
) -> dict[str, dict[str, float]]:
    nn_scores = [row.signals.ai_nn for row in train_humans]
    ratio_scores = [row.signals.log_distance_ratio for row in train_humans]
    and_scores = [
        and_score(
            human_percentile(distributions.nn_sorted, row.signals.ai_nn),
            human_percentile(distributions.ratio_sorted, row.signals.log_distance_ratio),
        )
        for row in train_humans
    ]
    return {
        "nn": _targets(nn_scores),
        "ratio": _targets(ratio_scores),
        "and": _targets(and_scores),
    }


def _targets(scores: Sequence[float]) -> dict[str, float]:
    return {
        "1%": conservative_threshold(scores, 0.01),
        "5%": conservative_threshold(scores, 0.05),
    }


def _transform_held_out(
    held: Sequence[GateRow],
    distributions: TrainDistributions,
    thresholds: Mapping[str, Mapping[str, float]],
    fold_id: int,
) -> list[OofRow]:
    rows: list[OofRow] = []
    for row in held:
        nn_p = human_percentile(distributions.nn_sorted, row.signals.ai_nn)
        ratio_p = human_percentile(distributions.ratio_sorted, row.signals.log_distance_ratio)
        combined = and_score(nn_p, ratio_p)
        rows.append(
            OofRow(
                row=row,
                fold_id=fold_id,
                nn_percentile=nn_p,
                ratio_percentile=ratio_p,
                and_score=combined,
                nn_flag_1=row.signals.ai_nn >= thresholds["nn"]["1%"],
                nn_flag_5=row.signals.ai_nn >= thresholds["nn"]["5%"],
                ratio_flag_1=row.signals.log_distance_ratio >= thresholds["ratio"]["1%"],
                ratio_flag_5=row.signals.log_distance_ratio >= thresholds["ratio"]["5%"],
                and_flag_1=combined >= thresholds["and"]["1%"],
                and_flag_5=combined >= thresholds["and"]["5%"],
            )
        )
    return rows


def _verify_b3_reuse(scored: Sequence[ScoredRow]) -> dict[str, object]:
    if not PHASE_B3_SCORES_PATH.is_file():
        return {"status": "skipped_missing_b3_json"}
    payload = json.loads(PHASE_B3_SCORES_PATH.read_text(encoding="utf-8"))
    metrics = payload["metrics"]["all_held_out"]
    checks = []
    for language in ("CPP", "PYTHON"):
        language_rows = [row for row in scored if row.language == language]
        checks.append(
            _compare_method_mean(
                language_rows,
                metrics[language]["methods"]["matched_population_nn"],
                "ai_nn",
            )
        )
        checks.append(
            _compare_method_mean(
                language_rows,
                metrics[language]["methods"]["log_distance_ratio"],
                "log_distance_ratio",
            )
        )
    failed = [item for item in checks if not item["ok"]]
    if failed:
        raise RuntimeError(f"Phase B.3 score reuse mismatch: {failed}")
    return {"status": "ok", "checks": checks}


def _compare_method_mean(
    rows: Sequence[ScoredRow],
    expected: Mapping[str, object],
    field: str,
) -> dict[str, object]:
    positives = [row for row in rows if row.label == LABEL_AI]
    negatives = [row for row in rows if row.label == LABEL_HUMAN]
    pos_mean = float(np.mean([_signal(row, field) for row in positives]))
    neg_mean = float(np.mean([_signal(row, field) for row in negatives]))
    expected_pos = expected["positives"]["mean"]
    expected_neg = expected["negatives"]["mean"]
    pos_ok = abs(pos_mean - float(expected_pos)) <= REUSE_TOLERANCE
    neg_ok = abs(neg_mean - float(expected_neg)) <= REUSE_TOLERANCE
    count_ok = (
        len(positives) == expected["n_positives"]
        and len(negatives) == expected["n_negatives"]
    )
    return {
        "field": field,
        "ok": pos_ok and neg_ok and count_ok,
        "n_positives": len(positives),
        "n_negatives": len(negatives),
        "pos_mean": pos_mean,
        "neg_mean": neg_mean,
    }


def _signal(row: ScoredRow, field: str) -> float:
    if field == "ai_nn":
        return row.ai_affinity.nn
    return row.contrast.log_distance_ratio


def _build_payload(
    population: object,
    subset: object,
    rows: Sequence[GateRow],
    oof_rows: Sequence[OofRow],
    fold_map: Mapping[str, int],
    match_keys: set[tuple[str, str, str]],
    verify: Mapping[str, object],
    eval_hash_before: str | None,
) -> dict[str, object]:
    variants = {
        "all_held_out": list(oof_rows),
        "exclude_exact_reference_match": [
            item
            for item in oof_rows
            if item.row.label != LABEL_AI or not item.row.exact_match
        ],
    }
    metrics = {
        name: _language_metrics(items, fold_map) for name, items in variants.items()
    }
    personas = {name: _persona_metrics(items) for name, items in variants.items()}
    tails = {name: _tail_diagnostics(items) for name, items in variants.items()}
    slices = _human_count_slices(variants["exclude_exact_reference_match"])
    verdict = decide_verdict(metrics, personas, slices)
    coverage = population.coverage
    return {
        "experiment": EXPERIMENT_NAME,
        "evaluation_mode": "cached_humans_only",
        "is_production": False,
        "is_final_all_human_evaluation": False,
        "bootstrap_iterations": 0,
        "reversible": True,
        "network_calls": False,
        "embeddings_generated": False,
        "percentile_note": (
            "Percentile distributions and AND thresholds use only training-fold "
            "human negatives. Exact-match AI positives are excluded only from "
            "the evaluated positive set; human empirical distributions are not retrained."
        ),
        "phase_b3_reuse": verify,
        "population": {
            "eligible_pairs": len(coverage.eligible_pairs),
            "eligible_pairs_cpp": sum(
                1 for _qid, language in coverage.eligible_pairs if language == "CPP"
            ),
            "eligible_pairs_python": sum(
                1 for _qid, language in coverage.eligible_pairs if language == "PYTHON"
            ),
            "cached_humans": sum(1 for row in rows if row.label == LABEL_HUMAN),
            "held_out_ai": sum(1 for row in rows if row.label == LABEL_AI),
            "same_cluster_ai_reference_matches": len(match_keys),
            "human_count_distribution": coverage.human_count_distribution,
            "pairs_excluded_no_cached_human_phase_b1": len(
                getattr(subset, "pairs_excluded_no_cached_human", ())
            ),
        },
        "metrics": metrics,
        "personas": personas,
        "tail_diagnostics": tails,
        "human_reference_slices": slices,
        "verdict": verdict,
        "eval_scores_sha256_before": eval_hash_before,
    }


def _language_metrics(
    oof_rows: Sequence[OofRow],
    fold_map: Mapping[str, int],
) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for language in ("CPP", "PYTHON"):
        language_rows = [item for item in oof_rows if item.row.language == language]
        nn_items = _labeled(language_rows, NN_METHOD)
        ratio_items = _labeled(language_rows, RATIO_METHOD)
        and_items = _labeled(language_rows, AND_METHOD)
        methods = {
            NN_METHOD: _bundle(NN_METHOD, nn_items, fold_map),
            RATIO_METHOD: _bundle(RATIO_METHOD, ratio_items, fold_map),
            AND_METHOD: _and_bundle(and_items, language_rows),
        }
        nn = methods[NN_METHOD]
        ratio = methods[RATIO_METHOD]
        and_metrics = methods[AND_METHOD]
        result[language] = {
            "n_pairs": len({(item.row.question_id, item.row.language) for item in language_rows}),
            "methods": methods,
            "and_minus_nn": _deltas(and_metrics, nn),
            "and_minus_ratio": _deltas(and_metrics, ratio),
        }
    return result


def _labeled(oof_rows: Sequence[OofRow], method: str) -> list[LabeledScore]:
    return [
        LabeledScore(
            question_id=item.row.question_id,
            pair_token=_pair_token(item.row.question_id, item.row.language),
            language=item.row.language,
            label=item.row.label,
            score=_method_score(item, method),
        )
        for item in oof_rows
    ]


def _method_score(item: OofRow, method: str) -> float:
    mapping = {
        NN_METHOD: item.row.signals.ai_nn,
        RATIO_METHOD: item.row.signals.log_distance_ratio,
        AND_METHOD: item.and_score,
    }
    return float(mapping[method])


def _bundle(
    method: str,
    items: Sequence[LabeledScore],
    fold_map: Mapping[str, int],
) -> dict[str, object]:
    positives = [item.score for item in items if item.label == LABEL_AI]
    negatives = [item.score for item in items if item.label == LABEL_HUMAN]
    pos_stats = score_statistics(positives)
    neg_stats = score_statistics(negatives)
    grouped = {
        "1%": _oof_point(items, fold_map, 0.01),
        "5%": _oof_point(items, fold_map, 0.05),
    }
    macro, macro_count = macro_auroc_by_question(items)
    row = MethodMetrics(
        method=method,
        n_positives=len(positives),
        n_negatives=len(negatives),
        pooled_auroc=pooled_auroc(positives, negatives),
        macro_auroc=macro,
        macro_question_count=macro_count,
        original_style={},
        grouped_out_of_fold=grouped,
        positives=pos_stats,
        negatives=neg_stats,
        mean_gap=_gap(pos_stats.mean, neg_stats.mean),
        median_gap=_gap(pos_stats.median, neg_stats.median),
    )
    return method_metrics_to_dict(row)


def _and_bundle(
    items: Sequence[LabeledScore],
    oof_rows: Sequence[OofRow],
) -> dict[str, object]:
    positives = [item.score for item in items if item.label == LABEL_AI]
    negatives = [item.score for item in items if item.label == LABEL_HUMAN]
    pos_stats = score_statistics(positives)
    neg_stats = score_statistics(negatives)
    macro, macro_count = macro_auroc_by_question(items)
    grouped = {
        "1%": _flag_point(oof_rows, 0.01),
        "5%": _flag_point(oof_rows, 0.05),
    }
    row = MethodMetrics(
        method=AND_METHOD,
        n_positives=len(positives),
        n_negatives=len(negatives),
        pooled_auroc=pooled_auroc(positives, negatives),
        macro_auroc=macro,
        macro_question_count=macro_count,
        original_style={},
        grouped_out_of_fold=grouped,
        positives=pos_stats,
        negatives=neg_stats,
        mean_gap=_gap(pos_stats.mean, neg_stats.mean),
        median_gap=_gap(pos_stats.median, neg_stats.median),
    )
    return method_metrics_to_dict(row)


def _oof_point(
    items: Sequence[LabeledScore],
    fold_map: Mapping[str, int],
    target_fpr: float,
) -> OperatingPoint:
    hits = {LABEL_AI: 0, LABEL_HUMAN: 0}
    totals = {LABEL_AI: 0, LABEL_HUMAN: 0}
    thresholds: list[float] = []
    for fold_id in range(GROUPED_FOLD_COUNT):
        train_neg = [
            item.score
            for item in items
            if item.label == LABEL_HUMAN and fold_map[item.question_id] != fold_id
        ]
        if not train_neg:
            continue
        threshold = conservative_threshold(train_neg, target_fpr)
        thresholds.append(threshold)
        for item in items:
            if fold_map[item.question_id] != fold_id:
                continue
            totals[item.label] += 1
            if item.score >= threshold:
                hits[item.label] += 1
    return OperatingPoint(
        target_fpr,
        float(np.mean(thresholds)) if thresholds else float("nan"),
        hits[LABEL_AI] / totals[LABEL_AI] if totals[LABEL_AI] else None,
        hits[LABEL_HUMAN] / totals[LABEL_HUMAN] if totals[LABEL_HUMAN] else None,
    )


def _flag_point(oof_rows: Sequence[OofRow], target_fpr: float) -> OperatingPoint:
    use_one = abs(target_fpr - 0.01) < 1e-12
    hits = {LABEL_AI: 0, LABEL_HUMAN: 0}
    totals = {LABEL_AI: 0, LABEL_HUMAN: 0}
    for item in oof_rows:
        totals[item.row.label] += 1
        flagged = item.and_flag_1 if use_one else item.and_flag_5
        if flagged:
            hits[item.row.label] += 1
    return OperatingPoint(
        target_fpr,
        float("nan"),
        hits[LABEL_AI] / totals[LABEL_AI] if totals[LABEL_AI] else None,
        hits[LABEL_HUMAN] / totals[LABEL_HUMAN] if totals[LABEL_HUMAN] else None,
    )


def _gap(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return float(left) - float(right)


def _deltas(
    left: Mapping[str, object],
    right: Mapping[str, object],
) -> dict[str, float | None]:
    return {
        "pooled_auroc": _optional_delta(left.get("pooled_auroc"), right.get("pooled_auroc")),
        "macro_auroc": _optional_delta(left.get("macro_auroc"), right.get("macro_auroc")),
        "oof_recall_1pct": _optional_delta(_oof_recall(left, "1%"), _oof_recall(right, "1%")),
        "oof_recall_5pct": _optional_delta(_oof_recall(left, "5%"), _oof_recall(right, "5%")),
    }


def _persona_metrics(oof_rows: Sequence[OofRow]) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for method in METHODS:
        result[method] = {}
        for persona in PERSONAS:
            positives = [
                item
                for item in oof_rows
                if item.row.label == LABEL_AI and item.row.source == persona
            ]
            result[method][persona] = {
                "n_positives": len(positives),
                "grouped_oof_recall_at_1pct_fpr": _persona_flag_rate(positives, method, 0.01),
                "grouped_oof_recall_at_5pct_fpr": _persona_flag_rate(positives, method, 0.05),
            }
    return result


def _persona_flag_rate(
    positives: Sequence[OofRow],
    method: str,
    target_fpr: float,
) -> float | None:
    if not positives:
        return None
    use_one = abs(target_fpr - 0.01) < 1e-12
    flags = []
    for item in positives:
        if method == AND_METHOD:
            flags.append(item.and_flag_1 if use_one else item.and_flag_5)
        elif method == NN_METHOD:
            flags.append(item.nn_flag_1 if use_one else item.nn_flag_5)
        else:
            flags.append(item.ratio_flag_1 if use_one else item.ratio_flag_5)
    return float(np.mean(flags))


def _tail_diagnostics(oof_rows: Sequence[OofRow]) -> dict[str, object]:
    by_language = {}
    for language in ("CPP", "PYTHON"):
        language_rows = [item for item in oof_rows if item.row.language == language]
        humans = [item for item in language_rows if item.row.label == LABEL_HUMAN]
        ais = [item for item in language_rows if item.row.label == LABEL_AI]
        nn_fp = [item for item in humans if item.nn_flag_1]
        and_fp = [item for item in humans if item.and_flag_1]
        nn_tp = [item for item in ais if item.nn_flag_1]
        and_tp = [item for item in ais if item.and_flag_1]
        nn_fp_ids = {(item.row.question_id, item.row.language, item.row.content_hash) for item in nn_fp}
        and_fp_ids = {(item.row.question_id, item.row.language, item.row.content_hash) for item in and_fp}
        nn_tp_ids = {(item.row.question_id, item.row.language, item.row.content_hash) for item in nn_tp}
        and_tp_ids = {(item.row.question_id, item.row.language, item.row.content_hash) for item in and_tp}
        by_language[language] = {
            "held_out_humans_above_1pct_and": len(and_fp),
            "held_out_ai_above_1pct_and": len(and_tp),
            "nn_false_positives": len(nn_fp),
            "and_false_positives": len(and_fp),
            "nn_false_positives_rejected_by_and": len(nn_fp_ids - and_fp_ids),
            "nn_true_positives_lost_by_and": len(nn_tp_ids - and_tp_ids),
            "new_ai_positives_caught_by_and": len(and_tp_ids - nn_tp_ids),
            "overlap_flagged_positives": len(nn_tp_ids & and_tp_ids),
            "and_false_positive_examples": [
                {
                    "question_id": item.row.question_id,
                    "language": item.row.language,
                    "nn_percentile": item.nn_percentile,
                    "ratio_percentile": item.ratio_percentile,
                    "and_score": item.and_score,
                }
                for item in and_fp[:30]
            ],
        }
    return by_language


def _human_count_slices(oof_rows: Sequence[OofRow]) -> dict[str, object]:
    buckets = {
        "2": [item for item in oof_rows if item.row.human_reference_count == 2],
        "3+": [item for item in oof_rows if item.row.human_reference_count >= 3],
    }
    result = {}
    for name, bucket in buckets.items():
        cpp = [item for item in bucket if item.row.language == "CPP"]
        result[name] = {
            "n_pairs": len({(item.row.question_id, item.row.language) for item in bucket}),
            "n_rows": len(bucket),
            "cpp_nn_recall_1pct": _flag_rate(cpp, "nn", LABEL_AI),
            "cpp_and_recall_1pct": _flag_rate(cpp, "and", LABEL_AI),
            "cpp_nn_fpr_1pct": _flag_rate(cpp, "nn", LABEL_HUMAN),
            "cpp_and_fpr_1pct": _flag_rate(cpp, "and", LABEL_HUMAN),
        }
    return result


def _flag_rate(rows: Sequence[OofRow], method: str, label: str) -> float | None:
    subset = [item for item in rows if item.row.label == label]
    if not subset:
        return None
    flags = [
        item.nn_flag_1 if method == "nn" else item.and_flag_1
        for item in subset
    ]
    return float(np.mean(flags))


def decide_verdict(
    metrics: Mapping[str, object],
    personas: Mapping[str, object],
    slices: Mapping[str, object],
) -> dict[str, object]:
    cpp = metrics["exclude_exact_reference_match"]["CPP"]
    and_metrics = cpp["methods"][AND_METHOD]
    nn = cpp["methods"][NN_METHOD]
    r1 = _oof_recall(and_metrics, "1%")
    fpr = _oof_fpr(and_metrics, "1%")
    r5_gain = _optional_delta(_oof_recall(and_metrics, "5%"), _oof_recall(nn, "5%"))
    macro_gain = _optional_delta(and_metrics.get("macro_auroc"), nn.get("macro_auroc"))
    persona_nn = _persona_value(personas, "exclude_exact_reference_match", NN_METHOD)
    persona_and = _persona_value(personas, "exclude_exact_reference_match", AND_METHOD)
    two = slices["2"]["cpp_and_recall_1pct"]
    two_nn = slices["2"]["cpp_nn_recall_1pct"]
    only_three = (
        two is not None
        and two_nn is not None
        and two <= two_nn
        and r1 is not None
        and r1 > (_oof_recall(nn, "1%") or 0)
    )
    promising = _is_promising(r1, fpr, r5_gain, macro_gain, persona_and, persona_nn, only_three)
    if promising:
        label = "promising"
    else:
        label = (
            "The percentile AND gate did not create sufficient low-FPR improvement. "
            "Keep NN and log-distance ratio as separate candidate combiner features. "
            "Stop canonicality-only tuning and proceed to structural AST features."
        )
    return {
        "label": label,
        "promising": promising,
        "cpp_and_oof_recall_1pct": r1,
        "cpp_and_oof_fpr_1pct": fpr,
        "and_minus_nn": cpp["and_minus_nn"],
        "and_minus_ratio": cpp["and_minus_ratio"],
        "production_review_nn": persona_nn,
        "production_review_and": persona_and,
        "two_human_cpp_and_recall_1pct": two,
        "two_human_cpp_nn_recall_1pct": two_nn,
        "gain_only_on_three_human_pairs": only_three,
    }


def _is_promising(
    r1: float | None,
    fpr: float | None,
    r5_gain: float | None,
    macro_gain: float | None,
    persona_and: float | None,
    persona_nn: float | None,
    only_three: bool,
) -> bool:
    if r1 is None or fpr is None or r5_gain is None or macro_gain is None:
        return False
    if r1 < CPP_RECALL_1_MIN:
        return False
    if fpr > MATERIAL_FPR_LIMIT:
        return False
    if r5_gain < -MATERIAL_R5_DROP:
        return False
    if macro_gain < -MACRO_DROP_MAX:
        return False
    if persona_and is not None and persona_nn is not None and persona_and < persona_nn:
        return False
    if only_three:
        return False
    return True


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


def _optional_delta(left: object, right: object) -> float | None:
    if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
        return None
    return float(left) - float(right)


def _persona_value(
    personas: Mapping[str, object],
    variant: str,
    method: str,
) -> float | None:
    payload = ((personas.get(variant) or {}).get(method) or {}).get("production_review")
    if not isinstance(payload, dict):
        return None
    value = payload.get("grouped_oof_recall_at_1pct_fpr")
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
        "delta_auroc_vs_nn",
        "delta_r1_vs_nn",
        "delta_auroc_vs_ratio",
        "delta_r1_vs_ratio",
    ]
    with SUMMARY_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(_summary_rows(payload))


def _summary_rows(payload: Mapping[str, object]) -> list[dict[str, object]]:
    rows = []
    for variant, languages in payload["metrics"].items():
        for language, body in languages.items():
            nn = body["methods"][NN_METHOD]
            ratio = body["methods"][RATIO_METHOD]
            for method, values in body["methods"].items():
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
                        "delta_auroc_vs_nn": _optional_delta(
                            values.get("pooled_auroc"),
                            nn.get("pooled_auroc"),
                        ),
                        "delta_r1_vs_nn": _optional_delta(
                            _oof_recall(values, "1%"),
                            _oof_recall(nn, "1%"),
                        ),
                        "delta_auroc_vs_ratio": _optional_delta(
                            values.get("pooled_auroc"),
                            ratio.get("pooled_auroc"),
                        ),
                        "delta_r1_vs_ratio": _optional_delta(
                            _oof_recall(values, "1%"),
                            _oof_recall(ratio, "1%"),
                        ),
                    }
                )
    return rows


def _render_report(payload: Mapping[str, object]) -> str:
    population = payload["population"]
    verdict = payload["verdict"]
    timings = payload.get("timings") or {}
    lines = [
        "# Percentile AND gate (Phase B.4)",
        "",
        "Fixed interpretable AND of NN and log-distance-ratio human percentiles.",
        "",
        f"- experiment: `{payload['experiment']}`",
        f"- evaluation_mode: `{payload['evaluation_mode']}`",
        f"- bootstrap_iterations: `{payload['bootstrap_iterations']}`",
        f"- reversible: `{payload['reversible']}`",
        f"- network_calls: `{payload['network_calls']}`",
        "",
        payload["percentile_note"],
        "",
        f"- Phase B.3 reuse: `{payload['phase_b3_reuse']['status']}`",
        "",
        "## Population",
        "",
        f"- eligible pairs: {population['eligible_pairs']} "
        f"(CPP {population['eligible_pairs_cpp']}, PYTHON {population['eligible_pairs_python']})",
        f"- cached humans: {population['cached_humans']}",
        f"- held-out AI: {population['held_out_ai']}",
        f"- same-cluster AI-reference exact matches: {population['same_cluster_ai_reference_matches']}",
        f"- human-count distribution: {population['human_count_distribution']}",
        "",
        "## Timings",
        "",
    ]
    for key, value in timings.items():
        lines.append(f"- {key}: {float(value):.3f}")
    lines.extend(["", "## CPP all held-out", ""])
    lines.extend(_table(payload, "all_held_out", "CPP"))
    lines.extend(["", "## Python all held-out", ""])
    lines.extend(_table(payload, "all_held_out", "PYTHON"))
    lines.extend(["", "## Exact-match-excluded", ""])
    lines.extend(_table(payload, "exclude_exact_reference_match", "CPP"))
    lines.extend(_table(payload, "exclude_exact_reference_match", "PYTHON"))
    lines.extend(["", "## production_review", ""])
    lines.extend(_persona_lines(payload))
    lines.extend(["", "## Tail diagnostics (exact-match-excluded)", ""])
    lines.append(str(payload["tail_diagnostics"]["exclude_exact_reference_match"]))
    lines.extend(["", "## Two-human versus three-human pairs", ""])
    lines.append(str(payload["human_reference_slices"]))
    lines.extend(
        [
            "",
            "## Verdict",
            "",
            f"**{verdict['label']}**",
            "",
            f"- C++ AND OOF R@1%: {_fmt(verdict['cpp_and_oof_recall_1pct'])} (need ≥ {CPP_RECALL_1_MIN})",
            f"- C++ AND FPR@1%: {_fmt(verdict['cpp_and_oof_fpr_1pct'])}",
            f"- AND minus NN: {verdict['and_minus_nn']}",
            f"- AND minus ratio: {verdict['and_minus_ratio']}",
            f"- production_review NN: {_fmt(verdict['production_review_nn'])}",
            f"- production_review AND: {_fmt(verdict['production_review_and'])}",
            f"- 2-human CPP AND R@1%: {_fmt(verdict['two_human_cpp_and_recall_1pct'])}",
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


if __name__ == "__main__":
    raise SystemExit(main())

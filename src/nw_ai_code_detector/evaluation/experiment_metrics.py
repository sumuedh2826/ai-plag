from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import time
from collections.abc import Mapping, Sequence

import numpy as np

from nw_ai_code_detector.constants import FPR_OPERATING_POINTS, SELECTION_RANDOM_SEED
from nw_ai_code_detector.scorer import _auroc

QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)
GROUPED_FOLD_COUNT = 5
UNIT_NORM_TOLERANCE = 1e-3
LABEL_AI = "ai"
LABEL_HUMAN = "human"


@dataclass(frozen=True)
class OperatingPoint:
    target_fpr: float
    threshold: float
    recall: float | None
    achieved_fpr: float | None


@dataclass(frozen=True)
class ScoreStats:
    count: int
    mean: float | None
    median: float | None
    std: float | None
    quantiles: dict[str, float]


@dataclass(frozen=True)
class MethodMetrics:
    method: str
    n_positives: int
    n_negatives: int
    pooled_auroc: float | None
    macro_auroc: float | None
    macro_question_count: int
    original_style: dict[str, OperatingPoint]
    grouped_out_of_fold: dict[str, OperatingPoint]
    positives: ScoreStats
    negatives: ScoreStats
    mean_gap: float | None
    median_gap: float | None


@dataclass(frozen=True)
class ClusterScoreGroup:
    token: str
    language: str
    nn_ai: tuple[float, ...]
    nn_human: tuple[float, ...]
    centroid_ai: tuple[float, ...]
    centroid_human: tuple[float, ...]


@dataclass(frozen=True)
class PairDeltaSummary:
    method: str
    n_pairs: int
    mean_delta: float | None
    median_delta: float | None
    quantiles: dict[str, float]
    pct_gt_zero: float | None
    pct_eq_zero: float | None
    pct_lt_zero: float | None
    per_question_mean_delta: dict[str, float]
    per_question_quantiles: dict[str, float]
    strongest_questions: list[dict[str, float | str]]
    weakest_questions: list[dict[str, float | str]]


@dataclass(frozen=True)
class LabeledScore:
    question_id: str
    pair_token: str
    language: str
    label: str
    score: float


def score_statistics(values: Sequence[float]) -> ScoreStats:
    if not values:
        return ScoreStats(count=0, mean=None, median=None, std=None, quantiles={})
    array = np.asarray(values, dtype=np.float64)
    quantile_map = {
        _quantile_key(q): float(np.quantile(array, q))
        for q in QUANTILES
    }
    return ScoreStats(
        count=len(values),
        mean=float(np.mean(array)),
        median=float(np.median(array)),
        std=float(np.std(array)),
        quantiles=quantile_map,
    )


def pooled_auroc(positives: Sequence[float], negatives: Sequence[float]) -> float | None:
    if not positives or not negatives:
        return None
    return _auroc(positives, negatives)


def original_operating_point(
    positives: Sequence[float],
    negatives: Sequence[float],
    target_fpr: float,
) -> OperatingPoint:
    if not negatives:
        return OperatingPoint(target_fpr, float("nan"), None, None)
    threshold = float(np.quantile(np.asarray(negatives, dtype=np.float64), 1.0 - target_fpr))
    return _apply_threshold(positives, negatives, target_fpr, threshold)


def conservative_threshold(negatives: Sequence[float], target_fpr: float) -> float:
    array = np.asarray(negatives, dtype=np.float64)
    unique_desc = np.sort(np.unique(array))[::-1]
    chosen = float(np.nextafter(float(unique_desc[0]), float("inf")))
    for candidate in unique_desc:
        achieved = float(np.mean(array >= candidate))
        if achieved <= target_fpr:
            chosen = float(candidate)
            continue
        break
    return chosen


def grouped_oof_operating_points(
    items: Sequence[LabeledScore],
    seed: int,
) -> dict[str, OperatingPoint]:
    fold_map = question_folds({item.question_id for item in items}, seed)
    return {
        f"{int(target * 100)}%": _oof_point_for_target(items, fold_map, target)
        for target in FPR_OPERATING_POINTS
    }


def language_calibrated_operating_points(
    items: Sequence[LabeledScore],
    seed: int,
) -> dict[str, dict[str, OperatingPoint]]:
    original = {
        f"{int(target * 100)}%": _language_calibrated_original(items, target)
        for target in FPR_OPERATING_POINTS
    }
    fold_map = question_folds({item.question_id for item in items}, seed)
    grouped_oof = {
        f"{int(target * 100)}%": _language_calibrated_oof(
            items,
            fold_map,
            target,
        )
        for target in FPR_OPERATING_POINTS
    }
    return {
        "original_style": original,
        "grouped_out_of_fold": grouped_oof,
    }


def macro_auroc_by_question(items: Sequence[LabeledScore]) -> tuple[float | None, int]:
    grouped: dict[str, dict[str, list[float]]] = {}
    for item in items:
        bucket = grouped.setdefault(item.pair_token, {"ai": [], "human": []})
        bucket[item.label].append(item.score)
    values = [
        _auroc(bucket["ai"], bucket["human"])
        for bucket in grouped.values()
        if bucket["ai"] and bucket["human"]
    ]
    if not values:
        return None, 0
    return float(np.mean(values)), len(values)


def method_metrics_bundle(
    method: str,
    labeled_scores: Sequence[LabeledScore],
    seed: int,
) -> MethodMetrics:
    positives = [item.score for item in labeled_scores if item.label == LABEL_AI]
    negatives = [item.score for item in labeled_scores if item.label == LABEL_HUMAN]
    pos_stats = score_statistics(positives)
    neg_stats = score_statistics(negatives)
    original = {
        f"{int(target * 100)}%": original_operating_point(positives, negatives, target)
        for target in FPR_OPERATING_POINTS
    }
    grouped = grouped_oof_operating_points(labeled_scores, seed)
    macro, macro_count = macro_auroc_by_question(labeled_scores)
    mean_gap = None
    median_gap = None
    if pos_stats.mean is not None and neg_stats.mean is not None:
        mean_gap = pos_stats.mean - neg_stats.mean
    if pos_stats.median is not None and neg_stats.median is not None:
        median_gap = pos_stats.median - neg_stats.median
    return MethodMetrics(
        method=method,
        n_positives=len(positives),
        n_negatives=len(negatives),
        pooled_auroc=pooled_auroc(positives, negatives),
        macro_auroc=macro,
        macro_question_count=macro_count,
        original_style=original,
        grouped_out_of_fold=grouped,
        positives=pos_stats,
        negatives=neg_stats,
        mean_gap=mean_gap,
        median_gap=median_gap,
    )


def cluster_bootstrap_deltas(
    groups: Sequence[ClusterScoreGroup],
    seed: int,
    iterations: int,
) -> dict[str, dict[str, float | None]]:
    rng = np.random.default_rng(seed)
    n_groups = len(groups)
    if n_groups == 0:
        return {}
    pooled_diffs: list[float] = []
    macro_diffs: list[float] = []
    recall_1_diffs: list[float] = []
    recall_5_diffs: list[float] = []
    grouped_oof_recall_1_diffs: list[float] = []
    grouped_oof_recall_5_diffs: list[float] = []
    calibrated_oof_recall_1_diffs: list[float] = []
    calibrated_oof_recall_5_diffs: list[float] = []
    started = time.perf_counter()
    for iteration in range(iterations):
        indexes = rng.integers(0, n_groups, size=n_groups)
        nn_labeled: list[LabeledScore] = []
        centroid_labeled: list[LabeledScore] = []
        for sample_index, group_index in enumerate(indexes):
            group = groups[int(group_index)]
            sample_id = f"{group.token}#{sample_index}"
            _append_labeled(
                nn_labeled,
                sample_id,
                group.language,
                group.nn_ai,
                group.nn_human,
            )
            _append_labeled(
                centroid_labeled,
                sample_id,
                group.language,
                group.centroid_ai,
                group.centroid_human,
            )
        pooled_diffs.append(_metric_delta(centroid_labeled, nn_labeled, "pooled"))
        macro_diffs.append(_metric_delta(centroid_labeled, nn_labeled, "macro"))
        recall_1_diffs.append(_metric_delta(centroid_labeled, nn_labeled, "r1"))
        recall_5_diffs.append(_metric_delta(centroid_labeled, nn_labeled, "r5"))
        grouped_oof_recall_1_diffs.append(
            _metric_delta(centroid_labeled, nn_labeled, "oof_r1")
        )
        grouped_oof_recall_5_diffs.append(
            _metric_delta(centroid_labeled, nn_labeled, "oof_r5")
        )
        calibrated_oof_recall_1_diffs.append(
            _metric_delta(centroid_labeled, nn_labeled, "calibrated_oof_r1")
        )
        calibrated_oof_recall_5_diffs.append(
            _metric_delta(centroid_labeled, nn_labeled, "calibrated_oof_r5")
        )
        completed = iteration + 1
        if completed % 25 == 0 or completed == iterations:
            elapsed = time.perf_counter() - started
            print(
                f"Bootstrap progress: {completed}/{iterations} "
                f"elapsed={elapsed:.1f}s"
            )
    return {
        "original_style_pooled_auroc": _ci(pooled_diffs),
        "original_style_macro_auroc": _ci(macro_diffs),
        "original_style_recall_at_1pct_fpr": _ci(recall_1_diffs),
        "original_style_recall_at_5pct_fpr": _ci(recall_5_diffs),
        "grouped_oof_pooled_single_threshold_recall_at_1pct_fpr": _ci(
            grouped_oof_recall_1_diffs
        ),
        "grouped_oof_pooled_single_threshold_recall_at_5pct_fpr": _ci(
            grouped_oof_recall_5_diffs
        ),
        "grouped_oof_language_calibrated_recall_at_1pct_fpr": _ci(
            calibrated_oof_recall_1_diffs
        ),
        "grouped_oof_language_calibrated_recall_at_5pct_fpr": _ci(
            calibrated_oof_recall_5_diffs
        ),
    }


def summarize_pair_deltas(
    method: str,
    deltas: Sequence[float],
    per_question: Mapping[str, float],
) -> PairDeltaSummary:
    stats = score_statistics(deltas)
    array = np.asarray(deltas, dtype=np.float64) if deltas else np.asarray([], dtype=np.float64)
    pct_gt = pct_eq = pct_lt = None
    if len(array):
        pct_gt = float(np.mean(array > 0) * 100)
        pct_eq = float(np.mean(array == 0) * 100)
        pct_lt = float(np.mean(array < 0) * 100)
    ranked = sorted(per_question.items(), key=lambda item: item[1], reverse=True)
    question_values = list(per_question.values())
    return PairDeltaSummary(
        method=method,
        n_pairs=len(deltas),
        mean_delta=stats.mean,
        median_delta=stats.median,
        quantiles=stats.quantiles,
        pct_gt_zero=pct_gt,
        pct_eq_zero=pct_eq,
        pct_lt_zero=pct_lt,
        per_question_mean_delta=dict(per_question),
        per_question_quantiles=score_statistics(question_values).quantiles,
        strongest_questions=_question_rank_rows(ranked[:10]),
        weakest_questions=_question_rank_rows(list(reversed(ranked[-10:]))),
    )


def operating_point_to_dict(point: OperatingPoint) -> dict[str, float | None]:
    return {
        "target_fpr": point.target_fpr,
        "threshold": point.threshold,
        "recall": point.recall,
        "achieved_fpr": point.achieved_fpr,
    }


def method_metrics_to_dict(row: MethodMetrics) -> dict[str, object]:
    return {
        "method": row.method,
        "n_positives": row.n_positives,
        "n_negatives": row.n_negatives,
        "pooled_auroc": row.pooled_auroc,
        "macro_auroc": row.macro_auroc,
        "macro_question_count": row.macro_question_count,
        "original_style": {
            key: operating_point_to_dict(value)
            for key, value in row.original_style.items()
        },
        "grouped_out_of_fold": {
            key: operating_point_to_dict(value)
            for key, value in row.grouped_out_of_fold.items()
        },
        "positives": _stats_to_dict(row.positives),
        "negatives": _stats_to_dict(row.negatives),
        "mean_gap": row.mean_gap,
        "median_gap": row.median_gap,
    }


def pair_delta_to_dict(row: PairDeltaSummary) -> dict[str, object]:
    return {
        "method": row.method,
        "n_pairs": row.n_pairs,
        "mean_delta": row.mean_delta,
        "median_delta": row.median_delta,
        "quantiles": row.quantiles,
        "pct_gt_zero": row.pct_gt_zero,
        "pct_eq_zero": row.pct_eq_zero,
        "pct_lt_zero": row.pct_lt_zero,
        "per_question_mean_delta": row.per_question_mean_delta,
        "per_question_quantiles": row.per_question_quantiles,
        "strongest_questions": row.strongest_questions,
        "weakest_questions": row.weakest_questions,
    }


def _oof_point_for_target(
    items: Sequence[LabeledScore],
    fold_map: Mapping[str, int],
    target_fpr: float,
) -> OperatingPoint:
    thresholds: list[float] = []
    for fold_id in range(GROUPED_FOLD_COUNT):
        threshold = _fold_threshold(items, fold_map, fold_id, target_fpr)
        if threshold is None:
            continue
        thresholds.append(threshold)
    recall = _oof_rate(items, fold_map, target_fpr, LABEL_AI)
    achieved = _oof_rate(items, fold_map, target_fpr, LABEL_HUMAN)
    threshold_mean = float(np.mean(thresholds)) if thresholds else float("nan")
    return OperatingPoint(target_fpr, threshold_mean, recall, achieved)


def _language_calibrated_original(
    items: Sequence[LabeledScore],
    target_fpr: float,
) -> OperatingPoint:
    thresholds = _language_thresholds(items, target_fpr, conservative=False)
    return _apply_language_thresholds(items, target_fpr, thresholds)


def _language_calibrated_oof(
    items: Sequence[LabeledScore],
    fold_map: Mapping[str, int],
    target_fpr: float,
) -> OperatingPoint:
    hits = {LABEL_AI: 0, LABEL_HUMAN: 0}
    totals = {LABEL_AI: 0, LABEL_HUMAN: 0}
    thresholds: list[float] = []
    for fold_id in range(GROUPED_FOLD_COUNT):
        training = [
            item
            for item in items
            if fold_map[item.question_id] != fold_id
        ]
        fold_thresholds = _language_thresholds(
            training,
            target_fpr,
            conservative=True,
        )
        thresholds.extend(fold_thresholds.values())
        for item in items:
            if fold_map[item.question_id] != fold_id:
                continue
            threshold = fold_thresholds.get(item.language)
            if threshold is None:
                continue
            totals[item.label] += 1
            if item.score >= threshold:
                hits[item.label] += 1
    recall = _safe_rate(hits[LABEL_AI], totals[LABEL_AI])
    achieved_fpr = _safe_rate(hits[LABEL_HUMAN], totals[LABEL_HUMAN])
    threshold_mean = float(np.mean(thresholds)) if thresholds else float("nan")
    return OperatingPoint(target_fpr, threshold_mean, recall, achieved_fpr)


def _language_thresholds(
    items: Sequence[LabeledScore],
    target_fpr: float,
    conservative: bool,
) -> dict[str, float]:
    languages = {item.language for item in items}
    thresholds: dict[str, float] = {}
    for language in languages:
        negatives = [
            item.score
            for item in items
            if item.language == language and item.label == LABEL_HUMAN
        ]
        if negatives:
            if conservative:
                thresholds[language] = conservative_threshold(
                    negatives,
                    target_fpr,
                )
            else:
                thresholds[language] = float(
                    np.quantile(
                        np.asarray(negatives, dtype=np.float64),
                        1.0 - target_fpr,
                    )
                )
    return thresholds


def _apply_language_thresholds(
    items: Sequence[LabeledScore],
    target_fpr: float,
    thresholds: Mapping[str, float],
) -> OperatingPoint:
    hits = {LABEL_AI: 0, LABEL_HUMAN: 0}
    totals = {LABEL_AI: 0, LABEL_HUMAN: 0}
    for item in items:
        threshold = thresholds.get(item.language)
        if threshold is None:
            continue
        totals[item.label] += 1
        if item.score >= threshold:
            hits[item.label] += 1
    recall = _safe_rate(hits[LABEL_AI], totals[LABEL_AI])
    achieved_fpr = _safe_rate(hits[LABEL_HUMAN], totals[LABEL_HUMAN])
    threshold_mean = float(np.mean(list(thresholds.values())))
    return OperatingPoint(target_fpr, threshold_mean, recall, achieved_fpr)


def _oof_rate(
    items: Sequence[LabeledScore],
    fold_map: Mapping[str, int],
    target_fpr: float,
    label: str,
) -> float | None:
    hits = 0
    total = 0
    for fold_id in range(GROUPED_FOLD_COUNT):
        threshold = _fold_threshold(items, fold_map, fold_id, target_fpr)
        if threshold is None:
            continue
        for item in items:
            if fold_map[item.question_id] != fold_id or item.label != label:
                continue
            total += 1
            if item.score >= threshold:
                hits += 1
    if total == 0:
        return None
    return hits / total


def _fold_threshold(
    items: Sequence[LabeledScore],
    fold_map: Mapping[str, int],
    fold_id: int,
    target_fpr: float,
) -> float | None:
    train_neg = [
        item.score
        for item in items
        if item.label == LABEL_HUMAN and fold_map[item.question_id] != fold_id
    ]
    if not train_neg:
        return None
    return conservative_threshold(train_neg, target_fpr)


def question_folds(question_ids: set[str], seed: int) -> dict[str, int]:
    assignments = _cached_question_folds(tuple(sorted(question_ids)), seed)
    return dict(assignments)


@lru_cache(maxsize=32)
def _cached_question_folds(
    question_ids: tuple[str, ...],
    seed: int,
) -> tuple[tuple[str, int], ...]:
    ordered = np.asarray(question_ids)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(ordered)
    mapping: dict[str, int] = {}
    folds = np.array_split(shuffled, GROUPED_FOLD_COUNT)
    for fold_id, fold in enumerate(folds):
        for question_id in fold:
            mapping[str(question_id)] = fold_id
    return tuple(sorted(mapping.items()))


def _apply_threshold(
    positives: Sequence[float],
    negatives: Sequence[float],
    target_fpr: float,
    threshold: float,
) -> OperatingPoint:
    recall = None
    achieved = None
    if positives:
        recall = float(np.mean(np.asarray(positives) >= threshold))
    if negatives:
        achieved = float(np.mean(np.asarray(negatives) >= threshold))
    return OperatingPoint(target_fpr, threshold, recall, achieved)


def _quantile_key(value: float) -> str:
    return f"p{int(value * 100)}"


def _stats_to_dict(stats: ScoreStats) -> dict[str, object]:
    return {
        "count": stats.count,
        "mean": stats.mean,
        "median": stats.median,
        "std": stats.std,
        "quantiles": stats.quantiles,
    }


def _question_rank_rows(
    ranked: Sequence[tuple[str, float]],
) -> list[dict[str, float | str]]:
    return [
        {"question_language": token, "mean_delta": delta}
        for token, delta in ranked
    ]


def _append_labeled(
    dest: list[LabeledScore],
    sample_id: str,
    language: str,
    ai_scores: Sequence[float],
    human_scores: Sequence[float],
) -> None:
    dest.extend(
        LabeledScore(sample_id, sample_id, language, LABEL_AI, float(score))
        for score in ai_scores
    )
    dest.extend(
        LabeledScore(sample_id, sample_id, language, LABEL_HUMAN, float(score))
        for score in human_scores
    )


def _metric_delta(
    centroid_items: Sequence[LabeledScore],
    nn_items: Sequence[LabeledScore],
    kind: str,
) -> float:
    centroid_value = _metric_value(centroid_items, kind)
    nn_value = _metric_value(nn_items, kind)
    if centroid_value is None or nn_value is None:
        return float("nan")
    return centroid_value - nn_value


def _metric_value(items: Sequence[LabeledScore], kind: str) -> float | None:
    positives = [item.score for item in items if item.label == LABEL_AI]
    negatives = [item.score for item in items if item.label == LABEL_HUMAN]
    if kind == "pooled":
        return pooled_auroc(positives, negatives)
    if kind == "macro":
        value, _count = macro_auroc_by_question(items)
        return value
    if kind == "r1":
        return original_operating_point(positives, negatives, 0.01).recall
    if kind == "r5":
        return original_operating_point(positives, negatives, 0.05).recall
    target_fpr = (
        0.01
        if kind in {"oof_r1", "calibrated_oof_r1"}
        else 0.05
    )
    if kind.startswith("calibrated_"):
        point = language_calibrated_operating_points(
            items,
            SELECTION_RANDOM_SEED,
        )["grouped_out_of_fold"][f"{int(target_fpr * 100)}%"]
        return point.recall
    point = grouped_oof_operating_points(items, SELECTION_RANDOM_SEED)[
        f"{int(target_fpr * 100)}%"
    ]
    return point.recall


def _safe_rate(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _ci(values: Sequence[float]) -> dict[str, float | None]:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return {"mean": None, "low": None, "high": None}
    return {
        "mean": float(np.mean(finite)),
        "low": float(np.percentile(finite, 2.5)),
        "high": float(np.percentile(finite, 97.5)),
    }

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import numpy as np
from sklearn.metrics import roc_auc_score

from nw_ai_code_detector.constants import FPR_OPERATING_POINTS
from nw_ai_code_detector.index import ClusterKey, ReferenceIndex

NN_NEIGHBOR_COUNT = 1


@dataclass(frozen=True)
class ScoredItem:
    question_id: str
    language: str
    label: str
    source: str
    nn_score: float


@dataclass(frozen=True)
class LanguageMetrics:
    language: str
    n_positives: int
    n_negatives: int
    excluded_pairs: int
    auroc_nn: float | None
    recall_at_fpr: dict[str, float | None]
    positive_nn_mean: float | None
    negative_nn_mean: float | None
    positive_nn_std: float | None
    negative_nn_std: float | None


def score_item(
    index: ReferenceIndex,
    key: ClusterKey,
    vector: Sequence[float],
) -> float:
    scores = index.search(key, vector, NN_NEIGHBOR_COUNT)
    return scores[0]


def metrics_for_items(
    items: Sequence[ScoredItem],
    language: str | None,
    excluded_pairs: int,
) -> LanguageMetrics:
    scoped = [
        item
        for item in items
        if language is None or item.language == language
    ]
    positives = [item.nn_score for item in scoped if item.label == "ai"]
    negatives = [item.nn_score for item in scoped if item.label == "human"]
    language_label = language or "COMBINED"
    if not positives or not negatives:
        return _empty_metrics(language_label, excluded_pairs, len(positives), len(negatives))
    recall = {
        f"{int(fpr * 100)}%": _recall_at_fpr(positives, negatives, fpr)
        for fpr in FPR_OPERATING_POINTS
    }
    return LanguageMetrics(
        language=language_label,
        n_positives=len(positives),
        n_negatives=len(negatives),
        excluded_pairs=excluded_pairs,
        auroc_nn=_auroc(positives, negatives),
        recall_at_fpr=recall,
        positive_nn_mean=float(np.mean(positives)),
        negative_nn_mean=float(np.mean(negatives)),
        positive_nn_std=float(np.std(positives)),
        negative_nn_std=float(np.std(negatives)),
    )


def histogram_lines(values: Sequence[float], title: str) -> list[str]:
    if not values:
        return [f"{title}: (empty)"]
    low = min(values)
    high = max(values)
    if low == high:
        return [f"{title}: all={low:.4f} n={len(values)}"]
    bins = np.linspace(min(values), max(values), 11)
    counts, edges = np.histogram(values, bins=bins)
    widest = max(int(count) for count in counts) or 1
    lines = [title]
    for count, left, right in zip(counts, edges[:-1], edges[1:]):
        bar = "#" * max(1, int(20 * count / widest)) if count else ""
        lines.append(f"  {left:.3f}-{right:.3f} {int(count):4d} {bar}")
    return lines


def _auroc(positives: Sequence[float], negatives: Sequence[float]) -> float:
    labels = [1] * len(positives) + [0] * len(negatives)
    scores = list(positives) + list(negatives)
    return float(roc_auc_score(labels, scores))


def _recall_at_fpr(
    positives: Sequence[float],
    negatives: Sequence[float],
    fpr: float,
) -> float:
    threshold = float(np.quantile(negatives, 1.0 - fpr))
    hits = sum(1 for score in positives if score >= threshold)
    return hits / len(positives)


def _empty_metrics(
    language: str,
    excluded_pairs: int,
    n_positives: int,
    n_negatives: int,
) -> LanguageMetrics:
    return LanguageMetrics(
        language=language,
        n_positives=n_positives,
        n_negatives=n_negatives,
        excluded_pairs=excluded_pairs,
        auroc_nn=None,
        recall_at_fpr={f"{int(fpr * 100)}%": None for fpr in FPR_OPERATING_POINTS},
        positive_nn_mean=None,
        negative_nn_mean=None,
        positive_nn_std=None,
        negative_nn_std=None,
    )

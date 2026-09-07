from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nw_ai_code_detector.canonicality_eligibility import (
    significant_token_threshold_for_language,
)
from nw_ai_code_detector.constants import (
    CLUSTER_LOW_DIVERSITY_DISTANCE,
    COMMENTED_OUT_CODE_DISCOUNT,
    DESCRIPTIVE_RAISE_FLOOR,
    HAS_CANONICALITY_SCORE_STATUSES,
    HIGH_AI_SCORE_FLOOR,
    HIGH_CONFIDENCE_LABEL,
    INSUFFICIENT_EVIDENCE_STATUS,
    INSUFFICIENT_TOKENS_REASON,
    LOW_CLUSTER_DIVERSITY_REASON,
    LOW_CONFIDENCE_SHORT_REASON,
    LOW_CONFIDENCE_SHORT_STATUS,
    LOW_CONFIDENCE_STATUS,
    SCORED_STATUS,
    SHORT_LOW_CONFIDENCE_MAX_TOKENS_BY_LANGUAGE,
)


@dataclass(frozen=True)
class CanonicalityDiscountRequest:
    raw_canonicality: float
    top3_mean: float
    token_count: int
    language: str
    cluster_diversity: float
    commented_out_code_present: bool
    frac_descriptive: float
    convention_frac: float


@dataclass(frozen=True)
class ConfidenceRouting:
    status: str
    reason: str | None
    short_code_below_floor: bool
    low_cluster_diversity: bool


@dataclass(frozen=True)
class CommentedOutDiscount:
    present: bool
    discount: float


@dataclass(frozen=True)
class DescriptiveRaise:
    high: bool
    applied: bool
    amount: float
    frac_descriptive: float


@dataclass(frozen=True)
class ConventionRaise:
    frac: float
    applied: bool
    amount: float


@dataclass(frozen=True)
class CanonicalityAssessment:
    routing: ConfidenceRouting
    raw_canonicality: float
    top3_mean: float
    token_count: int
    cluster_diversity: float
    commented_out_code: CommentedOutDiscount
    descriptive_raise: DescriptiveRaise
    convention_raise: ConventionRaise
    remaining_factor: float
    score: float | None
    confidence: str | None


def mean_pairwise_cosine_distance(vectors: np.ndarray) -> float:
    count = int(vectors.shape[0])
    if count < 2:
        return 1.0
    similarities = vectors @ vectors.T
    upper = np.triu_indices(count, k=1)
    distances = 1.0 - similarities[upper]
    return float(np.mean(distances))


def cluster_is_low_diversity(diversity: float) -> bool:
    return diversity < CLUSTER_LOW_DIVERSITY_DISTANCE


def assess_canonicality(request: CanonicalityDiscountRequest) -> CanonicalityAssessment:
    """Route unscoreable cases out; discount commented-out code; naming is confidence-only."""
    routing = _confidence_routing(request.token_count, request.language, request.cluster_diversity)
    commented_out = _commented_out_discount(request.commented_out_code_present)
    descriptive = _descriptive_reading(request.frac_descriptive)
    convention = _convention_raise(request.convention_frac)
    if routing.status not in HAS_CANONICALITY_SCORE_STATUSES:
        return CanonicalityAssessment(
            routing,
            request.raw_canonicality,
            request.top3_mean,
            request.token_count,
            request.cluster_diversity,
            commented_out,
            descriptive,
            convention,
            1.0,
            None,
            None,
        )
    remaining = 1.0 - commented_out.discount
    discounted = request.raw_canonicality * remaining
    if discounted > request.raw_canonicality:
        raise RuntimeError("Discount layer raised canonicality")
    score = discounted
    confidence = _naming_confidence_label(score, request.frac_descriptive)
    return CanonicalityAssessment(
        routing,
        request.raw_canonicality,
        request.top3_mean,
        request.token_count,
        request.cluster_diversity,
        commented_out,
        descriptive,
        convention,
        remaining,
        score,
        confidence,
    )


def _confidence_routing(
    token_count: int,
    language: str,
    cluster_diversity: float,
) -> ConfidenceRouting:
    floor = significant_token_threshold_for_language(language)
    if floor is None:
        raise ValueError(f"Unsupported language: {language}")
    short = token_count < floor
    tight = cluster_is_low_diversity(cluster_diversity)
    if short:
        return ConfidenceRouting(
            INSUFFICIENT_EVIDENCE_STATUS,
            INSUFFICIENT_TOKENS_REASON,
            True,
            tight,
        )
    if tight:
        return ConfidenceRouting(
            LOW_CONFIDENCE_STATUS,
            LOW_CLUSTER_DIVERSITY_REASON,
            False,
            True,
        )
    if token_count <= _short_low_confidence_max_tokens(language):
        return ConfidenceRouting(
            LOW_CONFIDENCE_SHORT_STATUS,
            LOW_CONFIDENCE_SHORT_REASON,
            False,
            False,
        )
    return ConfidenceRouting(SCORED_STATUS, None, False, False)


def _short_low_confidence_max_tokens(language: str) -> int:
    ceiling = SHORT_LOW_CONFIDENCE_MAX_TOKENS_BY_LANGUAGE.get(language)
    if ceiling is None:
        raise ValueError(f"Unsupported language: {language}")
    return ceiling


def _commented_out_discount(present: bool) -> CommentedOutDiscount:
    discount = COMMENTED_OUT_CODE_DISCOUNT if present else 0.0
    return CommentedOutDiscount(present, discount)


def _descriptive_reading(frac_descriptive: float) -> DescriptiveRaise:
    high = frac_descriptive >= DESCRIPTIVE_RAISE_FLOOR
    return DescriptiveRaise(high, False, 0.0, frac_descriptive)


def _convention_raise(convention_frac: float) -> ConventionRaise:
    return ConventionRaise(convention_frac, False, 0.0)


def _naming_confidence_label(score: float, frac_descriptive: float) -> str | None:
    if score < HIGH_AI_SCORE_FLOOR:
        return None
    if frac_descriptive < DESCRIPTIVE_RAISE_FLOOR:
        return None
    return HIGH_CONFIDENCE_LABEL

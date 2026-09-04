from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from collections.abc import Mapping, Set

from nw_ai_code_detector.canonicality_eligibility import (
    SubmissionEligibilityDecision,
    evaluate_submission_eligibility,
    significant_token_threshold_for_language,
)
from nw_ai_code_detector.constants import (
    EXACT_REFERENCE_MATCH_HINT,
    INSUFFICIENT_EVIDENCE_STATUS,
    INSUFFICIENT_TOKENS_REASON,
    LOW_CLUSTER_DIVERSITY_REASON,
    LOW_CONFIDENCE_STATUS,
    MISSING_OR_INVALID_STATUS,
    StyleSignalName,
    SubmissionExclusionReason,
)
from nw_ai_code_detector.discount_layer import (
    CanonicalityDiscountRequest,
    assess_canonicality,
    cluster_is_low_diversity,
    mean_pairwise_cosine_distance,
)
from nw_ai_code_detector.display_match import display_match_percent
from nw_ai_code_detector.embedder import TextEmbedder
from nw_ai_code_detector.index import ClusterKey, ReferenceIndex
from nw_ai_code_detector.scorer import score_item
from nw_ai_code_detector.significant_code_tokens import (
    SignificantCodeTokenizationError,
    significant_code_token_count,
)
from nw_ai_code_detector.explanation_card import (
    ExplanationCard,
    ExplanationCardRequest,
    build_explanation_card,
    build_unscored_explanation_card,
    format_explanation_card,
)
from nw_ai_code_detector.similarity_explanation import (
    find_nearest_generated_reference_from_cluster,
)
from nw_ai_code_detector.stripper import (
    SourceParseError,
    parses_source,
    strip_solution_body,
)
from nw_ai_code_detector.style_signals import (
    StyleFlag,
    extract_style_flags,
    naming_fractions,
    uncomputed_style_flags,
)


@dataclass(frozen=True)
class SubmissionScoreRequest:
    raw_code: str
    boilerplate: str
    question_id: str
    language: str


@dataclass(frozen=True)
class CanonicalityReading:
    stripped: str
    token_count: int
    canonicality: float
    top3_mean: float
    cluster_diversity: float
    query_vector: tuple[float, ...]
    reference_vectors: np.ndarray


@dataclass(frozen=True)
class DetectionResult:
    status: str
    reason: str | None
    score: float | None
    decision: str | None
    canonicality: float | None
    token_count: int | None
    exact_match_flag: bool
    signals: tuple[StyleFlag, ...]
    explanation: str | None
    confidence: str | None
    display_match_percent: int | None
    explanation_card: ExplanationCard | None


def score_submission(
    request: SubmissionScoreRequest,
    index: ReferenceIndex,
    embedder: TextEmbedder,
    reference_hashes: Mapping[tuple[str, str], Set[str]],
) -> DetectionResult:
    """Strip, gate, embed eligible code, then attach rank-and-flag evidence."""
    stripped, integrity = _integrity_result(request, index)
    if integrity is not None:
        return integrity
    token_count = _token_count(stripped, request.language)
    cluster_ok = _cluster_exists(request, index)
    decision = evaluate_submission_eligibility(
        token_count,
        request.language,
        cluster_ok,
        token_count is not None,
    )
    if not decision.eligible:
        return _ineligible_result(decision, token_count)
    diversity = _cluster_diversity(request, index)
    if cluster_is_low_diversity(diversity):
        return _blank_result(
            LOW_CONFIDENCE_STATUS,
            LOW_CLUSTER_DIVERSITY_REASON,
            token_count,
        )
    batch = embedder.embed_texts((stripped,))
    key = ClusterKey(request.question_id, request.language)
    nn_score, top3_mean = score_item(index, key, batch.vectors[0])
    reading = CanonicalityReading(
        stripped,
        token_count or 0,
        nn_score,
        top3_mean,
        diversity,
        batch.vectors[0],
        index.get_cluster(key).vectors,
    )
    return _scored_result(request, reading, reference_hashes)


def _scored_result(
    request: SubmissionScoreRequest,
    reading: CanonicalityReading,
    reference_hashes: Mapping[tuple[str, str], Set[str]],
) -> DetectionResult:
    flags = extract_style_flags(
        request.raw_code,
        reading.stripped,
        request.language,
        request.boilerplate,
    )
    exact_match = _exact_reference_match(reading.stripped, request, reference_hashes)
    hint = EXACT_REFERENCE_MATCH_HINT if exact_match else None
    commented = any(
        flag.fired and flag.name == StyleSignalName.COMMENTED_OUT_CODE.value
        for flag in flags
    )
    assessment = assess_canonicality(
        CanonicalityDiscountRequest(
            reading.canonicality,
            reading.top3_mean,
            reading.token_count,
            request.language,
            reading.cluster_diversity,
            commented,
            naming_fractions(reading.stripped, request.language).frac_descriptive,
        )
    )
    match = find_nearest_generated_reference_from_cluster(
        ClusterKey(request.question_id, request.language),
        reading.stripped,
        reading.query_vector,
        reading.reference_vectors,
    )
    card = build_explanation_card(
        ExplanationCardRequest(
            assessment,
            match,
            request.language,
            reading.cluster_diversity,
        )
    )
    explanation = format_explanation_card(card)
    if exact_match:
        explanation = f"{explanation} Exact reference match - review."
    percent = None
    if assessment.score is not None:
        percent = display_match_percent(assessment.score, request.language)
    return DetectionResult(
        status=assessment.routing.status,
        reason=assessment.routing.reason,
        score=assessment.score,
        decision=hint,
        canonicality=reading.canonicality,
        token_count=reading.token_count,
        exact_match_flag=exact_match,
        signals=flags,
        explanation=explanation,
        confidence=assessment.confidence,
        display_match_percent=percent,
        explanation_card=card,
    )


def _exact_reference_match(
    stripped: str,
    request: SubmissionScoreRequest,
    reference_hashes: Mapping[tuple[str, str], Set[str]],
) -> bool:
    digest = sha256(stripped.encode("utf-8")).hexdigest()
    cluster_hashes = reference_hashes.get((request.question_id, request.language), set())
    return digest in cluster_hashes


def _integrity_result(
    request: SubmissionScoreRequest,
    index: ReferenceIndex,
) -> tuple[str, DetectionResult | None]:
    if significant_token_threshold_for_language(request.language) is None:
        return "", _missing(SubmissionExclusionReason.UNSUPPORTED_LANGUAGE.value)
    stripped = _strip_or_none(request)
    if stripped is None or not stripped.strip():
        return "", _missing(SubmissionExclusionReason.MISSING_OR_INVALID.value)
    if not parses_source(stripped, request.language):
        return "", _missing(SubmissionExclusionReason.MISSING_OR_INVALID.value)
    try:
        significant_code_token_count(stripped, request.language)
    except SignificantCodeTokenizationError:
        return "", _missing(SubmissionExclusionReason.MISSING_OR_INVALID.value)
    if not _cluster_exists(request, index):
        return "", _missing(
            SubmissionExclusionReason.MISSING_EXACT_REFERENCE_CLUSTER.value
        )
    return stripped, None


def _strip_or_none(request: SubmissionScoreRequest) -> str | None:
    if not request.raw_code.strip():
        return None
    try:
        return strip_solution_body(
            request.raw_code,
            request.boilerplate,
            request.language,
        )
    except (SourceParseError, ValueError, KeyError):
        return None


def _token_count(stripped: str, language: str) -> int | None:
    try:
        return significant_code_token_count(stripped, language)
    except SignificantCodeTokenizationError:
        return None


def _cluster_exists(request: SubmissionScoreRequest, index: ReferenceIndex) -> bool:
    key = ClusterKey(request.question_id, request.language)
    try:
        cluster = index.get_cluster(key)
    except KeyError:
        return False
    if cluster.key.question_id != request.question_id:
        raise RuntimeError("Cross-question reference routing")
    if cluster.key.language != request.language:
        raise RuntimeError("Cross-language reference routing")
    return True


def _cluster_diversity(request: SubmissionScoreRequest, index: ReferenceIndex) -> float:
    key = ClusterKey(request.question_id, request.language)
    cluster = index.get_cluster(key)
    return mean_pairwise_cosine_distance(cluster.vectors)


def _ineligible_result(
    decision: SubmissionEligibilityDecision,
    token_count: int | None,
) -> DetectionResult:
    if (
        decision.exclusion_reason
        == SubmissionExclusionReason.INSUFFICIENT_SIGNIFICANT_CODE_TOKENS
    ):
        return _blank_result(
            INSUFFICIENT_EVIDENCE_STATUS,
            INSUFFICIENT_TOKENS_REASON,
            token_count,
        )
    return _blank_result(
        MISSING_OR_INVALID_STATUS,
        decision.exclusion_reason.value,
        token_count,
    )


def _missing(reason: str) -> DetectionResult:
    return _blank_result(MISSING_OR_INVALID_STATUS, reason, None)


def _blank_result(
    status: str,
    reason: str,
    token_count: int | None,
) -> DetectionResult:
    card = build_unscored_explanation_card(status, reason)
    return DetectionResult(
        status=status,
        reason=reason,
        score=None,
        decision=None,
        canonicality=None,
        token_count=token_count,
        exact_match_flag=False,
        signals=uncomputed_style_flags(),
        explanation=format_explanation_card(card),
        confidence=None,
        display_match_percent=None,
        explanation_card=card,
    )

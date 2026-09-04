from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from nw_ai_code_detector.constants import (
    COMMENTED_OUT_HUMAN_LEANING_BODY,
    DISPLAY_AI_REFERENCE_MATCH_PREFIX,
    DISPLAY_MATCH_PERCENT_MEDIUM_MIN,
    DISPLAY_MATCH_PERCENT_MIDPOINT,
    DIVERSITY_FEW_SOLUTIONS_BODY,
    DIVERSITY_MANY_SOLUTIONS_BODY,
    EXPLANATION_CARD_FOOTER,
    EXPLANATION_MATCH_LEVEL_HIGH,
    EXPLANATION_MATCH_LEVEL_LOW,
    EXPLANATION_MATCH_LEVEL_MEDIUM,
    EXPLANATION_SECTION_AI_REFERENCE_MATCH,
    EXPLANATION_SECTION_HUMAN_LEANING_SIGNALS,
    EXPLANATION_SECTION_MATCHING_STRUCTURE,
    EXPLANATION_SECTION_QUESTION_SOLUTION_DIVERSITY,
    EXPLANATION_SECTION_SCORING_STATUS,
    INSUFFICIENT_EVIDENCE_STATUS,
    INSUFFICIENT_TOKENS_REASON,
    LOW_CONFIDENCE_SHORT_STATUS,
    LOW_CONFIDENCE_STATUS,
    MATCH_NOT_SCORED_BODY,
    QUESTION_SOLUTION_FEW_DIVERSITY_MAX,
    SCORED_STATUS,
    SCORING_STATUS_NORMAL,
    SCORING_STATUS_SHORT,
    SCORING_STATUS_TIGHT_CLUSTER,
    SCORING_STATUS_TOO_SHORT,
)
from nw_ai_code_detector.discount_layer import CanonicalityAssessment
from nw_ai_code_detector.display_match import display_match_percent

if TYPE_CHECKING:
    from nw_ai_code_detector.similarity_explanation import NearestReferenceMatch


@dataclass(frozen=True)
class ExplanationSection:
    heading: str
    body: str


@dataclass(frozen=True)
class ExplanationCard:
    sections: tuple[ExplanationSection, ...]
    footer: str


@dataclass(frozen=True)
class ExplanationCardRequest:
    assessment: CanonicalityAssessment
    match: NearestReferenceMatch | None
    language: str
    cluster_diversity: float


def build_explanation_card(request: ExplanationCardRequest) -> ExplanationCard:
    sections = [
        _ai_reference_match_section(request),
        _matching_structure_section(request.match),
        _diversity_section(request.cluster_diversity),
    ]
    human_leaning = _human_leaning_section(request.assessment)
    if human_leaning is not None:
        sections.append(human_leaning)
    sections.append(_scoring_status_section(request.assessment))
    return ExplanationCard(tuple(sections), EXPLANATION_CARD_FOOTER)


def build_unscored_explanation_card(status: str, reason: str | None) -> ExplanationCard:
    body = _unscored_status_body(status, reason)
    return ExplanationCard(
        (
            ExplanationSection(EXPLANATION_SECTION_AI_REFERENCE_MATCH, MATCH_NOT_SCORED_BODY),
            ExplanationSection(EXPLANATION_SECTION_SCORING_STATUS, body),
        ),
        EXPLANATION_CARD_FOOTER,
    )


def format_explanation_card(card: ExplanationCard) -> str:
    blocks = [f"{section.heading}: {section.body}" for section in card.sections]
    blocks.append(card.footer)
    return " ".join(blocks)


def _ai_reference_match_section(request: ExplanationCardRequest) -> ExplanationSection:
    score = request.assessment.score
    if score is None:
        return ExplanationSection(EXPLANATION_SECTION_AI_REFERENCE_MATCH, MATCH_NOT_SCORED_BODY)
    percent = display_match_percent(score, request.language)
    level = _match_level(percent)
    body = (
        f"Match level: {level}. {DISPLAY_AI_REFERENCE_MATCH_PREFIX}: {percent}%."
    )
    return ExplanationSection(EXPLANATION_SECTION_AI_REFERENCE_MATCH, body)


def _match_level(percent: int) -> str:
    if percent >= DISPLAY_MATCH_PERCENT_MIDPOINT:
        return EXPLANATION_MATCH_LEVEL_HIGH
    if percent >= DISPLAY_MATCH_PERCENT_MEDIUM_MIN:
        return EXPLANATION_MATCH_LEVEL_MEDIUM
    return EXPLANATION_MATCH_LEVEL_LOW


def _matching_structure_section(match: NearestReferenceMatch | None) -> ExplanationSection:
    if match is None:
        body = "No nearest AI reference span was recovered."
    elif match.reference_line_start is None:
        body = (
            "Same question's AI reference cluster; no overlapping line span "
            "was recovered."
        )
    else:
        body = (
            "Overlapping structure around reference lines "
            f"{match.reference_line_start}-{match.reference_line_end}."
        )
    return ExplanationSection(EXPLANATION_SECTION_MATCHING_STRUCTURE, body)


def _diversity_section(cluster_diversity: float) -> ExplanationSection:
    if cluster_diversity < QUESTION_SOLUTION_FEW_DIVERSITY_MAX:
        body = DIVERSITY_FEW_SOLUTIONS_BODY
    else:
        body = DIVERSITY_MANY_SOLUTIONS_BODY
    return ExplanationSection(EXPLANATION_SECTION_QUESTION_SOLUTION_DIVERSITY, body)


def _human_leaning_section(assessment: CanonicalityAssessment) -> ExplanationSection | None:
    if not assessment.commented_out_code.present:
        return None
    return ExplanationSection(
        EXPLANATION_SECTION_HUMAN_LEANING_SIGNALS,
        COMMENTED_OUT_HUMAN_LEANING_BODY,
    )


def _scoring_status_section(assessment: CanonicalityAssessment) -> ExplanationSection:
    status = assessment.routing.status
    if status == SCORED_STATUS:
        body = SCORING_STATUS_NORMAL
    elif status == LOW_CONFIDENCE_SHORT_STATUS:
        body = SCORING_STATUS_SHORT
    elif status == LOW_CONFIDENCE_STATUS:
        body = SCORING_STATUS_TIGHT_CLUSTER
    elif status == INSUFFICIENT_EVIDENCE_STATUS:
        body = SCORING_STATUS_TOO_SHORT
    else:
        body = f"Not scored: {assessment.routing.reason}."
    return ExplanationSection(EXPLANATION_SECTION_SCORING_STATUS, body)


def _unscored_status_body(status: str, reason: str | None) -> str:
    if status == INSUFFICIENT_EVIDENCE_STATUS:
        return SCORING_STATUS_TOO_SHORT
    if status == LOW_CONFIDENCE_STATUS:
        return SCORING_STATUS_TIGHT_CLUSTER
    if reason == INSUFFICIENT_TOKENS_REASON:
        return SCORING_STATUS_TOO_SHORT
    return f"Not scored: {reason}."

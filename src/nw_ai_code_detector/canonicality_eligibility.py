from __future__ import annotations

from dataclasses import dataclass

from nw_ai_code_detector.constants import (
    SIGNIFICANT_TOKEN_THRESHOLDS_BY_LANGUAGE,
    SubmissionExclusionReason,
)


@dataclass(frozen=True)
class SubmissionEligibilityDecision:
    eligible: bool
    exclusion_reason: SubmissionExclusionReason


def significant_token_threshold_for_language(language: str) -> int | None:
    return SIGNIFICANT_TOKEN_THRESHOLDS_BY_LANGUAGE.get(language)


def evaluate_submission_eligibility(
    significant_code_token_count: int | None,
    language: str,
    cluster_exists: bool,
    valid: bool,
) -> SubmissionEligibilityDecision:
    threshold = significant_token_threshold_for_language(language)
    if threshold is None:
        return _decision(False, SubmissionExclusionReason.UNSUPPORTED_LANGUAGE)
    if not valid or significant_code_token_count is None:
        return _decision(False, SubmissionExclusionReason.MISSING_OR_INVALID)
    if not cluster_exists:
        return _decision(
            False,
            SubmissionExclusionReason.MISSING_EXACT_REFERENCE_CLUSTER,
        )
    if significant_code_token_count < threshold:
        return _decision(
            False,
            SubmissionExclusionReason.INSUFFICIENT_SIGNIFICANT_CODE_TOKENS,
        )
    return _decision(True, SubmissionExclusionReason.ELIGIBLE)


def _decision(
    eligible: bool,
    reason: SubmissionExclusionReason,
) -> SubmissionEligibilityDecision:
    return SubmissionEligibilityDecision(eligible=eligible, exclusion_reason=reason)

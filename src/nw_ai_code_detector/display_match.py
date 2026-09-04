from __future__ import annotations

from dataclasses import dataclass

from nw_ai_code_detector.constants import (
    DISPLAY_MATCH_HIGH_SCORE,
    DISPLAY_MATCH_LOW_SCORE_BY_LANGUAGE,
    DISPLAY_MATCH_MIDPOINT_SCORE_BY_LANGUAGE,
    DISPLAY_MATCH_PERCENT_BELOW_THRESHOLD_MAX,
    DISPLAY_MATCH_PERCENT_MAX,
    DISPLAY_MATCH_PERCENT_MIDPOINT,
    DISPLAY_MATCH_PERCENT_MIN,
)


@dataclass(frozen=True)
class DisplayMatchMapping:
    midpoint_score: float
    low_score: float
    high_score: float


def display_match_percent(score: float, language: str) -> int:
    mapping = _mapping_for_language(language)
    if score >= mapping.midpoint_score:
        return _high_band_percent(score, mapping)
    return _low_band_percent(score, mapping)


def _mapping_for_language(language: str) -> DisplayMatchMapping:
    midpoint = DISPLAY_MATCH_MIDPOINT_SCORE_BY_LANGUAGE.get(language)
    low_score = DISPLAY_MATCH_LOW_SCORE_BY_LANGUAGE.get(language)
    if midpoint is None or low_score is None:
        raise ValueError(f"Unsupported language: {language}")
    return DisplayMatchMapping(midpoint, low_score, DISPLAY_MATCH_HIGH_SCORE)


def _low_band_percent(score: float, mapping: DisplayMatchMapping) -> int:
    span = mapping.midpoint_score - mapping.low_score
    ratio = (score - mapping.low_score) / span
    percent = int(round(DISPLAY_MATCH_PERCENT_MIDPOINT * _clamp_unit(ratio)))
    return max(
        DISPLAY_MATCH_PERCENT_MIN,
        min(DISPLAY_MATCH_PERCENT_BELOW_THRESHOLD_MAX, percent),
    )


def _high_band_percent(score: float, mapping: DisplayMatchMapping) -> int:
    span = mapping.high_score - mapping.midpoint_score
    ratio = (score - mapping.midpoint_score) / span
    percent = int(
        round(
            DISPLAY_MATCH_PERCENT_MIDPOINT
            + DISPLAY_MATCH_PERCENT_MIDPOINT * _clamp_unit(ratio)
        )
    )
    return max(DISPLAY_MATCH_PERCENT_MIDPOINT, min(DISPLAY_MATCH_PERCENT_MAX, percent))


def _clamp_unit(ratio: float) -> float:
    return min(1.0, max(0.0, ratio))

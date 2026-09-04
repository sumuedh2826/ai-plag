from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

from nw_ai_code_detector.constants import (
    DETECTOR_CONSENSUS_HUMAN_PROBABILITY_FLOOR,
    DETECTOR_CONSENSUS_MIN_RESPONDING,
    DETECTOR_CONSENSUS_MIN_SIGNIFICANT_TOKENS,
    DetectorConsensusLabel,
    DetectorConsensusSkipReason,
    DetectorVerdict,
)


@dataclass(frozen=True)
class DetectorVote:
    detector_name: str
    verdict: DetectorVerdict
    human_probability: float | None


@dataclass(frozen=True)
class DetectorConsensusDecision:
    detector_consensus: DetectorConsensusLabel
    skip_reason: DetectorConsensusSkipReason | None
    responding_detector_names: tuple[str, ...]
    responding_detector_count: int


def decide_detector_consensus(
    significant_code_token_count: int | None,
    votes: Sequence[DetectorVote],
    raw_code_present: bool,
) -> DetectorConsensusDecision:
    if not raw_code_present:
        return _skipped(DetectorConsensusSkipReason.MISSING_RAW_CODE)
    if (
        significant_code_token_count is None
        or significant_code_token_count
        < DETECTOR_CONSENSUS_MIN_SIGNIFICANT_TOKENS
    ):
        return _skipped(DetectorConsensusSkipReason.TOO_SHORT)
    responding = _responding_votes(votes)
    if not responding:
        if any(vote.verdict is DetectorVerdict.ERROR for vote in votes):
            return _skipped(DetectorConsensusSkipReason.DETECTOR_ERRORS)
        return _skipped(DetectorConsensusSkipReason.NO_RESPONSES)
    names = tuple(vote.detector_name for vote in responding)
    if len(responding) < DETECTOR_CONSENSUS_MIN_RESPONDING:
        return _unsure(names)
    if _all_confident_human(responding):
        return DetectorConsensusDecision(
            detector_consensus=DetectorConsensusLabel.CONSENSUS_HUMAN,
            skip_reason=None,
            responding_detector_names=names,
            responding_detector_count=len(names),
        )
    if _all_confident_ai(responding):
        return DetectorConsensusDecision(
            detector_consensus=DetectorConsensusLabel.CONSENSUS_AI,
            skip_reason=None,
            responding_detector_names=names,
            responding_detector_count=len(names),
        )
    return _unsure(names)


def live_detector_keys_configured(environ: dict[str, str]) -> tuple[str, ...]:
    configured: list[str] = []
    if environ.get("SAPLING_API_KEY", "").strip():
        configured.append("sapling")
    if environ.get("GPTZERO_API_KEY", "").strip():
        configured.append("gptzero")
    copyleaks_ready = (
        environ.get("COPYLEAKS_API_KEY", "").strip()
        and environ.get("COPYLEAKS_EMAIL", "").strip()
    )
    if copyleaks_ready:
        configured.append("copyleaks")
    if environ.get("ZEROGPT_API_KEY", "").strip():
        configured.append("zerogpt")
    return tuple(configured)


def _responding_votes(votes: Sequence[DetectorVote]) -> tuple[DetectorVote, ...]:
    return tuple(
        vote
        for vote in votes
        if vote.verdict is not DetectorVerdict.ERROR
        and vote.human_probability is not None
    )


def _all_confident_human(votes: Sequence[DetectorVote]) -> bool:
    return all(
        vote.verdict is DetectorVerdict.HUMAN
        and vote.human_probability is not None
        and vote.human_probability >= DETECTOR_CONSENSUS_HUMAN_PROBABILITY_FLOOR
        for vote in votes
    )


def _all_confident_ai(votes: Sequence[DetectorVote]) -> bool:
    floor = DETECTOR_CONSENSUS_HUMAN_PROBABILITY_FLOOR
    return all(
        vote.verdict is DetectorVerdict.AI
        and vote.human_probability is not None
        and (100.0 - vote.human_probability) >= floor
        for vote in votes
    )


def _skipped(reason: DetectorConsensusSkipReason) -> DetectorConsensusDecision:
    return DetectorConsensusDecision(
        detector_consensus=DetectorConsensusLabel.SKIPPED,
        skip_reason=reason,
        responding_detector_names=(),
        responding_detector_count=0,
    )


def _unsure(names: tuple[str, ...]) -> DetectorConsensusDecision:
    return DetectorConsensusDecision(
        detector_consensus=DetectorConsensusLabel.UNSURE,
        skip_reason=None,
        responding_detector_names=names,
        responding_detector_count=len(names),
    )

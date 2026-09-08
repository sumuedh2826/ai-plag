"""Presentation-only wording for the demo.

Nothing here touches scoring, thresholds or the D10 bank.

Two deliberate constraints:
  * No line-level claims. The detector scores whole-code similarity, so pointing at
    "matching lines" would fabricate precision we do not have (and would mostly flag
    boilerplate and brackets everyone writes identically).
  * Similarity, never authorship. Phrasing is always "similar in structure to
    AI-generated solutions for this problem", never "matches one of our solutions",
    which would imply a specific copy.
"""
from __future__ import annotations

import json
import os
from typing import Any

from nw_ai_code_detector.constants import (
    INSUFFICIENT_EVIDENCE_STATUS,
    LOW_CONFIDENCE_SHORT_STATUS,
    LOW_CONFIDENCE_STATUS,
    MISSING_OR_INVALID_STATUS,
    QUESTION_SOLUTION_FEW_DIVERSITY_MAX,
    SCORED_STATUS,
)

POLISH_MODEL = "openai/gpt-5.6-luna"
POLISH_TIMEOUT_SECONDS = 20
POLISH_MAX_TOKENS = 400

POLISH_SYSTEM_PROMPT = (
    "Rewrite these detection facts into clean honest sentences for a reviewer. Rules: "
    "use ONLY the given facts; no percentage; never say 'AI-written'/'cheated'/"
    "'copied'/'probability'; phrase as SIMILARITY to AI-generated reference solutions, "
    "never as proof of authorship or copying; brief, natural, and help the reviewer "
    "know what to consider."
)

REVIEWER_QUESTION = (
    "Does this submission use the same approach, structure, and step order as the "
    "reference beyond what the problem requires? Did they independently arrive at the "
    "same non-obvious choices, or is this just the standard way to solve it?"
)

# --- plain-language status / confidence -------------------------------------

NOT_SCORED_REASONS = {
    INSUFFICIENT_EVIDENCE_STATUS:
        "Not scored: the submission is too short to assess reliably.",
    LOW_CONFIDENCE_STATUS:
        "Not scored: this problem has essentially one common solution, so a match "
        "isn't meaningful.",
    MISSING_OR_INVALID_STATUS:
        "Not scored: the submission could not be parsed, or this question has no "
        "reference set.",
}
# Confidence and status must not overlap:
#   STATUS      -> why a result was NOT scored.
#   CONFIDENCE  -> how much to trust a score that DOES exist; shown only when low, so
#                  that when it appears it actually means something. A not-scored
#                  result has no confidence line: there is no score to qualify.
LOW_CONFIDENCE_TEXT = {
    LOW_CONFIDENCE_SHORT_STATUS:
        "Low confidence: short code - weigh this result less",
}


def plain_status(status: str) -> tuple[str, str | None]:
    """(headline, reason). Reason is populated only for not-scored routes."""
    if status in (SCORED_STATUS, LOW_CONFIDENCE_SHORT_STATUS):
        return "Scored", None
    return "Not scored", NOT_SCORED_REASONS.get(
        status, "Not scored: this submission could not be assessed."
    )


def plain_confidence(status: str) -> str | None:
    """Only for scored results, and only when low. None means show nothing."""
    return LOW_CONFIDENCE_TEXT.get(status)


def is_low_confidence(status: str) -> bool:
    return status in LOW_CONFIDENCE_TEXT


# --- structured facts (the only thing the LLM ever sees) --------------------

SIMILARITY_SENTENCE = {
    "high": "This submission is highly similar in structure to AI-generated solutions "
            "for this problem.",
    "medium": "This submission is moderately similar in structure to AI-generated "
              "solutions for this problem.",
    "low": "This submission is only slightly similar in structure to AI-generated "
           "solutions for this problem.",
}
CANONICAL_TIGHT = (
    "This problem has essentially one common solution, so high similarity is expected "
    "and less meaningful."
)
CANONICAL_VARIED = (
    "This problem has many valid approaches, so a close match is more notable."
)

HEADINGS = {
    "similarity": "Overall Similarity",
    "question_canonicality": "How Canonical This Question Is",
    "scoring_status": "Scoring Status",
    "human_signals": "Human-Leaning Signals",
}


def explanation_facts(
    *,
    status: str,
    match_level: str | None,
    cluster_diversity: float | None,
    commented_out_code: bool,
) -> dict[str, Any]:
    headline, reason = plain_status(status)
    tight = (
        cluster_diversity is not None
        and cluster_diversity <= QUESTION_SOLUTION_FEW_DIVERSITY_MAX
    )
    facts: dict[str, Any] = {
        "similarity": SIMILARITY_SENTENCE.get(match_level) if match_level else None,
        "question_canonicality": CANONICAL_TIGHT if tight else CANONICAL_VARIED,
        "scoring_status": reason or "Scored normally.",
        "confidence": plain_confidence(status),
        "human_signals": (
            "Contains commented-out code (a debugging trace), which leans human."
            if commented_out_code else None
        ),
    }
    return {k: v for k, v in facts.items() if v is not None}


def _sections(facts: dict[str, Any]) -> list[tuple[str, str, str]]:
    """(section_key, heading, plain sentence) - one entry per section shown."""
    out: list[tuple[str, str, str]] = []
    if facts.get("similarity"):
        out.append(("similarity", HEADINGS["similarity"], facts["similarity"]))
    if facts.get("question_canonicality"):
        out.append(("question_canonicality", HEADINGS["question_canonicality"],
                    facts["question_canonicality"]))
    status_line = facts["scoring_status"]
    if facts.get("confidence"):
        status_line = f"{status_line} {facts['confidence']}."
    out.append(("scoring_status", HEADINGS["scoring_status"], status_line))
    if facts.get("human_signals"):
        out.append(("human_signals", HEADINGS["human_signals"], facts["human_signals"]))
    return out


def hardcoded_sentences(facts: dict[str, Any]) -> list[tuple[str, str]]:
    """The fallback and the source of truth."""
    return [(heading, sentence) for _key, heading, sentence in _sections(facts)]


def polish_facts(facts: dict[str, Any]) -> list[tuple[str, str]] | None:
    """Rephrase the plain sentences. Returns None on any failure -> caller falls back.
    The model receives only these sentences; never the code, score, or percentage."""
    try:
        from dotenv import load_dotenv

        from nw_ai_code_detector.config import ENV_PATH

        load_dotenv(ENV_PATH)
    except Exception:
        pass
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        return None
    sections = _sections(facts)
    try:
        from openai import OpenAI

        client = OpenAI(
            api_key=api_key,
            base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            timeout=POLISH_TIMEOUT_SECONDS,
        )
        response = client.chat.completions.create(
            model=POLISH_MODEL,
            messages=[
                {"role": "system", "content": POLISH_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps({
                    "facts": {k: s for k, _h, s in sections},
                    "instruction": (
                        "Return a JSON object with exactly these keys, each mapped to "
                        "one short rewritten sentence conveying the same fact."
                    ),
                })},
            ],
            temperature=0.2,
            max_tokens=POLISH_MAX_TOKENS,
            extra_body={"reasoning": {"effort": "low"}},
        )
        text = (response.choices[0].message.content or "").strip()
        parsed = json.loads(text[text.index("{"): text.rindex("}") + 1])
    except Exception:
        return None
    out: list[tuple[str, str]] = []
    for key, heading, fallback in sections:
        sentence = parsed.get(key)
        out.append((heading, sentence.strip()
                    if isinstance(sentence, str) and sentence.strip() else fallback))
    return out or None

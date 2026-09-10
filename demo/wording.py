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
    CLUSTER_LOW_DIVERSITY_DISTANCE as FLOOR_NOT_SCORED,
    INSUFFICIENT_EVIDENCE_STATUS,
    LOW_CONFIDENCE_SHORT_STATUS,
    LOW_CONFIDENCE_STATUS,
    MISSING_OR_INVALID_STATUS,
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

# Display-only vendor names. The real slug stays in the bank manifest and the logs;
# a reviewer has no use for "gemini-3.7-flash" and it dates the bank unnecessarily.
VENDOR_BY_PREFIX = (
    ("openai/", "ChatGPT"),
    ("anthropic/", "Anthropic"),
    ("deepseek/", "DeepSeek"),
    ("google/gemini", "Gemini"),
)
VENDOR_FALLBACK = "an AI model"


def vendor_name(model_slug: str | None) -> str:
    """Clean vendor label for a model slug. Unknown vendors degrade to a neutral
    phrase rather than leaking an internal identifier."""
    if not model_slug:
        return VENDOR_FALLBACK
    slug = model_slug.strip().lower()
    for prefix, name in VENDOR_BY_PREFIX:
        if slug.startswith(prefix):
            return name
    return VENDOR_FALLBACK


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

# Wording follows the band so the sentence and the colour can never disagree.
SIMILARITY_BY_BAND = {
    "red": "Strong similarity in structure to AI-generated solutions for this "
           "problem - flagged for review.",
    "yellow": "Similarity to AI-generated solutions for this problem is borderline - "
              "review carefully.",
    "green": "Limited similarity to AI-generated solutions for this problem - "
             "not flagged.",
}
# Three bands, keyed to the SCORING floor so the message can't contradict the flag.
#   below 0.030  -> not scored at all; the strong wording is safe there
#   0.030-0.050  -> scored, but few approaches; soften rather than undercut the flag
#   above 0.050  -> scored and varied; a close match is genuinely notable
# The previous single 0.060 cutoff sat above the 0.030 scoring floor, so two thirds of
# flagged submissions were told "essentially one common solution" alongside their flag.
CANONICAL_WORDING_MAX = 0.050
CANONICAL_TIGHT = (
    "This problem has essentially one common solution, so a match isn't meaningful."
)
CANONICAL_FEW = (
    "This problem has relatively few distinct valid approaches, so weigh the "
    "structural match accordingly."
)
CANONICAL_VARIED = (
    "This problem has many valid approaches, so a close match is more notable."
)


def canonicality_sentence(cluster_diversity: float | None, scored: bool) -> str | None:
    if cluster_diversity is None:
        return None
    if not scored and cluster_diversity < FLOOR_NOT_SCORED:
        return CANONICAL_TIGHT
    if cluster_diversity <= CANONICAL_WORDING_MAX:
        return CANONICAL_FEW
    return CANONICAL_VARIED

HEADINGS = {
    "similarity": "Overall Similarity",
    "question_canonicality": "How Canonical This Question Is",
    "naming": "Naming",
    "scoring_status": "Scoring Status",
}

# Appended to the similarity verdict rather than given its own section, so a whole
# heading doesn't appear and disappear between submissions. Phrased as a mitigating
# factor to weigh next to the verdict, not as a contradiction of it.
COMMENTED_OUT_CAVEAT = (
    "Note: it also contains commented-out code - a debugging trace, which leans "
    "human - weigh that alongside."
)

# Descriptive-naming share over DISTINCT local bindings only (fields, methods, types
# and parameters excluded by local_binding_names).
NAMING_DESCRIPTIVE_MIN = 0.45
NAMING_DESCRIPTIVE_TEXT = (
    "Uses many descriptive variable names, which is consistent with AI-generated code."
)
NAMING_PLAIN_TEXT = (
    "Variable names don't obviously suggest AI, but the structure still matches - "
    "review alongside the other signals."
)


def naming_note(frac_descriptive: float | None, flagged: bool) -> str | None:
    """Context for a flag, never a reason for one. Shown only when flagged."""
    if not flagged or frac_descriptive is None:
        return None
    return (NAMING_DESCRIPTIVE_TEXT if frac_descriptive >= NAMING_DESCRIPTIVE_MIN
            else NAMING_PLAIN_TEXT)


def explanation_facts(
    *,
    status: str,
    band: str,
    cluster_diversity: float | None,
    commented_out_code: bool,
    scored: bool,
    frac_descriptive: float | None = None,
    flagged: bool = False,
) -> dict[str, Any]:
    headline, reason = plain_status(status)
    facts: dict[str, Any] = {
        "similarity": SIMILARITY_BY_BAND.get(band),
        "question_canonicality": canonicality_sentence(cluster_diversity, scored),
        "naming": naming_note(frac_descriptive, flagged),
        "scoring_status": reason or "Scored normally.",
        "confidence": plain_confidence(status),
    }
    # The caveat rides on the verdict line. With no verdict (not scored) there is
    # nothing for it to qualify, so it is not shown.
    if commented_out_code and facts.get("similarity"):
        facts["similarity"] = f"{facts['similarity']} {COMMENTED_OUT_CAVEAT}"
    return {k: v for k, v in facts.items() if v is not None}


def _sections(facts: dict[str, Any]) -> list[tuple[str, str, str]]:
    """(section_key, heading, plain sentence) - one entry per section shown."""
    out: list[tuple[str, str, str]] = []
    if facts.get("similarity"):
        out.append(("similarity", HEADINGS["similarity"], facts["similarity"]))
    if facts.get("question_canonicality"):
        out.append(("question_canonicality", HEADINGS["question_canonicality"],
                    facts["question_canonicality"]))
    if facts.get("naming"):
        out.append(("naming", HEADINGS["naming"], facts["naming"]))
    status_line = facts.get("scoring_status") or "Scored normally."
    if facts.get("confidence"):
        status_line = f"{status_line} {facts['confidence']}."
    out.append(("scoring_status", HEADINGS["scoring_status"], status_line))
    return out


NOT_SCORED_GUIDANCE = (
    "No similarity comparison is available for this submission, so there is nothing "
    "to compare against the AI reference set."
)


def guidance_lines(facts: dict[str, Any], scored: bool) -> list[str]:
    """Body of the 'What to look at' panel. Pure, so the not-scored path is testable
    without Streamlit. Every lookup is a safe .get(): a not-scored submission has no
    similarity, canonicality or naming facts at all."""
    if not scored:
        reason = facts.get("scoring_status")
        return [reason, NOT_SCORED_GUIDANCE] if reason else [NOT_SCORED_GUIDANCE]
    return [
        line for line in (
            facts.get("similarity"),
            facts.get("question_canonicality"),
        ) if line
    ]


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
        if not (isinstance(sentence, str) and sentence.strip()):
            out.append((heading, fallback)); continue
        sentence = sentence.strip()
        if _dropped_a_fact(fallback, sentence):
            sentence = fallback
        out.append((heading, sentence))
    return out or None


# Facts the model has been observed to silently omit when rephrasing. Losing the
# commented-out-code caveat would delete a human-leaning mitigation from the verdict,
# so a rewrite that drops it is rejected in favour of the plain sentence.
REQUIRED_CUES = (("commented-out", ("comment", "debug")),)


def _dropped_a_fact(original: str, rewritten: str) -> bool:
    low_original, low_rewritten = original.lower(), rewritten.lower()
    for marker, cues in REQUIRED_CUES:
        if marker in low_original and not any(c in low_rewritten for c in cues):
            return True
    return False

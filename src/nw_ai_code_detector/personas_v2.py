from __future__ import annotations

from enum import Enum

from nw_ai_code_detector.constants import (
    MUST_PASS_EXAMPLES_DIRECTIVE,
    NO_COMMENTS_DIRECTIVE,
    OUTPUT_SHAPE_DIRECTIVE,
    PRINT_MISMATCH_PYTHON_QUESTION_IDS,
    PRINT_REQUIRED_OUTPUT_SHAPE_DIRECTIVE,
)
from nw_ai_code_detector.data_load import QuestionRecord
from nw_ai_code_detector.stripper import Language

REFS_V2_BANK_VERSION = "refs_v2"
# Salt keeps the v2 persona/model shuffle independent of v1's, which is seeded on
# the bare f"{qid}:{lang}". Same determinism, different draw.
REFS_V2_SEED_SALT = "v2"


class PersonaV2(str, Enum):
    BARE = "bare"
    SHORT_NAMES = "short_names"
    EVADE_DETECTION = "evade_detection"
    MOST_EFFICIENT = "most_efficient"
    DESCRIPTIVE_NAMES = "descriptive_names"
    LESS_OBVIOUS = "less_obvious"
    # Experimental, deliberately NOT part of PERSONA_V2_ORDER so the v2 bank
    # (6 refs per cluster) is unaffected.
    HUMANLIKE = "humanlike"


# Verbatim persona strings. BARE is deliberately absent: it sends no "Style:" line
# at all. This set drops the redundant "solve it" variants that collapsed cluster
# diversity in earlier probes and keeps only axes that measurably differentiate:
# naming (short vs descriptive), detector evasion, complexity, and algorithm choice.
PERSONA_V2_STYLE_DIRECTIVES = {
    PersonaV2.SHORT_NAMES: (
        "Give me the solution, use short variable names (one word / short)."
    ),
    PersonaV2.EVADE_DETECTION: "Write it so an AI-detection tool won't flag it.",
    PersonaV2.MOST_EFFICIENT: "Most efficient solution, best time complexity.",
    PersonaV2.DESCRIPTIVE_NAMES: (
        "Write it with clear, descriptive variable names and a clean, "
        "conventional structure."
    ),
    # Nudges the algorithm, not the quality bar - "must be fully correct" keeps
    # unexecuted refs from silently polluting the cluster with wrong answers.
    PersonaV2.LESS_OBVIOUS: (
        "Solve this using a less obvious approach - if multiple valid algorithms "
        "exist, choose a less common one. It must be fully correct."
    ),
    # "Looks hand-written", NOT "evades a detector" - deliberately a different axis
    # from EVADE_DETECTION.
    PersonaV2.HUMANLIKE: (
        "Write this the way a real developer would naturally write it - natural, "
        "human style, as if a person wrote it by hand."
    ),
}

PERSONA_V2_ORDER = (
    PersonaV2.BARE,
    PersonaV2.SHORT_NAMES,
    PersonaV2.EVADE_DETECTION,
    PersonaV2.MOST_EFFICIENT,
    PersonaV2.DESCRIPTIVE_NAMES,
    PersonaV2.LESS_OBVIOUS,
)


def output_shape_directive_for(question_id: str, language: Language) -> str:
    """Five known PYTHON questions are graded on printed output, not a return value.
    Same carve-out as v1; without it those clusters are built from wrong-shaped code."""
    if (
        language is Language.PYTHON
        and question_id in PRINT_MISMATCH_PYTHON_QUESTION_IDS
    ):
        return PRINT_REQUIRED_OUTPUT_SHAPE_DIRECTIVE
    return OUTPUT_SHAPE_DIRECTIVE


def build_prompt_v2(
    question: QuestionRecord,
    language: Language,
    persona: PersonaV2,
) -> str:
    boilerplate = question.boilerplates.get(language.value, "")
    blocks = [
        question.statement_content.strip(),
        f"Target language: {language.value}",
        f"Boilerplate:\n{boilerplate}",
    ]
    style_directive = PERSONA_V2_STYLE_DIRECTIVES.get(persona)
    if style_directive is not None:
        blocks.append(f"Style: {style_directive}")
    blocks.extend(
        [
            MUST_PASS_EXAMPLES_DIRECTIVE,
            NO_COMMENTS_DIRECTIVE,
            output_shape_directive_for(question.question_id, language),
        ]
    )
    return "\n\n".join(blocks)

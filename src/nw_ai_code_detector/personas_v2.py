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
    SOLVE_GIVE_CODE = "solve_give_code"
    COMPLETE_FUNCTION = "complete_function"
    MOST_EFFICIENT = "most_efficient"
    EVADE_DETECTION = "evade_detection"
    SHORT_NAMES = "short_names"
    BARE = "bare"


# Verbatim persona strings. BARE is deliberately absent: it sends no "Style:" line at all.
PERSONA_V2_STYLE_DIRECTIVES = {
    PersonaV2.SOLVE_GIVE_CODE: "Solve this problem. Give me the code.",
    PersonaV2.COMPLETE_FUNCTION: "Here is the starter code, complete the function.",
    PersonaV2.MOST_EFFICIENT: "Most efficient solution, best time complexity.",
    PersonaV2.EVADE_DETECTION: "Write it so an AI-detection tool won't flag it.",
    PersonaV2.SHORT_NAMES: (
        "Give me the solution, use short variable names (one word / short)."
    ),
}

PERSONA_V2_ORDER = tuple(PersonaV2)


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

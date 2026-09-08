import unittest
from pathlib import Path
from unittest import mock

from nw_ai_code_detector import generate_ai_refs_v2 as gen
from nw_ai_code_detector.config import AI_SOLUTIONS_DIR, AI_SOLUTIONS_V2_DIR
from nw_ai_code_detector.constants import (
    MUST_PASS_EXAMPLES_DIRECTIVE,
    NO_COMMENTS_DIRECTIVE,
    OUTPUT_SHAPE_DIRECTIVE,
    PRINT_MISMATCH_PYTHON_QUESTION_IDS,
    PRINT_REQUIRED_OUTPUT_SHAPE_DIRECTIVE,
)
from nw_ai_code_detector.data_load import QuestionRecord
from nw_ai_code_detector.openrouter_budget import Headroom
from nw_ai_code_detector.personas_v2 import (
    PERSONA_V2_ORDER,
    PERSONA_V2_STYLE_DIRECTIVES,
    PersonaV2,
    build_prompt_v2,
)
from nw_ai_code_detector.stripper import Language

MODELS = (
    "google/gemini-3.7-flash",
    "anthropic/claude-haiku-4.5",
    "openai/gpt-5.6-luna",
)

EXACT_PERSONA_STRINGS = {
    PersonaV2.SHORT_NAMES: "Give me the solution, use short variable names (one word / short).",
    PersonaV2.EVADE_DETECTION: "Write it so an AI-detection tool won't flag it.",
    PersonaV2.MOST_EFFICIENT: "Most efficient solution, best time complexity.",
    PersonaV2.DESCRIPTIVE_NAMES: (
        "Write it with clear, descriptive variable names and a clean, conventional structure."
    ),
    PersonaV2.LESS_OBVIOUS: (
        "Solve this using a less obvious approach - if multiple valid algorithms exist, "
        "choose a less common one. It must be fully correct."
    ),
}


def _question(question_id="q-1"):
    return QuestionRecord(
        question_id=question_id,
        difficulty="EASY",
        tags=(),
        primary_tag="TOPIC_X",
        is_function_completion=True,
        statement_content="Return the sum of an array.",
        boilerplates={"CPP": "class Solution {\n};", "PYTHON": "class Solution:\n    pass"},
    )


class PersonaStringTests(unittest.TestCase):
    def test_persona_strings_are_verbatim(self):
        for persona, expected in EXACT_PERSONA_STRINGS.items():
            self.assertEqual(PERSONA_V2_STYLE_DIRECTIVES[persona], expected)

    def test_there_are_six_personas_and_bare_has_no_style_string(self):
        self.assertEqual(len(PERSONA_V2_ORDER), 6)
        self.assertNotIn(PersonaV2.BARE, PERSONA_V2_STYLE_DIRECTIVES)


class PromptShapeTests(unittest.TestCase):
    def test_styled_personas_emit_their_exact_style_line(self):
        for persona, expected in EXACT_PERSONA_STRINGS.items():
            prompt = build_prompt_v2(_question(), Language.CPP, persona)
            self.assertIn(f"Style: {expected}", prompt)

    def test_bare_persona_emits_no_style_line(self):
        prompt = build_prompt_v2(_question(), Language.CPP, PersonaV2.BARE)
        self.assertNotIn("Style:", prompt)

    def test_every_persona_keeps_the_rest_of_the_scaffold(self):
        for persona in PERSONA_V2_ORDER:
            prompt = build_prompt_v2(_question(), Language.CPP, persona)
            self.assertIn("Return the sum of an array.", prompt)
            self.assertIn("Target language: CPP", prompt)
            self.assertIn("Boilerplate:\nclass Solution {", prompt)
            self.assertIn(MUST_PASS_EXAMPLES_DIRECTIVE, prompt)
            self.assertIn(NO_COMMENTS_DIRECTIVE, prompt)
            self.assertIn(OUTPUT_SHAPE_DIRECTIVE, prompt)

    def test_print_mismatch_python_questions_keep_the_carve_out(self):
        question = _question(PRINT_MISMATCH_PYTHON_QUESTION_IDS[0])
        prompt = build_prompt_v2(question, Language.PYTHON, PersonaV2.BARE)
        self.assertIn(PRINT_REQUIRED_OUTPUT_SHAPE_DIRECTIVE, prompt)

    def test_carve_out_does_not_leak_to_cpp_or_other_questions(self):
        carved = _question(PRINT_MISMATCH_PYTHON_QUESTION_IDS[0])
        self.assertIn(
            OUTPUT_SHAPE_DIRECTIVE,
            build_prompt_v2(carved, Language.CPP, PersonaV2.BARE),
        )
        self.assertIn(
            OUTPUT_SHAPE_DIRECTIVE,
            build_prompt_v2(_question(), Language.PYTHON, PersonaV2.BARE),
        )


class AssignmentTests(unittest.TestCase):
    def test_assignment_is_deterministic(self):
        first = gen._units_for_question_language("q-1", Language.CPP, MODELS)
        second = gen._units_for_question_language("q-1", Language.CPP, MODELS)
        self.assertEqual(first, second)

    def test_assignment_is_independent_of_v1_seed(self):
        from nw_ai_code_detector.generate_ai_refs import _assignment_seed

        self.assertNotEqual(
            gen._assignment_seed_v2("q-1", "CPP"),
            _assignment_seed("q-1", "CPP"),
        )

    def test_six_personas_split_two_each(self):
        units = gen._units_for_question_language("q-1", Language.CPP, MODELS)
        self.assertEqual(len(units), 6)
        counts = {}
        for unit in units:
            counts[unit.model] = counts.get(unit.model, 0) + 1
        self.assertEqual(sorted(counts.values()), [2, 2, 2])
        self.assertEqual(set(counts), set(MODELS))

    def test_all_six_personas_appear_once(self):
        units = gen._units_for_question_language("q-1", Language.CPP, MODELS)
        self.assertEqual(
            sorted(unit.persona.value for unit in units),
            sorted(persona.value for persona in PERSONA_V2_ORDER),
        )

    def test_persona_slots_always_cover_every_persona(self):
        import random as _random

        for count in (1, 2, 3, 7):
            models = tuple(f"m{i}" for i in range(count))
            slots = gen._persona_model_slots(models, _random.Random(0))
            self.assertEqual(len(slots), len(PERSONA_V2_ORDER))
            self.assertTrue(set(slots).issubset(set(models)))

    def test_persona_slots_reject_an_empty_model_list(self):
        import random as _random

        with self.assertRaises(ValueError):
            gen._persona_model_slots((), _random.Random(0))

    def test_temperature_stays_in_the_sampled_band(self):
        units = gen._units_for_question_language("q-1", Language.PYTHON, MODELS)
        for unit in units:
            self.assertGreaterEqual(unit.temperature, 0.7)
            self.assertLessEqual(unit.temperature, 1.0)


class ReasoningPolicyTests(unittest.TestCase):
    def test_anthropic_reasoning_is_disabled(self):
        self.assertEqual(
            gen._reasoning_for_model("anthropic/claude-haiku-4.5"),
            {"enabled": False},
        )

    def test_control_models_keep_low_effort(self):
        for model in ("openai/gpt-5.5", "google/gemini-3.7-flash"):
            self.assertEqual(gen._reasoning_for_model(model), {"effort": "low"})


class OutputPathGuardTests(unittest.TestCase):
    def test_v2_units_write_inside_the_v2_bank(self):
        unit = gen._units_for_question_language("q-1", Language.CPP, MODELS)[0]
        path = gen._unit_output_path(unit)
        self.assertIn(AI_SOLUTIONS_V2_DIR.resolve(), path.resolve().parents)
        gen._assert_v2_path(path)

    def test_a_path_inside_the_v1_bank_is_refused(self):
        v1_path = AI_SOLUTIONS_DIR / "q-1" / "CPP" / "bare_openai_gpt-5.5.json"
        with self.assertRaises(RuntimeError):
            gen._assert_v2_path(v1_path)

    def test_a_path_outside_both_banks_is_refused(self):
        with self.assertRaises(RuntimeError):
            gen._assert_v2_path(Path("/tmp/somewhere/else.json"))


class BudgetTests(unittest.TestCase):
    def _headroom(self, key_limit, key_usage, credits=8000.0, usage=6800.0):
        return Headroom(
            account_credits=credits,
            account_usage=usage,
            key_limit=key_limit,
            key_usage=key_usage,
            key_usage_monthly=0.0,
        )

    def test_key_limit_binds_when_lower_than_account_credit(self):
        headroom = self._headroom(key_limit=200.0, key_usage=101.43)
        self.assertAlmostEqual(headroom.key_remaining, 98.57, places=2)
        self.assertAlmostEqual(headroom.effective_remaining, 98.57, places=2)

    def test_unlimited_key_falls_back_to_account_credit(self):
        headroom = self._headroom(key_limit=None, key_usage=0.0)
        self.assertAlmostEqual(headroom.effective_remaining, 1200.0, places=2)

    def test_covers_applies_the_safety_factor(self):
        headroom = self._headroom(key_limit=200.0, key_usage=101.43)
        self.assertTrue(headroom.covers(28.0))
        self.assertFalse(headroom.covers(70.0))

    def test_projection_uses_per_model_rates(self):
        units = gen._units_for_question_language("q-1", Language.CPP, MODELS)
        self.assertGreater(gen._projected_cost(units), 0.0)

    def test_full_run_projection_is_in_the_expected_band(self):
        units = []
        for index in range(500):
            for language in (Language.CPP, Language.PYTHON):
                units.extend(
                    gen._units_for_question_language(f"q-{index}", language, MODELS)
                )
        self.assertEqual(len(units), 6000)
        projected = gen._projected_cost(units)
        self.assertGreater(projected, 3.0)
        self.assertLess(projected, 25.0)


class HeadroomFetchTests(unittest.TestCase):
    def test_fetch_headroom_parses_both_endpoints(self):
        from nw_ai_code_detector import openrouter_budget

        payloads = [
            {"data": {"total_credits": 8006.03, "total_usage": 6854.52}},
            {"data": {"limit": 200, "usage": 101.43, "usage_monthly": 7.11}},
        ]
        with mock.patch.object(openrouter_budget, "_get_json", side_effect=payloads):
            headroom = openrouter_budget.fetch_headroom("k", "https://example.invalid/api/v1")
        self.assertAlmostEqual(headroom.account_remaining, 1151.51, places=2)
        self.assertAlmostEqual(headroom.key_remaining, 98.57, places=2)


if __name__ == "__main__":
    unittest.main()


class CircuitBreakerTests(unittest.TestCase):
    def test_403_limit_exceeded_is_fatal(self):
        class Boom(Exception):
            status_code = 403

        self.assertTrue(gen._is_fatal_quota_error(Boom("Key limit exceeded (total limit)")))

    def test_limit_exceeded_text_is_fatal_without_a_status_code(self):
        self.assertTrue(gen._is_fatal_quota_error(Exception("Key limit exceeded")))

    def test_transient_errors_are_not_fatal(self):
        class Boom(Exception):
            status_code = 429

        self.assertFalse(gen._is_fatal_quota_error(Boom("slow down")))
        self.assertFalse(gen._is_fatal_quota_error(Exception("connection reset")))

    def test_retry_policy_does_not_retry_403(self):
        from openai import APIStatusError

        from nw_ai_code_detector.generate_ai_refs import _is_retryable

        error = APIStatusError.__new__(APIStatusError)
        error.status_code = 403
        self.assertFalse(_is_retryable(error))

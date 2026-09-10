import unittest
from unittest.mock import Mock, patch

from demo.runtime import (
    DemoScoreRequest,
    InMemoryVoyageEmbedder,
    score_demo_submission,
)
from nw_ai_code_detector.config import VoyageSettings
from nw_ai_code_detector.constants import (
    HIGH_CONFIDENCE_LABEL,
    INSUFFICIENT_EVIDENCE_STATUS,
    MISSING_OR_INVALID_STATUS,
    LOW_CONFIDENCE_SHORT_STATUS,
    LOW_CONFIDENCE_STATUS,
    SCORED_STATUS,
)
from nw_ai_code_detector.data_load import QuestionRecord
from nw_ai_code_detector.embedder import VoyageEmbedder
from nw_ai_code_detector.score_query import DetectionResult


def _detection(status, confidence):
    return DetectionResult(
        status=status,
        reason=None,
        score=None,
        decision=None,
        canonicality=None,
        token_count=80,
        exact_match_flag=False,
        signals=(),
        explanation=None,
        confidence=confidence,
        display_match_percent=None,
        explanation_card=None,
    )


class DemoRuntimeTests(unittest.TestCase):
    def test_demo_forwards_full_template_submission_to_production_scorer(self):
        boilerplate = """class solution {
public:
    int solve(int n) {
        // Write your code here
    }
};"""
        raw_code = """class solution {
public:
    int helper(int n) {
        return n + 1;
    }
    int solve(int n) {
        return helper(n);
    }
};"""
        question = QuestionRecord(
            "question-id",
            "EASY",
            (),
            "untagged",
            True,
            "Solve it.",
            {"CPP": boilerplate},
        )
        embedder = Mock(spec=InMemoryVoyageEmbedder)
        embedder.vector_for_text.return_value = None
        embedder.last_cache_hit = False
        with patch("demo.runtime.score_submission") as scorer:
            result = score_demo_submission(
                DemoScoreRequest(raw_code, question, "CPP"),
                Mock(),
                embedder,
                {},
            )
        production_request = scorer.call_args.args[0]
        self.assertEqual(production_request.raw_code, raw_code)
        self.assertEqual(production_request.boilerplate, boilerplate)
        self.assertEqual(result.completed_code, raw_code)
        self.assertIn("int helper(int n)", result.stripped_code)
        self.assertIn("return helper(n);", result.stripped_code)

    def test_embedder_cache_is_memory_only(self):
        settings = VoyageSettings("secret", "voyage-code-3", 1, 1, 25)
        embedder = InMemoryVoyageEmbedder(settings)
        embedder._client = Mock()
        embedder._client.embed.return_value = Mock(
            embeddings=[[3.0, 4.0]],
            total_tokens=2,
        )
        with patch("nw_ai_code_detector.embedder._write_cached_vector") as writer:
            first = embedder.embed_texts(("pasted student code",))
            second = embedder.embed_texts(("pasted student code",))
        self.assertEqual(first.cache_misses, 1)
        self.assertEqual(second.cache_hits, 1)
        writer.assert_not_called()
        embedder._client.embed.assert_called_once()

    def test_status_and_confidence_do_not_overlap(self):
        from demo.app import _band
        from demo.wording import plain_confidence, plain_status

        # STATUS carries *why it was not scored*; nothing else.
        self.assertEqual(plain_status(SCORED_STATUS), ("Scored", None))
        self.assertEqual(plain_status(LOW_CONFIDENCE_SHORT_STATUS), ("Scored", None))
        tight_head, tight_reason = plain_status(LOW_CONFIDENCE_STATUS)
        self.assertEqual(tight_head, "Not scored")
        self.assertIn("one common solution", tight_reason)
        short_head, short_reason = plain_status(INSUFFICIENT_EVIDENCE_STATUS)
        self.assertEqual(short_head, "Not scored")
        self.assertIn("too short", short_reason)

        # CONFIDENCE is a caveat on an existing score: shown only when low, and never
        # for a result that was not scored at all.
        self.assertIsNone(plain_confidence(SCORED_STATUS))
        self.assertIsNone(plain_confidence(LOW_CONFIDENCE_STATUS))
        self.assertIsNone(plain_confidence(INSUFFICIENT_EVIDENCE_STATUS))
        self.assertIsNone(plain_confidence(MISSING_OR_INVALID_STATUS))
        short_confidence = plain_confidence(LOW_CONFIDENCE_SHORT_STATUS)
        self.assertIsNotNone(short_confidence)
        self.assertIn("short code", short_confidence)

        # "High confidence" must never be rendered anywhere.
        for status in (SCORED_STATUS, LOW_CONFIDENCE_SHORT_STATUS, LOW_CONFIDENCE_STATUS,
                       INSUFFICIENT_EVIDENCE_STATUS, MISSING_OR_INVALID_STATUS):
            shown = f"{plain_status(status)[1] or ''} {plain_confidence(status) or ''}"
            self.assertNotIn("High confidence", shown)
            # No reason may appear in both fields.
            if plain_status(status)[1] and plain_confidence(status):
                self.fail(f"{status} populates both status reason and confidence")

        # The detector's naming label is additive and must never reach the wording.
        for status in (LOW_CONFIDENCE_STATUS, LOW_CONFIDENCE_SHORT_STATUS):
            self.assertNotIn(HIGH_CONFIDENCE_LABEL, plain_confidence(status) or "")
            self.assertNotIn(HIGH_CONFIDENCE_LABEL, plain_status(status)[1] or "")

        # Raw internal routing strings must never reach user-facing text.
        for status in (LOW_CONFIDENCE_STATUS, LOW_CONFIDENCE_SHORT_STATUS,
                       INSUFFICIENT_EVIDENCE_STATUS, SCORED_STATUS):
            shown = f"{plain_status(status)[1] or ''} {plain_confidence(status) or ''}"
            for internal in (LOW_CONFIDENCE_SHORT_STATUS, INSUFFICIENT_EVIDENCE_STATUS,
                             "short_code_low_confidence", "low_cluster_diversity"):
                self.assertNotIn(internal, shown)

    def test_colour_bands_anchor_to_the_flag_line(self):
        from demo.app import _band

        # green below the line - every verified human sits here (max observed 49%)
        self.assertEqual(_band(0), "green")
        self.assertEqual(_band(49), "green")
        # yellow 50-54: just above the line, borderline review
        self.assertEqual(_band(50), "yellow")
        self.assertEqual(_band(54), "yellow")
        # red from 55: where held-out AI concentrates
        self.assertEqual(_band(55), "red")
        self.assertEqual(_band(100), "red")
        # Colour follows the flag decision. Confidence only discriminates among
        # FLAGGED cases: a not-flagged result stays green however low the confidence.
        for percent in (0, 30, 49):
            self.assertEqual(_band(percent, low_confidence=True), "green")
        # Flagged at low confidence is yellow, and never reaches red.
        for percent in (50, 54, 55, 60, 100):
            self.assertEqual(_band(percent, low_confidence=True), "yellow")
        self.assertEqual(_band(None), "unscored")
        self.assertEqual(_band(None, low_confidence=True), "unscored")

    def test_polish_falls_back_when_the_model_is_unavailable(self):
        from unittest.mock import patch

        from demo.wording import explanation_facts, hardcoded_sentences, polish_facts

        facts = explanation_facts(
            status=SCORED_STATUS, band="red", cluster_diversity=0.09,
            commented_out_code=False, scored=True,
        )
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": ""}, clear=False):
            with patch("dotenv.load_dotenv", lambda *a, **k: None):
                self.assertIsNone(polish_facts(facts))
        self.assertTrue(hardcoded_sentences(facts))

    def test_wording_never_claims_authorship_or_copying(self):
        from demo.wording import explanation_facts, hardcoded_sentences

        for band in ("red", "yellow", "green"):
            for diversity in (0.035, 0.09):
                facts = explanation_facts(
                    status=SCORED_STATUS, band=band, cluster_diversity=diversity,
                    commented_out_code=False, scored=True,
                )
                text = " ".join(s for _h, s in hardcoded_sentences(facts)).lower()
                for banned in ("copied", "cheat", "ai-written", "probability", "%"):
                    self.assertNotIn(banned, text)
                self.assertIn("similar", text)

    def test_demo_scoring_rejects_disk_backed_embedder(self):
        settings = VoyageSettings("secret", "voyage-code-3", 1, 1, 25)
        disk_embedder = object.__new__(VoyageEmbedder)
        disk_embedder._settings = settings
        with self.assertRaises(TypeError):
            score_demo_submission(Mock(), Mock(), disk_embedder, {})


if __name__ == "__main__":
    unittest.main()


class NamingLayerTests(unittest.TestCase):
    def test_naming_note_only_appears_when_flagged(self):
        from demo.wording import naming_note

        self.assertIsNone(naming_note(0.9, flagged=False))
        self.assertIsNone(naming_note(0.1, flagged=False))
        self.assertIsNone(naming_note(None, flagged=True))
        self.assertIsNotNone(naming_note(0.9, flagged=True))

    def test_naming_note_threshold_is_0_45(self):
        from demo.wording import naming_note

        self.assertIn("descriptive", naming_note(0.45, flagged=True))
        self.assertIn("don't obviously suggest AI", naming_note(0.44, flagged=True))

    def test_commented_out_caveat_rides_on_the_verdict_line(self):
        from demo.wording import COMMENTED_OUT_CAVEAT, explanation_facts, hardcoded_sentences

        for band, flagged in (("red", True), ("yellow", True), ("green", False)):
            facts = explanation_facts(
                status=SCORED_STATUS, band=band, cluster_diversity=0.09,
                commented_out_code=True, scored=True, flagged=flagged,
            )
            sections = dict(hardcoded_sentences(facts))
            # it lives inside Overall Similarity, not its own section
            self.assertIn(COMMENTED_OUT_CAVEAT, sections["Overall Similarity"])
            self.assertNotIn("Human-Leaning Signals", sections)

    def test_no_caveat_when_no_commented_out_code(self):
        from demo.wording import explanation_facts, hardcoded_sentences

        facts = explanation_facts(
            status=SCORED_STATUS, band="red", cluster_diversity=0.09,
            commented_out_code=False, scored=True, flagged=True,
        )
        text = " ".join(s for _h, s in hardcoded_sentences(facts))
        self.assertNotIn("commented-out", text)

    def test_caveat_is_omitted_when_there_is_no_verdict_to_qualify(self):
        from demo.wording import explanation_facts, hardcoded_sentences

        facts = explanation_facts(
            status=INSUFFICIENT_EVIDENCE_STATUS, band="unscored", cluster_diversity=None,
            commented_out_code=True, scored=False, flagged=False,
        )
        text = " ".join(s for _h, s in hardcoded_sentences(facts))
        self.assertNotIn("commented-out", text)

    def test_polish_falls_back_if_the_model_drops_the_caveat(self):
        from demo.wording import _dropped_a_fact

        original = ("Limited similarity - not flagged. Note: it also contains "
                    "commented-out code - a debugging trace, which leans human.")
        self.assertTrue(_dropped_a_fact(original, "Limited similarity, not flagged."))
        self.assertFalse(_dropped_a_fact(original, "Limited similarity; commented-out code present."))
        self.assertFalse(_dropped_a_fact(original, "Limited similarity; a debugging trace is present."))
        self.assertFalse(_dropped_a_fact("Strong similarity - flagged.", "Strong similarity, flagged."))

    def test_naming_never_claims_to_be_the_reason_for_the_flag(self):
        from demo.wording import naming_note

        for frac in (0.1, 0.5, 0.99):
            text = naming_note(frac, flagged=True).lower()
            for banned in ("because", "therefore", "proves", "ai-written", "copied"):
                self.assertNotIn(banned, text)

    def test_naming_fraction_counts_distinct_bindings_only(self):
        from nw_ai_code_detector.style_signals import naming_fractions

        code = (
            "class solution {\npublic:\n    int f(vector<int>& arr) {\n"
            "        int totalCount = 0;\n        int totalCount2 = 0;\n"
            "        for (auto& temp : arr) { totalCount += temp.first; }\n"
            "        return totalCount;\n    }\n};"
        )
        result = naming_fractions(code, "CPP")
        # totalCount, totalCount2, temp - the `first` field and `arr` param excluded
        self.assertEqual(result.unique_identifier_count, 3)


class CanonicalityWordingTests(unittest.TestCase):
    def test_strong_wording_only_when_not_scored(self):
        from demo.wording import CANONICAL_FEW, CANONICAL_TIGHT, canonicality_sentence

        # below the 0.030 scoring floor and not scored -> strong wording is safe
        self.assertEqual(canonicality_sentence(0.01, scored=False), CANONICAL_TIGHT)
        # scored submissions never see "essentially one common solution"
        for diversity in (0.031, 0.040, 0.050):
            self.assertEqual(canonicality_sentence(diversity, scored=True), CANONICAL_FEW)

    def test_cutoff_is_0_050(self):
        from demo.wording import CANONICAL_FEW, CANONICAL_VARIED, canonicality_sentence

        self.assertEqual(canonicality_sentence(0.050, scored=True), CANONICAL_FEW)
        self.assertEqual(canonicality_sentence(0.051, scored=True), CANONICAL_VARIED)

    def test_a_flagged_submission_is_never_told_a_match_is_meaningless(self):
        from demo.wording import explanation_facts, hardcoded_sentences
        from nw_ai_code_detector.constants import SCORED_STATUS

        for band in ("red", "yellow"):
            for diversity in (0.031, 0.045, 0.09):
                facts = explanation_facts(
                    status=SCORED_STATUS, band=band, cluster_diversity=diversity,
                    commented_out_code=False, scored=True, flagged=True,
                )
                text = " ".join(s for _h, s in hardcoded_sentences(facts))
                self.assertNotIn("isn't meaningful", text)
                self.assertNotIn("one common solution", text)

    def test_similarity_wording_matches_the_band(self):
        from demo.wording import SIMILARITY_BY_BAND

        self.assertIn("flagged for review", SIMILARITY_BY_BAND["red"])
        self.assertIn("borderline", SIMILARITY_BY_BAND["yellow"])
        self.assertIn("not flagged", SIMILARITY_BY_BAND["green"])


class VendorNameTests(unittest.TestCase):
    def test_every_slug_in_the_production_bank_maps_to_a_clean_name(self):
        import json

        from demo.wording import VENDOR_FALLBACK, vendor_name
        from nw_ai_code_detector.config import REFERENCE_INDEX_D10_REFERENCES_PATH

        texts = json.loads(REFERENCE_INDEX_D10_REFERENCES_PATH.read_text(encoding="utf-8"))
        slugs = {ref["model"] for group in texts.values() for ref in group}
        self.assertTrue(slugs)
        for slug in slugs:
            self.assertNotEqual(vendor_name(slug), VENDOR_FALLBACK, slug)
        self.assertEqual(
            {vendor_name(s) for s in slugs},
            {"ChatGPT", "Anthropic", "DeepSeek", "Gemini"},
        )

    def test_vendor_mapping(self):
        from demo.wording import vendor_name

        self.assertEqual(vendor_name("openai/gpt-5.5"), "ChatGPT")
        self.assertEqual(vendor_name("anthropic/claude-haiku-4.5"), "Anthropic")
        self.assertEqual(vendor_name("deepseek/deepseek-v4-pro"), "DeepSeek")
        self.assertEqual(vendor_name("google/gemini-3.7-flash"), "Gemini")

    def test_unknown_vendor_never_leaks_a_slug(self):
        from demo.wording import VENDOR_FALLBACK, vendor_name

        for slug in (None, "", "mistral/large", "google/gemma-3"):
            self.assertEqual(vendor_name(slug), VENDOR_FALLBACK)
            self.assertNotIn("/", vendor_name(slug))

    def test_no_raw_slug_is_rendered_in_the_demo(self):
        import pathlib

        source = pathlib.Path("demo/app.py").read_text(encoding="utf-8")
        rendered = [
            line for line in source.splitlines()
            if "reference.generator" in line and "LOGGER" not in line
        ]
        # the slug may only reach the logger, never a caption/markdown call
        for line in rendered:
            self.assertNotIn("st.", line, line)


class NotScoredRenderTests(unittest.TestCase):
    """A not-scored submission has no similarity, canonicality, naming or nearest
    reference. Every render path must degrade cleanly instead of raising KeyError."""

    def _not_scored_facts(self, status):
        from demo.wording import explanation_facts

        return explanation_facts(
            status=status, band="unscored", cluster_diversity=None,
            commented_out_code=False, scored=False, frac_descriptive=None, flagged=False,
        )

    def test_facts_for_a_not_scored_result_contain_only_the_status(self):
        for status in (INSUFFICIENT_EVIDENCE_STATUS, LOW_CONFIDENCE_STATUS):
            facts = self._not_scored_facts(status)
            self.assertEqual(set(facts), {"scoring_status"})

    def test_guidance_panel_renders_without_error_when_not_scored(self):
        from demo.wording import NOT_SCORED_GUIDANCE, guidance_lines

        for status in (INSUFFICIENT_EVIDENCE_STATUS, LOW_CONFIDENCE_STATUS,
                       MISSING_OR_INVALID_STATUS):
            lines = guidance_lines(self._not_scored_facts(status), scored=False)
            self.assertTrue(lines)
            self.assertIn(NOT_SCORED_GUIDANCE, lines)
            self.assertTrue(any("Not scored" in line for line in lines))

    def test_explanation_sections_render_without_error_when_not_scored(self):
        from demo.wording import hardcoded_sentences

        for status in (INSUFFICIENT_EVIDENCE_STATUS, LOW_CONFIDENCE_STATUS):
            sections = hardcoded_sentences(self._not_scored_facts(status))
            self.assertTrue(sections)
            self.assertEqual([h for h, _s in sections], ["Scoring Status"])

    def test_guidance_survives_a_completely_empty_fact_set(self):
        from demo.wording import guidance_lines, hardcoded_sentences

        self.assertTrue(guidance_lines({}, scored=False))
        self.assertEqual(guidance_lines({}, scored=True), [])
        self.assertTrue(hardcoded_sentences({}))

    def test_no_render_path_uses_an_unguarded_fact_lookup(self):
        import pathlib
        import re

        for name in ("demo/app.py", "demo/wording.py"):
            source = pathlib.Path(name).read_text(encoding="utf-8")
            for line_no, line in enumerate(source.splitlines(), start=1):
                match = re.search(r'facts\[(["\'])(\w+)\1\]', line)
                if not match:
                    continue
                key = match.group(2)
                # allowed only immediately inside an `if facts.get("key")` guard
                window = source.splitlines()[max(0, line_no - 4):line_no]
                guarded = any(f'facts.get("{key}")' in w for w in window)
                self.assertTrue(guarded, f"{name}:{line_no} unguarded facts[{key!r}]")

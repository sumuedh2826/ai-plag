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

        self.assertEqual(_band(40), "borderline")
        self.assertEqual(_band(60), "borderline")
        self.assertEqual(_band(39), "low")
        self.assertEqual(_band(61), "high")

    def test_polish_falls_back_when_the_model_is_unavailable(self):
        from unittest.mock import patch

        from demo.wording import explanation_facts, hardcoded_sentences, polish_facts

        facts = explanation_facts(
            status=SCORED_STATUS, match_level="high",
            cluster_diversity=0.09, commented_out_code=False,
        )
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": ""}, clear=False):
            with patch("dotenv.load_dotenv", lambda *a, **k: None):
                self.assertIsNone(polish_facts(facts))
        self.assertTrue(hardcoded_sentences(facts))

    def test_wording_never_claims_authorship_or_copying(self):
        from demo.wording import explanation_facts, hardcoded_sentences

        for level in ("high", "medium", "low"):
            for tight in (0.01, 0.09):
                facts = explanation_facts(
                    status=SCORED_STATUS, match_level=level,
                    cluster_diversity=tight, commented_out_code=False,
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

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

    def test_confidence_mirrors_all_three_detector_routes(self):
        from demo.app import _confidence_label, _token_band_caption

        tight = _detection(LOW_CONFIDENCE_STATUS, None)
        too_short = _detection(INSUFFICIENT_EVIDENCE_STATUS, None)
        short_band = _detection(LOW_CONFIDENCE_SHORT_STATUS, None)
        short_band_named = _detection(LOW_CONFIDENCE_SHORT_STATUS, HIGH_CONFIDENCE_LABEL)
        scored_named = _detection(SCORED_STATUS, HIGH_CONFIDENCE_LABEL)
        scored_plain = _detection(SCORED_STATUS, None)
        self.assertIn("tight cluster", _confidence_label(tight))
        self.assertIn("below token floor", _confidence_label(too_short))
        self.assertIn("short code", _confidence_label(short_band))
        self.assertIn("short code", _confidence_label(short_band_named))
        self.assertIn(HIGH_CONFIDENCE_LABEL, _confidence_label(short_band_named))
        self.assertEqual(_confidence_label(scored_named), HIGH_CONFIDENCE_LABEL)
        self.assertEqual(_confidence_label(scored_plain), "standard")
        self.assertIn("70", _token_band_caption(short_band, "CPP"))
        self.assertIn("110", _token_band_caption(short_band, "CPP"))
        self.assertIn("55", _token_band_caption(short_band, "PYTHON"))
        self.assertIn("100", _token_band_caption(short_band, "PYTHON"))

    def test_demo_scoring_rejects_disk_backed_embedder(self):
        settings = VoyageSettings("secret", "voyage-code-3", 1, 1, 25)
        disk_embedder = object.__new__(VoyageEmbedder)
        disk_embedder._settings = settings
        with self.assertRaises(TypeError):
            score_demo_submission(Mock(), Mock(), disk_embedder, {})


if __name__ == "__main__":
    unittest.main()

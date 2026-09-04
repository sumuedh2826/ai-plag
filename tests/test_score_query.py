import unittest
from unittest.mock import Mock

import numpy as np

from nw_ai_code_detector.constants import (
    INSUFFICIENT_EVIDENCE_STATUS,
    INSUFFICIENT_TOKENS_REASON,
    LOW_CLUSTER_DIVERSITY_REASON,
    LOW_CONFIDENCE_STATUS,
    MISSING_OR_INVALID_STATUS,
    SCORED_STATUS,
)
from nw_ai_code_detector.embedder import EmbeddingBatch
from nw_ai_code_detector.index import ClusterKey, ClusterVectors, ReferenceIndex
from nw_ai_code_detector.score_query import SubmissionScoreRequest, score_submission


PYTHON_BOILERPLATE = """class solution:
    def solve(self, x):
        pass
"""
PYTHON_SHORT = """class solution:
    def solve(self, x):
        return x
"""
PYTHON_LONG = """class solution:
    def helper(self, values, limit):
        total = 0
        for value in values:
            if value < limit:
                total = total + value
            else:
                total = total - value
        return total

    def solve(self, values, limit):
        filtered = []
        for value in values:
            filtered.append(self.helper([value], limit))
        result = 0
        extra_total = 0
        extra_count = 0
        extra_limit = limit
        for item in filtered:
            result = result + item
            extra_total = extra_total + item
            extra_count = extra_count + 1
            extra_limit = extra_limit - 1
        return result + extra_total + extra_count + extra_limit
"""


class ScoreQueryTests(unittest.TestCase):
    def test_short_python_abstains_before_voyage(self):
        embedder = Mock()
        result = score_submission(
            SubmissionScoreRequest(PYTHON_SHORT, PYTHON_BOILERPLATE, "q1", "PYTHON"),
            _index("q1", "PYTHON"),
            embedder,
            {},
        )
        self.assertEqual(result.status, INSUFFICIENT_EVIDENCE_STATUS)
        self.assertEqual(result.reason, INSUFFICIENT_TOKENS_REASON)
        self.assertIsNone(result.score)
        self.assertIsNone(result.decision)
        embedder.embed_texts.assert_not_called()

    def test_missing_cluster_is_invalid_before_tokens(self):
        embedder = Mock()
        result = score_submission(
            SubmissionScoreRequest(PYTHON_SHORT, PYTHON_BOILERPLATE, "q1", "PYTHON"),
            _index("other", "PYTHON"),
            embedder,
            {},
        )
        self.assertEqual(result.status, MISSING_OR_INVALID_STATUS)
        self.assertIsNone(result.score)
        embedder.embed_texts.assert_not_called()

    def test_eligible_code_is_embedded_and_scored(self):
        embedder = Mock()
        embedder.embed_texts.return_value = EmbeddingBatch(
            vectors=((1.0, 0.0),),
            billed_tokens=4,
            cache_hits=0,
            cache_misses=1,
            cost_usd=0.0,
        )
        result = score_submission(
            SubmissionScoreRequest(PYTHON_LONG, PYTHON_BOILERPLATE, "q1", "PYTHON"),
            _index("q1", "PYTHON"),
            embedder,
            {},
        )
        self.assertEqual(result.status, SCORED_STATUS)
        self.assertIsNone(result.reason)
        self.assertIsNone(result.decision)
        self.assertGreater(result.score, 0.9)
        self.assertIsNotNone(result.display_match_percent)
        self.assertIn("Displayed AI-reference match", result.explanation)
        self.assertIsNotNone(result.explanation_card)
        self.assertNotIn("canonicality", result.explanation)
        self.assertNotIn("% AI-written", result.explanation)
        embedder.embed_texts.assert_called_once()

    def test_low_diversity_cluster_is_not_scored(self):
        embedder = Mock()
        vectors = np.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
        key = ClusterKey("q1", "PYTHON")
        index = ReferenceIndex.from_clusters((ClusterVectors(key, (0, 1), vectors),))
        result = score_submission(
            SubmissionScoreRequest(PYTHON_LONG, PYTHON_BOILERPLATE, "q1", "PYTHON"),
            index,
            embedder,
            {},
        )
        self.assertEqual(result.status, LOW_CONFIDENCE_STATUS)
        self.assertEqual(result.reason, LOW_CLUSTER_DIVERSITY_REASON)
        self.assertIsNone(result.score)
        embedder.embed_texts.assert_not_called()


def _index(question_id: str, language: str) -> ReferenceIndex:
    key = ClusterKey(question_id, language)
    vectors = np.asarray([[1.0, 0.0]], dtype=np.float32)
    cluster = ClusterVectors(key, (0,), vectors)
    return ReferenceIndex.from_clusters((cluster,))


if __name__ == "__main__":
    unittest.main()

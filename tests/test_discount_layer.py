import unittest

import numpy as np

from nw_ai_code_detector.constants import (
    CLUSTER_LOW_DIVERSITY_DISTANCE,
    COMMENTED_OUT_CODE_DISCOUNT,
    DESCRIPTIVE_RAISE_FLOOR,
    HIGH_AI_SCORE_FLOOR,
    HIGH_CONFIDENCE_LABEL,
    INSUFFICIENT_EVIDENCE_STATUS,
    INSUFFICIENT_TOKENS_REASON,
    LOW_CLUSTER_DIVERSITY_REASON,
    LOW_CONFIDENCE_SHORT_REASON,
    LOW_CONFIDENCE_SHORT_STATUS,
    LOW_CONFIDENCE_STATUS,
    PYTHON_SIGNIFICANT_TOKEN_THRESHOLD,
    SCORED_STATUS,
    SHORT_LOW_CONFIDENCE_MAX_TOKENS,
)
from nw_ai_code_detector.discount_layer import (
    CanonicalityDiscountRequest,
    assess_canonicality,
    cluster_is_low_diversity,
    mean_pairwise_cosine_distance,
)
from nw_ai_code_detector.style_signals import has_commented_out_code

PYTHON_BOILERPLATE = """class solution:
    def solve(self, x):
        pass
"""
PYTHON_COMMENTED_CODE = """class solution:
    def solve(self, x):
        # print(x)
        # return False
        return x
"""
PYTHON_CLEAN = """class solution:
    def solve(self, x):
        return x
"""


class DiscountLayerTests(unittest.TestCase):
    def test_short_code_routes_to_insufficient_evidence_without_a_score(self):
        reading = assess_canonicality(_python(30, 0.08, False, 0.0))
        self.assertEqual(reading.routing.status, INSUFFICIENT_EVIDENCE_STATUS)
        self.assertEqual(reading.routing.reason, INSUFFICIENT_TOKENS_REASON)
        self.assertTrue(reading.routing.short_code_below_floor)
        self.assertIsNone(reading.score)
        self.assertEqual(reading.raw_canonicality, 0.9)

    def test_low_diversity_routes_to_low_confidence_without_a_score(self):
        reading = assess_canonicality(_python(120, 0.005, False, 0.0))
        self.assertEqual(reading.routing.status, LOW_CONFIDENCE_STATUS)
        self.assertEqual(reading.routing.reason, LOW_CLUSTER_DIVERSITY_REASON)
        self.assertTrue(reading.routing.low_cluster_diversity)
        self.assertIsNone(reading.score)
        self.assertEqual(reading.remaining_factor, 1.0)

    def test_short_code_wins_when_cluster_is_also_tight(self):
        reading = assess_canonicality(_python(20, 0.005, True, 0.0))
        self.assertEqual(reading.routing.status, INSUFFICIENT_EVIDENCE_STATUS)
        self.assertTrue(reading.routing.low_cluster_diversity)
        self.assertIsNone(reading.score)

    def test_commented_out_code_discounts_when_present(self):
        self.assertTrue(
            has_commented_out_code(PYTHON_COMMENTED_CODE, PYTHON_BOILERPLATE, "PYTHON")
        )
        self.assertFalse(has_commented_out_code(PYTHON_CLEAN, PYTHON_BOILERPLATE, "PYTHON"))
        fired = assess_canonicality(_python(120, 0.08, True, 0.0))
        clean = assess_canonicality(_python(120, 0.08, False, 0.0))
        self.assertEqual(fired.routing.status, SCORED_STATUS)
        self.assertEqual(fired.commented_out_code.discount, COMMENTED_OUT_CODE_DISCOUNT)
        self.assertAlmostEqual(fired.score, 0.9 * (1.0 - COMMENTED_OUT_CODE_DISCOUNT))
        self.assertLess(fired.score, fired.raw_canonicality)
        self.assertEqual(clean.score, 0.9)
        self.assertEqual(clean.remaining_factor, 1.0)

    def test_single_letter_and_short_names_do_not_change_the_score(self):
        reading = assess_canonicality(_python(120, 0.08, False, 0.0))
        self.assertEqual(reading.score, 0.9)

    def test_descriptive_names_do_not_change_the_score(self):
        at_floor = assess_canonicality(_python(120, 0.08, False, DESCRIPTIVE_RAISE_FLOOR))
        below = assess_canonicality(
            _python(120, 0.08, False, DESCRIPTIVE_RAISE_FLOOR - 0.01)
        )
        self.assertEqual(at_floor.score, 0.9)
        self.assertEqual(below.score, 0.9)
        self.assertIsNone(at_floor.confidence)
        self.assertTrue(at_floor.descriptive_naming.high)

    def test_high_confidence_label_requires_score_already_above_flag_floor(self):
        flagged = assess_canonicality(
            CanonicalityDiscountRequest(
                0.99, 120, "PYTHON", 0.08, False, DESCRIPTIVE_RAISE_FLOOR
            )
        )
        below_band = assess_canonicality(
            CanonicalityDiscountRequest(0.964, 120, "PYTHON", 0.08, False, 0.50)
        )
        high_score_terse = assess_canonicality(
            CanonicalityDiscountRequest(0.99, 120, "PYTHON", 0.08, False, 0.0)
        )
        self.assertEqual(flagged.score, 0.99)
        self.assertEqual(flagged.confidence, HIGH_CONFIDENCE_LABEL)
        self.assertGreaterEqual(flagged.score, HIGH_AI_SCORE_FLOOR)
        self.assertEqual(below_band.score, 0.964)
        self.assertIsNone(below_band.confidence)
        self.assertLess(below_band.score, HIGH_AI_SCORE_FLOOR)
        self.assertEqual(high_score_terse.score, 0.99)
        self.assertEqual(high_score_terse.confidence, None)

    def test_commented_out_does_not_add_a_descriptive_raise(self):
        reading = assess_canonicality(_python(120, 0.08, True, 1.0))
        expected = 0.9 * (1.0 - COMMENTED_OUT_CODE_DISCOUNT)
        self.assertAlmostEqual(reading.score, expected)
        self.assertIsNone(reading.confidence)

    def test_token_floor_and_diversity_cutoff_are_unchanged(self):
        at_floor = assess_canonicality(
            _python(PYTHON_SIGNIFICANT_TOKEN_THRESHOLD, 0.08, False, 0.0)
        )
        below = assess_canonicality(
            _python(PYTHON_SIGNIFICANT_TOKEN_THRESHOLD - 1, 0.08, False, 0.0)
        )
        self.assertEqual(at_floor.routing.status, LOW_CONFIDENCE_SHORT_STATUS)
        self.assertEqual(at_floor.score, 0.9)
        self.assertEqual(at_floor.remaining_factor, 1.0)
        self.assertEqual(below.routing.status, INSUFFICIENT_EVIDENCE_STATUS)
        self.assertTrue(cluster_is_low_diversity(CLUSTER_LOW_DIVERSITY_DISTANCE - 1e-9))
        self.assertFalse(cluster_is_low_diversity(CLUSTER_LOW_DIVERSITY_DISTANCE))

    def test_medium_short_is_scored_with_low_confidence_flag_and_no_discount(self):
        flagged = assess_canonicality(
            _python(SHORT_LOW_CONFIDENCE_MAX_TOKENS, 0.08, False, 0.0)
        )
        full = assess_canonicality(
            _python(SHORT_LOW_CONFIDENCE_MAX_TOKENS + 1, 0.08, False, 0.0)
        )
        self.assertEqual(flagged.routing.status, LOW_CONFIDENCE_SHORT_STATUS)
        self.assertEqual(flagged.routing.reason, LOW_CONFIDENCE_SHORT_REASON)
        self.assertEqual(flagged.score, 0.9)
        self.assertEqual(flagged.remaining_factor, 1.0)
        self.assertEqual(full.routing.status, SCORED_STATUS)
        self.assertEqual(full.score, 0.9)

    def test_tight_cluster_wins_over_medium_short_band(self):
        reading = assess_canonicality(_python(80, 0.005, False, 0.0))
        self.assertEqual(reading.routing.status, LOW_CONFIDENCE_STATUS)
        self.assertIsNone(reading.score)

    def test_cpp_short_flag_extends_to_110_without_changing_score(self):
        at_ceiling = assess_canonicality(_cpp(110, 0.08, False, 0.0))
        above = assess_canonicality(_cpp(111, 0.08, False, 0.0))
        self.assertEqual(at_ceiling.routing.status, LOW_CONFIDENCE_SHORT_STATUS)
        self.assertEqual(at_ceiling.score, 0.9)
        self.assertEqual(above.routing.status, SCORED_STATUS)
        self.assertEqual(above.score, 0.9)

    def test_python_short_flag_still_ends_at_100(self):
        at_ceiling = assess_canonicality(_python(100, 0.08, False, 0.0))
        above = assess_canonicality(_python(101, 0.08, False, 0.0))
        self.assertEqual(at_ceiling.routing.status, LOW_CONFIDENCE_SHORT_STATUS)
        self.assertEqual(above.routing.status, SCORED_STATUS)

    def test_pairwise_distance_is_zero_for_identical_refs(self):
        repeated = np.ones((6, 4), dtype=np.float32)
        repeated = repeated / np.linalg.norm(repeated, axis=1, keepdims=True)
        self.assertAlmostEqual(mean_pairwise_cosine_distance(repeated), 0.0, places=5)


def _python(tokens, diversity, commented, frac_descriptive):
    return CanonicalityDiscountRequest(
        0.9, tokens, "PYTHON", diversity, commented, frac_descriptive
    )


def _cpp(tokens, diversity, commented, frac_descriptive):
    return CanonicalityDiscountRequest(
        0.9, tokens, "CPP", diversity, commented, frac_descriptive
    )

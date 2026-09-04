import unittest

from nw_ai_code_detector.constants import (
    DISPLAY_MATCH_LOW_SCORE_BY_LANGUAGE,
    DISPLAY_MATCH_MIDPOINT_SCORE_BY_LANGUAGE,
    DISPLAY_MATCH_PERCENT_BELOW_THRESHOLD_MAX,
    DISPLAY_MATCH_PERCENT_MIDPOINT,
)
from nw_ai_code_detector.display_match import display_match_percent


class DisplayMatchTests(unittest.TestCase):
    def test_score_below_threshold_stays_under_midpoint(self):
        for language, midpoint in DISPLAY_MATCH_MIDPOINT_SCORE_BY_LANGUAGE.items():
            percent = display_match_percent(midpoint - 1e-12, language)
            self.assertLessEqual(percent, DISPLAY_MATCH_PERCENT_BELOW_THRESHOLD_MAX)

    def test_score_at_threshold_is_midpoint(self):
        for language, midpoint in DISPLAY_MATCH_MIDPOINT_SCORE_BY_LANGUAGE.items():
            percent = display_match_percent(midpoint, language)
            self.assertEqual(percent, DISPLAY_MATCH_PERCENT_MIDPOINT)

    def test_labeled_human_median_anchor_is_low(self):
        cpp_median = 0.9499712884426117
        python_median = 0.932023823261261
        self.assertLess(display_match_percent(cpp_median, "CPP"), 30)
        self.assertLess(display_match_percent(python_median, "PYTHON"), 30)

    def test_high_ai_score_maps_high(self):
        self.assertGreaterEqual(display_match_percent(0.99, "CPP"), 70)
        self.assertGreaterEqual(display_match_percent(0.99, "PYTHON"), 70)

    def test_below_low_anchor_is_zero(self):
        cpp_low = DISPLAY_MATCH_LOW_SCORE_BY_LANGUAGE["CPP"]
        self.assertEqual(display_match_percent(cpp_low - 0.05, "CPP"), 0)

    def test_unknown_language_raises(self):
        with self.assertRaises(ValueError):
            display_match_percent(0.95, "JAVA")


if __name__ == "__main__":
    unittest.main()

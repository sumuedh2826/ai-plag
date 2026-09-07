import unittest

from nw_ai_code_detector.constants import (
    EXPLANATION_CARD_FOOTER,
    LOW_CONFIDENCE_SHORT_REASON,
    LOW_CONFIDENCE_SHORT_STATUS,
    SCORING_STATUS_SHORT,
)
from nw_ai_code_detector.discount_layer import (
    CanonicalityAssessment,
    CommentedOutDiscount,
    ConfidenceRouting,
    ConventionRaise,
    DescriptiveRaise,
)
from nw_ai_code_detector.similarity_explanation import (
    NearestReferenceMatch,
    build_reference_similarity_summary,
)


FORBIDDEN = (
    "written by AI",
    "matches ChatGPT",
    "AI-generated",
    "authored by",
    "single-letter",
    "human-leaning discount)",
)


class SimilarityExplanationTests(unittest.TestCase):
    def test_summary_claims_reference_similarity_not_authorship(self):
        text = build_reference_similarity_summary(_assessment(), _match(), "PYTHON")
        lowered = text.lower()
        self.assertIn("not proof of authorship", text)
        self.assertIn("lines 8-14", text)
        self.assertIn("Displayed AI-reference match", text)
        self.assertIn(EXPLANATION_CARD_FOOTER, text)
        self.assertNotIn("canonicality", text)
        self.assertNotIn("0.96", text)
        for phrase in FORBIDDEN:
            self.assertNotIn(phrase.lower(), lowered)

    def test_short_code_flag_is_explained_without_changing_authorship_claim(self):
        routing = ConfidenceRouting(
            LOW_CONFIDENCE_SHORT_STATUS, LOW_CONFIDENCE_SHORT_REASON, False, False
        )
        reading = CanonicalityAssessment(
            routing,
            0.96,
            0.94,
            80,
            0.08,
            CommentedOutDiscount(False, 0.0),
            DescriptiveRaise(False, False, 0.0, 0.1),
            ConventionRaise(0.0, False, 0.0),
            1.0,
            0.96,
            None,
        )
        text = build_reference_similarity_summary(reading, _match(), "PYTHON")
        self.assertIn(SCORING_STATUS_SHORT, text)
        self.assertIn("not proof of authorship", text)


def _assessment():
    routing = ConfidenceRouting("scored", None, False, False)
    return CanonicalityAssessment(
        routing,
        0.96,
        0.94,
        120,
        0.08,
        CommentedOutDiscount(False, 0.0),
        DescriptiveRaise(False, False, 0.0, 0.1),
        ConventionRaise(0.0, False, 0.0),
        1.0,
        0.96,
        None,
    )


def _match():
    return NearestReferenceMatch(
        "complete_function",
        "openai/gpt-5.5",
        0.96,
        8,
        14,
        ("for i in range(n):", "for j in range(m):"),
    )

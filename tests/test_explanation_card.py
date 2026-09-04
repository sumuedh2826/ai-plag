import unittest

from nw_ai_code_detector.constants import (
    COMMENTED_OUT_HUMAN_LEANING_BODY,
    EXPLANATION_CARD_FOOTER,
    EXPLANATION_SECTION_AI_REFERENCE_MATCH,
    EXPLANATION_SECTION_HUMAN_LEANING_SIGNALS,
    EXPLANATION_SECTION_SCORING_STATUS,
    INSUFFICIENT_EVIDENCE_STATUS,
    INSUFFICIENT_TOKENS_REASON,
    LOW_CONFIDENCE_SHORT_REASON,
    LOW_CONFIDENCE_SHORT_STATUS,
    MATCH_NOT_SCORED_BODY,
    SCORED_STATUS,
    SCORING_STATUS_SHORT,
)
from nw_ai_code_detector.discount_layer import (
    CanonicalityAssessment,
    CommentedOutDiscount,
    ConfidenceRouting,
    DescriptiveRaise,
)
from nw_ai_code_detector.explanation_card import (
    ExplanationCardRequest,
    build_explanation_card,
    build_unscored_explanation_card,
    format_explanation_card,
)
from nw_ai_code_detector.similarity_explanation import NearestReferenceMatch


class ExplanationCardTests(unittest.TestCase):
    def test_scored_card_has_display_percent_and_no_cosine(self):
        card = build_explanation_card(_request(0.99, SCORED_STATUS, None, False, 0.08))
        text = format_explanation_card(card)
        headings = [section.heading for section in card.sections]
        self.assertIn(EXPLANATION_SECTION_AI_REFERENCE_MATCH, headings)
        self.assertNotIn(EXPLANATION_SECTION_HUMAN_LEANING_SIGNALS, headings)
        self.assertIn("Match level: high", text)
        self.assertIn("Displayed AI-reference match:", text)
        self.assertIn(EXPLANATION_CARD_FOOTER, text)
        self.assertNotIn("% AI-written", text)
        self.assertNotIn("canonicality", text)
        self.assertNotIn("0.99", text)

    def test_commented_out_adds_human_leaning_section_only(self):
        card = build_explanation_card(_request(0.90, SCORED_STATUS, None, True, 0.08))
        headings = [section.heading for section in card.sections]
        self.assertIn(EXPLANATION_SECTION_HUMAN_LEANING_SIGNALS, headings)
        body = [s.body for s in card.sections if s.heading == EXPLANATION_SECTION_HUMAN_LEANING_SIGNALS][0]
        self.assertEqual(body, COMMENTED_OUT_HUMAN_LEANING_BODY)

    def test_short_status_and_unscored_card(self):
        short = build_explanation_card(
            _request(0.90, LOW_CONFIDENCE_SHORT_STATUS, LOW_CONFIDENCE_SHORT_REASON, False, 0.08)
        )
        status = [s.body for s in short.sections if s.heading == EXPLANATION_SECTION_SCORING_STATUS][0]
        self.assertEqual(status, SCORING_STATUS_SHORT)
        blank = build_unscored_explanation_card(
            INSUFFICIENT_EVIDENCE_STATUS,
            INSUFFICIENT_TOKENS_REASON,
        )
        match = [s.body for s in blank.sections if s.heading == EXPLANATION_SECTION_AI_REFERENCE_MATCH][0]
        self.assertEqual(match, MATCH_NOT_SCORED_BODY)


def _request(score, status, reason, commented, diversity):
    routing = ConfidenceRouting(status, reason, False, False)
    assessment = CanonicalityAssessment(
        routing,
        score,
        0.94,
        120,
        diversity,
        CommentedOutDiscount(commented, 0.12 if commented else 0.0),
        DescriptiveRaise(False, False, 0.0, 0.1),
        1.0,
        score if status != INSUFFICIENT_EVIDENCE_STATUS else None,
        None,
    )
    match = NearestReferenceMatch("complete_function", "openai/gpt-5.5", 0.96, 8, 14, ("x",))
    return ExplanationCardRequest(assessment, match, "PYTHON", diversity)


if __name__ == "__main__":
    unittest.main()

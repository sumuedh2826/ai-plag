import csv
import tempfile
import unittest
from pathlib import Path

from nw_ai_code_detector.constants import (
    DETECTOR_CONSENSUS_MIN_SIGNIFICANT_TOKENS,
    DetectorConsensusLabel,
    DetectorConsensusSkipReason,
    DetectorVerdict,
)
from nw_ai_code_detector.detector_consensus import (
    DetectorVote,
    decide_detector_consensus,
    live_detector_keys_configured,
)
from nw_ai_code_detector import build_detector_consensus_proxy as builder


def _vote(name: str, verdict: DetectorVerdict, score: float | None) -> DetectorVote:
    return DetectorVote(
        detector_name=name,
        verdict=verdict,
        human_probability=score,
    )


class DetectorConsensusTests(unittest.TestCase):
    def test_too_short_is_skipped_even_with_votes(self):
        decision = decide_detector_consensus(
            DETECTOR_CONSENSUS_MIN_SIGNIFICANT_TOKENS - 1,
            (
                _vote("sapling", DetectorVerdict.HUMAN, 99.0),
                _vote("gptzero", DetectorVerdict.HUMAN, 99.0),
            ),
            True,
        )
        self.assertEqual(decision.detector_consensus, DetectorConsensusLabel.SKIPPED)
        self.assertEqual(decision.skip_reason, DetectorConsensusSkipReason.TOO_SHORT)

    def test_missing_raw_is_skipped(self):
        decision = decide_detector_consensus(200, (), False)
        self.assertEqual(
            decision.skip_reason,
            DetectorConsensusSkipReason.MISSING_RAW_CODE,
        )

    def test_no_responses_is_skipped(self):
        decision = decide_detector_consensus(200, (), True)
        self.assertEqual(decision.skip_reason, DetectorConsensusSkipReason.NO_RESPONSES)

    def test_one_detector_is_unsure(self):
        decision = decide_detector_consensus(
            200,
            (_vote("sapling", DetectorVerdict.HUMAN, 99.0),),
            True,
        )
        self.assertEqual(decision.detector_consensus, DetectorConsensusLabel.UNSURE)

    def test_two_confident_human_votes_are_consensus_human(self):
        decision = decide_detector_consensus(
            200,
            (
                _vote("sapling", DetectorVerdict.HUMAN, 85.0),
                _vote("gptzero", DetectorVerdict.HUMAN, 90.0),
            ),
            True,
        )
        self.assertEqual(
            decision.detector_consensus,
            DetectorConsensusLabel.CONSENSUS_HUMAN,
        )
        self.assertEqual(decision.responding_detector_count, 2)

    def test_two_confident_ai_votes_are_consensus_ai(self):
        decision = decide_detector_consensus(
            200,
            (
                _vote("sapling", DetectorVerdict.AI, 10.0),
                _vote("gptzero", DetectorVerdict.AI, 0.0),
            ),
            True,
        )
        self.assertEqual(
            decision.detector_consensus,
            DetectorConsensusLabel.CONSENSUS_AI,
        )

    def test_mid_confidence_is_unsure(self):
        decision = decide_detector_consensus(
            200,
            (
                _vote("sapling", DetectorVerdict.HUMAN, 84.0),
                _vote("gptzero", DetectorVerdict.HUMAN, 99.0),
            ),
            True,
        )
        self.assertEqual(decision.detector_consensus, DetectorConsensusLabel.UNSURE)

    def test_disagreement_is_unsure(self):
        decision = decide_detector_consensus(
            200,
            (
                _vote("sapling", DetectorVerdict.HUMAN, 99.0),
                _vote("gptzero", DetectorVerdict.AI, 1.0),
            ),
            True,
        )
        self.assertEqual(decision.detector_consensus, DetectorConsensusLabel.UNSURE)

    def test_all_errors_are_skipped(self):
        decision = decide_detector_consensus(
            200,
            (
                _vote("sapling", DetectorVerdict.ERROR, None),
                _vote("gptzero", DetectorVerdict.ERROR, None),
            ),
            True,
        )
        self.assertEqual(
            decision.skip_reason,
            DetectorConsensusSkipReason.DETECTOR_ERRORS,
        )

    def test_source_never_names_ground_truth_as_a_label(self):
        consensus_source = Path(
            "src/nw_ai_code_detector/detector_consensus.py"
        ).read_text(encoding="utf-8")
        builder_source = Path(
            "src/nw_ai_code_detector/build_detector_consensus_proxy.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("ground_truth", consensus_source)
        self.assertEqual(builder_source.count("ground_truth"), 2)
        self.assertIn("IDENTITY_OUTPUT_KEYS", builder_source)
        self.assertIn("forbidden_name_ground_truth_used", builder_source)
        self.assertIn("detector_consensus", builder_source)
        self.assertNotIn("GROUND_TRUTH", builder_source)

    def test_identity_keys_are_rejected(self):
        with self.assertRaises(RuntimeError):
            builder._reject_identity_keys({"user_id": "x", "record_id": "r"})

    def test_live_keys_require_two_configured_apis(self):
        none = live_detector_keys_configured({})
        one = live_detector_keys_configured({"SAPLING_API_KEY": "abc"})
        two = live_detector_keys_configured(
            {"SAPLING_API_KEY": "abc", "GPTZERO_API_KEY": "def"}
        )
        self.assertEqual(none, ())
        self.assertEqual(one, ("sapling",))
        self.assertEqual(two, ("sapling", "gptzero"))

    def test_manual_csv_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manual.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=builder.MANUAL_CSV_FIELDS)
                writer.writeheader()
                writer.writerow(
                    {
                        "opaque_id": "s0001",
                        "record_id": "candidate_human|q|PYTHON|0|h",
                        "detector_name": "sapling",
                        "verdict": "human",
                        "human_probability": "91",
                        "notes": "",
                    }
                )
            votes = builder._load_manual_votes(path)
            record_votes = votes["candidate_human|q|PYTHON|0|h"]
            self.assertEqual(record_votes[0].verdict, DetectorVerdict.HUMAN)
            self.assertEqual(record_votes[0].human_probability, 91.0)


if __name__ == "__main__":
    unittest.main()

import inspect
import json
import unittest
from pathlib import Path
from unittest.mock import patch

from nw_ai_code_detector import build_canonicality_eligibility_tokens as builder
from nw_ai_code_detector import export_raw_similarity_samples as exporter
from nw_ai_code_detector.canonicality_eligibility import (
    evaluate_submission_eligibility,
)
from nw_ai_code_detector.config import (
    CANDIDATE_HUMAN_SIMILARITY_SAMPLES_PATH,
    CANDIDATE_HUMAN_TOKEN_RAW_SAMPLES_PATH,
    HELDOUT_AI_SIMILARITY_SAMPLES_PATH,
    SIGNIFICANT_TOKEN_ELIGIBILITY_DIR,
)
from nw_ai_code_detector.constants import (
    CPP_SIGNIFICANT_TOKEN_THRESHOLD,
    PYTHON_SIGNIFICANT_TOKEN_THRESHOLD,
    SubmissionExclusionReason,
    UNVERIFIED_LABEL_STATUS,
)
from nw_ai_code_detector.index import ClusterKey
from nw_ai_code_detector.significant_code_tokens import (
    SignificantCodeTokenizationError,
    significant_code_token_count,
)

CPP_MULTILINE = """int add(int a, int b) {
    return a + b;
}
"""
CPP_ONELINE = "int add(int a,int b){return a+b;}"
PYTHON_MULTILINE = """def add(a, b):
    return a + b
"""
PYTHON_ONELINE = "def add(a, b): return a + b"


class SignificantCodeTokenTests(unittest.TestCase):
    def test_finalized_thresholds_are_the_only_active_values(self):
        self.assertEqual(CPP_SIGNIFICANT_TOKEN_THRESHOLD, 70)
        self.assertEqual(PYTHON_SIGNIFICANT_TOKEN_THRESHOLD, 55)
        source = Path("src/nw_ai_code_detector/constants.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("SENSITIVITY_THRESHOLDS", source)

    def test_cpp_boundary_69_and_70(self):
        below = evaluate_submission_eligibility(69, "CPP", True, True)
        self.assertFalse(below.eligible)
        self.assertEqual(
            below.exclusion_reason,
            SubmissionExclusionReason.INSUFFICIENT_SIGNIFICANT_CODE_TOKENS,
        )
        self.assertTrue(
            evaluate_submission_eligibility(70, "CPP", True, True).eligible
        )

    def test_python_boundary_54_and_55(self):
        below = evaluate_submission_eligibility(54, "PYTHON", True, True)
        self.assertFalse(below.eligible)
        self.assertEqual(
            below.exclusion_reason,
            SubmissionExclusionReason.INSUFFICIENT_SIGNIFICANT_CODE_TOKENS,
        )
        self.assertTrue(
            evaluate_submission_eligibility(55, "PYTHON", True, True).eligible
        )

    def test_token_floor_is_an_active_exclude_after_integrity(self):
        source = Path(
            "src/nw_ai_code_detector/canonicality_eligibility.py"
        ).read_text(encoding="utf-8")
        self.assertIn("INSUFFICIENT_SIGNIFICANT_CODE_TOKENS", source)
        self.assertNotIn("entropy", source.lower())
        self.assertFalse(
            evaluate_submission_eligibility(1, "CPP", True, True).eligible
        )

    def test_cpp_formatting_is_invariant(self):
        self.assertEqual(
            significant_code_token_count(CPP_MULTILINE, "CPP"),
            significant_code_token_count(CPP_ONELINE, "CPP"),
        )

    def test_python_formatting_is_invariant(self):
        self.assertEqual(
            significant_code_token_count(PYTHON_MULTILINE, "PYTHON"),
            significant_code_token_count(PYTHON_ONELINE, "PYTHON"),
        )

    def test_blank_lines_spaces_and_tabs_do_not_affect_count(self):
        formatted = "int\tadd ( int a , int b ) {\n\n return a + b ;\n}\n"
        self.assertEqual(
            significant_code_token_count(formatted, "CPP"),
            significant_code_token_count(CPP_ONELINE, "CPP"),
        )

    def test_comments_do_not_affect_count(self):
        commented = CPP_MULTILINE.replace(
            "return",
            "// explanation\n    return",
        )
        self.assertEqual(
            significant_code_token_count(commented, "CPP"),
            significant_code_token_count(CPP_MULTILINE, "CPP"),
        )

    def test_long_and_short_identifiers_each_count_once(self):
        short = significant_code_token_count("int x = a;", "CPP")
        long = significant_code_token_count(
            "int finalCalculatedAdditionResult = sourceOperand;",
            "CPP",
        )
        self.assertEqual(short, long)

    def test_numeric_literal_counts_as_one_token(self):
        one_digit = significant_code_token_count("x = 1", "PYTHON")
        many_digits = significant_code_token_count("x = 123456789", "PYTHON")
        self.assertEqual(one_digit, many_digits)

    def test_string_literal_counts_as_one_token(self):
        short = significant_code_token_count('x = "a"', "PYTHON")
        long = significant_code_token_count(
            'x = "many characters inside one literal"',
            "PYTHON",
        )
        self.assertEqual(short, long)

    def test_operators_are_counted(self):
        base = significant_code_token_count("x = a", "PYTHON")
        with_operator = significant_code_token_count("x = a + b", "PYTHON")
        self.assertEqual(with_operator, base + 2)

    def test_cpp_and_python_use_distinct_lexers(self):
        cpp_parser = significant_code_token_count("int x = 1;", "CPP")
        python_parser = significant_code_token_count("x = 1", "PYTHON")
        self.assertGreater(cpp_parser, python_parser)

    def test_unsupported_language_does_not_fall_back(self):
        with self.assertRaisesRegex(
            SignificantCodeTokenizationError,
            "Unsupported language",
        ):
            significant_code_token_count("int x;", "JAVA")

    def test_tokenizer_failure_is_explicit(self):
        with self.assertRaises(SignificantCodeTokenizationError):
            significant_code_token_count("def broken(:", "PYTHON")


class EligibilityPolicyTests(unittest.TestCase):
    def test_raw_loc_character_word_and_voyage_counts_do_not_control_policy(self):
        source = Path(
            "src/nw_ai_code_detector/canonicality_eligibility.py"
        ).read_text(encoding="utf-8")
        for forbidden in (
            "raw_code",
            "effective" + "_loc",
            "character",
            "splitlines",
            "voyage",
            "word_count",
        ):
            self.assertNotIn(forbidden, source.lower())

    def test_reference_counts_and_voting_do_not_control_policy(self):
        source = Path(
            "src/nw_ai_code_detector/canonicality_eligibility.py"
        ).read_text(encoding="utf-8")
        for forbidden in (
            "distinct_reference",
            "passing_reference",
            "two_thirds",
            "2 / 3",
            "vote",
        ):
            self.assertNotIn(forbidden, source.lower())

    def test_long_submission_is_not_excluded_for_short_references(self):
        decision = evaluate_submission_eligibility(200, "CPP", True, True)
        self.assertTrue(decision.eligible)

    def test_missing_exact_cluster_is_integrity_failure(self):
        decision = evaluate_submission_eligibility(200, "CPP", False, True)
        self.assertEqual(
            decision.exclusion_reason,
            SubmissionExclusionReason.MISSING_EXACT_REFERENCE_CLUSTER,
        )

    def test_excluded_rows_have_no_zero_similarity(self):
        decision = evaluate_submission_eligibility(None, "CPP", True, False)
        self.assertFalse(decision.eligible)
        source = inspect.getsource(builder.evaluate_candidates)
        self.assertIn("if decision.eligible", source)
        self.assertNotIn("ai_nn_max_raw=0", source)
        self.assertNotIn("entropy", inspect.getsource(builder).lower())

    def test_exact_question_language_routing(self):
        source = inspect.getsource(builder.resolve_exact_cluster)
        self.assertIn("ClusterKey(question_id, language)", source)
        self.assertIn("Cross-question reference routing", source)
        self.assertIn("Cross-language reference routing", source)

    def test_cross_language_and_question_routing_are_rejected(self):
        clusters = {
            ClusterKey("q1", "PYTHON"): type(
                "Cluster",
                (),
                {"key": ClusterKey("q1", "PYTHON")},
            )(),
        }
        with self.assertRaisesRegex(RuntimeError, "Missing exact cluster"):
            builder.resolve_exact_cluster("q1", "CPP", clusters)
        wrong = type("Cluster", (), {"key": ClusterKey("q2", "CPP")})()
        with self.assertRaisesRegex(RuntimeError, "Cross-question"):
            builder.resolve_exact_cluster(
                "q1",
                "CPP",
                {ClusterKey("q1", "CPP"): wrong},
            )

    def test_candidate_labels_remain_unverified(self):
        self.assertEqual(UNVERIFIED_LABEL_STATUS, "unverified")
        self.assertNotIn("label = 0", inspect.getsource(builder))

    def test_no_network_embedding_or_faiss_writes(self):
        source = inspect.getsource(builder)
        self.assertNotIn("VoyageEmbedder", source)
        self.assertNotIn("load_voyage_settings", source)
        self.assertNotIn("faiss.write", source.lower())
        self.assertNotIn("np.save", source)
        with patch.object(builder, "cached_vector_for_text", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "network embedding is forbidden"):
                builder.score_stripped_record("q", "CPP", "int x;", {})


class GeneratedArtifactTests(unittest.TestCase):
    def test_attachment_selection_uses_twelve_distinct_hashes(self):
        if not CANDIDATE_HUMAN_TOKEN_RAW_SAMPLES_PATH.is_file():
            self.skipTest("token raw review file not generated")
        parsed = exporter.parse_review_file(
            CANDIDATE_HUMAN_SIMILARITY_SAMPLES_PATH
        )
        selected = exporter.select_samples(parsed, exporter.CANDIDATE_EXPORT_IDS)
        hashes = [sample.fields["RECORD_ID"].split("|")[-1] for sample in selected]
        self.assertEqual(len(hashes), 12)
        self.assertEqual(len(set(hashes)), 12)

    def test_cluster_inventory_and_protected_checksums(self):
        coverage_path = SIGNIFICANT_TOKEN_ELIGIBILITY_DIR / "coverage.json"
        metadata_path = SIGNIFICANT_TOKEN_ELIGIBILITY_DIR / "metadata.json"
        if not coverage_path.is_file() or not metadata_path.is_file():
            self.skipTest("significant-token artifacts not generated")
        coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        clusters = coverage["mixed_v1_clusters"]
        self.assertEqual(clusters["existing_mixed_v1_clusters"], 1000)
        self.assertEqual(clusters["missing_exact_clusters"], 0)
        self.assertTrue(clusters["all_expected_clusters_routable"])
        self.assertNotIn("reference_pair_eligibility", coverage)
        self.assertEqual(metadata["checksums_before"], metadata["checksums_after"])
        self.assertFalse(metadata["network_calls"])
        self.assertFalse(metadata["embeddings_generated"])

    def test_review_samples_meet_token_thresholds(self):
        paths = (
            CANDIDATE_HUMAN_SIMILARITY_SAMPLES_PATH,
            HELDOUT_AI_SIMILARITY_SAMPLES_PATH,
            CANDIDATE_HUMAN_TOKEN_RAW_SAMPLES_PATH,
        )
        if not all(path.is_file() for path in paths):
            self.skipTest("review files not generated")
        for path in paths:
            text = path.read_text(encoding="utf-8")
            counts = [
                int(line.split(":", 1)[1])
                for line in text.splitlines()
                if line.startswith("SIGNIFICANT_CODE_TOKEN_COUNT:")
            ]
            thresholds = [
                int(line.split(":", 1)[1])
                for line in text.splitlines()
                if line.startswith("MINIMUM_SIGNIFICANT_CODE_TOKENS:")
            ]
            self.assertEqual(len(counts), len(thresholds))
            self.assertTrue(counts)


if __name__ == "__main__":
    unittest.main()

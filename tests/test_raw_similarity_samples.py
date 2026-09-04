import inspect
import json
import tempfile
import unittest
from hashlib import sha256
from pathlib import Path
from unittest.mock import patch

from nw_ai_code_detector import export_raw_similarity_samples as exporter
from nw_ai_code_detector.config import (
    CANDIDATE_HUMAN_TOKEN_RAW_SAMPLES_PATH,
)
from nw_ai_code_detector.constants import (
    EXECUTION_CONTRACT_FULL_PROGRAM,
    EXECUTION_CONTRACT_FUNCTION_ONLY,
    EXECUTION_CONTRACT_UNKNOWN,
    RAW_CODE_UNAVAILABLE,
    STRIP_CANNOT_VERIFY,
    STRIP_EXACT_MATCH,
    STRIP_MISMATCH,
)
from nw_ai_code_detector.stripper import Language, strip_solution_body


FULL_PROGRAM = """#include <bits/stdc++.h>
using namespace std;
int main() {
    int n;
    cin >> n;
    cout << n << endl;
    return 0;
}
"""
FUNCTION_ONLY = """#include <bits/stdc++.h>
using namespace std;
class solution {
public:
    int run() {
        return 1;
    }
};
int main() { return 0; }
"""
FUNCTION_BOILERPLATE = """#include <bits/stdc++.h>
using namespace std;
class solution {
public:
    int run() {
    }
};
"""


class RawSimilarityExportTests(unittest.TestCase):
    def test_raw_code_is_retrieved_from_original_source(self):
        groups = {
            "q1:CPP": [{"raw_code": FULL_PROGRAM, "user_id": "secret"}],
        }
        record_id = "candidate_human|q1|CPP|0|abc"
        raw = exporter.retrieve_candidate_raw_code(record_id, groups)
        self.assertEqual(raw, FULL_PROGRAM)
        self.assertIn("#include", raw)

    def test_raw_code_is_never_reconstructed_from_stripped_code(self):
        source = inspect.getsource(exporter.retrieve_candidate_raw_code)
        self.assertNotIn("stripped_code", source)
        self.assertNotIn("STRIPPED_CODE_FIELD", source)
        groups = {"q1:CPP": [{"code": "int x;", "raw_code": ""}]}
        raw = exporter.retrieve_candidate_raw_code(
            "candidate_human|q1|CPP|0|abc",
            groups,
        )
        self.assertEqual(raw, RAW_CODE_UNAVAILABLE)

    def test_raw_exports_preserve_imports_and_includes(self):
        self.assertRegex(FULL_PROGRAM, r"#include")

    def test_raw_exports_preserve_main_input_and_output(self):
        main, inputs, outputs = exporter.detect_logic(FULL_PROGRAM)
        self.assertTrue(main)
        self.assertTrue(inputs)
        self.assertTrue(outputs)

    def test_full_program_main_is_not_classified_as_removable_boilerplate(self):
        issues = exporter.audit_required_removals(
            EXECUTION_CONTRACT_FULL_PROGRAM,
            FULL_PROGRAM,
            "int run() { return 1; }",
            "",
            "rid",
            "qid",
            "CPP",
            True,
            False,
            True,
            False,
            True,
            False,
        )
        self.assertTrue(any("main_or___main__" in item for item in issues))
        self.assertTrue(any("input_parsing" in item for item in issues))
        self.assertTrue(any("required_output" in item for item in issues))

    def test_required_output_formatting_is_preserved_in_raw(self):
        self.assertIn("cout", FULL_PROGRAM)

    def test_function_only_known_platform_drivers_are_identified_separately(self):
        self.assertTrue(exporter._driver_in_template(FUNCTION_ONLY))
        issues = exporter.audit_required_removals(
            EXECUTION_CONTRACT_FUNCTION_ONLY,
            FUNCTION_ONLY,
            "class solution { int run() { return 1; } };",
            FUNCTION_ONLY,
            "rid",
            "qid",
            "CPP",
            True,
            False,
            False,
            False,
            False,
            False,
        )
        self.assertFalse(any("main_not_matching_known_template" in item for item in issues))

    def test_user_authored_print_or_cout_is_not_automatic_boilerplate(self):
        raw = "class solution { int run() { cout << 1; return 1; } };"
        issues = exporter.audit_required_removals(
            EXECUTION_CONTRACT_FUNCTION_ONLY,
            raw,
            "class solution { int run() { return 1; } };",
            FUNCTION_BOILERPLATE,
            "rid",
            "qid",
            "CPP",
            False,
            False,
            False,
            False,
            True,
            False,
        )
        self.assertTrue(any("user_authored_output" in item for item in issues))

    def test_unknown_execution_contracts_remain_unknown(self):
        contract = exporter.resolve_execution_contract("missing", "CPP", {})
        self.assertEqual(contract, EXECUTION_CONTRACT_UNKNOWN)
        issues = exporter.audit_required_removals(
            EXECUTION_CONTRACT_UNKNOWN,
            FULL_PROGRAM,
            "int run() { return 1; }",
            "",
            "rid",
            "qid",
            "CPP",
            True,
            False,
            False,
            False,
            False,
            False,
        )
        self.assertTrue(
            any("main_removed_under_unknown_contract" in item for item in issues)
        )

    def test_stripping_reproduction_is_checked_against_stored_hash(self):
        reproduced = strip_solution_body(
            FUNCTION_ONLY,
            FUNCTION_BOILERPLATE,
            Language.CPP,
        )
        stored = sha256(reproduced.encode("utf-8")).hexdigest()
        status = exporter.reproduce_strip_status(
            FUNCTION_ONLY,
            stored,
            "CPP",
            FUNCTION_BOILERPLATE,
        )
        self.assertEqual(status, STRIP_EXACT_MATCH)

    def test_mismatches_are_reported_rather_than_overwritten(self):
        status = exporter.reproduce_strip_status(
            FUNCTION_ONLY,
            "deadbeef",
            "CPP",
            FUNCTION_BOILERPLATE,
        )
        self.assertEqual(status, STRIP_MISMATCH)
        unavailable = exporter.reproduce_strip_status(
            RAW_CODE_UNAVAILABLE,
            "hash",
            "CPP",
            "",
        )
        self.assertEqual(unavailable, STRIP_CANNOT_VERIFY)
        source = inspect.getsource(exporter.reproduce_strip_status)
        self.assertNotIn("write_text", source)

    def test_no_embeddings_faiss_or_network(self):
        source = inspect.getsource(exporter)
        self.assertNotIn("VoyageEmbedder", source)
        self.assertNotIn("faiss", source.lower())
        self.assertNotIn("load_voyage_settings", source)
        self.assertNotIn("np.save", source)

    def test_generated_files_contain_no_personal_identities(self):
        if not CANDIDATE_HUMAN_TOKEN_RAW_SAMPLES_PATH.is_file():
            self.skipTest("token raw review file not generated yet")
        text = CANDIDATE_HUMAN_TOKEN_RAW_SAMPLES_PATH.read_text(encoding="utf-8")
        for forbidden in (
            "user_id",
            "username",
            "email",
            "record_id",
            "question_id",
            "STRIPPED_CODE_USED_FOR_EMBEDDING",
        ):
            self.assertNotIn(forbidden.lower(), text.lower())
        self.assertEqual(text.count("SAMPLE:"), 12)
        self.assertEqual(text.count("LANGUAGE: CPP"), 6)
        self.assertEqual(text.count("LANGUAGE: PYTHON"), 6)
        self.assertEqual(text.count("CATEGORY: LOWEST"), 6)
        self.assertEqual(text.count("CATEGORY: HIGHEST"), 6)
        self.assertIn("RAW_CODE:", text)
        self.assertIn("EXECUTION_CONTRACT:", text)
        self.assertIn("STRIP_REPRODUCTION_STATUS:", text)
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
        self.assertEqual(len(counts), 12)
        self.assertEqual(len(thresholds), 12)
        self.assertEqual(text.count("EXACT_MATCH_TO_MIXED_V1:"), 12)


if __name__ == "__main__":
    unittest.main()

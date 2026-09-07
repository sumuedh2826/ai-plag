import unittest
from hashlib import sha256
from unittest.mock import Mock

import numpy as np

from nw_ai_code_detector.constants import (
    EXACT_REFERENCE_MATCH_HINT,
    INSUFFICIENT_EVIDENCE_STATUS,
    SCORED_STATUS,
    StyleSignalName,
)
from nw_ai_code_detector.embedder import EmbeddingBatch
from nw_ai_code_detector.index import ClusterKey, ClusterVectors, ReferenceIndex
from nw_ai_code_detector.score_query import SubmissionScoreRequest, score_submission
from nw_ai_code_detector.stripper import Language, strip_solution_body
from nw_ai_code_detector.style_signals import (
    _is_commented_out_code,
    extract_style_flags,
    naming_fractions,
)


PYTHON_BOILERPLATE = """class solution:
    def solve(self, x):
        pass
"""
CPP_BOILERPLATE = """class solution {
public:
    int solve(int x) {
    }
};
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
PYTHON_EXPLAIN = '''class solution:
    def solve(self, x):
        # This stores the running total from the left side
        return x
'''
PYTHON_TERSE = """class solution:
    def solve(self, x):
        # base case
        return x
"""
PYTHON_COMMENTED_CODE = """class solution:
    def solve(self, x):
        # print(x)
        # return False
        return x
"""
PYTHON_UNUSED = """class solution:
    def solve(self, x):
        unused_temp = 1
        return x
"""
PYTHON_PARAM_ONLY = """class solution:
    def solve(self, x):
        return x + 1
"""
PYTHON_LOOP_UNUSED = """class solution:
    def solve(self, values):
        total = 0
        for item in values:
            total = total + 1
        return total
"""
PYTHON_UNIFORM = """class solution:
    def solve(self, input_values, window_limit):
        left_bound_index = 0
        current_window_sum = 0
        maximum_window_sum = 0
        right_bound_index = window_limit
        return maximum_window_sum + left_bound_index + current_window_sum + len(input_values)
"""
CPP_COMMENTED_CODE = """class solution {
public:
    int solve(int x) {
        // int skipped = x;
        // return skipped;
        return x;
    }
};
"""
CPP_PROSE_COMMENTS = """class solution {
public:
    int solve(int x) {
        // No path from 0 to n-1
        if (x < 0) {
            return -1;
        }
        // Reconstruct path
        return x;
    }
};
"""
CPP_UNUSED = """class solution {
public:
    int solve(int x) {
        int unusedTemp = 0;
        return x;
    }
};
"""


class StyleSignalTests(unittest.TestCase):
    def test_explanatory_comment_is_ai_lean_not_terse_label(self):
        flags = _python_flags(PYTHON_EXPLAIN)
        self.assertTrue(_fired(flags, StyleSignalName.EXPLANATORY_COMMENTS))
        terse = _python_flags(PYTHON_TERSE)
        self.assertFalse(_fired(terse, StyleSignalName.EXPLANATORY_COMMENTS))
        self.assertFalse(_fired(terse, StyleSignalName.COMMENTED_OUT_CODE))

    def test_commented_out_code_is_human_lean_not_explanation(self):
        flags = _python_flags(PYTHON_COMMENTED_CODE)
        self.assertTrue(_fired(flags, StyleSignalName.COMMENTED_OUT_CODE))
        self.assertFalse(_fired(flags, StyleSignalName.EXPLANATORY_COMMENTS))
        cpp = _cpp_flags(CPP_COMMENTED_CODE)
        self.assertTrue(_fired(cpp, StyleSignalName.COMMENTED_OUT_CODE))

    def test_cpp_path_explanations_are_not_commented_out_code(self):
        flags = _cpp_flags(CPP_PROSE_COMMENTS)
        self.assertFalse(_fired(flags, StyleSignalName.COMMENTED_OUT_CODE))

    def test_comment_body_parse_classifies_code_not_prose(self):
        code_bodies = (
            "x = x + 1",
            "result.push_back(temp)",
            "count++",
            "arr[i] = 0",
            "return ans;",
            "if (x) return;",
        )
        prose_bodies = (
            "No path from 0 to n-1",
            "Reconstruct path",
            "TODO fix this",
            "base case",
            "handle edge case",
        )
        for body in code_bodies:
            self.assertTrue(_is_commented_out_code(body, Language.CPP), body)
        for body in prose_bodies:
            self.assertFalse(_is_commented_out_code(body, Language.CPP), body)

    def test_python_partial_assignment_and_flight_explanations_are_prose(self):
        self.assertTrue(
            _is_commented_out_code("dist[i] = cost + price", Language.PYTHON)
        )
        prose_bodies = (
            "dist[i] = cheapest cost to reach city i from src using at most the allowed edges.",
            "At most k stops means at most k+1 edges.",
            "Snapshot of current distances so we don't use more than one edge per round.",
            "Run the relaxation k+1 times.",
        )
        for body in prose_bodies:
            self.assertFalse(_is_commented_out_code(body, Language.PYTHON), body)

    def test_unused_local_ignores_params_and_loop_vars(self):
        self.assertTrue(_fired(_python_flags(PYTHON_UNUSED), StyleSignalName.UNUSED_LOCALS))
        self.assertTrue(_fired(_cpp_flags(CPP_UNUSED), StyleSignalName.UNUSED_LOCALS))
        self.assertFalse(
            _fired(_python_flags(PYTHON_PARAM_ONLY), StyleSignalName.UNUSED_LOCALS)
        )
        self.assertFalse(
            _fired(_python_flags(PYTHON_LOOP_UNUSED), StyleSignalName.UNUSED_LOCALS)
        )

    def test_uniform_naming_is_experimental_ai_flag(self):
        flags = _python_flags(PYTHON_UNIFORM)
        naming = _flag(flags, StyleSignalName.UNIFORM_VERBOSE_NAMING)
        self.assertTrue(naming.fired)
        self.assertTrue(naming.experimental)
        self.assertEqual(naming.direction, "ai")
        mixed = _python_flags(PYTHON_LONG)
        self.assertFalse(_fired(mixed, StyleSignalName.UNIFORM_VERBOSE_NAMING))

    def test_descriptive_fraction_excludes_fields_params_and_methods(self):
        tree = """class solution{
public:
    bool childrenSumProperty(Node* root) {
        if(root==nullptr)return true;
        int total = 0;
        if(root->left){
            total+=root->left->data;
        }
        if(root->right){
            total+=root->right->data;
        }
        return (root->data==total);
    }
};
"""
        fractions = naming_fractions(tree, "CPP")
        self.assertEqual(fractions.unique_identifier_count, 1)
        self.assertEqual(fractions.frac_descriptive, 0.0)
        verbose = """class solution {
public:
    int solve(int values) {
        int runningTotalValue = 0;
        int currentWindowLimit = values;
        return runningTotalValue + currentWindowLimit;
    }
};
"""
        raised = naming_fractions(verbose, "CPP")
        self.assertEqual(raised.unique_identifier_count, 2)
        self.assertEqual(raised.frac_descriptive, 1.0)

    def test_convention_frac_is_camel_for_cpp_and_snake_for_python(self):
        camel = naming_fractions(
            """class solution {
public:
    int solve(int values) {
        int maxNode = 0;
        int nextNode = values;
        int prev = maxNode + nextNode;
        return prev;
    }
};
""",
            "CPP",
        )
        snake = naming_fractions(PYTHON_UNIFORM, "PYTHON")
        self.assertGreater(camel.convention_frac, 0.42)
        self.assertGreater(snake.convention_frac, 0.42)


class RankAndFlagQueryTests(unittest.TestCase):
    def test_short_python_abstains_without_verdict(self):
        result = score_submission(
            SubmissionScoreRequest(PYTHON_SHORT, PYTHON_BOILERPLATE, "q1", "PYTHON"),
            _index("q1", "PYTHON"),
            Mock(),
            {},
        )
        self.assertEqual(result.status, INSUFFICIENT_EVIDENCE_STATUS)
        self.assertIsNone(result.decision)
        self.assertFalse(result.exact_match_flag)
        self.assertIsNone(result.canonicality)

    def test_eligible_ranks_by_canonicality_without_fused_risk(self):
        result = _score(PYTHON_LONG, {})
        self.assertEqual(result.status, SCORED_STATUS)
        self.assertEqual(result.score, result.canonicality)
        self.assertGreater(result.canonicality, 0.9)
        self.assertIsNotNone(result.token_count)
        self.assertIsNone(result.decision)
        self.assertIn("generated AI reference", result.explanation)
        self.assertIn("not proof of authorship", result.explanation)
        self.assertFalse(hasattr(result, "risk_score"))

    def test_commented_out_code_lowers_score_not_canonicality(self):
        raw = PYTHON_LONG.replace(
            "        filtered = []\n",
            "        # print(values)\n        filtered = []\n",
        )
        result = _score(raw, {})
        self.assertEqual(result.status, SCORED_STATUS)
        self.assertTrue(_fired(result.signals, StyleSignalName.COMMENTED_OUT_CODE))
        self.assertLess(result.score, result.canonicality)

    def test_exact_hash_is_the_only_decision_hint(self):
        stripped = strip_solution_body(PYTHON_LONG, PYTHON_BOILERPLATE, "PYTHON")
        digest = sha256(stripped.encode("utf-8")).hexdigest()
        result = _score(PYTHON_LONG, {("q1", "PYTHON"): {digest}})
        self.assertTrue(result.exact_match_flag)
        self.assertEqual(result.decision, EXACT_REFERENCE_MATCH_HINT)
        self.assertIn("Exact reference match", result.explanation)


def _python_flags(raw):
    stripped = strip_solution_body(raw, PYTHON_BOILERPLATE, "PYTHON")
    return extract_style_flags(raw, stripped, "PYTHON", PYTHON_BOILERPLATE)


def _cpp_flags(raw):
    stripped = strip_solution_body(raw, CPP_BOILERPLATE, "CPP")
    return extract_style_flags(raw, stripped, "CPP", CPP_BOILERPLATE)


def _fired(flags, name):
    return _flag(flags, name).fired


def _flag(flags, name):
    return next(item for item in flags if item.name == name.value)


def _score(raw, hashes):
    embedder = Mock()
    embedder.embed_texts.return_value = EmbeddingBatch(
        vectors=((1.0, 0.0),),
        billed_tokens=4,
        cache_hits=0,
        cache_misses=1,
        cost_usd=0.0,
    )
    return score_submission(
        SubmissionScoreRequest(raw, PYTHON_BOILERPLATE, "q1", "PYTHON"),
        _index("q1", "PYTHON"),
        embedder,
        hashes,
    )


def _index(question_id, language):
    key = ClusterKey(question_id, language)
    vectors = np.asarray([[1.0, 0.0]], dtype=np.float32)
    cluster = ClusterVectors(key, (0,), vectors)
    return ReferenceIndex.from_clusters((cluster,))


if __name__ == "__main__":
    unittest.main()

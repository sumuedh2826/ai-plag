import unittest

from nw_ai_code_detector.stripper import Language, parses_source, strip_solution_body


CPP_BOILERPLATE = """class solution {
public:
    int solve(int x) {
        // write your code here
    }
};
"""
CPP_RAW = """class solution {
public:
    int helper(int x) { return x + 1; } // helper note
    int solve(int x) {
        /* block
           explanation */
        int y = helper(x); // trailing note
        std::cout << "http://example"; // output is intentional
        return y;
    }
};
"""
PYTHON_BOILERPLATE = """class solution:
    def solve(self, x):
        # write your code here
        pass
"""
PYTHON_RAW = '''"""module documentation"""
class solution:
    """class documentation"""
    def helper(self, x):
        """helper documentation"""
        return x + 1  # helper note

    def solve(self, x):
        """target documentation"""
        # explanation
        value = self.helper(x)  # trailing note
        print("literal # and // stay")
        return value
'''


class StripperCommentRemovalTests(unittest.TestCase):
    def test_cpp_comments_are_removed_without_dropping_author_code(self):
        stripped = strip_solution_body(CPP_RAW, CPP_BOILERPLATE, Language.CPP)
        self.assertNotIn("// helper note", stripped)
        self.assertNotIn("/* block", stripped)
        self.assertNotIn("// trailing note", stripped)
        self.assertIn("int helper(int x) { return x + 1; }", stripped)
        self.assertIn('std::cout << "http://example";', stripped)
        self.assertTrue(parses_source(stripped, Language.CPP))

    def test_python_comments_and_docstrings_are_removed(self):
        stripped = strip_solution_body(
            PYTHON_RAW,
            PYTHON_BOILERPLATE,
            Language.PYTHON,
        )
        self.assertNotIn("documentation", stripped)
        self.assertNotIn("# explanation", stripped)
        self.assertNotIn("# trailing note", stripped)
        self.assertIn("def helper(self, x):", stripped)
        self.assertIn('print("literal # and // stay")', stripped)
        self.assertTrue(parses_source(stripped, Language.PYTHON))


if __name__ == "__main__":
    unittest.main()

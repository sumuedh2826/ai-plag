from __future__ import annotations

from dataclasses import dataclass

from tree_sitter import Node

from nw_ai_code_detector.significant_code_tokens import _parser
from nw_ai_code_detector.stripper import Language

HISTOGRAM_TYPES = (
    "if_statement",
    "elif_clause",
    "else_clause",
    "for_statement",
    "for_range_loop",
    "while_statement",
    "switch_statement",
    "case_statement",
    "try_statement",
    "with_statement",
    "return_statement",
    "break_statement",
    "continue_statement",
    "raise_statement",
    "expression_statement",
    "function_definition",
    "class_definition",
    "class_specifier",
    "declaration",
)
BRANCH_TYPES = frozenset(
    {
        "if_statement",
        "elif_clause",
        "else_clause",
        "case_statement",
        "conditional_expression",
        "ternary_expression",
    }
)
LOOP_TYPES = frozenset({"for_statement", "for_range_loop", "while_statement", "do_statement"})
NESTING_TYPES = frozenset(
    {
        "if_statement",
        "for_statement",
        "for_range_loop",
        "while_statement",
        "do_statement",
        "try_statement",
        "with_statement",
        "switch_statement",
        "compound_statement",
        "block",
    }
)
CYCLOMATIC_TYPES = BRANCH_TYPES | LOOP_TYPES | frozenset({"catch_clause", "except_clause"})
STATEMENT_SUFFIX = "_statement"


@dataclass(frozen=True)
class StructuralAstFeatures:
    max_nesting_depth: int
    control_flow_branch_count: int
    control_flow_loop_count: int
    statement_count: int
    statement_type_histogram: tuple[int, ...]
    cyclomatic_complexity: int
    ast_depth: int
    helper_function_ratio: float


def extract_structural_ast_features(stripped_code: str, language: str) -> StructuralAstFeatures | None:
    if not stripped_code.strip():
        return None
    try:
        source_language = Language(language)
    except ValueError:
        return None
    tree = _parser(source_language).parse(stripped_code.encode("utf-8"))
    if tree.root_node.has_error:
        return None
    return _features_from_root(tree.root_node)


def structural_feature_vector(features: StructuralAstFeatures) -> tuple[float, ...]:
    scalars = (
        float(features.max_nesting_depth),
        float(features.control_flow_branch_count),
        float(features.control_flow_loop_count),
        float(features.statement_count),
        float(features.cyclomatic_complexity),
        float(features.ast_depth),
        float(features.helper_function_ratio),
    )
    histogram = tuple(float(value) for value in features.statement_type_histogram)
    return scalars + histogram


def _features_from_root(root: Node) -> StructuralAstFeatures:
    counts = {name: 0 for name in HISTOGRAM_TYPES}
    _count_types(root, counts)
    function_count = counts["function_definition"]
    helper_ratio = 0.0
    if function_count > 0:
        helper_ratio = max(function_count - 1, 0) / function_count
    branch_count = _count_matching(root, BRANCH_TYPES)
    loop_count = _count_matching(root, LOOP_TYPES)
    return StructuralAstFeatures(
        max_nesting_depth=_max_nesting(root, 0),
        control_flow_branch_count=branch_count,
        control_flow_loop_count=loop_count,
        statement_count=_count_statements(root),
        statement_type_histogram=tuple(counts[name] for name in HISTOGRAM_TYPES),
        cyclomatic_complexity=1 + _count_matching(root, CYCLOMATIC_TYPES),
        ast_depth=_max_ast_depth(root),
        helper_function_ratio=helper_ratio,
    )


def _count_types(node: Node, counts: dict[str, int]) -> None:
    if node.type in counts:
        counts[node.type] += 1
    for child in node.children:
        _count_types(child, counts)


def _count_matching(node: Node, types: frozenset[str]) -> int:
    total = 1 if node.type in types else 0
    return total + sum(_count_matching(child, types) for child in node.children)


def _count_statements(node: Node) -> int:
    total = 1 if node.type.endswith(STATEMENT_SUFFIX) else 0
    return total + sum(_count_statements(child) for child in node.children)


def _max_nesting(node: Node, depth: int) -> int:
    next_depth = depth + 1 if node.type in NESTING_TYPES else depth
    if not node.children:
        return next_depth
    return max(_max_nesting(child, next_depth) for child in node.children)


def _max_ast_depth(node: Node) -> int:
    if not node.children:
        return 1
    return 1 + max(_max_ast_depth(child) for child in node.children)

from __future__ import annotations

from functools import lru_cache

import tree_sitter_cpp
import tree_sitter_python
from tree_sitter import Language as TreeSitterLanguage
from tree_sitter import Node, Parser

from nw_ai_code_detector.stripper import Language

IGNORED_NODE_TYPES = {
    "comment",
    "encoding",
}
LITERAL_NODE_TYPES = {
    "char_literal",
    "concatenated_string",
    "raw_string_literal",
    "string",
    "string_literal",
}


class SignificantCodeTokenizationError(ValueError):
    pass


def significant_code_token_count(stripped_code: str, language: str) -> int:
    normalized = _normalize_language(language)
    if not stripped_code.strip():
        raise SignificantCodeTokenizationError("Stripped code is empty")
    tree = _parser(normalized).parse(stripped_code.encode("utf-8"))
    if tree.root_node.has_error:
        raise SignificantCodeTokenizationError(
            f"Unable to tokenize stripped code as {normalized.value}"
        )
    return _count_significant_leaves(tree.root_node)


def _normalize_language(language: str) -> Language:
    try:
        return Language(language.upper())
    except ValueError as exc:
        raise SignificantCodeTokenizationError(
            f"Unsupported language: {language}"
        ) from exc


@lru_cache(maxsize=2)
def _parser(language: Language) -> Parser:
    if language is Language.CPP:
        capsule = tree_sitter_cpp.language()
    elif language is Language.PYTHON:
        capsule = tree_sitter_python.language()
    else:
        raise SignificantCodeTokenizationError(
            f"Unsupported language: {language.value}"
        )
    return Parser(TreeSitterLanguage(capsule))


def _count_significant_leaves(node: Node) -> int:
    if node.type in IGNORED_NODE_TYPES:
        return 0
    if node.type in LITERAL_NODE_TYPES:
        return 1
    if not node.children:
        return 1
    return sum(_count_significant_leaves(child) for child in node.children)

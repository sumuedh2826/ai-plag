from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

import tree_sitter_cpp
import tree_sitter_python
from tree_sitter import Language as TreeSitterLanguage
from tree_sitter import Node, Parser

from nw_ai_code_detector.constants import (
    BOILERPLATE_COMMENT_MARKERS,
    COMMENTED_CODE_STATEMENT_TYPES,
    EXPLANATORY_COMMENT_CUES,
    EXPLANATORY_COMMENT_MIN_WORDS,
    EXPLANATORY_COMMENT_SHORT_SENTENCE_WORDS,
    NAMING_DESCRIPTIVE_FRACTION_FLOOR,
    NAMING_DESCRIPTIVE_TOKEN_MIN_LENGTH,
    NAMING_MIN_UNIQUE_IDENTIFIERS,
    NAMING_SHORT_NAME_MAX_LENGTH,
    STYLE_SKIP_BUILTIN_IDENTIFIERS,
    STYLE_SKIP_IDENTIFIERS,
    StyleSignalDirection,
    StyleSignalName,
    UNUSED_PLACEHOLDER_NAMES,
)
from nw_ai_code_detector.local_bindings import local_binding_names
from nw_ai_code_detector.stripper import COMMENT_NODE_TYPE, Language

CAMEL_SPLIT_PATTERN = re.compile(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])")
SNAKE_SPLIT_PATTERN = re.compile(r"[a-zA-Z][a-zA-Z0-9]*")
WORD_PATTERN = re.compile(r"[A-Za-z]+")
CPP_LOOP_TYPES = frozenset({"for_statement", "for_range_loop", "while_statement"})
PYTHON_LOOP_TYPES = frozenset(
    {"for_statement", "for_in_clause", "while_statement", "list_comprehension"}
)
PYTHON_PARAM_ROOT_TYPES = frozenset({"parameters", "lambda_parameters"})
CPP_PARAM_ROOT_TYPES = frozenset({"parameter_list", "parameter_declaration"})


@dataclass(frozen=True)
class StyleFlag:
    name: str
    fired: bool
    direction: str
    experimental: bool


@dataclass(frozen=True)
class CommentSpan:
    start_row: int
    end_row: int
    body: str


def has_commented_out_code(raw_code: str, boilerplate: str, language: str) -> bool:
    source_language = Language(language)
    comments = _author_comment_bodies(raw_code, boilerplate, source_language)
    return any(_is_commented_out_code(body, source_language) for body in comments)


@dataclass(frozen=True)
class NamingFractions:
    frac_single_letter: float
    frac_short: float
    frac_descriptive: float
    unique_identifier_count: int


def naming_fractions(stripped_code: str, language: str) -> NamingFractions:
    source_language = Language(language)
    if not stripped_code.strip():
        return NamingFractions(0.0, 0.0, 0.0, 0)
    tree = _parse_tree(stripped_code, source_language)
    names = local_binding_names(tree.root_node, stripped_code.encode("utf-8"), source_language)
    total = len(names)
    if total == 0:
        return NamingFractions(0.0, 0.0, 0.0, 0)
    single = sum(1 for name in names if len(name) == 1) / total
    short = sum(1 for name in names if len(name) <= NAMING_SHORT_NAME_MAX_LENGTH) / total
    descriptive = sum(1 for name in names if _is_descriptive_name(name)) / total
    return NamingFractions(single, short, descriptive, total)


def extract_style_flags(
    raw_code: str,
    stripped_code: str,
    language: str,
    boilerplate: str,
) -> tuple[StyleFlag, ...]:
    source_language = Language(language)
    comments = _author_comment_bodies(raw_code, boilerplate, source_language)
    commented_out = any(_is_commented_out_code(body, source_language) for body in comments)
    explanatory = any(
        (not _is_commented_out_code(body, source_language)) and _is_explanatory_prose(body)
        for body in comments
    )
    unused_locals = _has_unused_local(stripped_code, source_language)
    uniform_naming = _has_uniform_verbose_naming(stripped_code, source_language)
    return (
        _flag(StyleSignalName.EXPLANATORY_COMMENTS, explanatory, StyleSignalDirection.AI, False),
        _flag(StyleSignalName.COMMENTED_OUT_CODE, commented_out, StyleSignalDirection.HUMAN, False),
        _flag(StyleSignalName.UNUSED_LOCALS, unused_locals, StyleSignalDirection.HUMAN, False),
        _flag(
            StyleSignalName.UNIFORM_VERBOSE_NAMING,
            uniform_naming,
            StyleSignalDirection.AI,
            True,
        ),
    )


def build_rank_explanation(
    canonicality: float,
    token_count: int,
    flags: tuple[StyleFlag, ...],
    exact_match: bool,
) -> str:
    parts = [
        f"canonicality {canonicality:.4f} with {token_count} tokens.",
    ]
    fired = [flag for flag in flags if flag.fired]
    if not fired:
        parts.append("No style flags fired.")
    for flag in fired:
        parts.append(_flag_sentence(flag))
    if exact_match:
        parts.append("Exact reference match - review.")
    return " ".join(parts)


def _flag_sentence(flag: StyleFlag) -> str:
    experimental = " experimental" if flag.experimental else ""
    return f"{flag.name} fired ({flag.direction} lean{experimental})."


def uncomputed_style_flags() -> tuple[StyleFlag, ...]:
    return (
        _flag(StyleSignalName.EXPLANATORY_COMMENTS, False, StyleSignalDirection.AI, False),
        _flag(StyleSignalName.COMMENTED_OUT_CODE, False, StyleSignalDirection.HUMAN, False),
        _flag(StyleSignalName.UNUSED_LOCALS, False, StyleSignalDirection.HUMAN, False),
        _flag(
            StyleSignalName.UNIFORM_VERBOSE_NAMING,
            False,
            StyleSignalDirection.AI,
            True,
        ),
    )


def _flag(
    name: StyleSignalName,
    fired: bool,
    direction: StyleSignalDirection,
    experimental: bool,
) -> StyleFlag:
    return StyleFlag(
        name=name.value,
        fired=fired,
        direction=direction.value,
        experimental=experimental,
    )


def _author_comment_bodies(
    raw_code: str,
    boilerplate: str,
    language: Language,
) -> tuple[str, ...]:
    if not raw_code.strip():
        return ()
    blocked = _boilerplate_comment_bodies(boilerplate, language)
    spans = _fused_comment_spans(raw_code, language)
    bodies = [
        span.body
        for span in spans
        if span.body and not _is_boilerplate_comment(span.body, blocked)
    ]
    return tuple(bodies)


def _boilerplate_comment_bodies(boilerplate: str, language: Language) -> frozenset[str]:
    if not boilerplate.strip():
        return frozenset()
    spans = _fused_comment_spans(boilerplate, language)
    return frozenset(span.body.lower() for span in spans if span.body)


def _is_boilerplate_comment(body: str, blocked: frozenset[str]) -> bool:
    lowered = body.lower()
    if lowered in blocked:
        return True
    return any(marker in lowered for marker in BOILERPLATE_COMMENT_MARKERS)


def _fused_comment_spans(source: str, language: Language) -> list[CommentSpan]:
    tree = _parse_tree(source, language)
    raw_spans = _collect_comment_spans(tree.root_node, source)
    if not raw_spans:
        return []
    fused = [raw_spans[0]]
    for span in raw_spans[1:]:
        previous = fused[-1]
        if span.start_row <= previous.end_row + 1:
            fused[-1] = CommentSpan(
                previous.start_row,
                span.end_row,
                f"{previous.body}\n{span.body}".strip(),
            )
            continue
        fused.append(span)
    return fused


def _collect_comment_spans(node: Node, source: str) -> list[CommentSpan]:
    if node.type == COMMENT_NODE_TYPE:
        body = _comment_body(_node_text(node, source))
        return [CommentSpan(node.start_point[0], node.end_point[0], body)]
    spans: list[CommentSpan] = []
    for child in node.named_children:
        spans.extend(_collect_comment_spans(child, source))
    return spans


def _comment_body(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("//"):
        return stripped[2:].strip()
    if stripped.startswith("#"):
        return stripped[1:].strip()
    if stripped.startswith("/*"):
        ended = stripped.endswith("*/")
        inner = stripped[2:-2] if ended else stripped[2:]
        return inner.strip()
    return stripped


def _is_commented_out_code(body: str, language: Language) -> bool:
    if not body.strip():
        return False
    return _parses_as_code(body, language)


def _parses_as_code(body: str, language: Language) -> bool:
    wrapped = _wrap_snippet(body, language)
    tree = _parse_tree(wrapped, language)
    body_nodes = _probe_body_nodes(tree.root_node, language)
    return any(_contains_code_statement(node) for node in body_nodes)


def _wrap_snippet(body: str, language: Language) -> str:
    if language is Language.PYTHON:
        indented = "\n".join(f"    {line}" for line in body.splitlines() or [""])
        return f"def probe():\n{indented}\n"
    return f"void probe() {{\n{body}\n}}\n"


def _probe_body_nodes(root: Node, language: Language) -> tuple[Node, ...]:
    if language is Language.PYTHON:
        return _python_probe_body(root)
    return _cpp_probe_body(root)


def _python_probe_body(root: Node) -> tuple[Node, ...]:
    for child in root.named_children:
        if child.type != "function_definition":
            continue
        for part in child.named_children:
            if part.type == "block":
                return tuple(part.named_children)
    return ()


def _cpp_probe_body(root: Node) -> tuple[Node, ...]:
    for child in root.named_children:
        if child.type == "function_definition":
            return _compound_statement_children(child)
        if child.type == "declaration":
            inner = _initializer_list_children(child)
            if inner:
                return inner
    return ()


def _compound_statement_children(function_node: Node) -> tuple[Node, ...]:
    for child in function_node.named_children:
        if child.type == "compound_statement":
            return tuple(child.named_children)
    return ()


def _initializer_list_children(declaration: Node) -> tuple[Node, ...]:
    for child in declaration.named_children:
        if child.type != "init_declarator":
            continue
        for part in child.named_children:
            if part.type == "initializer_list":
                return tuple(part.named_children)
    return ()


def _contains_code_statement(node: Node) -> bool:
    if node.type == "ERROR":
        return False
    if _is_substantial_statement(node):
        return True
    return any(_contains_code_statement(child) for child in node.named_children)


def _is_substantial_statement(node: Node) -> bool:
    if node.type not in COMMENTED_CODE_STATEMENT_TYPES:
        return False
    named = node.named_children
    if node.type == "declaration":
        return _is_executable_declaration(node)
    if node.type == "expression_statement" and len(named) == 1:
        return named[0].type not in {"identifier", "string", "string_literal"}
    return True


def _is_executable_declaration(node: Node) -> bool:
    named_types = {child.type for child in node.named_children}
    if "init_declarator" in named_types:
        return True
    if "primitive_type" in named_types:
        return True
    if "template_type" in named_types or "qualified_identifier" in named_types:
        return True
    return not node.has_error and "type_identifier" in named_types


def _is_explanatory_prose(body: str) -> bool:
    words = [word.lower() for word in WORD_PATTERN.findall(body)]
    if len(words) >= EXPLANATORY_COMMENT_MIN_WORDS:
        return True
    has_sentence_mark = any(mark in body for mark in ".?!")
    has_cue = any(word in EXPLANATORY_COMMENT_CUES for word in words)
    short_enough = len(words) >= EXPLANATORY_COMMENT_SHORT_SENTENCE_WORDS
    return has_sentence_mark and has_cue and short_enough


def _has_unused_local(stripped_code: str, language: Language) -> bool:
    if not stripped_code.strip():
        return False
    tree = _parse_tree(stripped_code, language)
    return any(
        _function_has_unused_local(node, stripped_code, language)
        for node in _function_nodes(tree.root_node)
    )


def _function_nodes(node: Node) -> list[Node]:
    found: list[Node] = []
    if node.type == "function_definition":
        found.append(node)
    for child in node.named_children:
        found.extend(_function_nodes(child))
    return found


def _function_has_unused_local(
    function: Node,
    source: str,
    language: Language,
) -> bool:
    excluded = _excluded_binding_names(function, source, language)
    locals_found = _local_binding_names(function, source, language)
    candidates = locals_found - excluded - UNUSED_PLACEHOLDER_NAMES
    identifier_counts = _identifier_counts(function, source)
    return any(identifier_counts.get(name, 0) == 1 for name in candidates)


def _excluded_binding_names(
    function: Node,
    source: str,
    language: Language,
) -> set[str]:
    names: set[str] = set()
    _collect_names_under(function, source, names, _is_parameter_root)
    loop_types = CPP_LOOP_TYPES if language is Language.CPP else PYTHON_LOOP_TYPES
    _collect_loop_targets(function, source, names, loop_types)
    return names


def _local_binding_names(
    function: Node,
    source: str,
    language: Language,
) -> set[str]:
    names: set[str] = set()
    if language is Language.PYTHON:
        _collect_python_locals(function, source, names)
        return names
    _collect_cpp_locals(function, source, names)
    return names


def _collect_names_under(
    node: Node,
    source: str,
    names: set[str],
    is_root,
) -> None:
    if is_root(node):
        _collect_identifiers(node, source, names)
        return
    for child in node.named_children:
        _collect_names_under(child, source, names, is_root)


def _is_parameter_root(node: Node) -> bool:
    return node.type in PYTHON_PARAM_ROOT_TYPES or node.type in CPP_PARAM_ROOT_TYPES


def _collect_loop_targets(
    node: Node,
    source: str,
    names: set[str],
    loop_types: frozenset[str],
) -> None:
    if node.type in loop_types:
        _collect_loop_target_identifiers(node, source, names)
    for child in node.named_children:
        _collect_loop_targets(child, source, names, loop_types)


def _collect_loop_target_identifiers(node: Node, source: str, names: set[str]) -> None:
    if node.type in {"identifier", "field_identifier"}:
        names.add(_node_text(node, source))
        return
    if node.type in {
        "assignment",
        "declaration",
        "init_declarator",
        "pattern_list",
        "left_hand_side",
    }:
        for child in node.named_children:
            _collect_loop_target_identifiers(child, source, names)
        return
    for child in node.named_children:
        if child.type in {"identifier", "declaration", "pattern_list", "assignment"}:
            _collect_loop_target_identifiers(child, source, names)


def _collect_python_locals(node: Node, source: str, names: set[str]) -> None:
    if node.type == "function_definition":
        for child in node.named_children:
            if child.type == "block":
                _collect_python_block_locals(child, source, names)
        return
    _collect_python_block_locals(node, source, names)


def _collect_python_block_locals(node: Node, source: str, names: set[str]) -> None:
    if node.type == "function_definition":
        return
    if node.type in {"assignment", "augmented_assignment"}:
        _collect_assignment_targets(node, source, names)
        return
    for child in node.named_children:
        _collect_python_block_locals(child, source, names)


def _collect_assignment_targets(node: Node, source: str, names: set[str]) -> None:
    left = node.named_children[0] if node.named_children else None
    if left is None:
        return
    if left.type == "identifier":
        names.add(_node_text(left, source))
        return
    if left.type in {"pattern_list", "tuple_pattern", "list_pattern"}:
        _collect_identifiers(left, source, names)


def _collect_cpp_locals(node: Node, source: str, names: set[str]) -> None:
    if node.type == "function_definition":
        body = next((child for child in node.named_children if child.type == "compound_statement"), None)
        if body is not None:
            _collect_cpp_locals(body, source, names)
        return
    if node.type == "declaration":
        _collect_identifiers(node, source, names)
        return
    for child in node.named_children:
        if child.type != "function_definition":
            _collect_cpp_locals(child, source, names)


def _collect_identifiers(node: Node, source: str, names: set[str]) -> None:
    if node.type in {"identifier", "field_identifier"}:
        names.add(_node_text(node, source))
        return
    for child in node.named_children:
        _collect_identifiers(child, source, names)


def _identifier_counts(node: Node, source: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    _count_identifiers(node, source, counts)
    return counts


def _count_identifiers(node: Node, source: str, counts: dict[str, int]) -> None:
    if node.type in {"identifier", "field_identifier"}:
        name = _node_text(node, source)
        counts[name] = counts.get(name, 0) + 1
        return
    for child in node.named_children:
        _count_identifiers(child, source, counts)


def _has_uniform_verbose_naming(stripped_code: str, language: Language) -> bool:
    names = _user_identifiers(stripped_code, language)
    if len(names) < NAMING_MIN_UNIQUE_IDENTIFIERS:
        return False
    descriptive = sum(1 for name in names if _is_descriptive_name(name))
    short = sum(1 for name in names if len(name) <= NAMING_SHORT_NAME_MAX_LENGTH)
    fraction = descriptive / len(names)
    return fraction >= NAMING_DESCRIPTIVE_FRACTION_FLOOR and short == 0


def _user_identifiers(stripped_code: str, language: Language) -> set[str]:
    if not stripped_code.strip():
        return set()
    tree = _parse_tree(stripped_code, language)
    names: set[str] = set()
    _collect_user_identifiers(tree.root_node, stripped_code, names, language)
    return names


def _collect_user_identifiers(
    node: Node,
    source: str,
    names: set[str],
    language: Language,
) -> None:
    if node.type == "identifier":
        if _should_skip_identifier(node, source):
            return
        names.add(_node_text(node, source))
        return
    if language is Language.CPP and node.type == "type_identifier":
        return
    for child in node.named_children:
        _collect_user_identifiers(child, source, names, language)


def _should_skip_identifier(node: Node, source: str) -> bool:
    name = _node_text(node, source)
    if name in STYLE_SKIP_IDENTIFIERS or name in STYLE_SKIP_BUILTIN_IDENTIFIERS:
        return True
    parent = node.parent
    if parent is None:
        return False
    if parent.type in {"function_definition", "class_definition"}:
        return bool(parent.named_children) and parent.named_children[0] == node
    return parent.type == "attribute"


def _is_descriptive_name(name: str) -> bool:
    tokens = _name_tokens(name)
    long_tokens = [
        token
        for token in tokens
        if len(token) >= NAMING_DESCRIPTIVE_TOKEN_MIN_LENGTH
    ]
    return len(long_tokens) >= 2


def _name_tokens(name: str) -> list[str]:
    if "_" in name:
        return [token.lower() for token in SNAKE_SPLIT_PATTERN.findall(name)]
    camel = CAMEL_SPLIT_PATTERN.findall(name)
    if camel:
        return [token.lower() for token in camel]
    return [name.lower()] if name else []


def _parse_tree(source: str, language: Language):
    return _parser(language).parse(source.encode("utf-8"))


@lru_cache(maxsize=2)
def _parser(language: Language) -> Parser:
    capsule = (
        tree_sitter_cpp.language()
        if language is Language.CPP
        else tree_sitter_python.language()
    )
    return Parser(TreeSitterLanguage(capsule))


def _node_text(node: Node, source: str) -> str:
    return source.encode("utf-8")[node.start_byte : node.end_byte].decode("utf-8")

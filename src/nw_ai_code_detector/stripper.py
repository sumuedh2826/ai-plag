from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
import re

import tree_sitter_cpp
import tree_sitter_python
from tree_sitter import Language as TreeSitterLanguage
from tree_sitter import Node, Parser, Tree


class Language(str, Enum):
    CPP = "CPP"
    PYTHON = "PYTHON"


class SourceParseError(ValueError):
    pass


PYTHON_CLASS_NODE_TYPES = frozenset({"class_definition"})
CPP_CLASS_NODE_TYPES = frozenset({"class_specifier", "struct_specifier"})
# Access specifiers are class structure, not driver code: dropping them changes member visibility.
CPP_STRUCTURE_NODE_TYPES = frozenset({"access_specifier"})
CPP_DECLARATOR_NAME_TYPES = frozenset({"field_identifier", "identifier", "operator_name"})
# Brace-only lines are structural: they must never be treated as removable scaffold content.
STRUCTURAL_PUNCTUATION_LINES = frozenset({"{", "}", "};", "{}"})
CPP_TERMINATOR_TOKENS = frozenset({":", ";"})
FUNCTION_NODE_TYPE = "function_definition"
PYTHON_CPP_COMMENT_PATTERN = re.compile(r"(?m)^([ \t]*)//")


@dataclass(frozen=True)
class TargetPair:
    raw_function: Node
    boilerplate_function: Node


@dataclass(frozen=True)
class StripResult:
    stripped_code: str
    target_count: int
    author_definition_count: int


@dataclass(frozen=True)
class StripContext:
    source: str
    boilerplate: str
    language: Language
    root: Node
    targets: tuple[TargetPair, ...]
    scaffold_fragments: frozenset[str]


def strip_solution_body(
    raw_code: str,
    boilerplate: str,
    language: Language | str,
) -> str:
    return strip_solution(raw_code, boilerplate, language).stripped_code


def strip_solution(
    raw_code: str,
    boilerplate: str,
    language: Language | str,
) -> StripResult:
    """Keep author-written code as a parseable unit, removing boilerplate-matched scaffolding."""
    source_language = Language(language)
    context = _build_strip_context(raw_code, boilerplate, source_language)
    chunks: list[str] = []
    for child in context.root.named_children:
        chunks.extend(_extract_node_chunks(child, context))
    chunk_texts = [chunk.strip("\n") for chunk in chunks if chunk.strip()]
    stripped_code = _normalize_lightly("\n".join(chunk_texts))
    if not stripped_code:
        raise ValueError("No author-written code remains after stripping boilerplate")

    target_count = len(context.targets)
    definition_count = count_function_definitions(stripped_code, source_language)
    author_definition_count = max(definition_count - target_count, 0)
    return StripResult(
        stripped_code=stripped_code,
        target_count=target_count,
        author_definition_count=author_definition_count,
    )


def parses_source(source: str, language: Language | str) -> bool:
    if not source.strip():
        return False
    source_language = Language(language)
    parse_source = _sanitize_source_for_parsing(source, source_language)
    tree = _parser(source_language).parse(parse_source.encode("utf-8"))
    return not tree.root_node.has_error


def count_function_definitions(source: str, language: Language | str) -> int:
    source_language = Language(language)
    parse_source = _sanitize_source_for_parsing(source, source_language)
    tree = _parser(source_language).parse(parse_source.encode("utf-8"))
    return _count_nodes_of_type(tree.root_node, FUNCTION_NODE_TYPE)


def validate_source_syntax(source: str, language: Language | str) -> None:
    _parse_source(source, Language(language), "source")


def _count_nodes_of_type(node: Node, node_type: str) -> int:
    match_count = 1 if node.type == node_type else 0
    child_counts = sum(
        _count_nodes_of_type(child, node_type) for child in node.named_children
    )
    return match_count + child_counts


def _build_strip_context(
    raw_code: str,
    boilerplate: str,
    language: Language,
) -> StripContext:
    raw_tree = _parse_source(raw_code, language, "raw code")
    boilerplate_tree = _parse_source(boilerplate, language, "boilerplate")
    boilerplate_functions = _find_outer_functions(boilerplate_tree.root_node)
    if not boilerplate_functions:
        raise ValueError("Boilerplate does not define any function")

    targets = _pair_targets_with_boilerplate(
        raw_tree.root_node,
        raw_code,
        boilerplate,
        (language, boilerplate_functions),
    )
    if not targets:
        raise ValueError("Raw code does not implement any boilerplate function")

    scaffold_fragments = _collect_scaffold_fragments(
        boilerplate_tree.root_node,
        boilerplate,
        tuple(pair.boilerplate_function for pair in targets),
    )
    return StripContext(
        source=raw_code,
        boilerplate=boilerplate,
        language=language,
        root=raw_tree.root_node,
        targets=targets,
        scaffold_fragments=scaffold_fragments,
    )


def _pair_targets_with_boilerplate(
    raw_root: Node,
    raw_code: str,
    boilerplate: str,
    boilerplate_spec: tuple[Language, tuple[Node, ...]],
) -> tuple[TargetPair, ...]:
    language, boilerplate_functions = boilerplate_spec
    pairs: list[TargetPair] = []
    for boilerplate_function in boilerplate_functions:
        function_name = _function_name(boilerplate_function, boilerplate, language)
        raw_function = _find_function_by_name(
            raw_root,
            raw_code,
            language,
            function_name,
        )
        if raw_function is None:
            continue
        pairs.append(
            TargetPair(
                raw_function=raw_function,
                boilerplate_function=boilerplate_function,
            )
        )
    return tuple(pairs)


def _parse_source(source: str, language: Language, label: str) -> Tree:
    if not source.strip():
        raise ValueError(f"{label.capitalize()} is empty")
    parse_source = _sanitize_source_for_parsing(source, language)
    tree = _parser(language).parse(parse_source.encode("utf-8"))
    if tree.root_node.has_error:
        raise SourceParseError(f"Unable to parse {label} as {language.value}")
    return tree


def _sanitize_source_for_parsing(source: str, language: Language) -> str:
    if language is not Language.PYTHON:
        return source
    return PYTHON_CPP_COMMENT_PATTERN.sub(r"\1# ", source)


@lru_cache(maxsize=2)
def _parser(language: Language) -> Parser:
    language_capsule = (
        tree_sitter_cpp.language()
        if language is Language.CPP
        else tree_sitter_python.language()
    )
    tree_sitter_language = TreeSitterLanguage(language_capsule)
    return Parser(tree_sitter_language)


def _find_outer_functions(node: Node) -> tuple[Node, ...]:
    if node.type == FUNCTION_NODE_TYPE:
        return (node,)
    functions: list[Node] = []
    for child in node.named_children:
        functions.extend(_find_outer_functions(child))
    return tuple(functions)


def _find_function_by_name(
    node: Node,
    source: str,
    language: Language,
    target_name: str,
) -> Node | None:
    if node.type == FUNCTION_NODE_TYPE:
        function_name = _function_name(node, source, language)
        if function_name == target_name:
            return node
    for child in node.named_children:
        function = _find_function_by_name(child, source, language, target_name)
        if function is not None:
            return function
    return None


def _function_name(node: Node, source: str, language: Language) -> str:
    if language is Language.PYTHON:
        name_node = node.child_by_field_name("name")
    else:
        name_node = _find_cpp_declarator_name(node.child_by_field_name("declarator"))
    if name_node is None:
        raise ValueError("Could not identify function name")
    return _node_text(name_node, source)


def _find_cpp_declarator_name(node: Node | None) -> Node | None:
    if node is None:
        return None
    if node.type in CPP_DECLARATOR_NAME_TYPES:
        return node
    for child in node.named_children:
        name_node = _find_cpp_declarator_name(child)
        if name_node is not None:
            return name_node
    return None


def _collect_scaffold_fragments(
    root: Node,
    source: str,
    boilerplate_functions: tuple[Node, ...],
) -> frozenset[str]:
    fragments = {
        _comparable_text(_node_text(node, source))
        for node in _nodes_outside_functions(root, boilerplate_functions)
    }
    return frozenset(fragment for fragment in fragments if fragment)


def _nodes_outside_functions(
    node: Node,
    boilerplate_functions: tuple[Node, ...],
) -> list[Node]:
    if any(node == function for function in boilerplate_functions):
        return []
    contains_function = any(
        _contains_node(node, function) for function in boilerplate_functions
    )
    if not contains_function:
        return [node]
    nodes: list[Node] = []
    for child in node.named_children:
        nodes.extend(_nodes_outside_functions(child, boilerplate_functions))
    return nodes


def _extract_node_chunks(node: Node, context: StripContext) -> list[str]:
    target_pair = _target_pair_for_node(node, context)
    if target_pair is not None:
        return _render_standalone_target(target_pair, context)
    if not _contains_any_target(node, context):
        return _keep_author_node(node, context)
    if _is_class_node(node, context.language):
        # Function-completion shape is uniform: keep the boilerplate class wrapper
        # for every solution of the same (question, language). Do not unwrap.
        wrapper_text = _render_class_wrapper(node, context)
        return [wrapper_text] if wrapper_text else []

    chunks: list[str] = []
    for child in _container_child_nodes(node, context.language):
        chunks.extend(_extract_node_chunks(child, context))
    return chunks


def _container_child_nodes(node: Node, language: Language) -> tuple[Node, ...]:
    """Descend through a container's body only, so header tokens such as a class
    name never leak out as standalone author code."""
    if _is_class_node(node, language):
        class_body = _class_body(node, language)
        if class_body is not None:
            return tuple(class_body.named_children)
    body = node.child_by_field_name("body")
    if body is not None:
        return tuple(body.named_children)
    return tuple(node.named_children)


def _target_pair_for_node(node: Node, context: StripContext) -> TargetPair | None:
    for pair in context.targets:
        if pair.raw_function == node:
            return pair
    return None


def _contains_any_target(node: Node, context: StripContext) -> bool:
    return any(_contains_node(node, pair.raw_function) for pair in context.targets)


def _keep_author_node(node: Node, context: StripContext) -> list[str]:
    node_text = _node_text_with_trailing_token(node, context.source)
    if _comparable_text(node_text) in context.scaffold_fragments:
        return []
    author_text = _remove_container_indentation(node_text, node.start_point.column)
    return [author_text]


def _contains_node(container: Node, candidate: Node) -> bool:
    starts_before = container.start_byte <= candidate.start_byte
    ends_after = container.end_byte >= candidate.end_byte
    return starts_before and ends_after


def _is_class_node(node: Node, language: Language) -> bool:
    class_node_types = (
        PYTHON_CLASS_NODE_TYPES
        if language is Language.PYTHON
        else CPP_CLASS_NODE_TYPES
    )
    return node.type in class_node_types


def _class_body(class_node: Node, language: Language) -> Node | None:
    body = class_node.child_by_field_name("body")
    if body is not None:
        return body
    if language is Language.CPP:
        for child in class_node.named_children:
            if child.type == "field_declaration_list":
                return child
    return None


def _render_standalone_target(pair: TargetPair, context: StripContext) -> list[str]:
    unit_text = _target_unit_text(pair, context)
    if not unit_text:
        return []
    dedented_text = _remove_container_indentation(
        unit_text,
        pair.raw_function.start_point.column,
    )
    return [dedented_text]


def _render_class_wrapper(class_node: Node, context: StripContext) -> str:
    class_body = _class_body(class_node, context.language)
    if class_body is None:
        raise ValueError("Class node does not expose a body")

    header_end = (
        class_body.start_byte + 1
        if context.language is Language.CPP
        else class_body.start_byte
    )
    header_text = _source_slice(context.source, class_node.start_byte, header_end)
    class_indent = " " * class_node.start_point.column
    header_lines = _prefix_first_line(header_text.rstrip(), class_indent)

    member_lines = _class_member_lines_with_gaps(class_body, context)
    if not member_lines:
        return ""

    unit_lines = [*header_lines, *member_lines]
    if context.language is Language.CPP:
        unit_lines.append(f"{class_indent}}};")
    unit_text = "\n".join(unit_lines)
    return _remove_container_indentation(unit_text, class_node.start_point.column)


def _class_member_lines_with_gaps(
    class_body: Node,
    context: StripContext,
) -> list[str]:
    lines: list[str] = []
    for child in class_body.named_children:
        member_lines = _class_member_lines(child, context)
        if member_lines:
            lines.extend(member_lines)
    return lines


def _class_member_lines(member: Node, context: StripContext) -> list[str]:
    target_pair = _target_pair_for_node(member, context)
    if target_pair is not None:
        return _target_unit_text(target_pair, context).splitlines()
    if _contains_any_target(member, context):
        nested_chunks = _extract_node_chunks(member, context)
        return "\n".join(nested_chunks).splitlines()

    member_indent = " " * member.start_point.column
    member_text = _node_text_with_trailing_token(member, context.source)
    if member.type in CPP_STRUCTURE_NODE_TYPES:
        return _prefix_first_line(member_text, member_indent)
    if _comparable_text(member_text) in context.scaffold_fragments:
        return []
    return _prefix_first_line(member_text, member_indent)


def _node_text_with_trailing_token(node: Node, source: str) -> str:
    """Absorb a trailing ':' or ';' sibling so C++ access specifiers and class
    definitions stay syntactically valid once lifted out of their container."""
    end_byte = node.end_byte
    next_sibling = node.next_sibling
    if next_sibling is not None and next_sibling.type in CPP_TERMINATOR_TOKENS:
        end_byte = next_sibling.end_byte
    return _source_slice(source, node.start_byte, end_byte)


def _target_unit_text(pair: TargetPair, context: StripContext) -> str:
    body_lines = _author_body_lines(pair, context)
    if not body_lines:
        return ""
    target_indent = " " * pair.raw_function.start_point.column
    signature_lines = _prefix_first_line(
        _target_signature_text(pair, context),
        target_indent,
    )
    if context.language is Language.PYTHON:
        return "\n".join([*signature_lines, *body_lines])
    return "\n".join([*signature_lines, *body_lines, f"{target_indent}}}"])


def _target_signature_text(pair: TargetPair, context: StripContext) -> str:
    return _signature_text(pair.raw_function, context.source, context.language)


def _signature_text(function_node: Node, source: str, language: Language) -> str:
    signature_end = _signature_end_byte(function_node, language)
    signature_text = _source_slice(source, function_node.start_byte, signature_end)
    if language is Language.PYTHON:
        return signature_text.rstrip()
    return signature_text


def _signature_end_byte(function_node: Node, language: Language) -> int:
    """Anchor the signature on the ':'/'{' token, because an empty stub attaches its
    comment outside the body node while a filled-in body keeps it inside."""
    body = function_node.child_by_field_name("body")
    if body is None:
        raise ValueError("Function does not contain a body")
    if language is Language.CPP:
        # Keeping the original opening brace avoids synthesizing whitespace.
        return body.start_byte + 1
    colon_end_bytes = [
        child.end_byte
        for child in function_node.children
        if child.type == ":" and child.end_byte <= body.start_byte
    ]
    return max(colon_end_bytes) if colon_end_bytes else body.start_byte


def _author_body_lines(pair: TargetPair, context: StripContext) -> list[str]:
    body_lines = _body_lines_with_indent(
        pair.raw_function,
        context.source,
        context.language,
    )
    scaffold_body_lines = _scaffold_body_lines(pair, context)
    author_lines = [
        line for line in body_lines if line.strip() not in scaffold_body_lines
    ]
    collapsed_lines = _normalize_lightly("\n".join(author_lines)).splitlines()
    return _trim_blank_edges(collapsed_lines)


def _scaffold_body_lines(pair: TargetPair, context: StripContext) -> frozenset[str]:
    stub_lines = _body_lines_with_indent(
        pair.boilerplate_function,
        context.boilerplate,
        context.language,
    )
    return frozenset(
        line.strip()
        for line in stub_lines
        if line.strip() and line.strip() not in STRUCTURAL_PUNCTUATION_LINES
    )


def _body_lines_with_indent(
    function_node: Node,
    source: str,
    language: Language,
) -> list[str]:
    body = function_node.child_by_field_name("body")
    if body is None:
        raise ValueError("Function does not contain a body")
    body_text = _source_slice(source, body.start_byte, body.end_byte)
    if language is Language.CPP:
        return _cpp_body_inner_text(body_text).splitlines()

    signature_end = _signature_end_byte(function_node, language)
    gap_text = _source_slice(source, signature_end, body.start_byte)
    body_indent = " " * body.start_point.column
    # Lines between ':' and the body node are author comments, not signature text.
    gap_lines = gap_text.splitlines()[1:]
    return [*gap_lines, *_prefix_first_line(body_text, body_indent)]


def _cpp_body_inner_text(body_text: str) -> str:
    opening_brace = body_text.find("{")
    closing_brace = body_text.rfind("}")
    if opening_brace < 0 or closing_brace < opening_brace:
        raise ValueError("C++ function body does not contain braces")
    return body_text[opening_brace + 1 : closing_brace]


def _prefix_first_line(text: str, indent: str) -> list[str]:
    lines = text.splitlines()
    if not lines:
        return []
    return [f"{indent}{lines[0]}", *lines[1:]]


def _trim_blank_edges(lines: list[str]) -> list[str]:
    trimmed = list(lines)
    while trimmed and not trimmed[0].strip():
        trimmed.pop(0)
    while trimmed and not trimmed[-1].strip():
        trimmed.pop()
    return trimmed


def _remove_container_indentation(text: str, indentation_width: int) -> str:
    if indentation_width <= 0:
        return text
    prefix = " " * indentation_width
    dedented_lines = [
        line[len(prefix) :] if line.startswith(prefix) else line
        for line in text.splitlines()
    ]
    return "\n".join(dedented_lines)


def _source_slice(source: str, start_byte: int, end_byte: int) -> str:
    source_bytes = source.encode("utf-8")
    return source_bytes[start_byte:end_byte].decode("utf-8")


def _node_text(node: Node, source: str) -> str:
    return _source_slice(source, node.start_byte, node.end_byte)


def _comparable_text(source: str) -> str:
    lines = [line.rstrip() for line in source.strip().splitlines()]
    return "\n".join(lines)


def _normalize_lightly(source: str) -> str:
    """Trim trailing whitespace and collapse blank runs without touching in-line spacing."""
    lines = _trim_blank_edges([line.rstrip() for line in source.splitlines()])
    normalized_lines: list[str] = []
    previous_line_blank = False
    for line in lines:
        line_blank = not line
        if line_blank and previous_line_blank:
            continue
        normalized_lines.append(line)
        previous_line_blank = line_blank
    return "\n".join(normalized_lines)

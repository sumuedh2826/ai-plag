from __future__ import annotations

import re

from tree_sitter import Node

from nw_ai_code_detector.constants import (
    STYLE_SKIP_BUILTIN_IDENTIFIERS,
    STYLE_SKIP_IDENTIFIERS,
)
from nw_ai_code_detector.stripper import Language

IDENTIFIER_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
CPP_MACRO_NAMES = frozenset(
    {
        "INT_MAX",
        "INT_MIN",
        "UINT_MAX",
        "LONG_MAX",
        "LONG_MIN",
        "ULONG_MAX",
        "LLONG_MAX",
        "LLONG_MIN",
        "ULLONG_MAX",
        "DBL_MAX",
        "DBL_MIN",
        "FLT_MAX",
        "FLT_MIN",
        "NAN",
        "INFINITY",
        "NULL",
    }
)
CPP_NEST_TYPES = frozenset(
    {
        "translation_unit",
        "class_specifier",
        "struct_specifier",
        "declaration_list",
        "field_declaration_list",
    }
)
CPP_NEST_CHILDREN = frozenset(
    {
        "function_definition",
        "class_specifier",
        "struct_specifier",
        "declaration_list",
        "field_declaration_list",
    }
)
CPP_DECLARATOR_TYPES = frozenset(
    {
        "identifier",
        "init_declarator",
        "pointer_declarator",
        "array_declarator",
        "reference_declarator",
        "structured_binding_declarator",
        "parenthesized_declarator",
    }
)
PY_PATTERN_CONTAINERS = frozenset(
    {
        "pattern_list",
        "tuple_pattern",
        "list_pattern",
        "tuple",
        "list",
        "as_pattern",
        "as_pattern_target",
        "parenthesized_expression",
    }
)
PY_TYPED_PARAMS = frozenset(
    {"typed_parameter", "default_parameter", "typed_default_parameter"}
)


def local_binding_names(root: Node, source: bytes, language: Language) -> set[str]:
    names: set[str] = set()
    if language is Language.PYTHON:
        _walk_python(root, source, names)
        return names
    _walk_cpp(root, source, names)
    return names


def _skip_name(name: str) -> bool:
    if not name or IDENTIFIER_NAME_PATTERN.match(name) is None:
        return True
    if name in STYLE_SKIP_IDENTIFIERS or name in STYLE_SKIP_BUILTIN_IDENTIFIERS:
        return True
    return name in CPP_MACRO_NAMES or name.startswith("__")


def _text(node: Node, source: bytes) -> str:
    return source[node.start_byte : node.end_byte].decode("utf-8")


def _cpp_declarator_names(node: Node, source: bytes) -> set[str]:
    if node.type == "function_declarator":
        return set()
    if node.type == "identifier":
        name = _text(node, source)
        return set() if _skip_name(name) else {name}
    names: set[str] = set()
    for index in range(node.child_count):
        child = node.children[index]
        field = node.field_name_for_child(index)
        if field == "value":
            continue
        if field == "declarator" or child.type in CPP_DECLARATOR_TYPES:
            names |= _cpp_declarator_names(child, source)
    return names


def _cpp_declaration_names(node: Node, source: bytes) -> set[str]:
    names: set[str] = set()
    for index in range(node.child_count):
        child = node.children[index]
        if node.field_name_for_child(index) == "declarator":
            names |= _cpp_declarator_names(child, source)
    return names


def _cpp_param_names(param_list: Node | None, source: bytes) -> set[str]:
    names: set[str] = set()
    if param_list is None:
        return names
    for child in param_list.named_children:
        if child.type != "parameter_declaration":
            continue
        declarator = child.child_by_field_name("declarator")
        if declarator is not None:
            names |= _cpp_declarator_names(declarator, source)
    return names


def _walk_cpp(node: Node, source: bytes, out: set[str]) -> None:
    if node.type in CPP_NEST_TYPES:
        for child in node.named_children:
            if child.type in CPP_NEST_CHILDREN:
                _walk_cpp(child, source, out)
        return
    if node.type == "parameter_list":
        return
    if node.type == "declaration":
        _collect_cpp_declaration(node, source, out)
        return
    if node.type == "for_range_loop":
        _collect_cpp_for_range(node, source, out)
        return
    if node.type == "function_definition":
        _collect_cpp_function(node, source, out)
        return
    for child in node.named_children:
        _walk_cpp(child, source, out)


def _collect_cpp_declaration(node: Node, source: bytes, out: set[str]) -> None:
    out |= _cpp_declaration_names(node, source)
    for child in node.children:
        if child.type != "init_declarator":
            continue
        value = child.child_by_field_name("value")
        if value is not None:
            _walk_cpp(value, source, out)


def _collect_cpp_for_range(node: Node, source: bytes, out: set[str]) -> None:
    declarator = node.child_by_field_name("declarator")
    if declarator is not None:
        out |= _cpp_declarator_names(declarator, source)
    body = node.child_by_field_name("body")
    if body is not None:
        _walk_cpp(body, source, out)
    right = node.child_by_field_name("right")
    if right is not None:
        _walk_cpp(right, source, out)


def _collect_cpp_function(node: Node, source: bytes, out: set[str]) -> None:
    declarator = node.child_by_field_name("declarator")
    params: set[str] = set()
    if declarator is not None:
        params = _cpp_param_names(declarator.child_by_field_name("parameters"), source)
    inner: set[str] = set()
    body = node.child_by_field_name("body")
    if body is not None:
        _walk_cpp(body, source, inner)
    out |= inner - params


def _py_pattern_names(node: Node, source: bytes) -> set[str]:
    if node.type == "identifier":
        name = _text(node, source)
        return set() if _skip_name(name) else {name}
    if node.type in {"attribute", "subscript"}:
        return set()
    if node.type == "as_pattern":
        alias = node.child_by_field_name("alias")
        return _py_pattern_names(alias, source) if alias is not None else set()
    names: set[str] = set()
    if node.type in PY_PATTERN_CONTAINERS or node.type.endswith("pattern"):
        for child in node.named_children:
            names |= _py_pattern_names(child, source)
    return names


def _py_param_names(params_node: Node | None, source: bytes) -> set[str]:
    names: set[str] = set()
    if params_node is None:
        return names
    for child in params_node.named_children:
        names |= _py_param_child_names(child, source)
    return names


def _py_param_child_names(child: Node, source: bytes) -> set[str]:
    if child.type == "identifier":
        name = _text(child, source)
        return set() if _skip_name(name) else {name}
    if child.type in PY_TYPED_PARAMS:
        ident = child.child_by_field_name("name")
        if ident is None:
            ident = next(
                (item for item in child.named_children if item.type == "identifier"),
                None,
            )
        if ident is None:
            return set()
        name = _text(ident, source)
        return set() if _skip_name(name) else {name}
    if child.type.endswith("pattern"):
        return _py_pattern_names(child, source)
    return set()


def _walk_python(node: Node, source: bytes, out: set[str]) -> None:
    if node.type == "class_definition":
        body = node.child_by_field_name("body")
        if body is not None:
            _walk_python(body, source, out)
        return
    if node.type == "module":
        for child in node.named_children:
            if child.type in {"function_definition", "class_definition"}:
                _walk_python(child, source, out)
        return
    if node.type in {"parameters", "lambda_parameters"}:
        return
    if node.type == "function_definition":
        _collect_python_function(node, source, out)
        return
    if node.type in {"assignment", "augmented_assignment"}:
        _collect_python_assignment(node, source, out)
        return
    if node.type == "for_statement":
        _collect_python_for(node, source, out)
        return
    if node.type == "for_in_clause":
        _collect_python_for_in(node, source, out)
        return
    if node.type in {"with_item", "except_clause"}:
        _collect_python_alias_clause(node, source, out)
        return
    if node.type == "named_expression":
        _collect_python_named_expression(node, source, out)
        return
    for child in node.named_children:
        _walk_python(child, source, out)


def _collect_python_function(node: Node, source: bytes, out: set[str]) -> None:
    params = _py_param_names(node.child_by_field_name("parameters"), source)
    inner: set[str] = set()
    body = node.child_by_field_name("body")
    if body is not None:
        _walk_python(body, source, inner)
    out |= inner - params


def _collect_python_assignment(node: Node, source: bytes, out: set[str]) -> None:
    left = node.child_by_field_name("left")
    if left is not None:
        out |= _py_pattern_names(left, source)
    right = node.child_by_field_name("right")
    if right is not None:
        _walk_python(right, source, out)


def _collect_python_for(node: Node, source: bytes, out: set[str]) -> None:
    left = node.child_by_field_name("left")
    if left is not None:
        out |= _py_pattern_names(left, source)
    right = node.child_by_field_name("right")
    if right is not None:
        _walk_python(right, source, out)
    body = node.child_by_field_name("body")
    if body is not None:
        _walk_python(body, source, out)


def _collect_python_for_in(node: Node, source: bytes, out: set[str]) -> None:
    left = node.child_by_field_name("left")
    if left is not None:
        out |= _py_pattern_names(left, source)
    right = node.child_by_field_name("right")
    if right is not None:
        _walk_python(right, source, out)


def _collect_python_alias_clause(node: Node, source: bytes, out: set[str]) -> None:
    for child in node.named_children:
        if child.type == "as_pattern":
            out |= _py_pattern_names(child, source)
        else:
            _walk_python(child, source, out)


def _collect_python_named_expression(node: Node, source: bytes, out: set[str]) -> None:
    name_node = node.child_by_field_name("name")
    if name_node is not None:
        out |= _py_pattern_names(name_node, source)
    value = node.child_by_field_name("value")
    if value is not None:
        _walk_python(value, source, out)

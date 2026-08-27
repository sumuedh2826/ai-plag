from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from nw_ai_code_detector.stripper import (
    Language,
    SourceParseError,
    StripResult,
    parses_source,
    strip_solution,
    strip_solution_body,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
QUESTIONS_PATH = DATA_DIR / "questions.json"
SUBMISSIONS_PATH = DATA_DIR / "scored_submissions.json"
PYTHON_BOILERPLATE_KEYS = ("PYTHON", "PYTHON39")
FUNCTION_NODE_TYPE = "function_definition"
SAMPLES_PER_LANGUAGE = 3
# Validation-display filters only. select_500 and generation must not reuse these.
MIN_RAW_LINE_COUNT = 10
MAX_RAW_LINE_COUNT = 60
PLACEHOLDER_MARKER = "write your code here"
PYTHON_AI_IMPORT = "import math as ai_sample_math"
CPP_AI_INCLUDE = "#include <deque>"


@dataclass(frozen=True)
class StripperSample:
    question_id: str
    language: Language
    raw_code: str
    boilerplate: str
    stripped_code: str
    target_count: int
    author_definition_count: int


def main() -> None:
    questions = _load_json_object(QUESTIONS_PATH)
    submissions = _load_json_object(SUBMISSIONS_PATH)
    groups = _read_mapping(submissions.get("groups"), "scored_submissions.groups")
    samples, skip_counts = _select_samples(groups, questions)

    print(
        "Offline stripper validation. Removal is boilerplate-match based and "
        "order-independent; AI samples below are synthetic fixtures and no API was called."
    )
    for sample_number, sample in enumerate(samples, start=1):
        _print_sample(sample_number, sample)

    print(f"\n{'=' * 88}")
    print(f"All {len(samples) * 2} stripped outputs parse under Tree-sitter.")
    print("No stripped line differs from its raw line except leading indentation.")
    print("No blank lines were inserted between adjacent raw content lines.")
    print(f"Skipped candidates by reason: {dict(sorted(skip_counts.items()))}")
    print(
        "Skip counts and the 10-60 line band apply only to this validation picker; "
        "they do not filter select_500 or generation."
    )


def _load_json_object(path: Path) -> Mapping[str, Any]:
    with path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    return _read_mapping(payload, path.name)


def _read_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a JSON object")
    return value


def _select_samples(
    groups: Mapping[str, Any],
    questions: Mapping[str, Any],
) -> tuple[list[StripperSample], Counter]:
    skip_counts: Counter = Counter()
    pools: dict[tuple[Language, bool], list[StripperSample]] = {
        (Language.PYTHON, True): [],
        (Language.PYTHON, False): [],
        (Language.CPP, True): [],
        (Language.CPP, False): [],
    }
    for group_key in sorted(groups):
        sample = _build_sample(group_key, groups[group_key], questions, skip_counts)
        if sample is None:
            continue
        has_author_helpers = sample.author_definition_count > 0
        pools[(sample.language, has_author_helpers)].append(sample)
    selected = _choose_samples_from_pools(pools)
    return selected, skip_counts


def _build_sample(
    group_key: str,
    records: Any,
    questions: Mapping[str, Any],
    skip_counts: Counter,
) -> StripperSample | None:
    if not isinstance(records, list) or not records:
        skip_counts["empty_group"] += 1
        return None
    question_id, language_token = group_key.rsplit(":", 1)
    language = Language(language_token)
    raw_code = records[0].get("raw_code") if isinstance(records[0], dict) else None
    if not isinstance(raw_code, str) or not raw_code.strip():
        skip_counts["missing_raw_code"] += 1
        return None
    raw_line_count = len(raw_code.splitlines())
    if not MIN_RAW_LINE_COUNT <= raw_line_count <= MAX_RAW_LINE_COUNT:
        skip_counts["outside_readability_band"] += 1
        return None

    question = questions.get(question_id)
    boilerplate = _find_boilerplate(question, language)
    if boilerplate is None:
        skip_counts["missing_boilerplate"] += 1
        return None

    strip_result = _strip_or_record_skip(
        raw_code,
        boilerplate,
        language,
        skip_counts,
    )
    if strip_result is None:
        return None
    return StripperSample(
        question_id=question_id,
        language=language,
        raw_code=raw_code,
        boilerplate=boilerplate,
        stripped_code=strip_result.stripped_code,
        target_count=strip_result.target_count,
        author_definition_count=strip_result.author_definition_count,
    )


def _strip_or_record_skip(
    raw_code: str,
    boilerplate: str,
    language: Language,
    skip_counts: Counter,
) -> StripResult | None:
    try:
        return strip_solution(raw_code, boilerplate, language)
    except SourceParseError:
        skip_counts["unparseable_source"] += 1
        return None
    except ValueError:
        skip_counts["no_author_code_or_target"] += 1
        return None


def _choose_samples_from_pools(
    pools: Mapping[tuple[Language, bool], list[StripperSample]],
) -> list[StripperSample]:
    used_question_ids: set[str] = set()
    selected: list[StripperSample] = []
    for language in (Language.PYTHON, Language.CPP):
        helper_pool = _sort_pool(pools[(language, True)])
        plain_pool = _sort_pool(pools[(language, False)])
        chosen = _take_distinct_samples(helper_pool, 1, used_question_ids)
        if not chosen:
            raise ValueError(f"No {language.value} sample with author helper definitions")
        remaining_count = SAMPLES_PER_LANGUAGE - len(chosen)
        chosen.extend(
            _take_distinct_samples(
                [*helper_pool, *plain_pool],
                remaining_count,
                used_question_ids,
            )
        )
        if len(chosen) != SAMPLES_PER_LANGUAGE:
            raise ValueError(f"Not enough distinct {language.value} samples")
        selected.extend(chosen)
    return selected


def _sort_pool(pool: Sequence[StripperSample]) -> list[StripperSample]:
    return sorted(
        pool,
        key=lambda sample: (
            -sample.author_definition_count,
            -len(sample.raw_code),
            sample.question_id,
        ),
    )


def _take_distinct_samples(
    pool: Sequence[StripperSample],
    count: int,
    used_question_ids: set[str],
) -> list[StripperSample]:
    taken: list[StripperSample] = []
    for sample in pool:
        if len(taken) == count:
            break
        if sample.question_id in used_question_ids:
            continue
        used_question_ids.add(sample.question_id)
        taken.append(sample)
    return taken


def _find_boilerplate(question: Any, language: Language) -> str | None:
    if not isinstance(question, dict):
        return None
    boilerplates = question.get("boilerplates")
    if not isinstance(boilerplates, dict):
        return None
    candidate_keys = (
        (Language.CPP.value,)
        if language is Language.CPP
        else PYTHON_BOILERPLATE_KEYS
    )
    for key in candidate_keys:
        boilerplate = _read_boilerplate_code(boilerplates.get(key))
        if boilerplate:
            return boilerplate
    return None


def _read_boilerplate_code(boilerplate_value: Any) -> str | None:
    if isinstance(boilerplate_value, str) and boilerplate_value.strip():
        return boilerplate_value
    if isinstance(boilerplate_value, dict):
        code_content = boilerplate_value.get("code_content")
        if isinstance(code_content, str) and code_content.strip():
            return code_content
    return None


def _print_sample(sample_number: int, sample: StripperSample) -> None:
    ai_raw_code = _build_synthetic_ai_output(sample.raw_code, sample.language)
    ai_stripped_code = strip_solution_body(
        ai_raw_code,
        sample.boilerplate,
        sample.language,
    )
    _check_stripped_output(sample, sample.stripped_code, sample.raw_code)
    _check_stripped_output(sample, ai_stripped_code, ai_raw_code)
    _check_ai_marker_kept(sample, ai_stripped_code)

    heading = (
        f"SAMPLE {sample_number}: {sample.question_id}:{sample.language.value} "
        f"boilerplate_targets={sample.target_count} "
        f"author_helper_defs={sample.author_definition_count}"
    )
    print(f"\n{'=' * 88}\n{heading}\n{'=' * 88}")
    print("\n--- HUMAN RAW ---\n")
    print(sample.raw_code)
    print("\n--- HUMAN STRIPPED (re-derived from raw_code) ---\n")
    print(sample.stripped_code)
    print(f"\nPARSE: human stripped parses as {sample.language.value} = OK")
    print("\n--- SYNTHETIC AI RAW ---\n")
    print(ai_raw_code)
    print("\n--- SYNTHETIC AI STRIPPED ---\n")
    print(ai_stripped_code)
    print(f"\nPARSE: AI stripped parses as {sample.language.value} = OK")


def _build_synthetic_ai_output(raw_code: str, language: Language) -> str:
    if language is Language.PYTHON:
        return f"{PYTHON_AI_IMPORT}\n\n{raw_code}"
    return f"{CPP_AI_INCLUDE}\n{raw_code}"


def _check_stripped_output(
    sample: StripperSample,
    stripped_code: str,
    raw_code: str,
) -> None:
    if not stripped_code.strip():
        raise AssertionError(f"{sample.question_id}: stripped output is empty")
    if not parses_source(stripped_code, sample.language):
        raise AssertionError(
            f"{sample.question_id}: stripped output does not parse as {sample.language.value}"
        )
    if PLACEHOLDER_MARKER in stripped_code.lower():
        raise AssertionError(f"{sample.question_id}: boilerplate placeholder survived")
    _check_lines_preserved(sample, stripped_code, raw_code)
    _check_no_inserted_blanks(sample, stripped_code, raw_code)


def _check_lines_preserved(
    sample: StripperSample,
    stripped_code: str,
    raw_code: str,
) -> None:
    raw_lines = {" ".join(line.split()) for line in raw_code.splitlines()}
    for line in stripped_code.splitlines():
        collapsed_line = " ".join(line.split())
        if collapsed_line and collapsed_line not in raw_lines:
            raise AssertionError(
                f"{sample.question_id}: stripped line not present in raw code: {line!r}"
            )


def _check_no_inserted_blanks(
    sample: StripperSample,
    stripped_code: str,
    raw_code: str,
) -> None:
    raw_keys = [_line_key(line) for line in raw_code.splitlines()]
    stripped_keys = [_line_key(line) for line in stripped_code.splitlines()]
    content_pairs = _consecutive_content_spans(stripped_keys)
    search_from = 0
    for left_key, right_key, stripped_blank_count in content_pairs:
        left_index = _find_key(raw_keys, left_key, search_from)
        right_index = _find_key(raw_keys, right_key, left_index + 1)
        if left_index is None or right_index is None:
            raise AssertionError(
                f"{sample.question_id}: cannot align stripped content to raw code"
            )
        raw_blank_count = sum(
            1 for key in raw_keys[left_index + 1 : right_index] if key == ""
        )
        if stripped_blank_count > raw_blank_count:
            raise AssertionError(
                f"{sample.question_id}: stripper inserted a blank line between {left_key!r} and {right_key!r}"
            )
        search_from = right_index


def _line_key(line: str) -> str:
    return " ".join(line.split())


def _consecutive_content_spans(
    keys: list[str],
) -> list[tuple[str, str, int]]:
    content_indexes = [index for index, key in enumerate(keys) if key]
    spans: list[tuple[str, str, int]] = []
    for left_index, right_index in zip(content_indexes, content_indexes[1:]):
        blank_count = sum(
            1 for key in keys[left_index + 1 : right_index] if key == ""
        )
        spans.append((keys[left_index], keys[right_index], blank_count))
    return spans


def _find_key(keys: list[str], target: str, start: int) -> int | None:
    for index in range(start, len(keys)):
        if keys[index] == target:
            return index
    return None


def _check_ai_marker_kept(sample: StripperSample, ai_stripped_code: str) -> None:
    expected_marker = (
        PYTHON_AI_IMPORT if sample.language is Language.PYTHON else CPP_AI_INCLUDE
    )
    if expected_marker not in ai_stripped_code:
        raise AssertionError(
            f"{sample.question_id}: author-added import/include was removed"
        )


if __name__ == "__main__":
    main()

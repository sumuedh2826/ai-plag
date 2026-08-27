from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from nw_ai_code_detector.config import DATA_DIR
from nw_ai_code_detector.constants import (
    BOILERPLATES_KEY,
    CODE_CONTENT_KEY,
    CONTENT_KEY,
    CPP_BOILERPLATE_KEYS,
    DIFFICULTY_KEY,
    FUNCTION_BASED_KEY,
    FUNCTION_CONFIG_KEY,
    GROUPS_KEY,
    PYTHON_BOILERPLATE_KEYS,
    STATEMENT_KEY,
    TAGS_KEY,
    UNTAGGED_PRIMARY_TAG,
    TOPIC_TAG_PREFIX,
)
from nw_ai_code_detector.stripper import Language

QUESTIONS_FILENAME = "questions.json"
SCORED_SUBMISSIONS_FILENAME = "scored_submissions.json"
COVERAGE_FILENAME = "coverage.json"
MANIFEST_FILENAME = "manifest.json"
REQUIRED_FILENAMES = (
    QUESTIONS_FILENAME,
    SCORED_SUBMISSIONS_FILENAME,
    COVERAGE_FILENAME,
    MANIFEST_FILENAME,
)
WAITING_FOR_DATA_MESSAGE = (
    "Waiting for data/: drop questions.json, scored_submissions.json, "
    "coverage.json, and manifest.json into ./data/ then re-run."
)


@dataclass(frozen=True)
class QuestionRecord:
    question_id: str
    difficulty: str
    tags: tuple[str, ...]
    primary_tag: str
    is_function_completion: bool
    statement_content: str
    boilerplates: Mapping[str, str]


@dataclass(frozen=True)
class Dataset:
    questions: Mapping[str, QuestionRecord]
    human_submission_counts: Mapping[str, int]
    coverage: Mapping[str, Any]
    manifest: Mapping[str, Any]


def load_dataset(data_dir: Path | None = None) -> Dataset:
    root = data_dir or DATA_DIR
    missing_paths = missing_data_files(root)
    if missing_paths:
        raise FileNotFoundError(WAITING_FOR_DATA_MESSAGE)

    questions_payload = _load_json(root / QUESTIONS_FILENAME)
    submissions_payload = _load_json(root / SCORED_SUBMISSIONS_FILENAME)
    coverage_payload = _load_json(root / COVERAGE_FILENAME)
    manifest_payload = _load_json(root / MANIFEST_FILENAME)
    questions = _parse_questions(_as_mapping(questions_payload, QUESTIONS_FILENAME))
    counts = _count_human_submissions(submissions_payload)
    return Dataset(
        questions=questions,
        human_submission_counts=counts,
        coverage=_as_mapping(coverage_payload, COVERAGE_FILENAME),
        manifest=_as_mapping(manifest_payload, MANIFEST_FILENAME),
    )


def missing_data_files(data_dir: Path | None = None) -> list[Path]:
    root = data_dir or DATA_DIR
    if not root.is_dir():
        return [root / filename for filename in REQUIRED_FILENAMES]
    return [
        root / filename
        for filename in REQUIRED_FILENAMES
        if not (root / filename).is_file()
    ]


def describe_dataset(dataset: Dataset) -> None:
    print(f"questions.json: {len(dataset.questions)} records")
    nonempty_groups = sum(
        1 for count in dataset.human_submission_counts.values() if count
    )
    print(f"scored_submissions.json: {nonempty_groups} question-ids with humans")
    print(f"coverage.json: object with {len(dataset.coverage)} keys")
    print(f"manifest.json: object with {len(dataset.manifest)} keys")
    function_completion_count = sum(
        1 for question in dataset.questions.values() if question.is_function_completion
    )
    print(f"function-completion questions: {function_completion_count}")


def main() -> int:
    missing_paths = missing_data_files()
    if missing_paths:
        print(WAITING_FOR_DATA_MESSAGE)
        for path in missing_paths:
            print(f"  missing: {path}")
        return 1
    dataset = load_dataset()
    describe_dataset(dataset)
    print("Field validation passed.")
    return 0


def _parse_questions(payload: Mapping[str, Any]) -> dict[str, QuestionRecord]:
    records: dict[str, QuestionRecord] = {}
    for question_id, raw_question in payload.items():
        if not isinstance(raw_question, dict):
            continue
        records[question_id] = _parse_question(question_id, raw_question)
    return records


def _parse_question(question_id: str, raw_question: Mapping[str, Any]) -> QuestionRecord:
    statement = _mapping_or_empty(raw_question.get(STATEMENT_KEY))
    tags = _parse_tag_names(statement.get(TAGS_KEY) or raw_question.get(TAGS_KEY))
    difficulty = str(
        statement.get(DIFFICULTY_KEY) or raw_question.get(DIFFICULTY_KEY) or ""
    ).upper()
    return QuestionRecord(
        question_id=str(raw_question.get("question_id") or question_id),
        difficulty=difficulty,
        tags=tags,
        primary_tag=_primary_tag(tags),
        is_function_completion=_is_function_completion(raw_question),
        statement_content=str(statement.get(CONTENT_KEY) or ""),
        boilerplates=_parse_boilerplates(raw_question.get(BOILERPLATES_KEY)),
    )


def _parse_tag_names(raw_tags: Any) -> tuple[str, ...]:
    if not isinstance(raw_tags, list):
        return ()
    names: list[str] = []
    for tag in raw_tags:
        if isinstance(tag, str) and tag:
            names.append(tag)
        elif isinstance(tag, dict):
            name = tag.get("name")
            if isinstance(name, str) and name:
                names.append(name)
    return tuple(names)


def _primary_tag(tags: Sequence[str]) -> str:
    topic_tags = [tag for tag in tags if tag.startswith(TOPIC_TAG_PREFIX)]
    if topic_tags:
        return topic_tags[0]
    return UNTAGGED_PRIMARY_TAG


def _is_function_completion(raw_question: Mapping[str, Any]) -> bool:
    config = _mapping_or_empty(raw_question.get(FUNCTION_CONFIG_KEY))
    cpp_config = _mapping_or_empty(config.get(Language.CPP.value))
    python_config = _mapping_or_empty(
        config.get("PYTHON39") or config.get(Language.PYTHON.value)
    )
    return bool(cpp_config.get(FUNCTION_BASED_KEY)) and bool(
        python_config.get(FUNCTION_BASED_KEY)
    )


def _parse_boilerplates(raw_boilerplates: Any) -> dict[str, str]:
    mapping = _mapping_or_empty(raw_boilerplates)
    parsed: dict[str, str] = {}
    cpp_code = _boilerplate_code(mapping, CPP_BOILERPLATE_KEYS)
    python_code = _boilerplate_code(mapping, PYTHON_BOILERPLATE_KEYS)
    if cpp_code:
        parsed[Language.CPP.value] = cpp_code
    if python_code:
        parsed[Language.PYTHON.value] = python_code
    return parsed


def _boilerplate_code(mapping: Mapping[str, Any], keys: Sequence[str]) -> str:
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, str) and value.strip():
            return value
        if isinstance(value, dict):
            code = value.get(CODE_CONTENT_KEY)
            if isinstance(code, str) and code.strip():
                return code
    return ""


def _count_human_submissions(payload: Any) -> dict[str, int]:
    groups = payload.get(GROUPS_KEY) if isinstance(payload, dict) else None
    if not isinstance(groups, dict):
        return {}
    counts: dict[str, int] = {}
    for group_key, records in groups.items():
        if not isinstance(records, list) or not records:
            continue
        question_id = str(group_key).rsplit(":", 1)[0]
        counts[question_id] = counts.get(question_id, 0) + len(records)
    return counts


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _as_mapping(payload: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(payload, dict):
        raise TypeError(f"{label} must be a JSON object")
    return payload


def _mapping_or_empty(value: Any) -> Mapping[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from collections.abc import Mapping, Sequence

from nw_ai_code_detector.config import (
    CANDIDATE_HUMAN_SIMILARITY_SAMPLES_PATH,
    CANDIDATE_HUMAN_TOKEN_RAW_SAMPLES_PATH,
    DATA_DIR,
    EVAL_AI_SOLUTIONS_DIR,
    HELDOUT_AI_RAW_SAMPLES_PATH,
    HELDOUT_AI_SIMILARITY_SAMPLES_PATH,
)
from nw_ai_code_detector.constants import (
    CANDIDATE_HUMAN_SOURCE,
    EXECUTION_CONTRACT_FULL_PROGRAM,
    EXECUTION_CONTRACT_FUNCTION_ONLY,
    EXECUTION_CONTRACT_UNKNOWN,
    FUNCTION_BASED_KEY,
    FUNCTION_CONFIG_KEY,
    GROUPS_KEY,
    HELDOUT_AI_SOURCE,
    RAW_CODE_FIELD,
    RAW_CODE_UNAVAILABLE,
    RAW_OUTPUT_FIELD,
    STRIP_CANNOT_VERIFY,
    STRIP_EXACT_MATCH,
    STRIP_MISMATCH,
    STRIPPED_CODE_FIELD,
)
from nw_ai_code_detector.data_load import load_dataset
from nw_ai_code_detector.stripper import Language, SourceParseError, strip_solution_body

SEPARATOR = "=" * 60
CANDIDATE_EXPORT_IDS = (
    "CPP_LOWEST_01",
    "CPP_LOWEST_02",
    "CPP_LOWEST_03",
    "CPP_HIGHEST_01",
    "CPP_HIGHEST_02",
    "CPP_HIGHEST_03",
    "PYTHON_LOWEST_01",
    "PYTHON_LOWEST_02",
    "PYTHON_LOWEST_03",
    "PYTHON_HIGHEST_01",
    "PYTHON_HIGHEST_02",
    "PYTHON_HIGHEST_03",
)
HELDOUT_EXPORT_IDS = (
    "CPP_LOWEST_01",
    "CPP_LOWEST_02",
    "PYTHON_LOWEST_01",
    "PYTHON_LOWEST_02",
)
MAIN_PATTERN = re.compile(
    r"\b(int|void)\s+main\s*\(|if\s+__name__\s*==|def\s+main\s*\(",
    re.MULTILINE,
)
INPUT_PATTERN = re.compile(
    r"\bcin\b|\bscanf\s*\(|\bgetline\s*\(|\bstd::cin\b|\binput\s*\(|sys\.stdin",
)
OUTPUT_PATTERN = re.compile(
    r"\bcout\b|\bprintf\s*\(|\bputs\s*\(|\bstd::cout\b|\bprint\s*\(|sys\.stdout",
)
INCLUDE_PATTERN = re.compile(r"^\s*#\s*include\b|^\s*import\b|^\s*from\s+\S+\s+import\b", re.MULTILINE)
IDENTITY_PATTERNS = (
    re.compile(r"\buser_id\b", re.IGNORECASE),
    re.compile(r"\bemail\b", re.IGNORECASE),
    re.compile(r"@"),
)
CANDIDATE_PREAMBLE = """CANDIDATE-HUMAN TOKEN-SIMILARITY RAW-CODE REVIEW

These records are unverified candidate-human submissions.
Low similarity does not prove human authorship.
High similarity does not prove AI authorship.
Exact mixed-v1 matches are possible contamination or copying and must not be used as verified human negatives.
"""
HELDOUT_PREAMBLE = """HELD-OUT AI RAW-CODE REVIEW SAMPLES

These records are known held-out AI solutions.
Selection population: integrity-scorable validation held-out AI.
Token count is recorded and is not an exclude.
RAW_CODE is copied from held-out source JSON (raw_output, else raw_code).
STRIPPED_CODE_USED_FOR_EMBEDDING is the exact stored embedding input.
The production stripper was not modified.
"""


@dataclass(frozen=True)
class ParsedReviewSample:
    sample_id: str
    fields: Mapping[str, str]
    stripped_code: str


@dataclass(frozen=True)
class RawExportSample:
    sample_id: str
    category: str
    language: str
    record_id: str
    question_id: str
    ai_nn_max: str
    ai_distance: str
    exact_match: str
    significant_code_token_count: str
    minimum_significant_code_tokens: str
    execution_contract: str
    strip_reproduction_status: str
    raw_code: str
    stripped_code: str
    main_present_in_raw: bool
    main_present_in_stripped: bool
    input_logic_present_in_raw: bool
    input_logic_present_in_stripped: bool
    output_logic_present_in_raw: bool
    output_logic_present_in_stripped: bool
    raw_code_hash: str
    stripped_code_hash: str
    stored_stripped_hash: str
    removal_audit: str
    required_issues: tuple[str, ...]


def main() -> int:
    dataset = load_dataset()
    questions = _load_questions_payload()
    groups = _load_submission_groups()
    candidate_parsed = parse_review_file(CANDIDATE_HUMAN_SIMILARITY_SAMPLES_PATH)
    candidate_exports = [
        build_candidate_export(sample, dataset, questions, groups)
        for sample in select_samples(candidate_parsed, CANDIDATE_EXPORT_IDS)
    ]
    write_raw_review_file(
        CANDIDATE_HUMAN_TOKEN_RAW_SAMPLES_PATH,
        CANDIDATE_PREAMBLE,
        candidate_exports,
    )
    print(f"Wrote {CANDIDATE_HUMAN_TOKEN_RAW_SAMPLES_PATH}")
    return 0


def parse_review_file(path: Path) -> list[ParsedReviewSample]:
    text = path.read_text(encoding="utf-8")
    parts = text.split(SEPARATOR)
    samples: list[ParsedReviewSample] = []
    index = 1
    while index + 1 < len(parts):
        header = parts[index]
        body = parts[index + 1]
        fields = _parse_fields(header)
        sample_id = fields.get("SAMPLE", "")
        if sample_id:
            samples.append(
                ParsedReviewSample(sample_id, fields, body.strip("\n"))
            )
        index += 2
    return samples


def select_samples(
    samples: Sequence[ParsedReviewSample],
    sample_ids: Sequence[str],
) -> list[ParsedReviewSample]:
    by_id = {sample.sample_id: sample for sample in samples}
    selected = []
    for sample_id in sample_ids:
        sample = by_id.get(sample_id)
        if sample is None:
            raise RuntimeError(f"Missing selected sample {sample_id}")
        selected.append(sample)
    return selected


def retrieve_candidate_raw_code(
    record_id: str,
    groups: Mapping[str, object],
) -> str:
    question_id, language, index = _candidate_locator(record_id)
    key = f"{question_id}:{language}"
    records = groups.get(key)
    if not isinstance(records, list) or index >= len(records):
        return RAW_CODE_UNAVAILABLE
    source_record = records[index]
    if not isinstance(source_record, dict):
        return RAW_CODE_UNAVAILABLE
    raw_code = source_record.get(RAW_CODE_FIELD)
    if not isinstance(raw_code, str) or not raw_code:
        return RAW_CODE_UNAVAILABLE
    return raw_code


def retrieve_heldout_raw_code(record_id: str) -> str:
    relative_path = _heldout_relative_path(record_id)
    path = EVAL_AI_SOLUTIONS_DIR / relative_path
    if not path.is_file():
        return RAW_CODE_UNAVAILABLE
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_output = payload.get(RAW_OUTPUT_FIELD)
    if isinstance(raw_output, str) and raw_output:
        return raw_output
    raw_code = payload.get(RAW_CODE_FIELD)
    if isinstance(raw_code, str) and raw_code:
        return raw_code
    return RAW_CODE_UNAVAILABLE


def resolve_execution_contract(
    question_id: str,
    language: str,
    questions: Mapping[str, object],
) -> str:
    payload = questions.get(question_id)
    if not isinstance(payload, dict):
        return EXECUTION_CONTRACT_UNKNOWN
    config = payload.get(FUNCTION_CONFIG_KEY)
    if not isinstance(config, dict):
        return EXECUTION_CONTRACT_UNKNOWN
    language_config = config.get(language)
    if language == "PYTHON" and not isinstance(language_config, dict):
        language_config = config.get("PYTHON39")
    if not isinstance(language_config, dict):
        return EXECUTION_CONTRACT_UNKNOWN
    if FUNCTION_BASED_KEY not in language_config:
        return EXECUTION_CONTRACT_UNKNOWN
    if language_config.get(FUNCTION_BASED_KEY) is True:
        return EXECUTION_CONTRACT_FUNCTION_ONLY
    if language_config.get(FUNCTION_BASED_KEY) is False:
        return EXECUTION_CONTRACT_FULL_PROGRAM
    return EXECUTION_CONTRACT_UNKNOWN


def reproduce_strip_status(
    raw_code: str,
    stored_hash: str,
    language: str,
    boilerplate: str,
) -> str:
    if raw_code == RAW_CODE_UNAVAILABLE:
        return STRIP_CANNOT_VERIFY
    try:
        reproduced = strip_solution_body(raw_code, boilerplate, language)
    except (SourceParseError, ValueError):
        return STRIP_CANNOT_VERIFY
    reproduced_hash = _text_hash(reproduced)
    if reproduced_hash == stored_hash:
        return STRIP_EXACT_MATCH
    return STRIP_MISMATCH


def detect_logic(code: str) -> tuple[bool, bool, bool]:
    main_present = bool(MAIN_PATTERN.search(code))
    input_present = bool(INPUT_PATTERN.search(code))
    output_present = bool(OUTPUT_PATTERN.search(code))
    return main_present, input_present, output_present


def build_candidate_export(
    sample: ParsedReviewSample,
    dataset,
    questions: Mapping[str, object],
    groups: Mapping[str, object],
) -> RawExportSample:
    record_id = sample.fields["RECORD_ID"]
    raw_code = retrieve_candidate_raw_code(record_id, groups)
    return _complete_export(sample, dataset, questions, raw_code)


def build_heldout_export(
    sample: ParsedReviewSample,
    dataset,
    questions: Mapping[str, object],
) -> RawExportSample:
    record_id = sample.fields["RECORD_ID"]
    raw_code = retrieve_heldout_raw_code(record_id)
    return _complete_export(sample, dataset, questions, raw_code)


def write_raw_review_file(
    path: Path,
    preamble: str,
    samples: Sequence[RawExportSample],
) -> None:
    blocks = [preamble.rstrip(), ""]
    for sample in samples:
        blocks.append(_format_export(sample))
    text = "\n".join(blocks) + "\n"
    _assert_no_identity(text)
    path.write_text(text, encoding="utf-8")


def _complete_export(
    sample: ParsedReviewSample,
    dataset,
    questions: Mapping[str, object],
    raw_code: str,
) -> RawExportSample:
    record_id = sample.fields["RECORD_ID"]
    question_id = sample.fields["QUESTION_ID"]
    language = sample.fields["LANGUAGE"]
    stored_hash = record_id.rsplit("|", 1)[-1]
    stripped = sample.stripped_code
    contract = resolve_execution_contract(question_id, language, questions)
    boilerplate = ""
    question = dataset.questions.get(question_id)
    if question is not None:
        boilerplate = question.boilerplates.get(language, "")
    status = reproduce_strip_status(raw_code, stored_hash, language, boilerplate)
    raw_main, raw_input, raw_output = detect_logic(
        "" if raw_code == RAW_CODE_UNAVAILABLE else raw_code
    )
    stripped_main, stripped_input, stripped_output = detect_logic(stripped)
    issues = audit_required_removals(
        contract,
        raw_code,
        stripped,
        boilerplate,
        record_id,
        question_id,
        language,
        raw_main,
        stripped_main,
        raw_input,
        stripped_input,
        raw_output,
        stripped_output,
    )
    return RawExportSample(
        sample_id=sample.sample_id,
        category=sample.fields.get("CATEGORY", ""),
        language=language,
        record_id=record_id,
        question_id=question_id,
        ai_nn_max=sample.fields.get("AI_NN_MAX_RAW", sample.fields.get("AI_NN_MAX", "")),
        ai_distance=sample.fields.get("AI_DISTANCE", ""),
        exact_match=sample.fields.get(
            "EXACT_MATCH_TO_MIXED_V1",
            sample.fields.get("EXACT_MATCH_TO_REFERENCE", ""),
        ),
        significant_code_token_count=sample.fields.get(
            "SIGNIFICANT_CODE_TOKEN_COUNT",
            "",
        ),
        minimum_significant_code_tokens=sample.fields.get(
            "MINIMUM_SIGNIFICANT_CODE_TOKENS",
            "",
        ),
        execution_contract=contract,
        strip_reproduction_status=status,
        raw_code=raw_code,
        stripped_code=stripped,
        main_present_in_raw=raw_main,
        main_present_in_stripped=stripped_main,
        input_logic_present_in_raw=raw_input,
        input_logic_present_in_stripped=stripped_input,
        output_logic_present_in_raw=raw_output,
        output_logic_present_in_stripped=stripped_output,
        raw_code_hash=_text_hash(raw_code) if raw_code != RAW_CODE_UNAVAILABLE else "",
        stripped_code_hash=_text_hash(stripped),
        stored_stripped_hash=stored_hash,
        removal_audit=_removal_summary(
            contract,
            raw_main,
            stripped_main,
            raw_input,
            stripped_input,
            raw_output,
            stripped_output,
            issues,
        ),
        required_issues=issues,
    )


def audit_required_removals(
    contract: str,
    raw_code: str,
    stripped: str,
    boilerplate: str,
    record_id: str,
    question_id: str,
    language: str,
    raw_main: bool,
    stripped_main: bool,
    raw_input: bool,
    stripped_input: bool,
    raw_output: bool,
    stripped_output: bool,
) -> tuple[str, ...]:
    if raw_code == RAW_CODE_UNAVAILABLE:
        return ()
    issues: list[str] = []
    driver_in_template = _driver_in_template(boilerplate)
    if contract == EXECUTION_CONTRACT_FULL_PROGRAM and raw_main and not stripped_main:
        issues.append(
            _issue_line(
                record_id,
                question_id,
                language,
                contract,
                "main_or___main__",
                _excerpt(raw_code, MAIN_PATTERN),
                "Full-program questions require main/input/output as solution code.",
            )
        )
    if contract == EXECUTION_CONTRACT_FULL_PROGRAM and raw_input and not stripped_input:
        issues.append(
            _issue_line(
                record_id,
                question_id,
                language,
                contract,
                "input_parsing",
                _excerpt(raw_code, INPUT_PATTERN),
                "Input reading is part of a full-program solution.",
            )
        )
    if contract == EXECUTION_CONTRACT_FULL_PROGRAM and raw_output and not stripped_output:
        issues.append(
            _issue_line(
                record_id,
                question_id,
                language,
                contract,
                "required_output",
                _excerpt(raw_code, OUTPUT_PATTERN),
                "Required print/cout belongs to the full-program solution.",
            )
        )
    if (
        contract == EXECUTION_CONTRACT_FUNCTION_ONLY
        and raw_output
        and not stripped_output
        and not _pattern_in_boilerplate(boilerplate, OUTPUT_PATTERN)
    ):
        issues.append(
            _issue_line(
                record_id,
                question_id,
                language,
                contract,
                "user_authored_output",
                _excerpt(raw_code, OUTPUT_PATTERN),
                "User-authored print/cout inside the required function must remain.",
            )
        )
    if (
        contract == EXECUTION_CONTRACT_FUNCTION_ONLY
        and raw_main
        and not stripped_main
        and not driver_in_template
    ):
        issues.append(
            _issue_line(
                record_id,
                question_id,
                language,
                contract,
                "main_not_matching_known_template",
                _excerpt(raw_code, MAIN_PATTERN),
                "main was removed but it is not an exact known platform driver.",
            )
        )
    if contract == EXECUTION_CONTRACT_UNKNOWN and raw_main and not stripped_main:
        issues.append(
            _issue_line(
                record_id,
                question_id,
                language,
                contract,
                "main_removed_under_unknown_contract",
                _excerpt(raw_code, MAIN_PATTERN),
                "Unknown contracts must not treat main as boilerplate.",
            )
        )
    return tuple(issues)


def _removal_summary(
    contract: str,
    raw_main: bool,
    stripped_main: bool,
    raw_input: bool,
    stripped_input: bool,
    raw_output: bool,
    stripped_output: bool,
    issues: Sequence[str],
) -> str:
    if not issues:
        return (
            f"No required-logic removal flagged for {contract}. "
            f"main raw/stripped={raw_main}/{stripped_main}; "
            f"input={raw_input}/{stripped_input}; "
            f"output={raw_output}/{stripped_output}."
        )
    return "POSSIBLE REQUIRED LOGIC REMOVED:\n" + "\n".join(issues)


def _issue_line(
    record_id: str,
    question_id: str,
    language: str,
    contract: str,
    removed_construct: str,
    excerpt: str,
    reason: str,
) -> str:
    return (
        f"record_id={record_id}; question_id={question_id}; "
        f"language={language}; execution_contract={contract}; "
        f"removed_construct={removed_construct}; "
        f"raw excerpt={excerpt!r}; reason it may be required={reason}"
    )


def _format_export(sample: RawExportSample) -> str:
    return "\n".join(
        [
            SEPARATOR,
            f"SAMPLE: {sample.sample_id}",
            f"CATEGORY: {sample.category}",
            f"LANGUAGE: {sample.language}",
            f"SIGNIFICANT_CODE_TOKEN_COUNT: "
            f"{sample.significant_code_token_count}",
            f"MINIMUM_SIGNIFICANT_CODE_TOKENS: "
            f"{sample.minimum_significant_code_tokens}",
            f"AI_NN_MAX: {sample.ai_nn_max}",
            f"AI_DISTANCE: {sample.ai_distance}",
            f"EXACT_MATCH_TO_MIXED_V1: {sample.exact_match}",
            f"EXECUTION_CONTRACT: {sample.execution_contract}",
            f"STRIP_REPRODUCTION_STATUS: {sample.strip_reproduction_status}",
            SEPARATOR,
            "",
            "RAW_CODE:",
            sample.raw_code,
            "",
        ]
    )


def _parse_fields(header: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in header.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        fields[key.strip()] = value.strip()
    return fields


def _candidate_locator(record_id: str) -> tuple[str, str, int]:
    parts = record_id.split("|")
    if len(parts) < 5 or parts[0] != CANDIDATE_HUMAN_SOURCE:
        raise RuntimeError(f"Invalid candidate record_id: {record_id}")
    return parts[1], parts[2], int(parts[3])


def _heldout_relative_path(record_id: str) -> str:
    parts = record_id.split("|")
    if len(parts) < 3 or parts[0] != HELDOUT_AI_SOURCE:
        raise RuntimeError(f"Invalid held-out record_id: {record_id}")
    return parts[1]


def _load_questions_payload() -> dict[str, object]:
    payload = json.loads((DATA_DIR / "questions.json").read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("questions.json is invalid")
    return payload


def _load_submission_groups() -> dict[str, object]:
    payload = json.loads(
        (DATA_DIR / "scored_submissions.json").read_text(encoding="utf-8")
    )
    groups = payload.get(GROUPS_KEY) if isinstance(payload, dict) else None
    if not isinstance(groups, dict):
        raise RuntimeError("scored_submissions.json groups are unavailable")
    return groups


def _driver_in_template(boilerplate: str) -> bool:
    main_present, input_present, output_present = detect_logic(boilerplate)
    return main_present or input_present or output_present


def _pattern_in_boilerplate(boilerplate: str, pattern: re.Pattern[str]) -> bool:
    return bool(pattern.search(boilerplate))


def _excerpt(code: str, pattern: re.Pattern[str]) -> str:
    match = pattern.search(code)
    if match is None:
        return ""
    start = max(match.start() - 40, 0)
    end = min(match.end() + 40, len(code))
    return re.sub(r"\s+", " ", code[start:end]).strip()


def _assert_no_identity(text: str) -> None:
    lowered = text.lower()
    if "user_id" in lowered or "email" in lowered:
        raise RuntimeError("Generated raw review file contains identity fields")


def _text_hash(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())

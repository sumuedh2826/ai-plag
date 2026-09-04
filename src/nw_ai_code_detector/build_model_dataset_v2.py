from __future__ import annotations

import csv
import json
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from collections.abc import Mapping, Sequence

from nw_ai_code_detector.config import (
    AI_SOLUTIONS_DIR,
    DATA_DIR,
    EMBEDDING_CACHE_DIR,
    EVAL_AI_SOLUTIONS_DIR,
    MODEL_DATASET_DIR,
    OUTPUTS_DIR,
)
from nw_ai_code_detector.constants import (
    CANONICALITY_INTERNAL_TEST_QUESTION_COUNT,
    CANONICALITY_SPLIT_SEED,
    CANONICALITY_TRAIN_QUESTION_COUNT,
    CANONICALITY_VALIDATION_QUESTION_COUNT,
    DatasetSplit,
    GENERATION_LANGUAGES,
    GROUPS_KEY,
    MODEL_DATASET_VERSION,
)
from nw_ai_code_detector.data_load import Dataset, load_dataset
from nw_ai_code_detector.embedder import cached_vector_for_text, embedding_cache_key
from nw_ai_code_detector.stripper import (
    Language,
    SourceParseError,
    strip_solution_body,
    validate_source_syntax,
)

CANONICAL_QUESTION_SPLIT_PATH = (
    OUTPUTS_DIR / "canonicality_dataset_split_v1" / "question_split.json"
)
GPT_HEAVY_CANDIDATES_DIR = DATA_DIR / "ai_solutions_gpt_pending"
EXPECTED_QUESTION_COUNT = 500
EXPECTED_MIXED_REFERENCE_COUNT = 6000
EXPECTED_MIXED_CLUSTER_COUNT = 1000
EXPECTED_MIXED_REFERENCES_PER_CLUSTER = 6
EXPECTED_GPT_HEAVY_COUNT = 2000


@dataclass(frozen=True)
class QuestionAssignment:
    question_id: str
    split: str
    difficulty: str
    split_seed: int


@dataclass(frozen=True)
class LabeledRecord:
    record_id: str
    question_id: str
    language: str
    split: str
    label: int
    source: str
    generator: str | None
    persona: str | None
    stripped_hash: str
    embedding_cache_key: str
    duplicate_group: str
    sample_weight: float


@dataclass(frozen=True)
class LoadedRecord:
    row: LabeledRecord
    text: str
    source_id: str
    user_group_hash: str | None


@dataclass(frozen=True)
class CandidateAudit:
    record_id: str
    question_id: str
    language: str
    split: str
    source_id: str
    generator: str | None
    persona: str | None
    stripped_hash: str
    embedding_cache_key: str
    disposition: str
    duplicate_group: str


@dataclass(frozen=True)
class RecordContext:
    question_id: str
    language: str
    split: str
    label: int


@dataclass(frozen=True)
class RecordSource:
    source_id: str
    source: str
    generator: str | None
    persona: str | None


@dataclass(frozen=True)
class RecordContent:
    text: str
    stripped_hash: str
    sample_weight: float


@dataclass(frozen=True)
class CandidateRules:
    mixed_source_ids: set[str]
    mixed_hashes: set[tuple[str, str, str]]
    protected_hashes: set[str]
    seen_hashes: set[tuple[str, str, str]]


@dataclass(frozen=True)
class BuildPopulation:
    assignments: tuple[QuestionAssignment, ...]
    humans: tuple[LoadedRecord, ...]
    held_out: tuple[LoadedRecord, ...]
    mixed_references: tuple[LoadedRecord, ...]
    gpt_audit: tuple[CandidateAudit, ...]
    gpt_train: tuple[LoadedRecord, ...]
    invalid_held_out: int


def main() -> int:
    started = time.perf_counter()
    before = snapshot_protected_artifacts()
    assignments = load_question_assignments()
    dataset = load_dataset()
    humans = load_human_records(dataset, assignments)
    held_out, invalid_held_out = load_ai_records(
        EVAL_AI_SOLUTIONS_DIR,
        assignments,
        label=1,
        source="heldout_ai",
    )
    mixed, invalid_mixed = load_ai_records(
        AI_SOLUTIONS_DIR,
        assignments,
        label=1,
        source="mixed_v1_reference",
    )
    if invalid_mixed:
        raise RuntimeError(f"Mixed-v1 contains {invalid_mixed} invalid records")
    validate_mixed_population(mixed)
    audit, gpt_train = audit_gpt_candidates(assignments, mixed, held_out, humans)
    population = BuildPopulation(
        assignments=tuple(assignments),
        humans=tuple(humans),
        held_out=tuple(held_out),
        mixed_references=tuple(mixed),
        gpt_audit=tuple(audit),
        gpt_train=tuple(gpt_train),
        invalid_held_out=invalid_held_out,
    )
    write_outputs(population, time.perf_counter() - started)
    after = snapshot_protected_artifacts()
    if before != after:
        raise RuntimeError("Protected split or mixed-v1 artifacts changed")
    print(f"Wrote model dataset to {MODEL_DATASET_DIR}")
    return 0


def load_question_assignments(
    path: Path = CANONICAL_QUESTION_SPLIT_PATH,
) -> list[QuestionAssignment]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assignments = [QuestionAssignment(**item) for item in payload]
    validate_question_assignments(assignments)
    return assignments


def validate_question_assignments(
    assignments: Sequence[QuestionAssignment],
) -> None:
    if len(assignments) != EXPECTED_QUESTION_COUNT:
        raise RuntimeError(f"Expected 500 questions, found {len(assignments)}")
    if len({item.question_id for item in assignments}) != len(assignments):
        raise RuntimeError("Question split contains duplicate question IDs")
    counts = Counter(item.split for item in assignments)
    expected = {
        DatasetSplit.TRAIN.value: CANONICALITY_TRAIN_QUESTION_COUNT,
        DatasetSplit.VALIDATION.value: CANONICALITY_VALIDATION_QUESTION_COUNT,
        DatasetSplit.INTERNAL_TEST.value: CANONICALITY_INTERNAL_TEST_QUESTION_COUNT,
    }
    if counts != expected:
        raise RuntimeError(f"Question split counts changed: {dict(counts)}")
    if {item.split_seed for item in assignments} != {CANONICALITY_SPLIT_SEED}:
        raise RuntimeError("Question split seed changed")


def load_human_records(
    dataset: Dataset,
    assignments: Sequence[QuestionAssignment],
) -> list[LoadedRecord]:
    assignments_by_question = {
        item.question_id: item for item in assignments
    }
    payload = json.loads(
        (DATA_DIR / "scored_submissions.json").read_text(encoding="utf-8")
    )
    groups = payload.get(GROUPS_KEY) if isinstance(payload, dict) else None
    if not isinstance(groups, dict):
        raise RuntimeError("scored_submissions.json groups are unavailable")
    records: list[LoadedRecord] = []
    for question_id in sorted(assignments_by_question):
        for language in GENERATION_LANGUAGES:
            records.extend(
                _load_humans_for_pair(
                    dataset,
                    groups,
                    assignments_by_question[question_id],
                    language,
                )
            )
    return _apply_duplicate_weights(records)


def load_ai_records(
    root: Path,
    assignments: Sequence[QuestionAssignment],
    label: int,
    source: str,
) -> tuple[list[LoadedRecord], int]:
    split_by_question = {item.question_id: item.split for item in assignments}
    records: list[LoadedRecord] = []
    invalid = 0
    for path in sorted(root.rglob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        question_id = str(payload.get("qid") or payload.get("question_id") or "")
        if question_id not in split_by_question:
            continue
        text = payload.get("stripped_code")
        if (
            not isinstance(text, str)
            or not text.strip()
            or payload.get("parse_ok") is not True
        ):
            invalid += 1
            continue
        language = str(payload.get("language") or "")
        relative = path.relative_to(root).as_posix()
        context = RecordContext(
            question_id,
            language,
            split_by_question[question_id],
            label,
        )
        record_source = RecordSource(
            relative,
            source,
            _optional_string(payload.get("model")),
            _optional_string(payload.get("persona")),
        )
        records.append(
            _loaded_ai_record(
                text,
                context,
                record_source,
            )
        )
    return _apply_duplicate_weights(records), invalid


def validate_mixed_population(records: Sequence[LoadedRecord]) -> None:
    if len(records) != EXPECTED_MIXED_REFERENCE_COUNT:
        raise RuntimeError(f"Expected 6000 mixed-v1 references, found {len(records)}")
    counts = Counter((item.row.question_id, item.row.language) for item in records)
    if len(counts) != EXPECTED_MIXED_CLUSTER_COUNT:
        raise RuntimeError(f"Expected 1000 mixed-v1 clusters, found {len(counts)}")
    invalid = [
        pair
        for pair, count in counts.items()
        if count != EXPECTED_MIXED_REFERENCES_PER_CLUSTER
    ]
    if invalid:
        raise RuntimeError(f"Mixed-v1 cluster size changed for {invalid[0]}")


def audit_gpt_candidates(
    assignments: Sequence[QuestionAssignment],
    mixed: Sequence[LoadedRecord],
    held_out: Sequence[LoadedRecord],
    humans: Sequence[LoadedRecord],
) -> tuple[list[CandidateAudit], list[LoadedRecord]]:
    split_by_question = {item.question_id: item.split for item in assignments}
    mixed_source_ids = {item.source_id for item in mixed}
    mixed_hashes = {
        (item.row.question_id, item.row.language, item.row.stripped_hash)
        for item in mixed
    }
    protected_hashes = {
        item.row.stripped_hash
        for item in (*held_out, *humans)
        if item.row.split != DatasetSplit.TRAIN.value
    }
    audit: list[CandidateAudit] = []
    selected: list[LoadedRecord] = []
    seen_hashes: set[tuple[str, str, str]] = set()
    rules = CandidateRules(
        mixed_source_ids,
        mixed_hashes,
        protected_hashes,
        seen_hashes,
    )
    paths = sorted(GPT_HEAVY_CANDIDATES_DIR.rglob("*.json"))
    if len(paths) != EXPECTED_GPT_HEAVY_COUNT:
        raise RuntimeError(f"Expected 2000 GPT-heavy candidates, found {len(paths)}")
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        item = _candidate_from_payload(path, payload, split_by_question)
        disposition = _candidate_disposition(item, rules)
        audit.append(_audit_row(item, disposition))
        if disposition == "eligible_train_positive":
            seen_hashes.add(
                (item.row.question_id, item.row.language, item.row.stripped_hash)
            )
            selected.append(item)
    return audit, selected


def write_outputs(population: BuildPopulation, runtime_seconds: float) -> None:
    MODEL_DATASET_DIR.mkdir(parents=True, exist_ok=True)
    _write_json(
        MODEL_DATASET_DIR / "question_split.json",
        [asdict(item) for item in population.assignments],
    )
    _write_jsonl(
        MODEL_DATASET_DIR / "labeled_humans.jsonl",
        [asdict(item.row) for item in population.humans],
    )
    _write_jsonl(
        MODEL_DATASET_DIR / "labeled_heldout_ai.jsonl",
        [asdict(item.row) for item in population.held_out],
    )
    _write_jsonl(
        MODEL_DATASET_DIR / "gpt_heavy_extra_audit.jsonl",
        [asdict(item) for item in population.gpt_audit],
    )
    _write_jsonl(
        MODEL_DATASET_DIR / "gpt_heavy_train_positives.jsonl",
        [asdict(item.row) for item in population.gpt_train],
    )
    manifests = _split_manifests(population)
    for split, rows in manifests.items():
        _write_jsonl(MODEL_DATASET_DIR / f"{split}_manifest.jsonl", rows)
    coverage = coverage_payload(population, manifests, runtime_seconds)
    _write_json(MODEL_DATASET_DIR / "coverage.json", coverage)
    _write_summary_csv(coverage)
    (MODEL_DATASET_DIR / "report.md").write_text(
        _report_markdown(coverage),
        encoding="utf-8",
    )


def coverage_payload(
    population: BuildPopulation,
    manifests: Mapping[str, Sequence[Mapping[str, object]]],
    runtime_seconds: float,
) -> dict[str, object]:
    dispositions = Counter(item.disposition for item in population.gpt_audit)
    disposition_names = (
        "eligible_train_positive",
        "excluded_non_train_question",
        "excluded_mixed_v1_reference_id_match",
        "excluded_mixed_v1_hash_match",
        "excluded_validation_or_test_hash_match",
        "excluded_invalid",
        "duplicate_within_gpt_heavy",
    )
    disposition_counts = {
        name: dispositions.get(name, 0) for name in disposition_names
    }
    return {
        "dataset_version": MODEL_DATASET_VERSION,
        "question_split": dict(
            Counter(item.split for item in population.assignments)
        ),
        "human_records": _record_coverage(population.humans),
        "held_out_ai": {
            **_record_coverage(population.held_out),
            "by_generator": dict(
                Counter(item.row.generator for item in population.held_out)
            ),
            "by_persona": dict(
                Counter(item.row.persona for item in population.held_out)
            ),
            "invalid_excluded": population.invalid_held_out,
        },
        "mixed_v1": {
            "references": len(population.mixed_references),
            "clusters": len(
                {
                    (item.row.question_id, item.row.language)
                    for item in population.mixed_references
                }
            ),
            "used_as_labeled_rows": 0,
        },
        "gpt_heavy": {
            "audited": len(population.gpt_audit),
            "eligible_train_positives": len(population.gpt_train),
            "disposition": disposition_counts,
            "overlap_diagnostics": _gpt_overlap_diagnostics(population),
        },
        "manifest_counts": {
            split: len(rows) for split, rows in manifests.items()
        },
        "leakage": leakage_payload(population, manifests),
        "user_overlap_across_splits": _user_overlap(population.humans),
        "runtime_seconds": runtime_seconds,
    }


def leakage_payload(
    population: BuildPopulation,
    manifests: Mapping[str, Sequence[Mapping[str, object]]],
) -> dict[str, int]:
    mixed_references = sum(
        row["source"] == "mixed_v1_reference"
        for rows in manifests.values()
        for row in rows
    )
    gpt_non_train = sum(
        1
        for split, rows in manifests.items()
        if split != DatasetSplit.TRAIN.value
        for row in rows
        if row["source"] == "gpt_heavy_extra"
    )
    wrong_split = sum(
        1
        for split, rows in manifests.items()
        for row in rows
        if row["split"] != split
    )
    return {
        "mixed_reference_in_labeled_manifest": mixed_references,
        "gpt_heavy_in_validation_or_test": gpt_non_train,
        "row_split_mismatch": wrong_split,
    }


def snapshot_protected_artifacts() -> dict[str, str]:
    split_hash = _file_sha256(CANONICAL_QUESTION_SPLIT_PATH)
    mixed_hash = _directory_sha256(AI_SOLUTIONS_DIR)
    index_hash = _directory_sha256(OUTPUTS_DIR / "reference_index")
    return {
        "question_split": split_hash,
        "mixed_solutions": mixed_hash,
        "mixed_index": index_hash,
        "embedding_cache_count": str(len(tuple(EMBEDDING_CACHE_DIR.glob("*.json")))),
    }


def _load_humans_for_pair(
    dataset: Dataset,
    groups: Mapping[str, object],
    assignment: QuestionAssignment,
    language: Language,
) -> list[LoadedRecord]:
    question_id = assignment.question_id
    source_records = groups.get(f"{question_id}:{language.value}")
    if not isinstance(source_records, list):
        return []
    boilerplate = dataset.questions[question_id].boilerplates.get(language.value, "")
    records: list[LoadedRecord] = []
    for index, source_record in enumerate(source_records):
        stripped = _strip_human(source_record, boilerplate, language)
        if stripped is None:
            continue
        stripped_hash = _text_hash(stripped)
        source_id = f"{question_id}/{language.value}/human_{index}"
        user_id = _user_id(source_record)
        context = RecordContext(question_id, language.value, assignment.split, 0)
        source = RecordSource(source_id, "human", None, None)
        content = RecordContent(stripped, stripped_hash, 1.0)
        records.append(
            LoadedRecord(
                row=_manifest_row(context, source, content),
                text=stripped,
                source_id=source_id,
                user_group_hash=_opaque_user_id(user_id),
            )
        )
    return records


def _loaded_ai_record(
    text: str,
    context: RecordContext,
    source: RecordSource,
) -> LoadedRecord:
    stripped_hash = _text_hash(text)
    content = RecordContent(text, stripped_hash, 1.0)
    row = _manifest_row(context, source, content)
    return LoadedRecord(
        row=row,
        text=text,
        source_id=source.source_id,
        user_group_hash=None,
    )


def _candidate_from_payload(
    path: Path,
    payload: Mapping[str, object],
    split_by_question: Mapping[str, str],
) -> LoadedRecord:
    question_id = str(payload.get("qid") or payload.get("question_id") or "")
    language = str(payload.get("language") or "")
    split = split_by_question.get(question_id, "invalid")
    text_value = payload.get("stripped_code")
    text = text_value if isinstance(text_value, str) else ""
    source_id = path.relative_to(GPT_HEAVY_CANDIDATES_DIR).as_posix()
    stripped_hash = _text_hash(text) if text else ""
    context = RecordContext(question_id, language, split, 1)
    source = RecordSource(
        source_id,
        "gpt_heavy_extra",
        _optional_string(payload.get("model")),
        _optional_string(payload.get("persona")),
    )
    content = RecordContent(text, stripped_hash, 1.0)
    row = _manifest_row(context, source, content)
    return LoadedRecord(row=row, text=text, source_id=source_id, user_group_hash=None)


def _candidate_disposition(
    item: LoadedRecord,
    rules: CandidateRules,
) -> str:
    row = item.row
    key = (row.question_id, row.language, row.stripped_hash)
    if row.split != DatasetSplit.TRAIN.value:
        return "excluded_non_train_question"
    if item.source_id in rules.mixed_source_ids:
        return "excluded_mixed_v1_reference_id_match"
    if key in rules.mixed_hashes:
        return "excluded_mixed_v1_hash_match"
    if row.stripped_hash in rules.protected_hashes:
        return "excluded_validation_or_test_hash_match"
    if not _is_valid_candidate(item):
        return "excluded_invalid"
    if key in rules.seen_hashes:
        return "duplicate_within_gpt_heavy"
    return "eligible_train_positive"


def _is_valid_candidate(item: LoadedRecord) -> bool:
    if not item.text.strip() or item.row.language not in {item.value for item in Language}:
        return False
    if cached_vector_for_text(item.text) is None:
        return False
    try:
        validate_source_syntax(item.text, Language(item.row.language))
    except (SourceParseError, ValueError):
        return False
    return True


def _audit_row(item: LoadedRecord, disposition: str) -> CandidateAudit:
    row = item.row
    return CandidateAudit(
        record_id=row.record_id,
        question_id=row.question_id,
        language=row.language,
        split=row.split,
        source_id=item.source_id,
        generator=row.generator,
        persona=row.persona,
        stripped_hash=row.stripped_hash,
        embedding_cache_key=row.embedding_cache_key,
        disposition=disposition,
        duplicate_group=row.duplicate_group,
    )


def _manifest_row(
    context: RecordContext,
    source: RecordSource,
    content: RecordContent,
) -> LabeledRecord:
    identity = (
        f"{MODEL_DATASET_VERSION}|{source.source}|{source.source_id}|"
        f"{content.stripped_hash}"
    )
    record_id = sha256(identity.encode("utf-8")).hexdigest()
    duplicate_group = (
        f"{context.question_id}:{context.language}:{content.stripped_hash}"
    )
    return LabeledRecord(
        record_id=record_id,
        question_id=context.question_id,
        language=context.language,
        split=context.split,
        label=context.label,
        source=source.source,
        generator=source.generator,
        persona=source.persona,
        stripped_hash=content.stripped_hash,
        embedding_cache_key=(
            embedding_cache_key(content.text) if content.text else ""
        ),
        duplicate_group=duplicate_group,
        sample_weight=content.sample_weight,
    )


def _apply_duplicate_weights(records: Sequence[LoadedRecord]) -> list[LoadedRecord]:
    counts = Counter(item.row.duplicate_group for item in records)
    weighted: list[LoadedRecord] = []
    for item in records:
        row = item.row
        weight = 1.0 / counts[row.duplicate_group]
        weighted.append(
            LoadedRecord(
                row=LabeledRecord(**{**asdict(row), "sample_weight": weight}),
                text=item.text,
                source_id=item.source_id,
                user_group_hash=item.user_group_hash,
            )
        )
    return weighted


def _split_manifests(
    population: BuildPopulation,
) -> dict[str, list[dict[str, object]]]:
    rows = [*population.humans, *population.held_out, *population.gpt_train]
    return {
        split.value: [
            asdict(item.row)
            for item in rows
            if item.row.split == split.value
        ]
        for split in DatasetSplit
    }


def _record_coverage(records: Sequence[LoadedRecord]) -> dict[str, object]:
    return {
        "total": len(records),
        "by_split": dict(Counter(item.row.split for item in records)),
        "by_language": dict(Counter(item.row.language for item in records)),
        "by_split_language": {
            split.value: dict(
                Counter(
                    item.row.language
                    for item in records
                    if item.row.split == split.value
                )
            )
            for split in DatasetSplit
        },
        "distinct_pair_hashes": len(
            {
                (item.row.question_id, item.row.language, item.row.stripped_hash)
                for item in records
            }
        ),
        "duplicate_logical_records": len(records)
        - len({item.row.duplicate_group for item in records}),
        "missing_cached_embeddings": sum(
            cached_vector_for_text(item.text) is None for item in records
        ),
    }


def _gpt_overlap_diagnostics(
    population: BuildPopulation,
) -> dict[str, int]:
    audit_keys = [
        (item.question_id, item.language, item.stripped_hash)
        for item in population.gpt_audit
    ]
    mixed_keys = {
        (item.row.question_id, item.row.language, item.row.stripped_hash)
        for item in population.mixed_references
    }
    protected_hashes = {
        item.row.stripped_hash
        for item in (*population.held_out, *population.humans)
        if item.row.split != DatasetSplit.TRAIN.value
    }
    mixed_source_ids = {item.source_id for item in population.mixed_references}
    return {
        "duplicate_pair_hash_records_all_candidates": len(audit_keys)
        - len(set(audit_keys)),
        "mixed_v1_source_id_matches_all_candidates": sum(
            item.source_id in mixed_source_ids for item in population.gpt_audit
        ),
        "mixed_v1_pair_hash_matches_all_candidates": sum(
            key in mixed_keys for key in audit_keys
        ),
        "validation_or_test_global_hash_matches_all_candidates": sum(
            item.stripped_hash in protected_hashes for item in population.gpt_audit
        ),
    }


def _user_overlap(records: Sequence[LoadedRecord]) -> dict[str, int]:
    splits_by_user: dict[str, set[str]] = defaultdict(set)
    for item in records:
        if item.user_group_hash:
            splits_by_user[item.user_group_hash].add(item.row.split)
    return {
        "user_groups_in_multiple_question_splits": sum(
            len(splits) > 1 for splits in splits_by_user.values()
        )
    }


def _strip_human(
    source_record: object,
    boilerplate: str,
    language: Language,
) -> str | None:
    if not isinstance(source_record, dict):
        return None
    raw_code = source_record.get("raw_code")
    if not isinstance(raw_code, str) or not raw_code.strip():
        return None
    try:
        return strip_solution_body(raw_code, boilerplate, language)
    except (SourceParseError, ValueError):
        return None


def _user_id(source_record: object) -> str | None:
    if not isinstance(source_record, dict):
        return None
    return _optional_string(source_record.get("user_id"))


def _opaque_user_id(user_id: str | None) -> str | None:
    if not user_id:
        return None
    value = f"{MODEL_DATASET_VERSION}|{user_id}"
    return sha256(value.encode("utf-8")).hexdigest()


def _optional_string(value: object) -> str | None:
    return str(value) if value is not None else None


def _text_hash(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    path.write_text(text, encoding="utf-8")


def _write_summary_csv(coverage: Mapping[str, object]) -> None:
    rows = _flatten_counts(coverage)
    with (MODEL_DATASET_DIR / "summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("section", "name", "count"))
        writer.writeheader()
        writer.writerows(rows)


def _flatten_counts(
    coverage: Mapping[str, object],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for section in ("question_split", "manifest_counts"):
        values = coverage[section]
        if isinstance(values, dict):
            rows.extend(
                {"section": section, "name": key, "count": value}
                for key, value in values.items()
            )
    gpt = coverage["gpt_heavy"]
    if isinstance(gpt, dict) and isinstance(gpt.get("disposition"), dict):
        rows.extend(
            {"section": "gpt_disposition", "name": key, "count": value}
            for key, value in gpt["disposition"].items()
        )
    return rows


def _report_markdown(coverage: Mapping[str, object]) -> str:
    return "\n".join(
        [
            "# Model dataset v2",
            "",
            "All eligible humans are labeled negatives. Mixed-v1 remains reference-only.",
            "GPT-heavy candidates augment training only after provenance and leakage checks.",
            "",
            f"- Human records: {coverage['human_records']}",
            f"- Held-out AI: {coverage['held_out_ai']}",
            f"- GPT-heavy: {coverage['gpt_heavy']}",
            f"- Leakage: {coverage['leakage']}",
            f"- User overlap diagnostic: {coverage['user_overlap_across_splits']}",
            "",
            "Cross-question user-style overlap remains an internal-evaluation limitation. "
            "A fresh or user-disjoint external test is still required.",
            "",
        ]
    )


def _file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _directory_sha256(directory: Path) -> str:
    digest = sha256()
    for path in sorted(item for item in directory.rglob("*") if item.is_file()):
        digest.update(path.relative_to(directory).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())

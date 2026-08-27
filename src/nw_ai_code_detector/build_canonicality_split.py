from __future__ import annotations

import csv
import json
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from math import ceil
from pathlib import Path
from collections.abc import Mapping, Sequence

import faiss
import numpy as np

from nw_ai_code_detector.config import (
    AI_SOLUTIONS_DIR,
    CANONICALITY_SPLIT_DIR,
    EMBEDDING_CACHE_DIR,
    EVAL_AI_SOLUTIONS_DIR,
    EVAL_SCORES_PATH,
    HUMAN_REFERENCE_BANK_DIR,
    REFERENCE_INDEX_DIR,
    SELECTED_500_PATH,
)
from nw_ai_code_detector.constants import (
    CANONICALITY_EMBEDDING_DIMENSION,
    CANONICALITY_INTERNAL_TEST_QUESTION_COUNT,
    CANONICALITY_SPLIT_SEED,
    CANONICALITY_SPLIT_VERSION,
    CANONICALITY_TRAIN_QUESTION_COUNT,
    CANONICALITY_VALIDATION_QUESTION_COUNT,
    DatasetSplit,
    Difficulty,
    GENERATION_LANGUAGES,
    HUMAN_REFERENCE_BANK_VERSION,
    HumanBankStatus,
    HumanRole,
    SELECTED_QUESTION_COUNT,
    VOYAGE_CODE_3_MODEL,
    VOYAGE_EMBED_INPUT_TYPE,
)
from nw_ai_code_detector.data_load import load_dataset
from nw_ai_code_detector.embedder import cached_vector_for_text, embedding_cache_key, l2_normalize
from nw_ai_code_detector.evaluation.evaluate_centroid_all_humans import (
    ROLE_HELD_OUT,
    ROLE_HUMAN,
    ROLE_REFERENCE,
    ExperimentRecord,
    _collect_records,
    _content_hash,
)
from nw_ai_code_detector.evaluation.experiment_metrics import UNIT_NORM_TOLERANCE
from nw_ai_code_detector.generate_ai_refs import _load_selected_questions
from nw_ai_code_detector.human_reference_index import (
    CHECKSUM_FILES,
    DISTANCE_METRIC,
    HumanReferenceIndex,
    cluster_offset_key,
)
from nw_ai_code_detector.select_500 import EligibleQuestion
from nw_ai_code_detector.stripper import Language

PHASE_LIMIT_SECONDS = 180
TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15
KNOWN_DIFFICULTIES = {Difficulty.EASY.value, Difficulty.MEDIUM.value, Difficulty.HARD.value}
UNAVAILABLE_DIFFICULTY = "UNAVAILABLE"
EXPECTED_HUMAN_RECORDS = 5387
EXPECTED_HUMAN_HASHES = 5344
EXPECTED_HELDOUT = 1997
COUNT_TOLERANCE = 20
FORBIDDEN_ARTIFACT_KEYS = {"raw_code", "user_id", "email", "name", "code"}
PROTECTED_HASH_KEYS = (
    "eval_scores",
    "mixed_index",
    "embedding_cache_count",
    "ai_solution_count",
    "held_out_count",
)


@dataclass(frozen=True)
class QuestionSplit:
    question_id: str
    split: str
    difficulty: str
    split_seed: int


@dataclass(frozen=True)
class HumanGroup:
    group_key: str
    records: tuple[ExperimentRecord, ...]
    sort_digest: str


@dataclass(frozen=True)
class RoleAssignment:
    record: ExperimentRecord
    role: str
    split: str
    record_id: str
    user_group_hash: str | None
    pair_status: str
    group_key: str


@dataclass(frozen=True)
class BankVector:
    question_id: str
    language: str
    reference_id: str
    stripped_hash: str
    embedding_cache_key: str
    source_record_count: int
    vector: np.ndarray
    split: str
    sparse: bool


def main() -> int:
    started = time.perf_counter()
    before = snapshot_protected_artifacts()
    print("Protected hashes before:")
    _print_snapshot(before)
    phase = time.perf_counter()
    questions, records, inspect = load_starting_population()
    _check_phase("load data", phase)
    _print_inspect(inspect)
    if inspect["stop"]:
        print("STOP: starting population does not match expected coverage")
        return 1
    phase = time.perf_counter()
    splits = assign_question_splits(questions, CANONICALITY_SPLIT_SEED)
    _check_phase("assign question split", phase)
    split_map = {item.question_id: item.split for item in splits}
    phase = time.perf_counter()
    assignments = assign_human_roles(records, split_map)
    held_out = build_heldout_rows(records, split_map)
    _check_phase("assign human roles", phase)
    phase = time.perf_counter()
    bank_vectors = collect_bank_vectors(assignments)
    _check_phase("load cached vectors", phase)
    empty_pairs = _empty_human_pairs(questions, assignments)
    phase = time.perf_counter()
    write_human_bank(bank_vectors, assignments, len(empty_pairs))
    _check_phase("build FAISS", phase)
    phase = time.perf_counter()
    loaded = HumanReferenceIndex.load(HUMAN_REFERENCE_BANK_DIR)
    mixed_compat = validate_mixed_v1(loaded, bank_vectors)
    leakages = leakage_report(assignments, held_out, splits, HUMAN_REFERENCE_BANK_DIR)
    if any(leakages.values()):
        print("STOP: leakage checks failed")
        print(json.dumps(leakages, indent=2))
        return 1
    _check_phase("validate artifacts", phase)
    phase = time.perf_counter()
    write_split_artifacts(splits, assignments, held_out, inspect, mixed_compat, leakages)
    _check_phase("write reports", phase)
    after = snapshot_protected_artifacts()
    print("Protected hashes after:")
    _print_snapshot(after)
    _assert_protected_unchanged(before, after)
    print(f"Total runtime: {time.perf_counter() - started:.1f}s")
    return 0


def load_starting_population() -> tuple[list[EligibleQuestion], list[ExperimentRecord], dict[str, object]]:
    dataset = load_dataset()
    questions = _load_selected_questions(SELECTED_500_PATH)
    records = _collect_records(dataset, questions)
    humans = [item for item in records if item.role == ROLE_HUMAN]
    references = [item for item in records if item.role == ROLE_REFERENCE]
    held = [item for item in records if item.role == ROLE_HELD_OUT]
    missing_humans = sum(1 for item in humans if cached_vector_for_text(item.text) is None)
    pairs = {(item.question_id, item.language) for item in references}
    inspect = {
        "selected_questions": len(questions),
        "ai_reference_pairs": len(pairs),
        "ai_reference_records": len(references),
        "human_records": len(humans),
        "distinct_human_hashes": len({item.content_hash for item in humans}),
        "missing_human_embeddings": missing_humans,
        "held_out_records": len(held),
        "stop": False,
    }
    inspect["stop"] = _coverage_failed(inspect)
    return questions, records, inspect


def assign_question_splits(
    questions: Sequence[EligibleQuestion],
    seed: int,
) -> tuple[QuestionSplit, ...]:
    buckets: dict[str, list[EligibleQuestion]] = defaultdict(list)
    unknown_ids = []
    for item in questions:
        difficulty = item.difficulty if item.difficulty in KNOWN_DIFFICULTIES else UNAVAILABLE_DIFFICULTY
        if difficulty == UNAVAILABLE_DIFFICULTY:
            unknown_ids.append(item.question_id)
        buckets[difficulty].append(item)
    rng = np.random.default_rng(seed)
    chosen: dict[str, list[EligibleQuestion]] = {
        DatasetSplit.TRAIN.value: [],
        DatasetSplit.VALIDATION.value: [],
        DatasetSplit.INTERNAL_TEST.value: [],
    }
    for difficulty in sorted(buckets):
        ranked = _shuffle_questions(buckets[difficulty], rng)
        n_train, n_val, n_test = _stratum_counts(len(ranked))
        chosen[DatasetSplit.TRAIN.value].extend(ranked[:n_train])
        chosen[DatasetSplit.VALIDATION.value].extend(ranked[n_train : n_train + n_val])
        chosen[DatasetSplit.INTERNAL_TEST.value].extend(ranked[n_train + n_val : n_train + n_val + n_test])
    _rebalance_split_counts(chosen, len(questions))
    splits = []
    for split_name, items in chosen.items():
        for item in items:
            difficulty = item.difficulty if item.difficulty in KNOWN_DIFFICULTIES else UNAVAILABLE_DIFFICULTY
            splits.append(QuestionSplit(item.question_id, split_name, difficulty, seed))
    _assert_split_invariants(splits, questions)
    if unknown_ids:
        print(f"Questions without known difficulty: {len(unknown_ids)}")
    return tuple(sorted(splits, key=lambda item: item.question_id))


def assign_human_roles(
    records: Sequence[ExperimentRecord],
    split_map: Mapping[str, str],
) -> tuple[RoleAssignment, ...]:
    humans = [item for item in records if item.role == ROLE_HUMAN]
    grouped: dict[tuple[str, str], list[ExperimentRecord]] = defaultdict(list)
    for item in humans:
        grouped[(item.question_id, item.language)].append(item)
    assignments: list[RoleAssignment] = []
    for pair, pair_records in sorted(grouped.items()):
        assignments.extend(_assign_pair_roles(pair_records, split_map[pair[0]]))
    _assert_role_invariants(assignments)
    return tuple(assignments)


def bank_group_count(distinct_group_count: int) -> int:
    if distinct_group_count <= 0:
        return 0
    count = ceil(distinct_group_count / 2)
    if distinct_group_count >= 2 and count == distinct_group_count:
        return distinct_group_count - 1
    return count


def collect_bank_vectors(assignments: Sequence[RoleAssignment]) -> tuple[BankVector, ...]:
    bank_rows = [item for item in assignments if item.role == HumanRole.REFERENCE_BANK.value]
    grouped: dict[tuple[str, str, str], list[RoleAssignment]] = defaultdict(list)
    status_by_pair = {
        (item.record.question_id, item.record.language): item.pair_status
        for item in assignments
    }
    for item in bank_rows:
        key = (item.record.question_id, item.record.language, item.record.content_hash)
        grouped[key].append(item)
    vectors: list[BankVector] = []
    for (question_id, language, stripped_hash), group in grouped.items():
        text = group[0].record.text
        cached = cached_vector_for_text(text)
        if cached is None:
            raise RuntimeError(
                f"Missing cached embedding for human bank {question_id}:{language}:{stripped_hash}"
            )
        array = np.asarray(l2_normalize(cached), dtype=np.float32)
        _assert_unit_vector(array)
        sparse = status_by_pair[(question_id, language)] == HumanBankStatus.SPARSE.value
        vectors.append(
            BankVector(
                question_id=question_id,
                language=language,
                reference_id=_human_reference_id(question_id, language, stripped_hash),
                stripped_hash=stripped_hash,
                embedding_cache_key=embedding_cache_key(text),
                source_record_count=len(group),
                vector=array,
                split=group[0].split,
                sparse=sparse,
            )
        )
    ordered = tuple(
        sorted(vectors, key=lambda item: (item.question_id, item.language, item.reference_id))
    )
    return ordered


def build_heldout_rows(
    records: Sequence[ExperimentRecord],
    split_map: Mapping[str, str],
) -> tuple[dict[str, object], ...]:
    mixed_hashes = _mixed_hashes(records)
    models = _heldout_models({item.question_id for item in records if item.role == ROLE_HELD_OUT})
    rows = []
    for item in records:
        if item.role != ROLE_HELD_OUT:
            continue
        pair = (item.question_id, item.language)
        exact = item.content_hash in mixed_hashes.get(pair, set())
        generator = models.get((item.question_id, item.language, item.source), "unknown")
        rows.append(
            {
                "record_id": _heldout_record_id(item),
                "question_id": item.question_id,
                "language": item.language,
                "split": split_map[item.question_id],
                "generator": generator,
                "persona": item.source,
                "stripped_hash": item.content_hash,
                "embedding_cache_key": embedding_cache_key(item.text),
                "exact_match_to_mixed_v1_reference": exact,
            }
        )
    return tuple(sorted(rows, key=lambda item: str(item["record_id"])))


def write_human_bank(
    vectors: Sequence[BankVector],
    assignments: Sequence[RoleAssignment],
    empty_pair_count: int,
) -> None:
    HUMAN_REFERENCE_BANK_DIR.mkdir(parents=True, exist_ok=True)
    matrix = np.asarray([item.vector for item in vectors], dtype=np.float32)
    index = faiss.IndexFlatIP(CANONICALITY_EMBEDDING_DIMENSION)
    index.add(np.ascontiguousarray(matrix))
    faiss.write_index(index, str(HUMAN_REFERENCE_BANK_DIR / "index.faiss"))
    _write_text(HUMAN_REFERENCE_BANK_DIR / "manifest.jsonl", _manifest_text(vectors))
    offsets = _cluster_offsets(vectors)
    _write_json(HUMAN_REFERENCE_BANK_DIR / "cluster_offsets.json", offsets)
    metadata = _bank_metadata(vectors, assignments, offsets, empty_pair_count)
    _write_json(HUMAN_REFERENCE_BANK_DIR / "metadata.json", metadata)
    checksums = {
        name: sha256((HUMAN_REFERENCE_BANK_DIR / name).read_bytes()).hexdigest()
        for name in CHECKSUM_FILES
    }
    _write_json(HUMAN_REFERENCE_BANK_DIR / "checksums.json", checksums)


def snapshot_protected_artifacts() -> dict[str, str]:
    return {
        "eval_scores": _file_sha256(EVAL_SCORES_PATH),
        "mixed_index": _dir_sha256(REFERENCE_INDEX_DIR),
        "embedding_cache_count": str(_cache_count()),
        "ai_solution_count": str(_json_count(AI_SOLUTIONS_DIR)),
        "held_out_count": str(_json_count(EVAL_AI_SOLUTIONS_DIR)),
    }


def validate_mixed_v1(
    human_index: HumanReferenceIndex,
    bank_vectors: Sequence[BankVector],
) -> dict[str, object]:
    manifest = json.loads((REFERENCE_INDEX_DIR / "manifest.json").read_text(encoding="utf-8"))
    clusters = manifest["clusters"]
    ai_tokens = set(clusters)
    human_pairs = {(item.question_id, item.language) for item in bank_vectors}
    matched = 0
    missing = 0
    for question_id, language in sorted(human_pairs):
        token = f"{question_id}:{language}"
        if token in ai_tokens:
            matched += 1
        else:
            missing += 1
    dimension = int(manifest["dimension"])
    if dimension != CANONICALITY_EMBEDDING_DIMENSION:
        raise RuntimeError("mixed-v1 dimension is not 1024")
    logical = sum(int(item["count"]) for item in clusters.values())
    sample = next(REFERENCE_INDEX_DIR.glob("*.npy"))
    array = np.load(sample)
    if array.shape[1] != CANONICALITY_EMBEDDING_DIMENSION:
        raise RuntimeError("mixed-v1 npy dimension mismatch")
    if not np.isfinite(array).all():
        raise RuntimeError("mixed-v1 sample is non-finite")
    return {
        "mixed_v1_logical_references": logical,
        "mixed_v1_clusters": len(clusters),
        "mixed_v1_dimension": dimension,
        "human_clusters_with_ai_match": matched,
        "human_clusters_missing_ai": missing,
        "human_index_loaded": human_index is not None,
    }


def leakage_report(
    assignments: Sequence[RoleAssignment],
    held_out: Sequence[Mapping[str, object]],
    splits: Sequence[QuestionSplit],
    bank_dir: Path,
) -> dict[str, int]:
    bank_ids = {item.record_id for item in assignments if item.role == HumanRole.REFERENCE_BANK.value}
    labeled_ids = {item.record_id for item in assignments if item.role == HumanRole.LABELED.value}
    both_roles = len(bank_ids & labeled_ids)
    hash_both = _hash_role_overlap(assignments)
    labeled_split_dupes = _duplicate_labeled_across_splits(assignments)
    question_multi = _question_multi_split(splits)
    language_mismatch = _language_split_mismatch(assignments, splits)
    heldout_mismatch = sum(
        1
        for row in held_out
        if row["split"] != next(item.split for item in splits if item.question_id == row["question_id"])
    )
    wrong_scope = _wrong_scope_vectors(bank_dir)
    raw_fields = _forbidden_fields(bank_dir)
    return {
        "human_record_in_both_roles": both_roles,
        "within_pair_hash_in_both_roles": hash_both,
        "labeled_record_duplicated_across_splits": labeled_split_dupes,
        "question_id_in_more_than_one_split": question_multi,
        "cpp_python_split_mismatch": language_mismatch,
        "heldout_split_mismatch": heldout_mismatch,
        "human_bank_wrong_question_or_language": wrong_scope,
        "raw_code_or_user_id_in_bank_artifacts": raw_fields,
    }


def write_split_artifacts(
    splits: Sequence[QuestionSplit],
    assignments: Sequence[RoleAssignment],
    held_out: Sequence[Mapping[str, object]],
    inspect: Mapping[str, object],
    mixed_compat: Mapping[str, object],
    leakages: Mapping[str, int],
) -> None:
    CANONICALITY_SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    _write_json(
        CANONICALITY_SPLIT_DIR / "question_split.json",
        [
            {
                "question_id": item.question_id,
                "split": item.split,
                "difficulty": item.difficulty,
                "split_seed": item.split_seed,
            }
            for item in splits
        ],
    )
    _write_text(CANONICALITY_SPLIT_DIR / "human_roles.jsonl", _roles_text(assignments))
    labeled = [item for item in assignments if item.role == HumanRole.LABELED.value]
    _write_text(CANONICALITY_SPLIT_DIR / "labeled_humans.jsonl", _labeled_text(labeled))
    _write_text(CANONICALITY_SPLIT_DIR / "labeled_heldout_ai.jsonl", _jsonl(held_out))
    coverage = _coverage_payload(splits, assignments, held_out, inspect, mixed_compat, leakages)
    _write_json(CANONICALITY_SPLIT_DIR / "coverage.json", coverage)
    _write_summary_csv(coverage)
    (CANONICALITY_SPLIT_DIR / "report.md").write_text(_report_markdown(coverage), encoding="utf-8")


def _assign_pair_roles(
    records: Sequence[ExperimentRecord],
    split: str,
) -> list[RoleAssignment]:
    groups = _human_groups(records)
    bank_n = bank_group_count(len(groups))
    status = _pair_status(bank_n, len(groups))
    assignments = []
    for index, group in enumerate(groups):
        role = HumanRole.REFERENCE_BANK.value if index < bank_n else HumanRole.LABELED.value
        for record in group.records:
            assignments.append(
                RoleAssignment(
                    record=record,
                    role=role,
                    split=split,
                    record_id=_human_record_id(record),
                    user_group_hash=_user_group_hash(record.user_id),
                    pair_status=status,
                    group_key=group.group_key,
                )
            )
    return assignments


def _human_groups(records: Sequence[ExperimentRecord]) -> tuple[HumanGroup, ...]:
    hashes = [item.content_hash for item in records]
    parent = {value: value for value in hashes}
    by_user: dict[str, list[str]] = defaultdict(list)
    for item in records:
        if item.user_id:
            by_user[item.user_id].append(item.content_hash)
    for values in by_user.values():
        first = values[0]
        for other in values[1:]:
            _union(parent, first, other)
    components: dict[str, list[ExperimentRecord]] = defaultdict(list)
    for item in records:
        components[_find(parent, item.content_hash)].append(item)
    groups = []
    question_id = records[0].question_id
    language = records[0].language
    for records_in_group in components.values():
        group_key = "|".join(sorted({item.content_hash for item in records_in_group}))
        digest = sha256(
            f"{CANONICALITY_SPLIT_VERSION}|{question_id}|{language}|{group_key}".encode("utf-8")
        ).hexdigest()
        groups.append(HumanGroup(group_key, tuple(records_in_group), digest))
    return tuple(sorted(groups, key=lambda item: item.sort_digest))


def _pair_status(bank_n: int, group_n: int) -> str:
    if bank_n <= 0:
        return HumanBankStatus.UNAVAILABLE.value
    if bank_n == 1:
        return HumanBankStatus.SPARSE.value
    return HumanBankStatus.AVAILABLE.value


def _shuffle_questions(
    questions: Sequence[EligibleQuestion],
    rng: np.random.Generator,
) -> list[EligibleQuestion]:
    ordered = sorted(questions, key=lambda item: item.question_id)
    permutation = rng.permutation(len(ordered))
    return [ordered[int(index)] for index in permutation]


def _stratum_counts(count: int) -> tuple[int, int, int]:
    n_train = int(round(count * TRAIN_FRACTION))
    n_val = int(round(count * VALIDATION_FRACTION))
    if n_train + n_val > count:
        n_val = max(0, count - n_train)
    n_test = count - n_train - n_val
    return n_train, n_val, n_test


def _rebalance_split_counts(
    chosen: dict[str, list[EligibleQuestion]],
    total_questions: int,
) -> None:
    if total_questions != SELECTED_QUESTION_COUNT:
        return
    targets = {
        DatasetSplit.TRAIN.value: CANONICALITY_TRAIN_QUESTION_COUNT,
        DatasetSplit.VALIDATION.value: CANONICALITY_VALIDATION_QUESTION_COUNT,
        DatasetSplit.INTERNAL_TEST.value: CANONICALITY_INTERNAL_TEST_QUESTION_COUNT,
    }
    donor_order = (
        DatasetSplit.TRAIN.value,
        DatasetSplit.VALIDATION.value,
        DatasetSplit.INTERNAL_TEST.value,
    )
    for split_name, target in targets.items():
        while len(chosen[split_name]) > target:
            item = chosen[split_name].pop()
            receiver = next(name for name in donor_order if len(chosen[name]) < targets[name])
            chosen[receiver].append(item)
        while len(chosen[split_name]) < target:
            donor = next(name for name in donor_order if name != split_name and len(chosen[name]) > targets[name])
            chosen[split_name].append(chosen[donor].pop())


def _assert_split_invariants(
    splits: Sequence[QuestionSplit],
    questions: Sequence[EligibleQuestion],
) -> None:
    if len(splits) != len(questions):
        raise RuntimeError("Question split count mismatch")
    if len({item.question_id for item in splits}) != len(splits):
        raise RuntimeError("A question_id was assigned to more than one split")
    if len(questions) != SELECTED_QUESTION_COUNT:
        return
    counts = Counter(item.split for item in splits)
    if counts[DatasetSplit.TRAIN.value] != CANONICALITY_TRAIN_QUESTION_COUNT:
        raise RuntimeError("Train split is not 350 questions")
    if counts[DatasetSplit.VALIDATION.value] != CANONICALITY_VALIDATION_QUESTION_COUNT:
        raise RuntimeError("Validation split is not 75 questions")
    if counts[DatasetSplit.INTERNAL_TEST.value] != CANONICALITY_INTERNAL_TEST_QUESTION_COUNT:
        raise RuntimeError("Internal-test split is not 75 questions")


def _assert_role_invariants(assignments: Sequence[RoleAssignment]) -> None:
    ids = [item.record_id for item in assignments]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Duplicate human record IDs")
    if _hash_role_overlap(assignments):
        raise RuntimeError("A stripped hash was assigned to both roles in a pair")


def _manifest_text(vectors: Sequence[BankVector]) -> str:
    rows = []
    for index, item in enumerate(vectors):
        rows.append(
            {
                "faiss_row_id": index,
                "reference_id": item.reference_id,
                "question_id": item.question_id,
                "language": item.language,
                "stripped_hash": item.stripped_hash,
                "embedding_cache_key": item.embedding_cache_key,
                "source_record_count": item.source_record_count,
                "split_version": CANONICALITY_SPLIT_VERSION,
                "bank_version": HUMAN_REFERENCE_BANK_VERSION,
            }
        )
    return _jsonl(rows)


def _cluster_offsets(vectors: Sequence[BankVector]) -> dict[str, dict[str, object]]:
    offsets: dict[str, dict[str, object]] = {}
    start = 0
    index = 0
    while index < len(vectors):
        question_id = vectors[index].question_id
        language = vectors[index].language
        count = 0
        while (
            index + count < len(vectors)
            and vectors[index + count].question_id == question_id
            and vectors[index + count].language == language
        ):
            count += 1
        offsets[cluster_offset_key(question_id, language)] = {
            "start": start,
            "count": count,
            "sparse": vectors[index].sparse,
        }
        start += count
        index += count
    return offsets


def _bank_metadata(
    vectors: Sequence[BankVector],
    assignments: Sequence[RoleAssignment],
    offsets: Mapping[str, Mapping[str, object]],
    empty_pair_count: int,
) -> dict[str, object]:
    bank_assignments = [item for item in assignments if item.role == HumanRole.REFERENCE_BANK.value]
    languages = Counter(item.language for item in vectors)
    cluster_languages = Counter(json.loads(token)[1] for token in offsets)
    sizes = Counter(int(item["count"]) for item in offsets.values())
    pair_status = {
        (item.record.question_id, item.record.language): item.pair_status
        for item in assignments
    }
    sparse = sum(1 for status in pair_status.values() if status == HumanBankStatus.SPARSE.value)
    unavailable = sum(
        1 for status in pair_status.values() if status == HumanBankStatus.UNAVAILABLE.value
    )
    return {
        "artifact_type": "human_reference_bank",
        "bank_version": HUMAN_REFERENCE_BANK_VERSION,
        "dataset_split_version": CANONICALITY_SPLIT_VERSION,
        "embedding_model": VOYAGE_CODE_3_MODEL,
        "embedding_input_type": VOYAGE_EMBED_INPUT_TYPE,
        "embedding_dimension": CANONICALITY_EMBEDDING_DIMENSION,
        "distance_metric": DISTANCE_METRIC,
        "l2_normalized": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "total_vector_count": len(vectors),
        "total_cluster_count": len(offsets),
        "CPP_cluster_count": cluster_languages.get(Language.CPP.value, 0),
        "CPP_vector_count": languages.get(Language.CPP.value, 0),
        "PYTHON_cluster_count": cluster_languages.get(Language.PYTHON.value, 0),
        "PYTHON_vector_count": languages.get(Language.PYTHON.value, 0),
        "bank_count_distribution": {str(key): value for key, value in sorted(sizes.items())},
        "sparse_pair_count": sparse,
        "missing_pair_count": unavailable + empty_pair_count,
        "source_human_record_count": len(bank_assignments),
        "distinct_stripped_hash_count": len(vectors),
        "network_calls": False,
        "embeddings_generated": False,
    }


def _coverage_payload(
    splits: Sequence[QuestionSplit],
    assignments: Sequence[RoleAssignment],
    held_out: Sequence[Mapping[str, object]],
    inspect: Mapping[str, object],
    mixed_compat: Mapping[str, object],
    leakages: Mapping[str, int],
) -> dict[str, object]:
    labeled = [item for item in assignments if item.role == HumanRole.LABELED.value]
    bank = [item for item in assignments if item.role == HumanRole.REFERENCE_BANK.value]
    return {
        "inspect": dict(inspect),
        "question_split": _question_split_coverage(splits, held_out),
        "human_roles": _human_role_coverage(assignments, labeled, bank),
        "leakage": dict(leakages),
        "mixed_v1": dict(mixed_compat),
        "user_overlap_across_splits": _user_overlap(labeled),
    }


def _question_split_coverage(
    splits: Sequence[QuestionSplit],
    held_out: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    counts = Counter(item.split for item in splits)
    difficulty = defaultdict(Counter)
    for item in splits:
        difficulty[item.split][item.difficulty] += 1
    return {
        "total_question_ids": len(splits),
        "train": counts[DatasetSplit.TRAIN.value],
        "validation": counts[DatasetSplit.VALIDATION.value],
        "internal_test": counts[DatasetSplit.INTERNAL_TEST.value],
        "difficulty_by_split": {key: dict(value) for key, value in difficulty.items()},
        "cpp_python_pair_count": len(splits) * 2,
        "held_out_by_split": dict(Counter(str(item["split"]) for item in held_out)),
        "held_out_by_generator": dict(Counter(str(item["generator"]) for item in held_out)),
        "held_out_by_persona": dict(Counter(str(item["persona"]) for item in held_out)),
    }


def _human_role_coverage(
    assignments: Sequence[RoleAssignment],
    labeled: Sequence[RoleAssignment],
    bank: Sequence[RoleAssignment],
) -> dict[str, object]:
    bank_only = [
        item
        for item in bank
        if (item.record.question_id, item.record.language)
        not in {(row.record.question_id, row.record.language) for row in labeled}
    ]
    return {
        "total_eligible_humans": len(assignments),
        "human_bank_humans": len(bank),
        "labeled_humans": len(labeled),
        "bank_only_humans": len(bank_only),
        "train_labeled_humans": sum(1 for item in labeled if item.split == DatasetSplit.TRAIN.value),
        "validation_labeled_humans": sum(
            1 for item in labeled if item.split == DatasetSplit.VALIDATION.value
        ),
        "internal_test_labeled_humans": sum(
            1 for item in labeled if item.split == DatasetSplit.INTERNAL_TEST.value
        ),
        "distinct_hashes_total": len({item.record.content_hash for item in assignments}),
        "distinct_hashes_bank": len({item.record.content_hash for item in bank}),
        "distinct_hashes_labeled": len({item.record.content_hash for item in labeled}),
        "by_language": {
            "bank": dict(Counter(item.record.language for item in bank)),
            "labeled": dict(Counter(item.record.language for item in labeled)),
        },
        "by_split_labeled": dict(Counter(item.split for item in labeled)),
    }


def _user_overlap(labeled: Sequence[RoleAssignment]) -> dict[str, int]:
    by_hash: dict[str, set[str]] = defaultdict(set)
    for item in labeled:
        if item.user_group_hash:
            by_hash[item.user_group_hash].add(item.split)
    multi = sum(1 for splits in by_hash.values() if len(splits) > 1)
    return {"labeled_user_groups_in_multiple_splits": multi}


def _write_summary_csv(coverage: Mapping[str, object]) -> None:
    human = coverage["human_roles"]
    split = coverage["question_split"]
    rows = [
        {"metric": "train_questions", "value": split["train"]},
        {"metric": "validation_questions", "value": split["validation"]},
        {"metric": "internal_test_questions", "value": split["internal_test"]},
        {"metric": "human_bank_humans", "value": human["human_bank_humans"]},
        {"metric": "labeled_humans", "value": human["labeled_humans"]},
        {"metric": "train_labeled_humans", "value": human["train_labeled_humans"]},
        {"metric": "validation_labeled_humans", "value": human["validation_labeled_humans"]},
        {"metric": "internal_test_labeled_humans", "value": human["internal_test_labeled_humans"]},
    ]
    path = CANONICALITY_SPLIT_DIR / "summary.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "value"])
        writer.writeheader()
        writer.writerows(rows)


def _report_markdown(coverage: Mapping[str, object]) -> str:
    return "\n".join(
        [
            "# Canonicality dataset split v1",
            "",
            json.dumps(coverage, indent=2),
            "",
        ]
    )


def _roles_text(assignments: Sequence[RoleAssignment]) -> str:
    rows = [
        {
            "record_id": item.record_id,
            "question_id": item.record.question_id,
            "language": item.record.language,
            "role": item.role,
            "split": item.split,
            "stripped_hash": item.record.content_hash,
            "embedding_cache_key": embedding_cache_key(item.record.text),
            "user_group_hash": item.user_group_hash,
            "pair_status": item.pair_status,
            "group_key": item.group_key,
        }
        for item in assignments
    ]
    return _jsonl(rows)


def _labeled_text(assignments: Sequence[RoleAssignment]) -> str:
    rows = [
        {
            "record_id": item.record_id,
            "question_id": item.record.question_id,
            "language": item.record.language,
            "split": item.split,
            "stripped_hash": item.record.content_hash,
            "embedding_cache_key": embedding_cache_key(item.record.text),
            "user_group_hash": item.user_group_hash,
            "label": ROLE_HUMAN,
        }
        for item in assignments
    ]
    return _jsonl(rows)


def _mixed_hashes(records: Sequence[ExperimentRecord]) -> dict[tuple[str, str], set[str]]:
    mapping: dict[tuple[str, str], set[str]] = defaultdict(set)
    for item in records:
        if item.role == ROLE_REFERENCE:
            mapping[(item.question_id, item.language)].add(item.content_hash)
    return mapping


def _empty_human_pairs(
    questions: Sequence[EligibleQuestion],
    assignments: Sequence[RoleAssignment],
) -> set[tuple[str, str]]:
    expected = {
        (item.question_id, language.value)
        for item in questions
        for language in GENERATION_LANGUAGES
    }
    present = {(item.record.question_id, item.record.language) for item in assignments}
    return expected - present


def _heldout_models(selected_ids: set[str]) -> dict[tuple[str, str, str], str]:
    mapping: dict[tuple[str, str, str], str] = {}
    if not EVAL_AI_SOLUTIONS_DIR.is_dir():
        return mapping
    for path in EVAL_AI_SOLUTIONS_DIR.rglob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        question_id = str(payload.get("qid") or "")
        if question_id not in selected_ids:
            continue
        key = (question_id, str(payload.get("language")), str(payload.get("persona")))
        mapping[key] = str(payload.get("model") or "unknown")
    return mapping


def _human_record_id(record: ExperimentRecord) -> str:
    payload = (
        f"{CANONICALITY_SPLIT_VERSION}|{record.question_id}|{record.language}|"
        f"{record.source}|{record.content_hash}"
    )
    return sha256(payload.encode("utf-8")).hexdigest()


def _heldout_record_id(record: ExperimentRecord) -> str:
    payload = (
        f"{CANONICALITY_SPLIT_VERSION}|heldout|{record.question_id}|{record.language}|"
        f"{record.source}|{record.content_hash}"
    )
    return sha256(payload.encode("utf-8")).hexdigest()


def _human_reference_id(question_id: str, language: str, stripped_hash: str) -> str:
    payload = f"{HUMAN_REFERENCE_BANK_VERSION}|{question_id}|{language}|{stripped_hash}"
    return sha256(payload.encode("utf-8")).hexdigest()


def _user_group_hash(user_id: str | None) -> str | None:
    if not user_id:
        return None
    return sha256(f"{CANONICALITY_SPLIT_VERSION}|{user_id}".encode("utf-8")).hexdigest()


def _hash_role_overlap(assignments: Sequence[RoleAssignment]) -> int:
    bank: set[tuple[str, str, str]] = set()
    labeled: set[tuple[str, str, str]] = set()
    for item in assignments:
        key = (item.record.question_id, item.record.language, item.record.content_hash)
        if item.role == HumanRole.REFERENCE_BANK.value:
            bank.add(key)
        else:
            labeled.add(key)
    return len(bank & labeled)


def _duplicate_labeled_across_splits(assignments: Sequence[RoleAssignment]) -> int:
    seen: dict[str, str] = {}
    dupes = 0
    for item in assignments:
        if item.role != HumanRole.LABELED.value:
            continue
        previous = seen.get(item.record_id)
        if previous is not None and previous != item.split:
            dupes += 1
        seen[item.record_id] = item.split
    return dupes


def _question_multi_split(splits: Sequence[QuestionSplit]) -> int:
    grouped: dict[str, set[str]] = defaultdict(set)
    for item in splits:
        grouped[item.question_id].add(item.split)
    return sum(1 for values in grouped.values() if len(values) > 1)


def _language_split_mismatch(
    assignments: Sequence[RoleAssignment],
    splits: Sequence[QuestionSplit],
) -> int:
    split_map = {item.question_id: item.split for item in splits}
    mismatches = 0
    for item in assignments:
        if item.split != split_map[item.record.question_id]:
            mismatches += 1
    return mismatches


def _wrong_scope_vectors(bank_dir: Path) -> int:
    offsets = json.loads((bank_dir / "cluster_offsets.json").read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in (bank_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    errors = 0
    for token, info in offsets.items():
        question_id, language = json.loads(token)
        start = int(info["start"])
        count = int(info["count"])
        for row in rows[start : start + count]:
            if row["question_id"] != question_id or row["language"] != language:
                errors += 1
    return errors


def _forbidden_fields(bank_dir: Path) -> int:
    hits = 0
    for path in bank_dir.iterdir():
        if not path.is_file() or path.suffix not in {".json", ".jsonl"}:
            continue
        text = path.read_text(encoding="utf-8")
        for key in FORBIDDEN_ARTIFACT_KEYS:
            if f'"{key}"' in text:
                hits += 1
    return hits


def _coverage_failed(inspect: Mapping[str, object]) -> bool:
    if inspect["selected_questions"] != SELECTED_QUESTION_COUNT:
        return True
    if inspect["ai_reference_pairs"] != 1000:
        return True
    if inspect["ai_reference_records"] != 6000:
        return True
    if abs(int(inspect["human_records"]) - EXPECTED_HUMAN_RECORDS) > COUNT_TOLERANCE:
        return True
    if abs(int(inspect["distinct_human_hashes"]) - EXPECTED_HUMAN_HASHES) > COUNT_TOLERANCE:
        return True
    if inspect["missing_human_embeddings"] != 0:
        return True
    if abs(int(inspect["held_out_records"]) - EXPECTED_HELDOUT) > COUNT_TOLERANCE:
        return True
    return False


def _assert_unit_vector(vector: np.ndarray) -> None:
    if vector.shape != (CANONICALITY_EMBEDDING_DIMENSION,):
        raise RuntimeError("Unexpected bank vector dimension")
    if not np.isfinite(vector).all():
        raise RuntimeError("Non-finite bank vector")
    norm = float(np.linalg.norm(vector))
    if abs(norm - 1.0) > UNIT_NORM_TOLERANCE:
        raise RuntimeError("Bank vector is not unit normalized")


def _find(parent: dict[str, str], item: str) -> str:
    while parent[item] != item:
        parent[item] = parent[parent[item]]
        item = parent[item]
    return item


def _union(parent: dict[str, str], left: str, right: str) -> None:
    root_left = _find(parent, left)
    root_right = _find(parent, right)
    if root_left != root_right:
        parent[root_right] = root_left


def _jsonl(rows: Sequence[Mapping[str, object]]) -> str:
    return "\n".join(json.dumps(row) for row in rows) + ("\n" if rows else "")


def _print_inspect(inspect: Mapping[str, object]) -> None:
    print("Starting population:")
    for key, value in inspect.items():
        print(f"  {key}: {value}")


def _assert_protected_unchanged(before: Mapping[str, str], after: Mapping[str, str]) -> None:
    for key in PROTECTED_HASH_KEYS:
        if before[key] != after[key]:
            raise RuntimeError(f"Protected artifact changed: {key}")


def _check_phase(name: str, started: float) -> None:
    elapsed = time.perf_counter() - started
    print(f"Phase {name} runtime: {elapsed:.1f}s")
    if elapsed > PHASE_LIMIT_SECONDS:
        raise RuntimeError(f"Phase {name} exceeded three minutes ({elapsed:.1f}s)")


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temp_path.replace(path)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _file_sha256(path: Path) -> str:
    if not path.is_file():
        return "missing"
    return sha256(path.read_bytes()).hexdigest()


def _dir_sha256(directory: Path) -> str:
    if not directory.is_dir():
        return "missing"
    digest = sha256()
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            digest.update(str(path.relative_to(directory)).encode("utf-8"))
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _cache_count() -> int:
    if not EMBEDDING_CACHE_DIR.is_dir():
        return 0
    return sum(1 for path in EMBEDDING_CACHE_DIR.glob("*.json"))


def _json_count(directory: Path) -> int:
    if not directory.is_dir():
        return 0
    return sum(1 for path in directory.rglob("*.json"))


def _print_snapshot(snapshot: Mapping[str, str]) -> None:
    for key, value in snapshot.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    raise SystemExit(main())

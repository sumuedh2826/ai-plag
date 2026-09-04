from __future__ import annotations

import json
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from collections.abc import Mapping, Sequence, Set

from nw_ai_code_detector.config import (
    DATA_DIR,
    DISCOUNT_LAYER_DIR,
    SIGNIFICANT_TOKEN_ELIGIBILITY_DIR,
)
from nw_ai_code_detector.constants import (
    CANDIDATE_HUMAN_SOURCE,
    CLUSTER_LOW_DIVERSITY_DISTANCE,
    GROUPS_KEY,
    HAS_CANONICALITY_SCORE_STATUSES,
    RAW_CODE_FIELD,
    SIGNIFICANT_TOKEN_THRESHOLDS_BY_LANGUAGE,
)
from nw_ai_code_detector.data_load import load_dataset
from nw_ai_code_detector.stripper import Language, SourceParseError, strip_solution_body
from tools.labeling.constants import (
    FORBIDDEN_IDENTITY_KEYS,
    LABELING_OUTPUT_DIR,
    LABELS_BACKUP_NAME,
    MANUAL_LABELS_PATH,
    REVIEW_QUEUE_PATH,
    SCOREABLE_BATCH_DESCRIPTIVE_COUNT,
    SCOREABLE_BATCH_DESCRIPTIVE_PYTHON_COUNT,
    SCOREABLE_BATCH_MIN_PER_LANGUAGE_DIFFICULTY,
    SCOREABLE_BATCH_MIN_PER_SCORE_TERTILE,
    SCOREABLE_BATCH_PYTHON_COUNT,
    SCOREABLE_BATCH_SEED,
    SCOREABLE_BATCH_SIZE,
)
from tools.labeling.labels_store import load_labels
from tools.labeling.select_review_set import ReviewPoolRecord, parse_record_id


DIFFICULTIES = ("EASY", "MEDIUM", "HARD")
LANGUAGES = ("PYTHON", "CPP")
SCORE_TERTILES = ("low", "mid", "high")


@dataclass(frozen=True)
class ScoreableReviewRecord:
    record_id: str
    question_id: str
    language: str
    difficulty: str
    group_index: int
    significant_code_token_count: int
    stripped_hash: str
    ai_nn_max_raw: float
    exact_match_to_ai: bool
    descriptive_raise: bool
    frac_descriptive: float


@dataclass(frozen=True)
class ScoreableBatchSpec:
    size: int
    seed: int
    python_count: int
    descriptive_count: int
    descriptive_python_count: int
    min_per_language_difficulty: int
    min_per_score_tertile: int


@dataclass
class _SelectionState:
    eligible: list[ScoreableReviewRecord]
    selected_ids: list[str]
    selected: set[str]
    rng: random.Random
    spec: ScoreableBatchSpec
    by_id: dict[str, ScoreableReviewRecord]


def select_scoreable_batch(
    rows: Sequence[ScoreableReviewRecord],
    spec: ScoreableBatchSpec,
    blocked_ids: Set[str],
    blocked_locators: Set[tuple[str, str, int]],
) -> list[ScoreableReviewRecord]:
    eligible = [
        row
        for row in rows
        if row.record_id not in blocked_ids
        and (row.question_id, row.language, row.group_index) not in blocked_locators
    ]
    rng = random.Random(spec.seed)
    state = _SelectionState(
        eligible, [], set(), rng, spec, {row.record_id: row for row in eligible}
    )
    _add_descriptive_slice(state)
    _fill_language_difficulty_floors(state)
    _fill_score_tertiles(state)
    _fill_language_targets(state)
    ordered = [state.by_id[record_id] for record_id in state.selected_ids]
    rng.shuffle(ordered)
    return ordered[: spec.size]


def backup_labels(labels_path: Path, backup_dir: Path) -> Path:
    backup_dir.mkdir(parents=True, exist_ok=True)
    destination = backup_dir / LABELS_BACKUP_NAME
    shutil.copy2(labels_path, destination)
    return destination


def main() -> int:
    before = _file_digest(MANUAL_LABELS_PATH)
    backup = backup_labels(MANUAL_LABELS_PATH, LABELING_OUTPUT_DIR)
    labels = load_labels()
    blocked_ids = set(labels)
    blocked_locators = {_locator_from_record_id(record_id) for record_id in labels}
    pool = load_scoreable_candidates()
    spec = ScoreableBatchSpec(
        SCOREABLE_BATCH_SIZE,
        SCOREABLE_BATCH_SEED,
        SCOREABLE_BATCH_PYTHON_COUNT,
        SCOREABLE_BATCH_DESCRIPTIVE_COUNT,
        SCOREABLE_BATCH_DESCRIPTIVE_PYTHON_COUNT,
        SCOREABLE_BATCH_MIN_PER_LANGUAGE_DIFFICULTY,
        SCOREABLE_BATCH_MIN_PER_SCORE_TERTILE,
    )
    selected = select_scoreable_batch(pool, spec, blocked_ids, blocked_locators)
    anchored = _anchor_current_hashes(selected)
    write_scoreable_queue(anchored, REVIEW_QUEUE_PATH, spec)
    after = _file_digest(MANUAL_LABELS_PATH)
    if before != after:
        raise RuntimeError("manual_labels.jsonl changed during batch prep")
    _print_report(anchored, labels, backup, before)
    return 0


def load_scoreable_candidates() -> list[ScoreableReviewRecord]:
    readings = _load_human_readings()
    pool = _load_eligibility_pool()
    rows: list[ScoreableReviewRecord] = []
    used: set[int] = set()
    for item in pool:
        match = _match_reading(item, readings, used)
        if match is None:
            continue
        if match["routing"]["status"] not in HAS_CANONICALITY_SCORE_STATUSES:
            continue
        if not _passes_scoreable_gates(item, match):
            continue
        raise_info = match["descriptive_raise"]
        rows.append(
            ScoreableReviewRecord(
                item.record_id,
                item.question_id,
                item.language,
                item.difficulty,
                item.group_index,
                item.significant_code_token_count,
                item.stripped_hash,
                item.ai_nn_max_raw,
                item.exact_match_to_ai,
                bool(raise_info["high"]),
                float(raise_info["frac_descriptive"]),
            )
        )
    return rows


def write_scoreable_queue(
    records: Sequence[ScoreableReviewRecord],
    path: Path,
    spec: ScoreableBatchSpec,
) -> None:
    items = [_queue_item(index, row) for index, row in enumerate(records)]
    payload = {
        "seed": spec.seed,
        "size": len(items),
        "selection": "scoreable_locked_config_balanced",
        "python_count": spec.python_count,
        "descriptive_count": spec.descriptive_count,
        "min_per_language_difficulty": spec.min_per_language_difficulty,
        "min_per_score_tertile": spec.min_per_score_tertile,
        "items": items,
    }
    _reject_identity_keys(payload)
    for item in items:
        _reject_identity_keys(item)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _add_descriptive_slice(state: _SelectionState) -> None:
    flagged = [row for row in state.eligible if row.descriptive_raise]
    python_rows = [row for row in flagged if row.language == "PYTHON"]
    cpp_rows = [row for row in flagged if row.language == "CPP"]
    state.rng.shuffle(python_rows)
    state.rng.shuffle(cpp_rows)
    python_take = min(state.spec.descriptive_python_count, len(python_rows))
    cpp_take = min(state.spec.descriptive_count - python_take, len(cpp_rows))
    for row in python_rows[:python_take] + cpp_rows[:cpp_take]:
        _try_add(state, row)


def _fill_language_difficulty_floors(state: _SelectionState) -> None:
    by_cell: dict[tuple[str, str], list[ScoreableReviewRecord]] = defaultdict(list)
    for row in state.eligible:
        by_cell[(row.language, row.difficulty)].append(row)
    for language in LANGUAGES:
        for difficulty in DIFFICULTIES:
            needed = state.spec.min_per_language_difficulty - _cell_count(
                state, language, difficulty
            )
            candidates = [
                row
                for row in by_cell[(language, difficulty)]
                if row.record_id not in state.selected
            ]
            state.rng.shuffle(candidates)
            for row in candidates:
                if needed <= 0:
                    break
                if _try_add(state, row):
                    needed -= 1


def _fill_score_tertiles(state: _SelectionState) -> None:
    for language in LANGUAGES:
        bands = _tertile_bands(state.eligible, language)
        for band in SCORE_TERTILES:
            needed = state.spec.min_per_score_tertile - _tertile_count(
                state, language, bands, band
            )
            candidates = [row for row in bands[band] if row.record_id not in state.selected]
            state.rng.shuffle(candidates)
            for row in candidates:
                if needed <= 0:
                    break
                if _try_add(state, row):
                    needed -= 1


def _fill_language_targets(state: _SelectionState) -> None:
    leftover = [row for row in state.eligible if row.record_id not in state.selected]
    state.rng.shuffle(leftover)
    for row in leftover:
        if len(state.selected_ids) >= state.spec.size:
            return
        _try_add(state, row)


def _try_add(state: _SelectionState, row: ScoreableReviewRecord) -> bool:
    if row.record_id in state.selected:
        return False
    if len(state.selected_ids) >= state.spec.size:
        return False
    if _language_count(state, row.language) >= _language_cap(state.spec, row.language):
        return False
    if row.descriptive_raise and _flagged_count(state) >= state.spec.descriptive_count:
        return False
    state.selected_ids.append(row.record_id)
    state.selected.add(row.record_id)
    return True


def _flagged_count(state: _SelectionState) -> int:
    return sum(
        1 for record_id in state.selected_ids if state.by_id[record_id].descriptive_raise
    )


def _language_cap(spec: ScoreableBatchSpec, language: str) -> int:
    if language == "PYTHON":
        return spec.python_count
    return spec.size - spec.python_count


def _language_count(state: _SelectionState, language: str) -> int:
    return sum(1 for record_id in state.selected_ids if state.by_id[record_id].language == language)


def _cell_count(state: _SelectionState, language: str, difficulty: str) -> int:
    return sum(
        1
        for record_id in state.selected_ids
        if state.by_id[record_id].language == language
        and state.by_id[record_id].difficulty == difficulty
    )


def _tertile_count(
    state: _SelectionState,
    language: str,
    bands: Mapping[str, Sequence[ScoreableReviewRecord]],
    band: str,
) -> int:
    ids = {row.record_id for row in bands[band]}
    return sum(
        1
        for record_id in state.selected_ids
        if record_id in ids and state.by_id[record_id].language == language
    )


def _tertile_bands(
    rows: Sequence[ScoreableReviewRecord],
    language: str,
) -> dict[str, list[ScoreableReviewRecord]]:
    members = sorted(
        [row for row in rows if row.language == language],
        key=lambda row: (row.ai_nn_max_raw, row.record_id),
    )
    if not members:
        return {band: [] for band in SCORE_TERTILES}
    size = len(members)
    low_end = size // 3
    mid_end = (2 * size) // 3
    return {
        "low": members[:low_end],
        "mid": members[low_end:mid_end],
        "high": members[mid_end:],
    }


def _passes_scoreable_gates(item: ReviewPoolRecord, reading: Mapping[str, object]) -> bool:
    floor = SIGNIFICANT_TOKEN_THRESHOLDS_BY_LANGUAGE[item.language]
    diversity = float(reading["cluster_diversity"])
    long_enough = item.significant_code_token_count >= floor
    diverse_enough = diversity >= CLUSTER_LOW_DIVERSITY_DISTANCE
    return long_enough and diverse_enough


def _match_reading(
    item: ReviewPoolRecord,
    readings: Sequence[Mapping[str, object]],
    used: set[int],
) -> Mapping[str, object] | None:
    for index, reading in enumerate(readings):
        if index in used:
            continue
        if reading["question_id"] != item.question_id:
            continue
        if reading["language"] != item.language:
            continue
        if int(reading["token_count"]) != item.significant_code_token_count:
            continue
        if abs(float(reading["raw_canonicality"]) - item.ai_nn_max_raw) > 1e-9:
            continue
        used.add(index)
        return reading
    return None


def _load_human_readings() -> list[dict[str, object]]:
    path = DISCOUNT_LAYER_DIR / "readings.jsonl"
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        payload = json.loads(line)
        if payload.get("source") != CANDIDATE_HUMAN_SOURCE:
            continue
        rows.append(payload)
    return rows


def _load_eligibility_pool() -> list[ReviewPoolRecord]:
    path = SIGNIFICANT_TOKEN_ELIGIBILITY_DIR / "candidate_human_scores.jsonl"
    rows: list[ReviewPoolRecord] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        payload = json.loads(line)
        record_id = str(payload["record_id"])
        question_id, language, group_index = parse_record_id(record_id)
        rows.append(
            ReviewPoolRecord(
                record_id,
                question_id,
                language,
                str(payload["difficulty"]),
                group_index,
                int(payload["significant_code_token_count"]),
                str(payload["stripped_hash"]),
                float(payload["ai_nn_max_raw"]),
                bool(payload.get("exact_match_to_mixed_v1", False)),
            )
        )
    return rows


def _anchor_current_hashes(
    records: Sequence[ScoreableReviewRecord],
) -> list[ScoreableReviewRecord]:
    dataset = load_dataset()
    groups = json.loads((DATA_DIR / "scored_submissions.json").read_text(encoding="utf-8"))
    mapping = groups.get(GROUPS_KEY)
    return [_with_current_hash(row, dataset, mapping) for row in records]


def _with_current_hash(row: ScoreableReviewRecord, dataset, mapping) -> ScoreableReviewRecord:
    current_hash = _current_stripped_hash(row, dataset, mapping)
    record_id = (
        f"{CANDIDATE_HUMAN_SOURCE}|{row.question_id}|{row.language}|"
        f"{row.group_index}|{current_hash}"
    )
    return ScoreableReviewRecord(
        record_id,
        row.question_id,
        row.language,
        row.difficulty,
        row.group_index,
        row.significant_code_token_count,
        current_hash,
        row.ai_nn_max_raw,
        row.exact_match_to_ai,
        row.descriptive_raise,
        row.frac_descriptive,
    )


def _current_stripped_hash(row: ScoreableReviewRecord, dataset, mapping) -> str:
    raw = _raw_code(mapping, row.question_id, row.language, row.group_index)
    boilerplate = dataset.questions[row.question_id].boilerplates.get(row.language, "")
    try:
        stripped = strip_solution_body(raw, boilerplate, Language(row.language))
    except (SourceParseError, ValueError) as exc:
        raise RuntimeError(f"Strip failed for {row.record_id}") from exc
    return sha256(stripped.encode("utf-8")).hexdigest()


def _raw_code(mapping: object, question_id: str, language: str, index: int) -> str:
    if not isinstance(mapping, dict):
        raise RuntimeError("scored_submissions groups unavailable")
    records = mapping.get(f"{question_id}:{language}")
    if not isinstance(records, list) or index >= len(records):
        raise RuntimeError(f"Missing raw_code for {question_id}:{language}:{index}")
    record = records[index]
    if not isinstance(record, dict):
        raise RuntimeError(f"Invalid raw record for {question_id}:{language}:{index}")
    raw = record.get(RAW_CODE_FIELD)
    if not isinstance(raw, str) or not raw.strip():
        raise RuntimeError(f"Empty raw_code for {question_id}:{language}:{index}")
    return raw


def _locator_from_record_id(record_id: str) -> tuple[str, str, int]:
    question_id, language, group_index = parse_record_id(record_id)
    return question_id, language, group_index


def _queue_item(index: int, row: ScoreableReviewRecord) -> dict[str, object]:
    return {
        "order_index": index,
        "record_id": row.record_id,
        "question_id": row.question_id,
        "language": row.language,
        "difficulty": row.difficulty,
        "group_index": row.group_index,
        "significant_code_token_count": row.significant_code_token_count,
        "ai_nn_max_raw": row.ai_nn_max_raw,
        "exact_match_to_ai": row.exact_match_to_ai,
        "descriptive_raise": row.descriptive_raise,
        "frac_descriptive": row.frac_descriptive,
    }


def _file_digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _reject_identity_keys(payload: Mapping[str, object]) -> None:
    overlap = FORBIDDEN_IDENTITY_KEYS.intersection(payload)
    if overlap:
        raise RuntimeError(f"Forbidden identity keys: {sorted(overlap)}")


def _print_report(
    selected: Sequence[ScoreableReviewRecord],
    labels: Mapping[str, object],
    backup: Path,
    labels_digest: str,
) -> None:
    languages = Counter(row.language for row in selected)
    cells = Counter((row.language, row.difficulty) for row in selected)
    flagged = sum(1 for row in selected if row.descriptive_raise)
    overlap = [row.record_id for row in selected if row.record_id in labels]
    locators = {(row.question_id, row.language, row.group_index) for row in selected}
    labeled_locators = {_locator_from_record_id(record_id) for record_id in labels}
    locator_overlap = locators.intersection(labeled_locators)
    print(f"selected={len(selected)}")
    print(f"languages={dict(languages)}")
    print(
        "cells="
        + str({f"{lang}:{diff}": count for (lang, diff), count in sorted(cells.items())})
    )
    print(f"descriptive_raise={flagged}")
    print(f"record_id_overlap={len(overlap)}")
    print(f"locator_overlap={len(locator_overlap)}")
    print(f"labels_count={len(labels)} digest={labels_digest}")
    print(f"labels_backup={backup}")
    print(f"queue={REVIEW_QUEUE_PATH}")


if __name__ == "__main__":
    raise SystemExit(main())

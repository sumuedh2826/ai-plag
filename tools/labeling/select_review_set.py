from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from collections.abc import Mapping, Sequence

from nw_ai_code_detector.config import (
    AI_SOLUTIONS_DIR,
    EVAL_AI_SOLUTIONS_DIR,
    SIGNIFICANT_TOKEN_ELIGIBILITY_DIR,
)
from nw_ai_code_detector.constants import (
    CANDIDATE_HUMAN_SOURCE,
    STRIPPED_CODE_FIELD,
)
from tools.labeling.constants import (
    FORBIDDEN_IDENTITY_KEYS,
    MANUAL_REVIEW_MIN_PER_LANGUAGE_DIFFICULTY,
    MANUAL_REVIEW_SEED,
    MANUAL_REVIEW_SET_SIZE,
    REVIEW_QUEUE_PATH,
)


@dataclass(frozen=True)
class ReviewPoolRecord:
    record_id: str
    question_id: str
    language: str
    difficulty: str
    group_index: int
    significant_code_token_count: int
    stripped_hash: str
    ai_nn_max_raw: float
    exact_match_to_ai: bool


@dataclass
class SelectionBuckets:
    rows: Sequence[ReviewPoolRecord]
    selected_ids: list[str]
    selected: set[str]
    rng: random.Random


def select_review_records(
    rows: Sequence[ReviewPoolRecord],
    size: int,
    seed: int,
    min_per_cell: int,
) -> list[ReviewPoolRecord]:
    by_id = {row.record_id: row for row in rows}
    selected_ids: list[str] = []
    selected = set()
    rng = random.Random(seed)
    for row in sorted(rows, key=_record_sort_key):
        if row.exact_match_to_ai:
            _add_id(row.record_id, selected_ids, selected)
    buckets = SelectionBuckets(rows, selected_ids, selected, rng)
    _fill_language_difficulty_floors(buckets, min_per_cell)
    leftover = [row for row in rows if row.record_id not in selected]
    rng.shuffle(leftover)
    for row in leftover:
        if len(selected_ids) >= size:
            break
        _add_id(row.record_id, selected_ids, selected)
    ordered = [by_id[record_id] for record_id in selected_ids]
    rng.shuffle(ordered)
    return ordered[:size]


def load_pool_records() -> list[ReviewPoolRecord]:
    path = SIGNIFICANT_TOKEN_ELIGIBILITY_DIR / "candidate_human_scores.jsonl"
    ai_hashes = _load_ai_stripped_hashes(AI_SOLUTIONS_DIR)
    ai_hashes.update(_load_ai_stripped_hashes(EVAL_AI_SOLUTIONS_DIR))
    rows: list[ReviewPoolRecord] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        payload = json.loads(line)
        record_id = str(payload["record_id"])
        question_id, language, group_index = parse_record_id(record_id)
        stripped_hash = str(payload["stripped_hash"])
        rows.append(
            ReviewPoolRecord(
                record_id=record_id,
                question_id=question_id,
                language=language,
                difficulty=str(payload["difficulty"]),
                group_index=group_index,
                significant_code_token_count=int(
                    payload["significant_code_token_count"]
                ),
                stripped_hash=stripped_hash,
                ai_nn_max_raw=float(payload["ai_nn_max_raw"]),
                exact_match_to_ai=stripped_hash in ai_hashes,
            )
        )
    return rows


def parse_record_id(record_id: str) -> tuple[str, str, int]:
    prefix = f"{CANDIDATE_HUMAN_SOURCE}|"
    if not record_id.startswith(prefix):
        raise ValueError(f"Unexpected record_id: {record_id}")
    remainder = record_id[len(prefix) :]
    question_id, language, index_text, _hash = remainder.split("|", 3)
    return question_id, language, int(index_text)


def write_review_queue(records: Sequence[ReviewPoolRecord], path: Path) -> None:
    items = [_queue_item(index, row) for index, row in enumerate(records)]
    payload = {
        "seed": MANUAL_REVIEW_SEED,
        "size": len(items),
        "selection": "random_with_exact_and_cell_floor",
        "min_per_language_difficulty": MANUAL_REVIEW_MIN_PER_LANGUAGE_DIFFICULTY,
        "items": items,
    }
    _reject_identity_keys(payload)
    for item in items:
        _reject_identity_keys(item)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> int:
    if REVIEW_QUEUE_PATH.is_file():
        print(f"Review queue already exists: {REVIEW_QUEUE_PATH}")
        return 0
    rows = load_pool_records()
    selected = select_review_records(
        rows,
        MANUAL_REVIEW_SET_SIZE,
        MANUAL_REVIEW_SEED,
        MANUAL_REVIEW_MIN_PER_LANGUAGE_DIFFICULTY,
    )
    write_review_queue(selected, REVIEW_QUEUE_PATH)
    print(f"Wrote {len(selected)} records to {REVIEW_QUEUE_PATH}")
    return 0


def _fill_language_difficulty_floors(
    buckets: SelectionBuckets,
    min_per_cell: int,
) -> None:
    by_cell: dict[tuple[str, str], list[ReviewPoolRecord]] = defaultdict(list)
    for row in buckets.rows:
        by_cell[(row.language, row.difficulty)].append(row)
    counts = _cell_counts(buckets.rows, buckets.selected_ids)
    for cell in sorted(by_cell):
        needed = min_per_cell - counts[cell]
        if needed <= 0:
            continue
        candidates = [
            row for row in by_cell[cell] if row.record_id not in buckets.selected
        ]
        buckets.rng.shuffle(candidates)
        for row in candidates[:needed]:
            _add_id(row.record_id, buckets.selected_ids, buckets.selected)


def _cell_counts(
    rows: Sequence[ReviewPoolRecord],
    selected_ids: Sequence[str],
) -> dict[tuple[str, str], int]:
    by_id = {row.record_id: row for row in rows}
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for record_id in selected_ids:
        row = by_id[record_id]
        counts[(row.language, row.difficulty)] += 1
    return counts


def _add_id(record_id: str, selected_ids: list[str], selected: set[str]) -> None:
    if record_id in selected:
        return
    selected_ids.append(record_id)
    selected.add(record_id)


def _record_sort_key(row: ReviewPoolRecord) -> str:
    return row.record_id


def _queue_item(index: int, row: ReviewPoolRecord) -> dict[str, object]:
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
    }


def _load_ai_stripped_hashes(root: Path) -> set[str]:
    hashes: set[str] = set()
    for path in root.rglob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        stripped = payload.get(STRIPPED_CODE_FIELD)
        if isinstance(stripped, str) and stripped.strip():
            hashes.add(sha256(stripped.encode("utf-8")).hexdigest())
    return hashes


def _reject_identity_keys(payload: Mapping[str, object]) -> None:
    overlap = FORBIDDEN_IDENTITY_KEYS.intersection(payload)
    if overlap:
        raise RuntimeError(f"Forbidden identity keys: {sorted(overlap)}")


if __name__ == "__main__":
    raise SystemExit(main())

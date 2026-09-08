from __future__ import annotations

from functools import lru_cache
import json
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Mapping, Sequence

import numpy as np

from nw_ai_code_detector.config import (
    AI_SOLUTIONS_DIR,
    DATA_DIR,
    REFERENCE_INDEX_V1_DIR,
)
from nw_ai_code_detector.constants import (
    GROUPS_KEY,
    RAW_CODE_FIELD,
    RAW_OUTPUT_FIELD,
    STRIPPED_CODE_FIELD,
)
from nw_ai_code_detector.data_load import Dataset, load_dataset
from nw_ai_code_detector.embedder import cached_vector_for_text
from nw_ai_code_detector.stripper import Language, SourceParseError, strip_solution_body
from tools.labeling.constants import REVIEW_QUEUE_PATH


@dataclass(frozen=True)
class QueueItem:
    order_index: int
    record_id: str
    question_id: str
    language: str
    difficulty: str
    group_index: int
    significant_code_token_count: int
    ai_nn_max_raw: float
    exact_match_to_ai: bool


@dataclass(frozen=True)
class NearestAiReference:
    stripped_code: str
    raw_output: str | None
    similarity: float
    source_filename: str


@dataclass(frozen=True)
class ReviewPage:
    item: QueueItem
    statement_content: str
    boilerplate: str
    raw_code: str
    stripped_code: str | None
    strip_error: str | None
    nearest_ai: NearestAiReference | None
    nearest_ai_error: str | None


def load_queue(path: Path = REVIEW_QUEUE_PATH) -> list[QueueItem]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [_queue_item(item) for item in payload["items"]]


def load_review_page(item: QueueItem) -> ReviewPage:
    dataset = _cached_dataset()
    groups = _cached_groups()
    question = dataset.questions[item.question_id]
    boilerplate = question.boilerplates.get(item.language, "")
    raw_code = _raw_code(groups, item)
    stripped, strip_error = _strip_candidate(raw_code, boilerplate, item.language)
    nearest, nearest_error = _nearest_ai_reference(item, stripped)
    return ReviewPage(
        item=item,
        statement_content=question.statement_content,
        boilerplate=boilerplate,
        raw_code=raw_code,
        stripped_code=stripped,
        strip_error=strip_error,
        nearest_ai=nearest,
        nearest_ai_error=nearest_error,
    )


@lru_cache(maxsize=1)
def _cached_dataset() -> Dataset:
    return load_dataset()


@lru_cache(maxsize=1)
def _cached_groups() -> Mapping[str, Sequence[Mapping[str, object]]]:
    return _load_groups()


def _queue_item(payload: Mapping[str, object]) -> QueueItem:
    return QueueItem(
        order_index=int(payload["order_index"]),
        record_id=str(payload["record_id"]),
        question_id=str(payload["question_id"]),
        language=str(payload["language"]),
        difficulty=str(payload["difficulty"]),
        group_index=int(payload["group_index"]),
        significant_code_token_count=int(payload.get("significant_code_token_count", 0)),
        ai_nn_max_raw=float(payload.get("ai_nn_max_raw", 0.0)),
        exact_match_to_ai=bool(payload.get("exact_match_to_ai", False)),
    )


def _load_groups() -> Mapping[str, Sequence[Mapping[str, object]]]:
    payload = json.loads(
        (DATA_DIR / "scored_submissions.json").read_text(encoding="utf-8")
    )
    groups = payload.get(GROUPS_KEY)
    if not isinstance(groups, dict):
        raise RuntimeError("scored_submissions.json groups are unavailable")
    return groups


def _raw_code(
    groups: Mapping[str, Sequence[Mapping[str, object]]],
    item: QueueItem,
) -> str:
    key = f"{item.question_id}:{item.language}"
    records = groups.get(key)
    if not isinstance(records, list) or item.group_index >= len(records):
        raise RuntimeError(f"Missing raw_code for {item.record_id}")
    record = records[item.group_index]
    if not isinstance(record, dict):
        raise RuntimeError(f"Invalid submission record for {item.record_id}")
    raw_code = record.get(RAW_CODE_FIELD)
    if not isinstance(raw_code, str) or not raw_code.strip():
        raise RuntimeError(f"Empty raw_code for {item.record_id}")
    return raw_code


def _strip_candidate(
    raw_code: str,
    boilerplate: str,
    language: str,
) -> tuple[str | None, str | None]:
    try:
        stripped = strip_solution_body(raw_code, boilerplate, Language(language))
    except (SourceParseError, ValueError) as exc:
        return None, str(exc)
    return stripped, None


def _nearest_ai_reference(
    item: QueueItem,
    stripped_code: str | None,
) -> tuple[NearestAiReference | None, str | None]:
    if stripped_code is None:
        return None, "No stripped candidate code to compare"
    query = cached_vector_for_text(stripped_code)
    if query is None:
        return None, "Missing cached embedding; network embedding is forbidden"
    files = _mixed_v1_files(item.question_id, item.language)
    vector_path = REFERENCE_INDEX_V1_DIR / f"{item.question_id}__{item.language}.npy"
    if not vector_path.is_file():
        return None, f"Missing mixed-v1 vectors at {vector_path.name}"
    vectors = np.asarray(np.load(vector_path), dtype=np.float32)
    if len(files) != len(vectors):
        return None, "Mixed-v1 file count does not match cluster vectors"
    query_vector = np.asarray(query, dtype=np.float32)
    similarities = np.asarray(vectors @ query_vector, dtype=np.float64)
    nearest_index = int(np.argmax(similarities))
    payload = json.loads(files[nearest_index].read_text(encoding="utf-8"))
    stripped = payload.get(STRIPPED_CODE_FIELD)
    if not isinstance(stripped, str):
        return None, "Nearest mixed-v1 file has no stripped_code"
    raw_output = payload.get(RAW_OUTPUT_FIELD)
    raw_text = raw_output if isinstance(raw_output, str) and raw_output.strip() else None
    return (
        NearestAiReference(
            stripped_code=stripped,
            raw_output=raw_text,
            similarity=float(similarities[nearest_index]),
            source_filename=files[nearest_index].name,
        ),
        None,
    )


def _mixed_v1_files(question_id: str, language: str) -> list[Path]:
    directory = AI_SOLUTIONS_DIR / question_id / language
    return sorted(directory.glob("*.json"))

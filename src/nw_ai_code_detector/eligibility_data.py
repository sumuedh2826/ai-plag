from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from collections.abc import Mapping

from nw_ai_code_detector.build_model_dataset_v2 import (
    snapshot_protected_artifacts as snapshot_dataset_artifacts,
)
from nw_ai_code_detector.config import (
    EVAL_AI_SOLUTIONS_DIR,
    EVAL_SCORES_PATH,
    SELECTED_500_PATH,
)
from nw_ai_code_detector.constants import (
    RAW_CODE_FIELD,
    RAW_OUTPUT_FIELD,
    STRIPPED_CODE_FIELD,
)


@dataclass(frozen=True)
class LoadedSolution:
    question_id: str
    language: str
    raw_code: str | None
    stripped_code: str | None
    parse_ok: bool
    generator: str | None
    persona: str | None
    relative_path: str
    source: str


def load_solution_records(root: Path, source: str) -> list[LoadedSolution]:
    records = []
    for path in sorted(root.rglob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        records.append(_loaded_solution(payload, path, root, source))
    return records


def snapshot_protected_bundle() -> dict[str, str]:
    snapshot = snapshot_dataset_artifacts()
    snapshot["eval_scores"] = _file_sha256(EVAL_SCORES_PATH)
    snapshot["selected_500"] = _file_sha256(SELECTED_500_PATH)
    snapshot["eval_ai_solutions"] = _directory_sha256(EVAL_AI_SOLUTIONS_DIR)
    return snapshot


def display_similarity(raw: float) -> float:
    return min(raw, 1.0)


def _loaded_solution(
    payload: Mapping[str, object],
    path: Path,
    root: Path,
    source: str,
) -> LoadedSolution:
    raw_code = payload.get(RAW_CODE_FIELD)
    if not isinstance(raw_code, str) or not raw_code:
        raw_output = payload.get(RAW_OUTPUT_FIELD)
        raw_code = raw_output if isinstance(raw_output, str) else None
    stripped = payload.get(STRIPPED_CODE_FIELD)
    generator = payload.get("model")
    persona = payload.get("persona")
    return LoadedSolution(
        question_id=str(payload.get("qid") or payload.get("question_id") or ""),
        language=str(payload.get("language") or ""),
        raw_code=raw_code,
        stripped_code=stripped if isinstance(stripped, str) else None,
        parse_ok=payload.get("parse_ok") is True,
        generator=str(generator) if generator is not None else None,
        persona=str(persona) if persona is not None else None,
        relative_path=path.relative_to(root).as_posix(),
        source=source,
    )


def _file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _directory_sha256(directory: Path) -> str:
    digest = sha256()
    for path in sorted(item for item in directory.rglob("*") if item.is_file()):
        digest.update(path.relative_to(directory).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()

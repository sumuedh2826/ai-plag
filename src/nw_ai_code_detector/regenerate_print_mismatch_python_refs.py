from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Mapping, Sequence
from typing import Any

from openai import AsyncOpenAI

from nw_ai_code_detector.config import (
    AI_SOLUTIONS_DIR,
    EVAL_AI_SOLUTIONS_DIR,
    load_openrouter_settings,
    load_voyage_settings,
)
from nw_ai_code_detector.constants import (
    EVAL_PERSONA_STYLE_DIRECTIVES,
    EvalPersona,
    GenerationStatus,
    MUST_PASS_EXAMPLES_DIRECTIVE,
    NO_COMMENTS_DIRECTIVE,
    PERSONA_STYLE_DIRECTIVES,
    PRINT_MISMATCH_PYTHON_QUESTION_IDS,
    PRINT_MISMATCH_REGENERATE_ALL_PYTHON_IDS,
    PRINT_MISMATCH_REGENERATION_ATTEMPTS,
    PRINT_REQUIRED_OUTPUT_SHAPE_DIRECTIVE,
    Persona,
    STRIPPED_CODE_FIELD,
)
from nw_ai_code_detector.data_load import Dataset, QuestionRecord, load_dataset
from nw_ai_code_detector.embedder import VoyageEmbedder
from nw_ai_code_detector.export_raw_similarity_samples import OUTPUT_PATTERN
from nw_ai_code_detector.generate_ai_refs import (
    GenerationUnit,
    _complete_chat,
    _extract_code,
    _quality_gate,
    _response_text,
    _usage_dict,
)
from nw_ai_code_detector.refresh_stripped_embeddings import (
    CorpusChange,
    _assert_cluster_invariants,
    _rebuild_affected_clusters,
)
from nw_ai_code_detector.stripper import Language


@dataclass(frozen=True)
class PrintMismatchTarget:
    path: Path
    question_id: str
    is_reference: bool
    persona_key: str
    style_directive: str
    model: str
    temperature: float


def main() -> int:
    dataset = load_dataset()
    targets = _select_targets()
    print(f"Regenerating {len(targets)} PYTHON print-mismatch refs")
    outcomes = asyncio.run(_regenerate_targets(dataset, targets))
    failed = [item for item in outcomes if not item["ok"]]
    if failed:
        raise RuntimeError(f"{len(failed)} print-mismatch regenerations failed")
    changes = _reference_changes(outcomes)
    _embed_reference_changes(changes)
    rebuilt = _rebuild_affected_clusters(changes)
    _assert_cluster_invariants()
    confirmation = _confirm_python_clusters_print()
    print(json.dumps({"rebuilt_clusters": rebuilt, "confirmation": confirmation}, indent=2))
    if not all(row["all_print"] for row in confirmation):
        raise RuntimeError("PYTHON print-mismatch clusters still contain non-printing refs")
    return 0


def _select_targets() -> list[PrintMismatchTarget]:
    targets = []
    for question_id in PRINT_MISMATCH_PYTHON_QUESTION_IDS:
        regenerate_all = question_id in PRINT_MISMATCH_REGENERATE_ALL_PYTHON_IDS
        targets.extend(_targets_from_root(AI_SOLUTIONS_DIR, question_id, True, regenerate_all))
        targets.extend(
            _targets_from_root(EVAL_AI_SOLUTIONS_DIR, question_id, False, regenerate_all)
        )
    return targets


def _targets_from_root(
    root: Path,
    question_id: str,
    is_reference: bool,
    regenerate_all: bool,
) -> list[PrintMismatchTarget]:
    folder = root / question_id / Language.PYTHON.value
    if not folder.is_dir():
        return []
    targets = []
    for path in sorted(folder.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not regenerate_all and _stripped_prints(payload):
            continue
        targets.append(_target_from_payload(path, question_id, is_reference, payload))
    return targets


def _target_from_payload(
    path: Path,
    question_id: str,
    is_reference: bool,
    payload: Mapping[str, Any],
) -> PrintMismatchTarget:
    persona_key = str(payload.get("persona") or "")
    return PrintMismatchTarget(
        path=path,
        question_id=question_id,
        is_reference=is_reference,
        persona_key=persona_key,
        style_directive=_style_directive(persona_key, is_reference),
        model=str(payload.get("model") or ""),
        temperature=float(payload.get("temperature") or 0.8),
    )


def _style_directive(persona_key: str, is_reference: bool) -> str:
    if is_reference:
        return PERSONA_STYLE_DIRECTIVES[Persona(persona_key)]
    return EVAL_PERSONA_STYLE_DIRECTIVES[EvalPersona(persona_key)]


def _stripped_prints(payload: Mapping[str, Any]) -> bool:
    stripped = payload.get(STRIPPED_CODE_FIELD)
    if not isinstance(stripped, str):
        return False
    return bool(OUTPUT_PATTERN.search(stripped))


async def _regenerate_targets(
    dataset: Dataset,
    targets: Sequence[PrintMismatchTarget],
) -> list[dict[str, Any]]:
    settings = load_openrouter_settings()
    client = AsyncOpenAI(
        api_key=settings.api_key,
        base_url=settings.base_url,
        timeout=settings.timeout_seconds,
    )
    semaphore = asyncio.Semaphore(min(6, settings.concurrency))
    tasks = [
        asyncio.create_task(_regenerate_one(client, semaphore, dataset, target))
        for target in targets
    ]
    return [await task for task in tasks]


async def _regenerate_one(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    dataset: Dataset,
    target: PrintMismatchTarget,
) -> dict[str, Any]:
    async with semaphore:
        return await _regenerate_one_locked(client, dataset, target)


async def _regenerate_one_locked(
    client: AsyncOpenAI,
    dataset: Dataset,
    target: PrintMismatchTarget,
) -> dict[str, Any]:
    question = dataset.questions[target.question_id]
    prompt = _print_mismatch_prompt(question, target)
    last_error = "no_attempt"
    for attempt in range(1, PRINT_MISMATCH_REGENERATION_ATTEMPTS + 1):
        try:
            payload = await _generate_payload(client, question, target, prompt)
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            continue
        if payload.get("parse_ok") is True and _stripped_prints(payload):
            _write_json(target.path, payload)
            print(f"ok {target.path.name} {target.question_id[:8]} attempt={attempt}")
            return {"ok": True, "target": target, "payload": payload}
        last_error = str(payload.get("reject_reason") or "missing_print")
    print(f"FAIL {target.path} {last_error}")
    return {"ok": False, "target": target, "payload": {}, "error": last_error}


async def _generate_payload(
    client: AsyncOpenAI,
    question: QuestionRecord,
    target: PrintMismatchTarget,
    prompt: str,
) -> dict[str, Any]:
    unit = GenerationUnit(
        question_id=target.question_id,
        language=Language.PYTHON,
        persona=Persona.NAIVE_DUMP,
        model=target.model,
        temperature=target.temperature,
    )
    response = await _complete_chat(client, unit, prompt)
    raw_output = _extract_code(_response_text(response), Language.PYTHON)
    parse_ok, reject_reason, stripped_code = _quality_gate(
        raw_output,
        question.boilerplates.get(Language.PYTHON.value, ""),
        Language.PYTHON,
    )
    usage = _usage_dict(response)
    return {
        "qid": target.question_id,
        "language": Language.PYTHON.value,
        "persona": target.persona_key,
        "model": target.model,
        "temperature": target.temperature,
        "prompt": prompt,
        "raw_output": raw_output,
        "stripped_code": stripped_code,
        "parse_ok": parse_ok,
        "reject_reason": reject_reason,
        "status": GenerationStatus.OK.value if parse_ok else GenerationStatus.FAILED.value,
        "token_usage": dict(usage),
        "cost": usage.get("cost"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **({"split": "held_out"} if not target.is_reference else {}),
    }


def _print_mismatch_prompt(question: QuestionRecord, target: PrintMismatchTarget) -> str:
    boilerplate = question.boilerplates.get(Language.PYTHON.value, "")
    return "\n\n".join(
        [
            question.statement_content.strip(),
            f"Target language: {Language.PYTHON.value}",
            f"Boilerplate:\n{boilerplate}",
            f"Style: {target.style_directive}",
            MUST_PASS_EXAMPLES_DIRECTIVE,
            NO_COMMENTS_DIRECTIVE,
            PRINT_REQUIRED_OUTPUT_SHAPE_DIRECTIVE,
        ]
    )


def _reference_changes(outcomes: Sequence[Mapping[str, Any]]) -> list[CorpusChange]:
    changes = []
    for item in outcomes:
        if not item.get("ok"):
            continue
        target = item["target"]
        if not target.is_reference:
            continue
        payload = item["payload"]
        changes.append(
            CorpusChange(
                source="ai_solutions",
                language=Language.PYTHON.value,
                identifier=target.path.relative_to(AI_SOLUTIONS_DIR).as_posix(),
                question_id=target.question_id,
                stripped=str(payload[STRIPPED_CODE_FIELD]),
                is_reference=True,
            )
        )
    return changes


def _embed_reference_changes(changes: Sequence[CorpusChange]) -> None:
    texts = tuple(dict.fromkeys(item.stripped for item in changes))
    embedder = VoyageEmbedder(load_voyage_settings())
    batch = embedder.embed_texts(texts)
    print(
        json.dumps(
            {
                "embedded": len(texts),
                "cache_hits": batch.cache_hits,
                "cache_misses": batch.cache_misses,
                "cost_usd": batch.cost_usd,
            }
        )
    )


def _confirm_python_clusters_print() -> list[dict[str, Any]]:
    rows = []
    for question_id in PRINT_MISMATCH_PYTHON_QUESTION_IDS:
        mixed = _print_counts(AI_SOLUTIONS_DIR / question_id / Language.PYTHON.value)
        held = _print_counts(EVAL_AI_SOLUTIONS_DIR / question_id / Language.PYTHON.value)
        rows.append(
            {
                "qid": question_id,
                "mixed_print": f"{mixed[0]}/{mixed[1]}",
                "held_print": f"{held[0]}/{held[1]}",
                "all_print": mixed[0] == mixed[1] and mixed[1] == 6 and held[0] == held[1],
            }
        )
    return rows


def _print_counts(folder: Path) -> tuple[int, int]:
    files = list(folder.glob("*.json"))
    printing = 0
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("parse_ok") is True and _stripped_prints(payload):
            printing += 1
    return printing, len(files)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temp_path = path.with_suffix(".json.tmp")
    temp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temp_path.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())

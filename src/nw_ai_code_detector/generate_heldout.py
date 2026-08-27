from __future__ import annotations

import asyncio
import json
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from collections.abc import Mapping, Sequence
from typing import Any

from openai import AsyncOpenAI

from nw_ai_code_detector.config import (
    EVAL_AI_SOLUTIONS_DIR,
    OpenRouterSettings,
    load_openrouter_settings,
    model_slugs,
)
from nw_ai_code_detector.constants import (
    DEFAULT_GENERATION_CONCURRENCY,
    DEFAULT_TEMPERATURE,
    EVAL_PERSONA_ORDER,
    EVAL_PERSONA_STYLE_DIRECTIVES,
    EvalPersona,
    GENERATION_LANGUAGES,
    GenerationStatus,
    MUST_PASS_EXAMPLES_DIRECTIVE,
    NO_COMMENTS_DIRECTIVE,
    OUTPUT_SHAPE_DIRECTIVE,
    TEMPERATURE_MAX,
    TEMPERATURE_MIN,
)
from nw_ai_code_detector.data_load import Dataset, QuestionRecord
from nw_ai_code_detector.generate_ai_refs import (
    GenerationOutcome,
    GenerationUnit,
    _complete_chat,
    _extract_code,
    _file_needs_work,
    _quality_gate,
    _response_text,
    _usage_dict,
    _with_concurrency,
)
from nw_ai_code_detector.select_500 import EligibleQuestion
from nw_ai_code_detector.stripper import Language


@dataclass(frozen=True)
class HeldOutUnit:
    question_id: str
    language: Language
    persona: EvalPersona
    model: str
    temperature: float


def generate_held_out_refs(
    dataset: Dataset,
    questions: Sequence[EligibleQuestion],
    concurrency: int,
) -> int:
    settings = _with_concurrency(load_openrouter_settings(), concurrency)
    units = _build_held_out_units(questions, model_slugs(settings))
    pending = _pending_held_out(units)
    if not pending:
        print(f"Held-out AI refs already present: {len(units)}")
        return 0
    return asyncio.run(_run_held_out(dataset, pending, settings))


def _build_held_out_units(
    questions: Sequence[EligibleQuestion],
    models: tuple[str, ...],
) -> list[HeldOutUnit]:
    units: list[HeldOutUnit] = []
    for question in questions:
        for language in GENERATION_LANGUAGES:
            units.extend(_units_for_question_language(question.question_id, language, models))
    return units


def _units_for_question_language(
    question_id: str,
    language: Language,
    models: tuple[str, ...],
) -> list[HeldOutUnit]:
    rng = random.Random(_held_out_seed(question_id, language.value))
    personas = list(EVAL_PERSONA_ORDER)
    chosen_models = rng.sample(list(models), k=len(EVAL_PERSONA_ORDER))
    rng.shuffle(personas)
    units: list[HeldOutUnit] = []
    for persona, model in zip(personas, chosen_models):
        temperature = TEMPERATURE_MIN + rng.random() * (TEMPERATURE_MAX - TEMPERATURE_MIN)
        units.append(
            HeldOutUnit(
                question_id=question_id,
                language=language,
                persona=persona,
                model=model,
                temperature=round(temperature, 2) or DEFAULT_TEMPERATURE,
            )
        )
    return units


def _held_out_seed(question_id: str, language: str) -> int:
    digest = sha256(f"heldout:{question_id}:{language}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _pending_held_out(units: Sequence[HeldOutUnit]) -> list[HeldOutUnit]:
    pending: list[HeldOutUnit] = []
    for unit in units:
        path = _held_out_path(unit)
        if not path.is_file() or _file_needs_work(path):
            pending.append(unit)
    return pending


def _held_out_path(unit: HeldOutUnit) -> Path:
    model_token = unit.model.replace("/", "_")
    filename = f"{unit.persona.value}_{model_token}.json"
    return EVAL_AI_SOLUTIONS_DIR / unit.question_id / unit.language.value / filename


async def _run_held_out(
    dataset: Dataset,
    pending: Sequence[HeldOutUnit],
    settings: OpenRouterSettings,
) -> int:
    client = AsyncOpenAI(
        api_key=settings.api_key,
        base_url=settings.base_url,
        timeout=settings.timeout_seconds,
    )
    semaphore = asyncio.Semaphore(settings.concurrency)
    started = time.perf_counter()
    total = len(pending)
    print(f"Generating {total} held-out AI units at concurrency {settings.concurrency}")
    outcomes: list[GenerationOutcome] = []
    tasks = [
        asyncio.create_task(_generate_one(client, semaphore, dataset, unit, index, total, started))
        for index, unit in enumerate(pending, start=1)
    ]
    for task in asyncio.as_completed(tasks):
        outcome = await task
        outcomes.append(outcome)
        parse_ok = sum(1 for item in outcomes if item.parse_ok)
        elapsed = time.perf_counter() - started
        print(
            f"held-out {len(outcomes)}/{total} parse_ok={parse_ok} "
            f"elapsed={elapsed:.1f}s"
        )
    print(
        f"Held-out generation done: parse_ok="
        f"{sum(1 for item in outcomes if item.parse_ok)}/{total} "
        f"wall={time.perf_counter() - started:.1f}s"
    )
    return 0


async def _generate_one(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    dataset: Dataset,
    unit: HeldOutUnit,
    index: int,
    total: int,
    started: float,
) -> GenerationOutcome:
    async with semaphore:
        return await _generate_locked(client, dataset, unit, index, total, started)


async def _generate_locked(
    client: AsyncOpenAI,
    dataset: Dataset,
    unit: HeldOutUnit,
    index: int,
    total: int,
    started: float,
) -> GenerationOutcome:
    question = dataset.questions[unit.question_id]
    prompt = _build_held_out_prompt(question, unit)
    request_started = time.perf_counter()
    generation_unit = _as_generation_unit(unit)
    try:
        response = await _complete_chat(client, generation_unit, prompt)
        return _persist_held_out(unit, question, prompt, response, request_started)
    except Exception as exc:
        return _persist_held_out_failure(unit, prompt, request_started, exc)


def _as_generation_unit(unit: HeldOutUnit) -> GenerationUnit:
    from nw_ai_code_detector.constants import Persona

    return GenerationUnit(
        question_id=unit.question_id,
        language=unit.language,
        persona=Persona.NAIVE_DUMP,
        model=unit.model,
        temperature=unit.temperature,
    )


def _build_held_out_prompt(question: QuestionRecord, unit: HeldOutUnit) -> str:
    boilerplate = question.boilerplates.get(unit.language.value, "")
    style_directive = EVAL_PERSONA_STYLE_DIRECTIVES[unit.persona]
    return "\n\n".join(
        [
            question.statement_content.strip(),
            f"Target language: {unit.language.value}",
            f"Boilerplate:\n{boilerplate}",
            f"Style: {style_directive}",
            MUST_PASS_EXAMPLES_DIRECTIVE,
            NO_COMMENTS_DIRECTIVE,
            OUTPUT_SHAPE_DIRECTIVE,
        ]
    )


def _persist_held_out(
    unit: HeldOutUnit,
    question: QuestionRecord,
    prompt: str,
    response: Any,
    request_started: float,
) -> GenerationOutcome:
    raw_output = _extract_code(_response_text(response), unit.language)
    parse_ok, reject_reason, stripped_code = _quality_gate(
        raw_output,
        question.boilerplates.get(unit.language.value, ""),
        unit.language,
    )
    usage = _usage_dict(response)
    payload = {
        "qid": unit.question_id,
        "language": unit.language.value,
        "persona": unit.persona.value,
        "model": unit.model,
        "temperature": unit.temperature,
        "prompt": prompt,
        "raw_output": raw_output,
        "stripped_code": stripped_code,
        "parse_ok": parse_ok,
        "reject_reason": reject_reason,
        "status": GenerationStatus.OK.value,
        "token_usage": dict(usage),
        "cost": usage.get("cost"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "split": "held_out",
    }
    path = _write_held_out(unit, payload)
    return GenerationOutcome(
        unit=_as_generation_unit(unit),
        status=GenerationStatus.OK,
        elapsed_seconds=time.perf_counter() - request_started,
        prompt_tokens=int(usage.get("prompt_tokens") or 0),
        completion_tokens=int(usage.get("completion_tokens") or 0),
        cost=float(usage.get("cost") or 0.0),
        parse_ok=parse_ok,
        output_path=path,
    )


def _persist_held_out_failure(
    unit: HeldOutUnit,
    prompt: str,
    request_started: float,
    exc: Exception,
) -> GenerationOutcome:
    payload = {
        "qid": unit.question_id,
        "language": unit.language.value,
        "persona": unit.persona.value,
        "model": unit.model,
        "temperature": unit.temperature,
        "prompt": prompt,
        "raw_output": "",
        "stripped_code": "",
        "parse_ok": False,
        "reject_reason": f"{type(exc).__name__}: {exc}",
        "status": GenerationStatus.FAILED.value,
        "token_usage": {},
        "cost": None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "split": "held_out",
    }
    path = _write_held_out(unit, payload)
    return GenerationOutcome(
        unit=_as_generation_unit(unit),
        status=GenerationStatus.FAILED,
        elapsed_seconds=time.perf_counter() - request_started,
        prompt_tokens=0,
        completion_tokens=0,
        cost=0.0,
        parse_ok=False,
        output_path=path,
    )


def _write_held_out(unit: HeldOutUnit, payload: Mapping[str, object]) -> Path:
    path = _held_out_path(unit)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(".json.tmp")
    temp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temp_path.replace(path)
    return path

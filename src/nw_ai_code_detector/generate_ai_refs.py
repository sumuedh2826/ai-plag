from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from collections.abc import Mapping, Sequence
from typing import Any

from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI, RateLimitError
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_random_exponential

from nw_ai_code_detector.config import (
    AI_SOLUTIONS_DIR,
    OpenRouterSettings,
    PROGRESS_LOG_PATH,
    SELECTED_500_PATH,
    load_openrouter_settings,
    model_slugs,
)
from nw_ai_code_detector.constants import (
    DEEPSEEK_MAX_TOKENS,
    DEEPSEEK_MODEL_FRAGMENT,
    DEFAULT_GENERATION_CONCURRENCY,
    DEFAULT_TEMPERATURE,
    GENERATION_LANGUAGES,
    GENERATION_MAX_TOKENS,
    GENERATION_RETRY_ATTEMPTS,
    GenerationStatus,
    MAX_STRIPPED_CHAR_COUNT,
    MIN_STRIPPED_CHAR_COUNT,
    MUST_PASS_EXAMPLES_DIRECTIVE,
    NO_COMMENTS_DIRECTIVE,
    OUTPUT_SHAPE_DIRECTIVE,
    PERSONA_ORDER,
    PERSONA_STYLE_DIRECTIVES,
    Persona,
    TEMPERATURE_MAX,
    TEMPERATURE_MIN,
)
from nw_ai_code_detector.data_load import Dataset, QuestionRecord, load_dataset
from nw_ai_code_detector.select_500 import EligibleQuestion
from nw_ai_code_detector.stripper import Language, SourceParseError, parses_source, strip_solution_body

CODE_FENCE_PATTERN = re.compile(
    r"^[ \t]*```[^\r\n]*\r?\n(.*?)(?:^[ \t]*```[ \t]*(?:\r?\n|$)|\Z)",
    re.DOTALL | re.MULTILINE,
)
PYTHON_CODE_START_PATTERN = re.compile(
    r"^[ \t]*(?:from\s+\S+\s+import|import\s+\S+|@|class\s+\w+|def\s+\w+)",
)
CPP_CODE_START_PATTERN = re.compile(
    r"^[ \t]*(?:#\s*include|using\s+namespace|namespace\s+\w+|"
    r"template\s*<|class\s+\w+|struct\s+\w+|enum\s+\w+)",
)
TRAILING_PROSE_PATTERN = re.compile(
    r"^[ \t]*(?:explanation|note|time complexity|space complexity|"
    r"this code|the code|here is why)\s*:",
    re.IGNORECASE,
)
RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})


@dataclass(frozen=True)
class GenerationUnit:
    question_id: str
    language: Language
    persona: Persona
    model: str
    temperature: float


@dataclass(frozen=True)
class GenerationCliOptions:
    limit: int | None
    concurrency: int
    dry_run: bool
    retry_failed: bool


@dataclass(frozen=True)
class GenerationOutcome:
    unit: GenerationUnit
    status: GenerationStatus
    elapsed_seconds: float
    prompt_tokens: int
    completion_tokens: int
    cost: float
    parse_ok: bool
    output_path: Path


def main() -> int:
    options = _parse_cli_options()
    dataset = load_dataset()
    selected = _load_selected_questions(SELECTED_500_PATH)
    units = _build_generation_units(
        _limit_questions(selected, options.limit),
        model_slugs(load_openrouter_settings()) if not options.dry_run else model_slugs(
            _placeholder_settings(options.concurrency)
        ),
    )
    pending = _pending_units(units, options.retry_failed)
    if options.dry_run:
        _print_dry_run(units, pending)
        return 0
    settings = load_openrouter_settings()
    effective_settings = _with_concurrency(settings, options.concurrency)
    return asyncio.run(_run_generation(dataset, pending, units, effective_settings))


def _parse_cli_options() -> GenerationCliOptions:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=DEFAULT_GENERATION_CONCURRENCY)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()
    return GenerationCliOptions(
        limit=args.limit,
        concurrency=args.concurrency,
        dry_run=args.dry_run,
        retry_failed=args.retry_failed,
    )


def _placeholder_settings(concurrency: int) -> OpenRouterSettings:
    return OpenRouterSettings(
        api_key="dry-run",
        base_url="https://openrouter.ai/api/v1",
        gemini_model="google/gemini-3.7-flash",
        deepseek_model="deepseek/deepseek-v4-pro",
        openai_model="openai/gpt-5.5",
        concurrency=concurrency,
        timeout_seconds=60,
    )


def _with_concurrency(
    settings: OpenRouterSettings,
    concurrency: int,
) -> OpenRouterSettings:
    return OpenRouterSettings(
        api_key=settings.api_key,
        base_url=settings.base_url,
        gemini_model=settings.gemini_model,
        deepseek_model=settings.deepseek_model,
        openai_model=settings.openai_model,
        concurrency=concurrency,
        timeout_seconds=settings.timeout_seconds,
    )


def _load_selected_questions(path: Path) -> list[EligibleQuestion]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [
        EligibleQuestion(
            question_id=item["question_id"],
            difficulty=item["difficulty"],
            tags=tuple(item["tags"]),
            primary_tag=item["primary_tag"],
            coverage_count=item["coverage_count"],
            coverage_bucket=item["coverage_bucket"],
        )
        for item in payload
    ]


def _limit_questions(
    selected: Sequence[EligibleQuestion],
    limit: int | None,
) -> list[EligibleQuestion]:
    if limit is None:
        return list(selected)
    return list(selected[:limit])


def _build_generation_units(
    questions: Sequence[EligibleQuestion],
    models: tuple[str, ...],
) -> list[GenerationUnit]:
    units: list[GenerationUnit] = []
    for question in questions:
        for language in GENERATION_LANGUAGES:
            units.extend(_units_for_question_language(question.question_id, language, models))
    return units


def _units_for_question_language(
    question_id: str,
    language: Language,
    models: tuple[str, ...],
) -> list[GenerationUnit]:
    rng = random.Random(_assignment_seed(question_id, language.value))
    personas = list(PERSONA_ORDER)
    rng.shuffle(personas)
    model_slots = _persona_model_slots(models)
    rng.shuffle(model_slots)
    units: list[GenerationUnit] = []
    for persona, model in zip(personas, model_slots):
        temperature = TEMPERATURE_MIN + rng.random() * (TEMPERATURE_MAX - TEMPERATURE_MIN)
        units.append(
            GenerationUnit(
                question_id=question_id,
                language=language,
                persona=persona,
                model=model,
                temperature=round(temperature, 2) or DEFAULT_TEMPERATURE,
            )
        )
    return units


def _persona_model_slots(models: tuple[str, ...]) -> list[str]:
    if not models or len(PERSONA_ORDER) % len(models) != 0:
        raise ValueError("Persona count must divide evenly across models")
    copies_each = len(PERSONA_ORDER) // len(models)
    slots: list[str] = []
    for model in models:
        slots.extend([model] * copies_each)
    return slots
    units: list[GenerationUnit] = []
    for persona, model in zip(personas, model_slots):
        temperature = TEMPERATURE_MIN + rng.random() * (TEMPERATURE_MAX - TEMPERATURE_MIN)
        units.append(
            GenerationUnit(
                question_id=question_id,
                language=language,
                persona=persona,
                model=model,
                temperature=round(temperature, 2) or DEFAULT_TEMPERATURE,
            )
        )
    return units


def _assignment_seed(question_id: str, language: str) -> int:
    digest = sha256(f"{question_id}:{language}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _pending_units(
    units: Sequence[GenerationUnit],
    retry_failed: bool,
) -> list[GenerationUnit]:
    pending: list[GenerationUnit] = []
    for unit in units:
        path = _unit_output_path(unit)
        if not path.is_file():
            pending.append(unit)
            continue
        if retry_failed and _file_needs_work(path):
            pending.append(unit)
    return pending


def _file_needs_work(path: Path) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return True
    status = payload.get("status")
    raw_output = payload.get("raw_output")
    return (
        status == GenerationStatus.FAILED.value
        or payload.get("parse_ok") is not True
        or not isinstance(raw_output, str)
        or not raw_output.strip()
    )


def _unit_output_path(unit: GenerationUnit) -> Path:
    model_token = unit.model.replace("/", "_")
    filename = f"{unit.persona.value}_{model_token}.json"
    return AI_SOLUTIONS_DIR / unit.question_id / unit.language.value / filename


def _print_dry_run(units: Sequence[GenerationUnit], pending: Sequence[GenerationUnit]) -> None:
    print(f"Work plan: {len(units)} units, {len(pending)} pending")
    model_counts = {}
    for unit in units:
        model_counts[unit.model] = model_counts.get(unit.model, 0) + 1
    print("Persona->model assignment counts across the plan:")
    for model, count in sorted(model_counts.items()):
        print(f"  {model}: {count}")
    print("First 12 pending units:")
    for unit in pending[:12]:
        print(
            f"  {unit.question_id} {unit.language.value} {unit.persona.value} "
            f"-> {unit.model} t={unit.temperature}"
        )


async def _run_generation(
    dataset: Dataset,
    pending: Sequence[GenerationUnit],
    all_units: Sequence[GenerationUnit],
    settings: OpenRouterSettings,
) -> int:
    if not pending:
        print(f"Nothing to generate. {len(all_units)} units already present.")
        return 0
    client = AsyncOpenAI(
        api_key=settings.api_key,
        base_url=settings.base_url,
        timeout=settings.timeout_seconds,
    )
    semaphore = asyncio.Semaphore(settings.concurrency)
    started = time.perf_counter()
    outcomes: list[GenerationOutcome] = []
    total = len(pending)
    print(f"Generating {total} units at concurrency {settings.concurrency}")
    tasks = [
        asyncio.create_task(
            _generate_one_unit(client, semaphore, dataset, unit, index, total, started)
        )
        for index, unit in enumerate(pending, start=1)
    ]
    for task in asyncio.as_completed(tasks):
        outcome = await task
        outcomes.append(outcome)
        _print_live_progress(outcomes, total, started)
    _print_run_summary(outcomes, time.perf_counter() - started)
    return 0


async def _generate_one_unit(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    dataset: Dataset,
    unit: GenerationUnit,
    index: int,
    total: int,
    started: float,
) -> GenerationOutcome:
    async with semaphore:
        return await _generate_unit_locked(client, dataset, unit, index, total, started)


async def _generate_unit_locked(
    client: AsyncOpenAI,
    dataset: Dataset,
    unit: GenerationUnit,
    index: int,
    total: int,
    started: float,
) -> GenerationOutcome:
    question = dataset.questions[unit.question_id]
    prompt = _build_prompt(question, unit)
    request_started = time.perf_counter()
    try:
        response = await _complete_chat(client, unit, prompt)
        outcome = _persist_success(unit, question, prompt, response, request_started)
    except Exception as exc:
        outcome = _persist_failure(unit, prompt, request_started, exc)
    _append_progress(outcome, index, total, started)
    return outcome


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, (APITimeoutError, APIConnectionError, RateLimitError)):
        return True
    if isinstance(exc, APIStatusError):
        return exc.status_code in RETRYABLE_STATUS_CODES
    return False


@retry(
    retry=retry_if_exception(_is_retryable),
    wait=wait_random_exponential(multiplier=1, max=30),
    stop=stop_after_attempt(GENERATION_RETRY_ATTEMPTS),
    reraise=True,
)
async def _complete_chat(client: AsyncOpenAI, unit: GenerationUnit, prompt: str) -> Any:
    return await client.chat.completions.create(
        model=unit.model,
        messages=[{"role": "user", "content": prompt}],
        temperature=unit.temperature,
        max_tokens=_max_tokens_for_model(unit.model),
        extra_body={"reasoning": _reasoning_for_model(unit.model)},
    )


def _max_tokens_for_model(model: str) -> int:
    if DEEPSEEK_MODEL_FRAGMENT in model:
        return DEEPSEEK_MAX_TOKENS
    return GENERATION_MAX_TOKENS


def _reasoning_for_model(model: str) -> dict[str, object]:
    if DEEPSEEK_MODEL_FRAGMENT in model:
        return {"effort": "none", "exclude": True}
    return {"effort": "low"}


def _build_prompt(question: QuestionRecord, unit: GenerationUnit) -> str:
    boilerplate = question.boilerplates.get(unit.language.value, "")
    style_directive = PERSONA_STYLE_DIRECTIVES[unit.persona]
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


def _persist_success(
    unit: GenerationUnit,
    question: QuestionRecord,
    prompt: str,
    response: Any,
    request_started: float,
) -> GenerationOutcome:
    raw_text = _response_text(response)
    raw_output = _extract_code(raw_text, unit.language)
    parse_ok, reject_reason, stripped_code = _quality_gate(
        raw_output,
        question.boilerplates.get(unit.language.value, ""),
        unit.language,
    )
    usage = _usage_dict(response)
    payload = _result_payload(
        unit,
        prompt,
        raw_output,
        stripped_code,
        parse_ok,
        reject_reason,
        usage,
        GenerationStatus.OK,
    )
    path = _write_result(unit, payload)
    return GenerationOutcome(
        unit=unit,
        status=GenerationStatus.OK,
        elapsed_seconds=time.perf_counter() - request_started,
        prompt_tokens=int(usage.get("prompt_tokens") or 0),
        completion_tokens=int(usage.get("completion_tokens") or 0),
        cost=float(usage.get("cost") or 0.0),
        parse_ok=parse_ok,
        output_path=path,
    )


def _persist_failure(
    unit: GenerationUnit,
    prompt: str,
    request_started: float,
    exc: Exception,
) -> GenerationOutcome:
    payload = _result_payload(
        unit,
        prompt,
        "",
        "",
        False,
        f"{type(exc).__name__}: {exc}",
        {},
        GenerationStatus.FAILED,
    )
    path = _write_result(unit, payload)
    return GenerationOutcome(
        unit=unit,
        status=GenerationStatus.FAILED,
        elapsed_seconds=time.perf_counter() - request_started,
        prompt_tokens=0,
        completion_tokens=0,
        cost=0.0,
        parse_ok=False,
        output_path=path,
    )


def _quality_gate(
    raw_output: str,
    boilerplate: str,
    language: Language,
) -> tuple[bool, str | None, str]:
    if not raw_output.strip():
        return False, "empty_raw_output", ""
    try:
        stripped_code = strip_solution_body(raw_output, boilerplate, language)
    except (SourceParseError, ValueError) as exc:
        return False, f"strip_failed: {exc}", ""
    if not stripped_code.strip():
        return False, "empty_stripped_code", stripped_code
    if not MIN_STRIPPED_CHAR_COUNT <= len(stripped_code) <= MAX_STRIPPED_CHAR_COUNT:
        return False, "length_sanity_failed", stripped_code
    if not parses_source(stripped_code, language):
        return False, "tree_sitter_parse_failed", stripped_code
    return True, None, stripped_code


def _response_text(response: Any) -> str:
    choices = getattr(response, "choices", None) or []
    if not choices:
        return ""
    message = choices[0].message
    return str(getattr(message, "content", None) or "")


def _extract_code(raw_text: str, language: Language) -> str:
    fences = CODE_FENCE_PATTERN.findall(raw_text)
    if fences:
        return max(fences, key=len).strip()
    lines = raw_text.strip().splitlines()
    code_start_pattern = (
        PYTHON_CODE_START_PATTERN
        if language is Language.PYTHON
        else CPP_CODE_START_PATTERN
    )
    code_start = next(
        (index for index, line in enumerate(lines) if code_start_pattern.match(line)),
        0,
    )
    code_lines = lines[code_start:]
    prose_start = next(
        (
            index
            for index, line in enumerate(code_lines)
            if TRAILING_PROSE_PATTERN.match(line)
        ),
        len(code_lines),
    )
    return "\n".join(code_lines[:prose_start]).strip()


def _usage_dict(response: Any) -> dict[str, Any]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    return dict(usage)


def _result_payload(
    unit: GenerationUnit,
    prompt: str,
    raw_output: str,
    stripped_code: str,
    parse_ok: bool,
    reject_reason: str | None,
    usage: Mapping[str, Any],
    status: GenerationStatus,
) -> dict[str, Any]:
    return {
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
        "status": status.value,
        "token_usage": dict(usage),
        "cost": usage.get("cost"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _write_result(unit: GenerationUnit, payload: Mapping[str, Any]) -> Path:
    path = _unit_output_path(unit)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(".json.tmp")
    temp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temp_path.replace(path)
    return path


def _append_progress(
    outcome: GenerationOutcome,
    index: int,
    total: int,
    started: float,
) -> None:
    AI_SOLUTIONS_DIR.mkdir(parents=True, exist_ok=True)
    elapsed = time.perf_counter() - started
    record = {
        "qid": outcome.unit.question_id,
        "language": outcome.unit.language.value,
        "persona": outcome.unit.persona.value,
        "model": outcome.unit.model,
        "status": outcome.status.value,
        "parse_ok": outcome.parse_ok,
        "elapsed_seconds": round(outcome.elapsed_seconds, 3),
        "index": index,
        "total": total,
        "run_elapsed_seconds": round(elapsed, 3),
    }
    with PROGRESS_LOG_PATH.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record) + "\n")


def _print_live_progress(
    outcomes: Sequence[GenerationOutcome],
    total: int,
    started: float,
) -> None:
    done = len(outcomes)
    failures = sum(1 for item in outcomes if item.status is GenerationStatus.FAILED)
    elapsed = time.perf_counter() - started
    remaining = (elapsed / done) * (total - done) if done else 0.0
    print(
        f"progress {done}/{total} failures={failures} "
        f"elapsed={elapsed:.1f}s est_remaining={remaining:.1f}s",
        flush=True,
    )


def _print_run_summary(outcomes: Sequence[GenerationOutcome], elapsed: float) -> None:
    parse_ok_count = sum(1 for item in outcomes if item.parse_ok)
    failed_count = sum(1 for item in outcomes if item.status is GenerationStatus.FAILED)
    total_tokens = sum(item.prompt_tokens + item.completion_tokens for item in outcomes)
    total_cost = sum(item.cost for item in outcomes)
    print("\n=== generation summary ===")
    print(f"units: {len(outcomes)}  wall-clock: {elapsed:.1f}s")
    print(f"failures: {failed_count}  parse_ok: {parse_ok_count}/{len(outcomes)}")
    print(f"tokens: {total_tokens}  cost_usd: {total_cost:.6f}")
    _print_per_model_latency(outcomes)
    sample_paths = [str(item.output_path) for item in outcomes[:3]]
    print("sample paths:")
    for path in sample_paths:
        print(f"  {path}")


def _print_per_model_latency(outcomes: Sequence[GenerationOutcome]) -> None:
    grouped: dict[str, list[float]] = {}
    for item in outcomes:
        grouped.setdefault(item.unit.model, []).append(item.elapsed_seconds)
    print("per-model latency seconds:")
    for model, values in sorted(grouped.items()):
        mean = sum(values) / len(values)
        print(f"  {model}: n={len(values)} mean={mean:.2f} min={min(values):.2f} max={max(values):.2f}")


if __name__ == "__main__":
    raise SystemExit(main())

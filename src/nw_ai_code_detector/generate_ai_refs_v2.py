from __future__ import annotations

import argparse
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
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_random_exponential

from nw_ai_code_detector.config import (
    AI_SOLUTIONS_DIR,
    AI_SOLUTIONS_V2_DIR,
    OpenRouterSettings,
    PROGRESS_LOG_V2_PATH,
    REFS_V2_REPORT_PATH,
    SELECTED_500_PATH,
    load_openrouter_settings,
    model_slugs_v2,
)
from nw_ai_code_detector.constants import (
    DEFAULT_GENERATION_CONCURRENCY,
    DEFAULT_TEMPERATURE,
    GENERATION_LANGUAGES,
    GENERATION_MAX_TOKENS,
    GENERATION_RETRY_ATTEMPTS,
    GenerationStatus,
    TEMPERATURE_MAX,
    TEMPERATURE_MIN,
)
from nw_ai_code_detector.data_load import Dataset, load_dataset
from nw_ai_code_detector.generate_ai_refs import (
    _extract_code,
    _is_retryable,
    _limit_questions,
    _load_selected_questions,
    _quality_gate,
    _response_text,
    _usage_dict,
)
from nw_ai_code_detector.openrouter_budget import (
    HEADROOM_SAFETY_FACTOR,
    HeadroomError,
    fetch_headroom,
    headroom_lines,
)
from nw_ai_code_detector.refs_v2_guard import ensure_v1_untouched
from nw_ai_code_detector.personas_v2 import (
    PERSONA_V2_ORDER,
    PersonaV2,
    REFS_V2_BANK_VERSION,
    REFS_V2_SEED_SALT,
    build_prompt_v2,
)
from nw_ai_code_detector.select_500 import EligibleQuestion
from nw_ai_code_detector.stripper import Language

ANTHROPIC_MODEL_FRAGMENT = "anthropic"
# A 403 "limit exceeded" is terminal, not transient. Retrying it is what turned one
# failure into 3,296 in the previous run: trip the breaker and drain instead.
FATAL_STATUS_CODES = frozenset({401, 402, 403})
FATAL_MESSAGE_FRAGMENTS = ("limit exceeded", "insufficient credits", "quota")
PROGRESS_EVERY = 25
# v1 measured 712 prompt / 208 completion tokens per call; round up for projection.
PROJECTION_PROMPT_TOKENS = 750
PROJECTION_COMPLETION_TOKENS = 260
USD_PER_MILLION_BY_MODEL = {
    "anthropic/claude-haiku-4.5": (1.00, 5.00),
    "openai/gpt-5.5": (5.00, 30.00),
    "google/gemini-3.7-flash": (0.75, 3.75),
}


@dataclass(frozen=True)
class GenerationUnitV2:
    question_id: str
    language: Language
    persona: PersonaV2
    model: str
    temperature: float


@dataclass(frozen=True)
class CliOptionsV2:
    limit: int | None
    concurrency: int
    dry_run: bool
    retry_failed: bool
    budget_usd: float | None


@dataclass(frozen=True)
class OutcomeV2:
    unit: GenerationUnitV2
    status: GenerationStatus
    elapsed_seconds: float
    prompt_tokens: int
    completion_tokens: int
    cost: float
    parse_ok: bool
    reject_reason: str | None
    output_path: Path


class BudgetExceeded(RuntimeError):
    pass


def _is_fatal_quota_error(exc: BaseException) -> bool:
    status = getattr(exc, "status_code", None)
    if status in FATAL_STATUS_CODES:
        return True
    text = str(exc).lower()
    return any(fragment in text for fragment in FATAL_MESSAGE_FRAGMENTS)


def main() -> int:
    options = _parse_cli_options()
    settings = load_openrouter_settings()
    models = model_slugs_v2(settings)
    selected = _limit_questions(_load_selected_questions(SELECTED_500_PATH), options.limit)
    units = _build_units(selected, models)
    pending = _pending_units(units, options.retry_failed)
    projected = _projected_cost(pending)

    print(f"refs_v2 plan: {len(units)} units, {len(pending)} pending, models={list(models)}")
    _print_assignment_summary(units)

    try:
        headroom = fetch_headroom(settings.api_key, settings.base_url)
    except HeadroomError as exc:
        print(f"ERROR: {exc}")
        return 1
    for line in headroom_lines(headroom, projected):
        print(line)

    if options.dry_run:
        _print_dry_run(pending)
        print("\ndry-run only; no paid calls made.")
        return 0

    if not pending:
        print("Nothing to generate.")
        return 0

    if not headroom.covers(projected):
        print(
            f"ABORT: effective headroom ${headroom.effective_remaining:,.2f} does not cover "
            f"${projected:,.2f} x{HEADROOM_SAFETY_FACTOR} safety factor."
        )
        return 1

    budget = options.budget_usd if options.budget_usd is not None else projected * HEADROOM_SAFETY_FACTOR
    print(f"budget cap for this run: ${budget:,.2f}\n")
    effective = _with_concurrency(settings, options.concurrency)
    return asyncio.run(_run(load_dataset(), pending, effective, budget))


def _parse_cli_options() -> CliOptionsV2:
    parser = argparse.ArgumentParser(description="Generate the refs_v2 AI reference bank.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=DEFAULT_GENERATION_CONCURRENCY)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--budget-usd", type=float, default=None)
    args = parser.parse_args()
    return CliOptionsV2(
        limit=args.limit,
        concurrency=args.concurrency,
        dry_run=args.dry_run,
        retry_failed=args.retry_failed,
        budget_usd=args.budget_usd,
    )


def _with_concurrency(settings: OpenRouterSettings, concurrency: int) -> OpenRouterSettings:
    return OpenRouterSettings(
        api_key=settings.api_key,
        base_url=settings.base_url,
        gemini_model=settings.gemini_model,
        deepseek_model=settings.deepseek_model,
        openai_model=settings.openai_model,
        concurrency=concurrency,
        timeout_seconds=settings.timeout_seconds,
        anthropic_model=settings.anthropic_model,
    )


def _assignment_seed_v2(question_id: str, language: str) -> int:
    digest = sha256(
        f"{REFS_V2_SEED_SALT}:{question_id}:{language}".encode("utf-8")
    ).hexdigest()
    return int(digest[:16], 16)


def _persona_model_slots(models: Sequence[str]) -> list[str]:
    if not models or len(PERSONA_V2_ORDER) % len(models) != 0:
        raise ValueError("Persona count must divide evenly across models")
    copies_each = len(PERSONA_V2_ORDER) // len(models)
    slots: list[str] = []
    for model in models:
        slots.extend([model] * copies_each)
    return slots


def _units_for_question_language(
    question_id: str,
    language: Language,
    models: Sequence[str],
) -> list[GenerationUnitV2]:
    rng = random.Random(_assignment_seed_v2(question_id, language.value))
    personas = list(PERSONA_V2_ORDER)
    rng.shuffle(personas)
    slots = _persona_model_slots(models)
    rng.shuffle(slots)
    units: list[GenerationUnitV2] = []
    for persona, model in zip(personas, slots):
        temperature = TEMPERATURE_MIN + rng.random() * (TEMPERATURE_MAX - TEMPERATURE_MIN)
        units.append(
            GenerationUnitV2(
                question_id=question_id,
                language=language,
                persona=persona,
                model=model,
                temperature=round(temperature, 2) or DEFAULT_TEMPERATURE,
            )
        )
    return units


def _build_units(
    questions: Sequence[EligibleQuestion],
    models: Sequence[str],
) -> list[GenerationUnitV2]:
    units: list[GenerationUnitV2] = []
    for question in questions:
        for language in GENERATION_LANGUAGES:
            units.extend(
                _units_for_question_language(question.question_id, language, models)
            )
    return units


def _unit_output_path(unit: GenerationUnitV2) -> Path:
    model_token = unit.model.replace("/", "_")
    filename = f"{unit.persona.value}_{model_token}.json"
    return AI_SOLUTIONS_V2_DIR / unit.question_id / unit.language.value / filename


def _pending_units(
    units: Sequence[GenerationUnitV2],
    retry_failed: bool,
) -> list[GenerationUnitV2]:
    pending: list[GenerationUnitV2] = []
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
    raw_output = payload.get("raw_output")
    return (
        payload.get("status") == GenerationStatus.FAILED.value
        or payload.get("parse_ok") is not True
        or not isinstance(raw_output, str)
        or not raw_output.strip()
    )


def _projected_cost(units: Sequence[GenerationUnitV2]) -> float:
    total = 0.0
    for unit in units:
        prompt_rate, completion_rate = USD_PER_MILLION_BY_MODEL.get(unit.model, (5.0, 30.0))
        total += PROJECTION_PROMPT_TOKENS * prompt_rate / 1_000_000
        total += PROJECTION_COMPLETION_TOKENS * completion_rate / 1_000_000
    return total


def _print_assignment_summary(units: Sequence[GenerationUnitV2]) -> None:
    by_model: dict[str, int] = {}
    by_persona: dict[str, int] = {}
    for unit in units:
        by_model[unit.model] = by_model.get(unit.model, 0) + 1
        by_persona[unit.persona.value] = by_persona.get(unit.persona.value, 0) + 1
    print("  units per model:")
    for model, count in sorted(by_model.items()):
        print(f"    {model:34s} {count}")
    print("  units per persona:")
    for persona, count in sorted(by_persona.items()):
        print(f"    {persona:34s} {count}")


def _print_dry_run(pending: Sequence[GenerationUnitV2]) -> None:
    print("\nfirst 12 pending units:")
    for unit in pending[:12]:
        print(
            f"  {unit.question_id} {unit.language.value:7s} {unit.persona.value:18s} "
            f"-> {unit.model} t={unit.temperature}"
        )


def _reasoning_for_model(model: str) -> dict[str, object]:
    # Claude bills thinking tokens as output; these are short DSA functions.
    if ANTHROPIC_MODEL_FRAGMENT in model:
        return {"enabled": False}
    return {"effort": "low"}


@retry(
    retry=retry_if_exception(_is_retryable),
    wait=wait_random_exponential(multiplier=1, max=30),
    stop=stop_after_attempt(GENERATION_RETRY_ATTEMPTS),
    reraise=True,
)
async def _complete_chat_v2(client: AsyncOpenAI, unit: GenerationUnitV2, prompt: str) -> Any:
    return await client.chat.completions.create(
        model=unit.model,
        messages=[{"role": "user", "content": prompt}],
        temperature=unit.temperature,
        max_tokens=GENERATION_MAX_TOKENS,
        extra_body={"reasoning": _reasoning_for_model(unit.model)},
    )


async def _run(
    dataset: Dataset,
    pending: Sequence[GenerationUnitV2],
    settings: OpenRouterSettings,
    budget: float,
) -> int:
    ensure_v1_untouched()
    client = AsyncOpenAI(
        api_key=settings.api_key,
        base_url=settings.base_url,
        timeout=settings.timeout_seconds,
    )
    semaphore = asyncio.Semaphore(settings.concurrency)
    state = {"spent": 0.0, "stopped": False, "stop_reason": ""}
    started = time.perf_counter()
    total = len(pending)
    print(f"generating {total} units at concurrency {settings.concurrency}")
    tasks = [
        asyncio.create_task(
            _generate_one(client, semaphore, dataset, unit, state, budget)
        )
        for unit in pending
    ]
    outcomes: list[OutcomeV2] = []
    for index, task in enumerate(asyncio.as_completed(tasks), start=1):
        outcome = await task
        if outcome is None:
            continue
        outcomes.append(outcome)
        _append_progress(outcome, index, total, started)
        if index % PROGRESS_EVERY == 0 or index == total:
            _print_live_progress(outcomes, total, started, state, budget)
    _print_summary(outcomes, time.perf_counter() - started, state, budget)
    _write_report(outcomes, time.perf_counter() - started)
    ensure_v1_untouched()
    return 1 if state["stopped"] else 0


async def _generate_one(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    dataset: Dataset,
    unit: GenerationUnitV2,
    state: dict[str, Any],
    budget: float,
) -> OutcomeV2 | None:
    async with semaphore:
        if state["stopped"]:
            return None
        if state["spent"] >= budget:
            state["stopped"] = True
            state["stop_reason"] = f"budget cap ${budget:.2f} reached"
            return None
        question = dataset.questions[unit.question_id]
        prompt = build_prompt_v2(question, unit.language, unit.persona)
        request_started = time.perf_counter()
        try:
            response = await _complete_chat_v2(client, unit, prompt)
            outcome = _persist_success(unit, question, prompt, response, request_started)
        except Exception as exc:
            if _is_fatal_quota_error(exc):
                if not state["stopped"]:
                    state["stopped"] = True
                    state["stop_reason"] = f"quota/auth error, run halted: {exc}"
                return None
            outcome = _persist_failure(unit, prompt, request_started, exc)
        state["spent"] += outcome.cost
        return outcome


def _persist_success(
    unit: GenerationUnitV2,
    question: Any,
    prompt: str,
    response: Any,
    request_started: float,
) -> OutcomeV2:
    raw_output = _extract_code(_response_text(response), unit.language)
    parse_ok, reject_reason, stripped_code = _quality_gate(
        raw_output,
        question.boilerplates.get(unit.language.value, ""),
        unit.language,
    )
    usage = _usage_dict(response)
    path = _write_result(
        unit,
        _payload(unit, prompt, raw_output, stripped_code, parse_ok, reject_reason, usage,
                 GenerationStatus.OK),
    )
    return OutcomeV2(
        unit=unit,
        status=GenerationStatus.OK,
        elapsed_seconds=time.perf_counter() - request_started,
        prompt_tokens=int(usage.get("prompt_tokens") or 0),
        completion_tokens=int(usage.get("completion_tokens") or 0),
        cost=float(usage.get("cost") or 0.0),
        parse_ok=parse_ok,
        reject_reason=reject_reason,
        output_path=path,
    )


def _persist_failure(
    unit: GenerationUnitV2,
    prompt: str,
    request_started: float,
    exc: Exception,
) -> OutcomeV2:
    reason = f"{type(exc).__name__}: {exc}"
    path = _write_result(
        unit,
        _payload(unit, prompt, "", "", False, reason, {}, GenerationStatus.FAILED),
    )
    return OutcomeV2(
        unit=unit,
        status=GenerationStatus.FAILED,
        elapsed_seconds=time.perf_counter() - request_started,
        prompt_tokens=0,
        completion_tokens=0,
        cost=0.0,
        parse_ok=False,
        reject_reason=reason,
        output_path=path,
    )


def _payload(
    unit: GenerationUnitV2,
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
        "question_id": unit.question_id,
        "language": unit.language.value,
        "persona": unit.persona.value,
        "model": unit.model,
        "temperature": unit.temperature,
        "bank_version": REFS_V2_BANK_VERSION,
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


def _write_result(unit: GenerationUnitV2, payload: Mapping[str, Any]) -> Path:
    path = _unit_output_path(unit)
    _assert_v2_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(".json.tmp")
    temp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temp_path.replace(path)
    return path


def _assert_v2_path(path: Path) -> None:
    resolved = path.resolve()
    if AI_SOLUTIONS_DIR.resolve() in resolved.parents:
        raise RuntimeError(f"refs_v2 refused to write into the v1 bank: {resolved}")
    if AI_SOLUTIONS_V2_DIR.resolve() not in resolved.parents:
        raise RuntimeError(f"refs_v2 output escaped its bank: {resolved}")


def _append_progress(outcome: OutcomeV2, index: int, total: int, started: float) -> None:
    PROGRESS_LOG_V2_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "qid": outcome.unit.question_id,
        "language": outcome.unit.language.value,
        "persona": outcome.unit.persona.value,
        "model": outcome.unit.model,
        "status": outcome.status.value,
        "parse_ok": outcome.parse_ok,
        "reject_reason": outcome.reject_reason,
        "cost": outcome.cost,
        "elapsed_seconds": round(outcome.elapsed_seconds, 3),
        "index": index,
        "total": total,
        "run_elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    with PROGRESS_LOG_V2_PATH.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record) + "\n")


def _print_live_progress(
    outcomes: Sequence[OutcomeV2],
    total: int,
    started: float,
    state: Mapping[str, Any],
    budget: float,
) -> None:
    done = len(outcomes)
    failures = sum(1 for item in outcomes if item.status is GenerationStatus.FAILED)
    rejects = sum(1 for item in outcomes if item.status is GenerationStatus.OK and not item.parse_ok)
    elapsed = time.perf_counter() - started
    remaining = (elapsed / done) * (total - done) if done else 0.0
    print(
        f"progress {done}/{total} failures={failures} gate_rejects={rejects} "
        f"spent=${state['spent']:.4f}/${budget:.2f} "
        f"elapsed={elapsed:.0f}s eta={remaining:.0f}s",
        flush=True,
    )


def _by_model(outcomes: Sequence[OutcomeV2]) -> dict[str, dict[str, float]]:
    grouped: dict[str, dict[str, float]] = {}
    for item in outcomes:
        row = grouped.setdefault(
            item.unit.model,
            {"n": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost": 0.0, "parse_ok": 0},
        )
        row["n"] += 1
        row["prompt_tokens"] += item.prompt_tokens
        row["completion_tokens"] += item.completion_tokens
        row["cost"] += item.cost
        row["parse_ok"] += 1 if item.parse_ok else 0
    return grouped


def _print_summary(
    outcomes: Sequence[OutcomeV2],
    elapsed: float,
    state: Mapping[str, Any],
    budget: float,
) -> None:
    grouped = _by_model(outcomes)
    total_cost = sum(row["cost"] for row in grouped.values())
    total_tokens = sum(row["prompt_tokens"] + row["completion_tokens"] for row in grouped.values())
    parse_ok = sum(int(row["parse_ok"]) for row in grouped.values())
    failures = sum(1 for item in outcomes if item.status is GenerationStatus.FAILED)
    print("\n=== refs_v2 generation summary ===")
    print(f"units {len(outcomes)}  wall-clock {elapsed:.1f}s")
    print(f"failures {failures}  parse_ok {parse_ok}/{len(outcomes)}")
    print(f"tokens {total_tokens}  cost ${total_cost:.4f}  budget ${budget:.2f}")
    if state.get("stopped"):
        print(f"!! run stopped early: {state.get('stop_reason') or 'unknown'}")
    print(f"\n{'model':34s} {'n':>5s} {'prompt':>10s} {'completion':>11s} {'cost$':>9s} {'parse_ok':>9s}")
    for model, row in sorted(grouped.items()):
        print(
            f"{model:34s} {int(row['n']):5d} {int(row['prompt_tokens']):10d} "
            f"{int(row['completion_tokens']):11d} {row['cost']:9.4f} "
            f"{int(row['parse_ok']):4d}/{int(row['n']):<4d}"
        )
    _print_reject_reasons(outcomes)


def _print_reject_reasons(outcomes: Sequence[OutcomeV2]) -> None:
    reasons: dict[str, int] = {}
    for item in outcomes:
        if item.parse_ok or item.reject_reason is None:
            continue
        key = item.reject_reason.split(":")[0]
        reasons[key] = reasons.get(key, 0) + 1
    if not reasons:
        return
    print("\nreject reasons:")
    for reason, count in sorted(reasons.items(), key=lambda pair: -pair[1]):
        print(f"  {reason:40s} {count}")


def _write_report(outcomes: Sequence[OutcomeV2], elapsed: float) -> None:
    grouped = _by_model(outcomes)
    payload = {
        "bank_version": REFS_V2_BANK_VERSION,
        "units": len(outcomes),
        "wall_clock_seconds": round(elapsed, 1),
        "total_cost_usd": sum(row["cost"] for row in grouped.values()),
        "by_model": grouped,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    REFS_V2_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REFS_V2_REPORT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())

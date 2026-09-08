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
from collections.abc import Mapping
from typing import Any

from openai import AsyncOpenAI

from nw_ai_code_detector.config import (
    AI_SOLUTIONS_DIR,
    AI_SOLUTIONS_HUMANLIKE_DIR,
    AI_SOLUTIONS_V2_DIR,
    PROGRESS_LOG_HUMANLIKE_PATH,
    SELECTED_500_PATH,
    load_openrouter_settings,
)
from nw_ai_code_detector.constants import (
    DEFAULT_TEMPERATURE,
    GENERATION_LANGUAGES,
    GenerationStatus,
    TEMPERATURE_MAX,
    TEMPERATURE_MIN,
)
from nw_ai_code_detector.data_load import load_dataset
from nw_ai_code_detector.generate_ai_refs import (
    _extract_code,
    _limit_questions,
    _load_selected_questions,
    _quality_gate,
    _response_text,
    _usage_dict,
)
from nw_ai_code_detector.generate_ai_refs_v2 import (
    GenerationUnitV2,
    _complete_chat_v2,
    _is_fatal_quota_error,
)
from nw_ai_code_detector.openrouter_budget import (
    HEADROOM_SAFETY_FACTOR,
    HeadroomError,
    fetch_headroom,
    headroom_lines,
)
from nw_ai_code_detector.personas_v2 import PersonaV2, build_prompt_v2
from nw_ai_code_detector.refs_v2_guard import ensure_v1_untouched
from nw_ai_code_detector.stripper import Language

HUMANLIKE_MODEL = "openai/gpt-5.6-luna"
HUMANLIKE_SEED_SALT = "humanlike"
USD_PER_MILLION_BY_MODEL = {
    "openai/gpt-5.6-luna": (0.20, 1.20),
    "openai/gpt-5.5": (5.00, 30.00),
    "anthropic/claude-haiku-4.5": (1.00, 5.00),
    "google/gemini-3.7-flash": (0.75, 3.75),
}
PROJECTION_PROMPT_TOKENS = 750
PROJECTION_COMPLETION_TOKENS = 300
PROGRESS_EVERY = 50


_PERSONA = PersonaV2.HUMANLIKE
_MODEL = HUMANLIKE_MODEL
_OUT_DIR = AI_SOLUTIONS_HUMANLIKE_DIR
_SALT = HUMANLIKE_SEED_SALT


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate the humanlike persona refs.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=28)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--budget-usd", type=float, default=None)
    parser.add_argument("--retry-salt", type=int, default=0)
    parser.add_argument("--persona", default=PersonaV2.HUMANLIKE.value)
    parser.add_argument("--model", default=HUMANLIKE_MODEL)
    parser.add_argument("--out-dir", default=str(AI_SOLUTIONS_HUMANLIKE_DIR))
    parser.add_argument("--salt", default=HUMANLIKE_SEED_SALT)
    args = parser.parse_args()
    global _PERSONA, _MODEL, _OUT_DIR, _SALT
    _PERSONA = PersonaV2(args.persona)
    _MODEL = args.model
    _OUT_DIR = Path(args.out_dir)
    _SALT = args.salt

    settings = load_openrouter_settings()
    questions = _limit_questions(_load_selected_questions(SELECTED_500_PATH), args.limit)
    units = [
        _unit(q.question_id, language, args.retry_salt)
        for q in questions
        for language in GENERATION_LANGUAGES
    ]
    pending = [u for u in units if _needs_work(u, args.retry_failed)]
    rate = USD_PER_MILLION_BY_MODEL.get(_MODEL, (5.00, 30.00))
    projected = len(pending) * (
        PROJECTION_PROMPT_TOKENS * rate[0] + PROJECTION_COMPLETION_TOKENS * rate[1]
    ) / 1_000_000

    print(f"plan: persona={_PERSONA.value} model={_MODEL} out={_OUT_DIR}")
    print(f"  {len(units)} units, {len(pending)} pending")
    try:
        headroom = fetch_headroom(settings.api_key, settings.base_url)
    except HeadroomError as exc:
        print(f"ERROR: {exc}")
        return 1
    for line in headroom_lines(headroom, projected):
        print(line)

    if args.dry_run:
        for unit in pending[:6]:
            print(f"  {unit.question_id} {unit.language.value} t={unit.temperature}")
        print("\ndry-run only; no paid calls made.")
        return 0
    if not pending:
        print("Nothing to generate.")
        return 0
    if not headroom.covers(projected):
        print(f"ABORT: headroom ${headroom.effective_remaining:,.2f} < "
              f"${projected:,.2f} x{HEADROOM_SAFETY_FACTOR}")
        return 1

    budget = args.budget_usd if args.budget_usd is not None else max(
        projected * HEADROOM_SAFETY_FACTOR, 0.50
    )
    print(f"budget cap: ${budget:.2f}\n")
    return asyncio.run(_run(pending, settings, args.concurrency, budget))


def _unit(question_id: str, language: Language, salt: int) -> GenerationUnitV2:
    digest = sha256(
        f"{_SALT}:{salt}:{question_id}:{language.value}".encode("utf-8")
    ).hexdigest()
    rng = random.Random(int(digest[:16], 16))
    temperature = TEMPERATURE_MIN + rng.random() * (TEMPERATURE_MAX - TEMPERATURE_MIN)
    return GenerationUnitV2(
        question_id=question_id,
        language=language,
        persona=_PERSONA,
        model=_MODEL,
        temperature=round(temperature, 2) or DEFAULT_TEMPERATURE,
    )


def _path(unit: GenerationUnitV2) -> Path:
    name = f"{unit.persona.value}_{unit.model.replace('/', '_')}.json"
    return _OUT_DIR / unit.question_id / unit.language.value / name


def _needs_work(unit: GenerationUnitV2, retry_failed: bool) -> bool:
    path = _path(unit)
    if not path.is_file():
        return True
    if not retry_failed:
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return True
    return payload.get("parse_ok") is not True


async def _run(pending, settings, concurrency: int, budget: float) -> int:
    ensure_v1_untouched()
    client = AsyncOpenAI(
        api_key=settings.api_key, base_url=settings.base_url, timeout=settings.timeout_seconds
    )
    sem = asyncio.Semaphore(concurrency)
    state = {"spent": 0.0, "stopped": False, "reason": ""}
    dataset = load_dataset()
    started = time.perf_counter()
    total = len(pending)
    print(f"generating {total} units at concurrency {concurrency}")

    async def one(unit):
        async with sem:
            if state["stopped"]:
                return None
            if state["spent"] >= budget:
                state["stopped"] = True
                state["reason"] = f"budget cap ${budget:.2f} reached"
                return None
            question = dataset.questions[unit.question_id]
            prompt = build_prompt_v2(question, unit.language, unit.persona)
            try:
                response = await _complete_chat_v2(client, unit, prompt)
            except Exception as exc:
                if _is_fatal_quota_error(exc):
                    state["stopped"] = True
                    state["reason"] = f"quota/auth error: {exc}"
                    return None
                return _persist(unit, prompt, "", {}, GenerationStatus.FAILED,
                                f"{type(exc).__name__}: {exc}", question)
            usage = _usage_dict(response)
            state["spent"] += float(usage.get("cost") or 0.0)
            raw = _extract_code(_response_text(response), unit.language)
            return _persist(unit, prompt, raw, usage, GenerationStatus.OK, None, question)

    done = ok = 0
    for task in asyncio.as_completed([asyncio.create_task(one(u)) for u in pending]):
        result = await task
        if result is None:
            continue
        done += 1
        ok += 1 if result else 0
        if done % PROGRESS_EVERY == 0 or done == total:
            print(f"progress {done}/{total} parse_ok={ok} spent=${state['spent']:.4f}/${budget:.2f} "
                  f"elapsed={time.perf_counter()-started:.0f}s", flush=True)
    print(f"\n=== {_PERSONA.value} / {_MODEL} summary ===")
    print(f"units {done}  parse_ok {ok}  cost ${state['spent']:.4f}  "
          f"wall {time.perf_counter()-started:.0f}s")
    if state["stopped"]:
        print(f"!! stopped early: {state['reason']}")
    ensure_v1_untouched()
    return 0


def _persist(unit, prompt, raw, usage: Mapping[str, Any], status, reason, question) -> bool:
    parse_ok, reject, stripped = (False, reason, "")
    if raw.strip():
        parse_ok, reject, stripped = _quality_gate(
            raw, question.boilerplates.get(unit.language.value, ""), unit.language
        )
    elif reason is None:
        reject = "empty_raw_output"
    payload = {
        "qid": unit.question_id, "question_id": unit.question_id,
        "language": unit.language.value, "persona": unit.persona.value,
        "model": unit.model, "temperature": unit.temperature,
        "bank_version": f"gen_{_PERSONA.value}_{_MODEL.replace('/', '_')}", "prompt": prompt,
        "raw_output": raw, "stripped_code": stripped, "parse_ok": parse_ok,
        "reject_reason": reject, "status": status.value,
        "token_usage": dict(usage), "cost": usage.get("cost"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    path = _path(unit)
    resolved = path.resolve()
    for forbidden in (AI_SOLUTIONS_DIR, AI_SOLUTIONS_V2_DIR):
        if forbidden.resolve() in resolved.parents:
            raise RuntimeError(f"humanlike refused to write into {forbidden}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)
    progress_path = _OUT_DIR / "_progress.jsonl"
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    with progress_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"qid": unit.question_id, "language": unit.language.value,
                                 "parse_ok": parse_ok, "reject_reason": reject}) + "\n")
    return parse_ok


if __name__ == "__main__":
    raise SystemExit(main())

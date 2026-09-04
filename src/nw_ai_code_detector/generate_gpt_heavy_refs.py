from __future__ import annotations

import argparse
import asyncio
import json
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Mapping, Sequence
from typing import Any

import faiss
import numpy as np
from openai import AsyncOpenAI

from nw_ai_code_detector.build_model_dataset_v2 import GPT_HEAVY_CANDIDATES_DIR
from nw_ai_code_detector.config import (
    AI_SOLUTIONS_DIR,
    OUTPUTS_DIR,
    OpenRouterSettings,
    REFERENCE_INDEX_DIR,
    SELECTED_500_PATH,
    load_openrouter_settings,
    load_voyage_settings,
)
from nw_ai_code_detector.constants import (
    DEFAULT_GENERATION_CONCURRENCY,
    GENERATION_LANGUAGES,
    GenerationStatus,
    MUST_PASS_EXAMPLES_DIRECTIVE,
    NO_COMMENTS_DIRECTIVE,
    STRIPPED_CODE_FIELD,
)
from nw_ai_code_detector.data_load import Dataset, QuestionRecord, load_dataset
from nw_ai_code_detector.embedder import EmbeddingBatch, VoyageEmbedder, cached_vector_for_text
from nw_ai_code_detector.generate_ai_refs import (
    GenerationUnit,
    _complete_chat,
    _extract_code,
    _quality_gate,
    _response_text,
    _usage_dict,
)
from nw_ai_code_detector.index import ClusterKey
from nw_ai_code_detector.select_500 import EligibleQuestion
from nw_ai_code_detector.stripper import Language

GPT_HEAVY_REF_COUNT_PER_CLUSTER = 2
GPT_HEAVY_EXPECTED_CLUSTER_COUNT = 1000
ORIGINAL_REF_COUNT_PER_CLUSTER = 6
GPT_HEAVY_PROGRESS_PATH = GPT_HEAVY_CANDIDATES_DIR / "_progress.jsonl"
GPT_HEAVY_REPORT_PATH = OUTPUTS_DIR / "gpt_heavy_refs_report.json"
GPT_HEAVY_PROMPT_VERSION = "gpt_heavy_simple_full_boilerplate_v2"
SIMPLE_EXAM_DIRECTIVE = "Solve this problem. Complete the function."
GPT_HEAVY_OUTPUT_SHAPE_DIRECTIVE = (
    "Return the entire completed boilerplate class exactly as provided, with "
    "only the target function body filled in. Do NOT output only the function "
    "body. Do NOT omit the class wrapper or function signature. "
    "Do NOT write a main() function, do NOT write an if __name__ == '__main__' "
    "block, do NOT read input, and do NOT add any driver/test/harness code. "
    "If the provided function is expected to print or display output, print "
    "from inside that function; otherwise return the required value. Output "
    "only the completed code."
)


@dataclass(frozen=True)
class GptHeavyUnit:
    question_id: str
    language: Language
    variant: int
    model: str
    temperature: float


@dataclass(frozen=True)
class GptHeavyOutcome:
    unit: GptHeavyUnit
    status: str
    parse_ok: bool
    generated_this_run: bool
    output_path: Path
    prompt_tokens: int
    completion_tokens: int
    cost: float
    reject_reason: str | None


@dataclass(frozen=True)
class GptHeavyReport:
    total_units: int
    generated_this_run: int
    existing_ok: int
    skipped_failed: int
    parse_ok: int
    generated_by_language: dict[str, int]
    parse_ok_by_language: dict[str, int]
    skipped_by_language: dict[str, int]
    openrouter_prompt_tokens: int
    openrouter_completion_tokens: int
    openrouter_cost_usd: float
    voyage_billed_tokens: int
    voyage_cost_usd: float
    voyage_cache_hits: int
    voyage_cache_misses: int
    clusters_rebuilt: int
    refs_per_cluster_distribution: dict[str, int]


def main() -> int:
    options = _parse_args()
    dataset = load_dataset()
    selected = _load_selected_questions()
    settings = load_openrouter_settings()
    if settings.openai_model != "openai/gpt-5.5":
        raise RuntimeError(f"Expected MODEL_OPENAI=openai/gpt-5.5, got {settings.openai_model}")
    units = _build_units(selected, settings.openai_model)
    units = units[: options.limit] if options.limit is not None else units
    outcomes = asyncio.run(
        _run_generation(dataset, units, settings, options.concurrency, options.retry_failed)
    )
    all_payloads = _load_all_ok_payloads()
    embedding_batch = _embed_ok_payloads(all_payloads)
    rebuilt = _rebuild_clusters(all_payloads)
    distribution = _refs_per_cluster_distribution()
    report = _build_report(outcomes, embedding_batch, rebuilt, distribution)
    _write_json(GPT_HEAVY_REPORT_PATH, report.__dict__)
    _print_report(report)
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=DEFAULT_GENERATION_CONCURRENCY)
    parser.add_argument("--retry-failed", action="store_true")
    return parser.parse_args()


def _load_selected_questions() -> list[EligibleQuestion]:
    payload = json.loads(SELECTED_500_PATH.read_text(encoding="utf-8"))
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


def _build_units(
    selected: Sequence[EligibleQuestion],
    model: str,
) -> list[GptHeavyUnit]:
    units = []
    for question in selected:
        for language in GENERATION_LANGUAGES:
            for variant in range(1, GPT_HEAVY_REF_COUNT_PER_CLUSTER + 1):
                units.append(
                    GptHeavyUnit(
                        question_id=question.question_id,
                        language=language,
                        variant=variant,
                        model=model,
                        temperature=0.8,
                    )
                )
    return units


async def _run_generation(
    dataset: Dataset,
    units: Sequence[GptHeavyUnit],
    settings: OpenRouterSettings,
    concurrency: int,
    retry_failed: bool,
) -> list[GptHeavyOutcome]:
    semaphore = asyncio.Semaphore(min(concurrency, settings.concurrency))
    client = AsyncOpenAI(
        api_key=settings.api_key,
        base_url=settings.base_url,
        timeout=settings.timeout_seconds,
    )
    started = time.perf_counter()
    tasks = [
        asyncio.create_task(
            _handle_unit(
                client,
                semaphore,
                dataset,
                unit,
                retry_failed,
                index,
                len(units),
                started,
            )
        )
        for index, unit in enumerate(units, start=1)
    ]
    outcomes = []
    for task in asyncio.as_completed(tasks):
        outcomes.append(await task)
        _print_progress(outcomes, len(units), started)
    return outcomes


async def _handle_unit(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    dataset: Dataset,
    unit: GptHeavyUnit,
    retry_failed: bool,
    index: int,
    total: int,
    started: float,
) -> GptHeavyOutcome:
    existing = _existing_outcome(unit, retry_failed)
    if existing is not None:
        return existing
    async with semaphore:
        return await _generate_with_retry(client, dataset, unit, index, total, started)


def _existing_outcome(unit: GptHeavyUnit, retry_failed: bool) -> GptHeavyOutcome | None:
    path = _unit_path(unit)
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    status = str(payload.get("status") or "")
    parse_ok = payload.get("parse_ok") is True
    if status == GenerationStatus.OK.value and parse_ok and str(payload.get(STRIPPED_CODE_FIELD) or "").strip():
        return GptHeavyOutcome(unit, status, True, False, path, 0, 0, 0.0, None)
    if (
        status == GenerationStatus.FAILED.value
        and payload.get("prompt_version") == GPT_HEAVY_PROMPT_VERSION
        and not retry_failed
    ):
        return GptHeavyOutcome(
            unit,
            status,
            False,
            False,
            path,
            0,
            0,
            0.0,
            str(payload.get("reject_reason") or "previous_failed_skip"),
        )
    return None


async def _generate_with_retry(
    client: AsyncOpenAI,
    dataset: Dataset,
    unit: GptHeavyUnit,
    index: int,
    total: int,
    started: float,
) -> GptHeavyOutcome:
    last_payload: dict[str, Any] | None = None
    for attempt in (1, 2):
        payload, outcome = await _generate_once(client, dataset, unit, attempt)
        last_payload = payload
        if outcome.parse_ok:
            _write_payload(unit, payload)
            _append_progress(outcome, index, total, started)
            return outcome
    failed_payload = last_payload or _failure_payload(unit, "missing_generation_payload")
    _write_payload(unit, failed_payload)
    outcome = _outcome_from_payload(unit, failed_payload, True)
    _append_progress(outcome, index, total, started)
    return outcome


async def _generate_once(
    client: AsyncOpenAI,
    dataset: Dataset,
    unit: GptHeavyUnit,
    attempt: int,
) -> tuple[dict[str, Any], GptHeavyOutcome]:
    question = dataset.questions[unit.question_id]
    prompt = _build_prompt(question, unit)
    try:
        started = time.perf_counter()
        generation_unit = _as_generation_unit(unit)
        response = await _complete_chat(client, generation_unit, prompt)
        raw_output = _extract_code(_response_text(response), unit.language)
        parse_ok, reject_reason, stripped_code = _quality_gate(
            raw_output,
            question.boilerplates.get(unit.language.value, ""),
            unit.language,
        )
        usage = _usage_dict(response)
        if not parse_ok and attempt == 1:
            reject_reason = f"{reject_reason}; retrying once"
        payload = _payload(unit, prompt, raw_output, stripped_code, parse_ok, reject_reason, usage)
        outcome = GptHeavyOutcome(
            unit=unit,
            status=str(payload["status"]),
            parse_ok=parse_ok,
            generated_this_run=True,
            output_path=_unit_path(unit),
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            cost=float(usage.get("cost") or 0.0),
            reject_reason=reject_reason,
        )
        _ = time.perf_counter() - started
        return payload, outcome
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
        if attempt == 1:
            reason = f"{reason}; retrying once"
        payload = _failure_payload(unit, reason)
        return payload, _outcome_from_payload(unit, payload, True)


def _build_prompt(question: QuestionRecord, unit: GptHeavyUnit) -> str:
    boilerplate = question.boilerplates.get(unit.language.value, "")
    return "\n\n".join(
        [
            question.statement_content.strip(),
            f"Target language: {unit.language.value}",
            f"Boilerplate:\n{boilerplate}",
            f"Style: {SIMPLE_EXAM_DIRECTIVE}",
            MUST_PASS_EXAMPLES_DIRECTIVE,
            NO_COMMENTS_DIRECTIVE,
            GPT_HEAVY_OUTPUT_SHAPE_DIRECTIVE,
        ]
    )


def _payload(
    unit: GptHeavyUnit,
    prompt: str,
    raw_output: str,
    stripped_code: str,
    parse_ok: bool,
    reject_reason: str | None,
    usage: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "qid": unit.question_id,
        "language": unit.language.value,
        "source": "gpt_heavy_v2",
        "prompt_version": GPT_HEAVY_PROMPT_VERSION,
        "persona": "simple_exam",
        "variant": unit.variant,
        "model": unit.model,
        "temperature": unit.temperature,
        "prompt": prompt,
        "raw_output": raw_output,
        "stripped_code": stripped_code,
        "parse_ok": parse_ok,
        "reject_reason": reject_reason,
        "status": GenerationStatus.OK.value if parse_ok else GenerationStatus.FAILED.value,
        "token_usage": dict(usage),
        "cost": usage.get("cost"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _failure_payload(unit: GptHeavyUnit, reject_reason: str) -> dict[str, Any]:
    return _payload(unit, "", "", "", False, reject_reason, {})


def _outcome_from_payload(
    unit: GptHeavyUnit,
    payload: Mapping[str, Any],
    generated_this_run: bool,
) -> GptHeavyOutcome:
    usage = payload.get("token_usage")
    usage_map = usage if isinstance(usage, dict) else {}
    return GptHeavyOutcome(
        unit=unit,
        status=str(payload.get("status") or GenerationStatus.FAILED.value),
        parse_ok=payload.get("parse_ok") is True,
        generated_this_run=generated_this_run,
        output_path=_unit_path(unit),
        prompt_tokens=int(usage_map.get("prompt_tokens") or 0),
        completion_tokens=int(usage_map.get("completion_tokens") or 0),
        cost=float(usage_map.get("cost") or payload.get("cost") or 0.0),
        reject_reason=str(payload.get("reject_reason") or "") or None,
    )


def _as_generation_unit(unit: GptHeavyUnit) -> GenerationUnit:
    from nw_ai_code_detector.constants import Persona

    return GenerationUnit(
        question_id=unit.question_id,
        language=unit.language,
        persona=Persona.NAIVE_DUMP,
        model=unit.model,
        temperature=unit.temperature,
    )


def _unit_path(unit: GptHeavyUnit) -> Path:
    filename = f"gpt_5_5_simple_{unit.variant:02d}.json"
    return GPT_HEAVY_CANDIDATES_DIR / unit.question_id / unit.language.value / filename


def _write_payload(unit: GptHeavyUnit, payload: Mapping[str, Any]) -> Path:
    path = _unit_path(unit)
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(path, payload)
    return path


def _append_progress(
    outcome: GptHeavyOutcome,
    index: int,
    total: int,
    started: float,
) -> None:
    GPT_HEAVY_CANDIDATES_DIR.mkdir(parents=True, exist_ok=True)
    record = {
        "qid": outcome.unit.question_id,
        "language": outcome.unit.language.value,
        "variant": outcome.unit.variant,
        "model": outcome.unit.model,
        "status": outcome.status,
        "parse_ok": outcome.parse_ok,
        "generated_this_run": outcome.generated_this_run,
        "reject_reason": outcome.reject_reason,
        "index": index,
        "total": total,
        "run_elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    with GPT_HEAVY_PROGRESS_PATH.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record) + "\n")


def _print_progress(
    outcomes: Sequence[GptHeavyOutcome],
    total: int,
    started: float,
) -> None:
    done = len(outcomes)
    if done % 25 != 0 and done != total:
        return
    generated = sum(item.generated_this_run for item in outcomes)
    ok = sum(item.parse_ok for item in outcomes)
    skipped = sum(1 for item in outcomes if not item.parse_ok)
    elapsed = time.perf_counter() - started
    print(
        f"gpt-heavy {done}/{total} generated={generated} ok={ok} "
        f"skipped={skipped} elapsed={elapsed:.1f}s",
        flush=True,
    )


def _load_all_ok_payloads() -> list[dict[str, Any]]:
    payloads = []
    for path in _simple_variant_paths():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("parse_ok") is True and str(payload.get(STRIPPED_CODE_FIELD) or "").strip():
            payloads.append(payload)
    return payloads


def _simple_variant_paths() -> list[Path]:
    paths: list[Path] = []
    for variant in range(1, GPT_HEAVY_REF_COUNT_PER_CLUSTER + 1):
        paths.extend(
            sorted(GPT_HEAVY_CANDIDATES_DIR.rglob(f"gpt_5_5_simple_{variant:02d}.json"))
        )
    return sorted(paths)


def _embed_ok_payloads(payloads: Sequence[Mapping[str, Any]]) -> EmbeddingBatch:
    texts = list(dict.fromkeys(str(item[STRIPPED_CODE_FIELD]) for item in payloads))
    if not texts:
        return EmbeddingBatch(vectors=(), billed_tokens=0, cache_hits=0, cache_misses=0, cost_usd=0.0)
    embedder = VoyageEmbedder(load_voyage_settings())
    return embedder.embed_texts(texts)


def _rebuild_clusters(extra_payloads: Sequence[Mapping[str, Any]]) -> int:
    grouped = _reference_texts_by_cluster(extra_payloads)
    manifest = json.loads((REFERENCE_INDEX_DIR / "manifest.json").read_text(encoding="utf-8"))
    cluster_manifest = manifest.get("clusters")
    if not isinstance(cluster_manifest, dict):
        raise RuntimeError("reference_index manifest missing clusters")
    rebuilt = 0
    for key, texts in sorted(grouped.items(), key=lambda item: item[0].token):
        vectors = _cached_matrix(texts)
        _write_cluster(key, vectors)
        cluster_manifest[key.token] = {
            "question_id": key.question_id,
            "language": key.language,
            "vector_ids": list(range(len(texts))),
            "count": len(texts),
        }
        rebuilt += 1
    _write_json(REFERENCE_INDEX_DIR / "manifest.json", manifest)
    return rebuilt


def _reference_texts_by_cluster(
    extra_payloads: Sequence[Mapping[str, Any]],
) -> dict[ClusterKey, list[str]]:
    grouped: dict[ClusterKey, list[str]] = defaultdict(list)
    for path in sorted(AI_SOLUTIONS_DIR.rglob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("parse_ok") is not True:
            continue
        text = payload.get(STRIPPED_CODE_FIELD)
        if not isinstance(text, str) or not text.strip():
            continue
        key = ClusterKey(str(payload.get("qid") or ""), str(payload.get("language") or ""))
        grouped[key].append(text)
    for payload in extra_payloads:
        key = ClusterKey(str(payload.get("qid") or ""), str(payload.get("language") or ""))
        grouped[key].append(str(payload[STRIPPED_CODE_FIELD]))
    _assert_cluster_text_counts(grouped)
    return grouped


def _assert_cluster_text_counts(grouped: Mapping[ClusterKey, Sequence[str]]) -> None:
    if len(grouped) != GPT_HEAVY_EXPECTED_CLUSTER_COUNT:
        raise RuntimeError(f"Expected 1000 clusters, got {len(grouped)}")
    for key, texts in grouped.items():
        if len(texts) < ORIGINAL_REF_COUNT_PER_CLUSTER:
            raise RuntimeError(f"Cluster {key.token} has fewer than original refs: {len(texts)}")
        if len(texts) > ORIGINAL_REF_COUNT_PER_CLUSTER + GPT_HEAVY_REF_COUNT_PER_CLUSTER:
            raise RuntimeError(f"Cluster {key.token} has too many refs: {len(texts)}")


def _cached_matrix(texts: Sequence[str]) -> np.ndarray:
    vectors = []
    for text in texts:
        vector = cached_vector_for_text(text)
        if vector is None:
            raise RuntimeError("Reference text is missing from embedding cache")
        vectors.append(vector)
    return np.asarray(vectors, dtype=np.float32)


def _write_cluster(key: ClusterKey, vectors: np.ndarray) -> None:
    index = faiss.IndexFlatIP(int(vectors.shape[1]))
    index.add(np.ascontiguousarray(vectors, dtype=np.float32))
    stem = key.token.replace(":", "__")
    faiss.write_index(index, str(REFERENCE_INDEX_DIR / f"{stem}.faiss"))
    np.save(REFERENCE_INDEX_DIR / f"{stem}.npy", vectors)


def _refs_per_cluster_distribution() -> dict[str, int]:
    manifest = json.loads((REFERENCE_INDEX_DIR / "manifest.json").read_text(encoding="utf-8"))
    counts = Counter(str(item["count"]) for item in manifest["clusters"].values())
    return dict(sorted(counts.items()))


def _build_report(
    outcomes: Sequence[GptHeavyOutcome],
    embedding_batch: EmbeddingBatch,
    rebuilt: int,
    distribution: Mapping[str, int],
) -> GptHeavyReport:
    generated = [item for item in outcomes if item.generated_this_run]
    generated_by_language = Counter(item.unit.language.value for item in generated)
    ok_by_language = Counter(item.unit.language.value for item in outcomes if item.parse_ok)
    skipped_by_language = Counter(item.unit.language.value for item in outcomes if not item.parse_ok)
    return GptHeavyReport(
        total_units=len(outcomes),
        generated_this_run=len(generated),
        existing_ok=sum(1 for item in outcomes if item.parse_ok and not item.generated_this_run),
        skipped_failed=sum(1 for item in outcomes if not item.parse_ok),
        parse_ok=sum(item.parse_ok for item in outcomes),
        generated_by_language=dict(generated_by_language),
        parse_ok_by_language=dict(ok_by_language),
        skipped_by_language=dict(skipped_by_language),
        openrouter_prompt_tokens=sum(item.prompt_tokens for item in generated),
        openrouter_completion_tokens=sum(item.completion_tokens for item in generated),
        openrouter_cost_usd=sum(item.cost for item in generated),
        voyage_billed_tokens=embedding_batch.billed_tokens,
        voyage_cost_usd=embedding_batch.cost_usd,
        voyage_cache_hits=embedding_batch.cache_hits,
        voyage_cache_misses=embedding_batch.cache_misses,
        clusters_rebuilt=rebuilt,
        refs_per_cluster_distribution=dict(distribution),
    )


def _print_report(report: GptHeavyReport) -> None:
    print(json.dumps(report.__dict__, indent=2))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(".json.tmp")
    temp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temp_path.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())

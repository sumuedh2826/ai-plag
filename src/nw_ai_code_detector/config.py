from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from nw_ai_code_detector.constants import (
    DEFAULT_EMBED_BATCH_SIZE,
    DEFAULT_EMBED_CONCURRENCY,
    DEFAULT_GENERATION_CONCURRENCY,
    GENERATION_TIMEOUT_SECONDS,
    INCLUDE_DEEPSEEK,
    VOYAGE_CODE_3_MODEL,
    VOYAGE_EMBED_TIMEOUT_SECONDS,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
OUTPUTS_DIR = REPO_ROOT / "outputs"
AI_SOLUTIONS_DIR = DATA_DIR / "ai_solutions"
EVAL_AI_SOLUTIONS_DIR = DATA_DIR / "eval_ai_solutions"
EMBEDDING_CACHE_DIR = DATA_DIR / "embedding_cache"
REFERENCE_INDEX_DIR = OUTPUTS_DIR / "reference_index"
EVAL_SCORES_PATH = OUTPUTS_DIR / "eval_scores.json"
CENTROID_ALL_HUMANS_SCORES_PATH = OUTPUTS_DIR / "centroid_all_humans_scores.json"
CENTROID_ALL_HUMANS_REPORT_PATH = OUTPUTS_DIR / "centroid_all_humans_report.md"
CENTROID_ALL_HUMANS_SUMMARY_PATH = OUTPUTS_DIR / "centroid_all_humans_summary.csv"
CENTROID_ALL_HUMANS_PAIR_DELTAS_PATH = OUTPUTS_DIR / "centroid_all_humans_pair_deltas.csv"
CENTROID_ALL_HUMANS_COVERAGE_PATH = OUTPUTS_DIR / "centroid_all_humans_coverage.json"
CENTROID_CACHED_HUMANS_SCORES_PATH = OUTPUTS_DIR / "centroid_cached_humans_scores.json"
CENTROID_CACHED_HUMANS_REPORT_PATH = OUTPUTS_DIR / "centroid_cached_humans_report.md"
CENTROID_CACHED_HUMANS_SUMMARY_PATH = OUTPUTS_DIR / "centroid_cached_humans_summary.csv"
CENTROID_CACHED_HUMANS_PAIR_DELTAS_PATH = OUTPUTS_DIR / "centroid_cached_humans_pair_deltas.csv"
CENTROID_CACHED_HUMANS_COVERAGE_PATH = OUTPUTS_DIR / "centroid_cached_humans_coverage.json"
PROGRESS_LOG_PATH = AI_SOLUTIONS_DIR / "_progress.jsonl"
SELECTED_500_PATH = OUTPUTS_DIR / "selected_500.json"
CANONICALITY_SPLIT_DIR = OUTPUTS_DIR / "canonicality_dataset_split_v1"
MODEL_DATASET_DIR = OUTPUTS_DIR / "model_dataset_v2"
AI_REFERENCE_EVALUATION_DIR = OUTPUTS_DIR / "ai_reference_scores_v2"
ENV_PATH = REPO_ROOT / ".env"


@dataclass(frozen=True)
class OpenRouterSettings:
    api_key: str
    base_url: str
    gemini_model: str
    deepseek_model: str
    openai_model: str
    concurrency: int
    timeout_seconds: int


def load_openrouter_settings() -> OpenRouterSettings:
    load_dotenv(ENV_PATH)
    api_key = _required_env("OPENROUTER_API_KEY")
    concurrency = _int_env("GENERATION_CONCURRENCY", DEFAULT_GENERATION_CONCURRENCY)
    timeout_seconds = _int_env(
        "GENERATION_TIMEOUT_SECONDS",
        GENERATION_TIMEOUT_SECONDS,
    )
    return OpenRouterSettings(
        api_key=api_key,
        base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        gemini_model=_required_env("MODEL_GEMINI"),
        deepseek_model=_required_env("MODEL_DEEPSEEK"),
        openai_model=_required_env("MODEL_OPENAI"),
        concurrency=concurrency,
        timeout_seconds=timeout_seconds,
    )


@dataclass(frozen=True)
class VoyageSettings:
    api_key: str
    model: str
    batch_size: int
    concurrency: int
    timeout_seconds: int


def load_voyage_settings() -> VoyageSettings:
    load_dotenv(ENV_PATH)
    return VoyageSettings(
        api_key=_required_env("VOYAGE_API_KEY"),
        model=VOYAGE_CODE_3_MODEL,
        batch_size=_int_env("VOYAGE_EMBED_BATCH_SIZE", DEFAULT_EMBED_BATCH_SIZE),
        concurrency=_int_env("VOYAGE_EMBED_CONCURRENCY", DEFAULT_EMBED_CONCURRENCY),
        timeout_seconds=_int_env(
            "VOYAGE_EMBED_TIMEOUT_SECONDS",
            VOYAGE_EMBED_TIMEOUT_SECONDS,
        ),
    )


def model_slugs(settings: OpenRouterSettings) -> tuple[str, ...]:
    models = [settings.gemini_model, settings.openai_model]
    if INCLUDE_DEEPSEEK:
        models = [
            settings.gemini_model,
            settings.deepseek_model,
            settings.openai_model,
        ]
    return tuple(models)


def _required_env(name: str) -> str:
    value = os.getenv(name, "").strip().strip('"')
    if not value:
        raise ValueError(f"Missing required environment variable {name}")
    return value


def _int_env(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if not raw_value:
        return default
    return int(raw_value)

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
# v1 bank: retained as the reported baseline and for v1-maintenance scripts.
REFERENCE_INDEX_V1_DIR = OUTPUTS_DIR / "reference_index"
# PRODUCTION bank. Serving code reads REFERENCE_INDEX_DIR, so the cutover is here.
# Currently D10-swap (D10 minus v1:complete_function, plus the d11 hardened ref).
# D10 remains on disk, pinned below, as the fallback.
REFERENCE_INDEX_DIR = OUTPUTS_DIR / "reference_index_d10_swap"
# Side-cars of whichever bank is live. Serving reads these, never the D10-pinned ones.
ACTIVE_BANK_HASHES_PATH = REFERENCE_INDEX_DIR / "reference_hashes.json"
ACTIVE_BANK_REFERENCES_PATH = REFERENCE_INDEX_DIR / "reference_texts.json"
EVAL_SCORES_PATH = OUTPUTS_DIR / "eval_scores.json"
PROGRESS_LOG_PATH = AI_SOLUTIONS_DIR / "_progress.jsonl"
SELECTED_500_PATH = OUTPUTS_DIR / "selected_500.json"
CANONICALITY_SPLIT_DIR = OUTPUTS_DIR / "canonicality_dataset_split_v1"
MODEL_DATASET_DIR = OUTPUTS_DIR / "model_dataset_v2"
AI_REFERENCE_EVALUATION_DIR = OUTPUTS_DIR / "ai_reference_scores_v2"
CANONICALITY_ELIGIBILITY_DIR = OUTPUTS_DIR / "canonicality_eligibility_v1"
# Token counts and very-short floors (CPP>=70, PYTHON>=55) live here.
# Raised from CPP 80 / PYTHON 60; re-check against a labeled test set.
SIGNIFICANT_TOKEN_ELIGIBILITY_DIR = OUTPUTS_DIR / "canonicality_eligibility_tokens"
STYLE_SIGNALS_DIR = OUTPUTS_DIR / "style_signals_v0"
DISCOUNT_LAYER_DIR = OUTPUTS_DIR / "discount_layer_v0"
DETECTOR_CONSENSUS_DIR = OUTPUTS_DIR / "detector_consensus"
DETECTOR_CONSENSUS_LABELS_PATH = (
    DETECTOR_CONSENSUS_DIR / "proxy_labels_detector_consensus.jsonl"
)
DETECTOR_CONSENSUS_CACHE_DIR = DETECTOR_CONSENSUS_DIR / "api_cache"
DETECTOR_CONSENSUS_PROGRESS_PATH = DETECTOR_CONSENSUS_DIR / "_progress.jsonl"
DETECTOR_CONSENSUS_MANUAL_BATCH_DIR = DETECTOR_CONSENSUS_DIR / "manual_batch"
DETECTOR_CONSENSUS_MANUAL_RESULTS_CSV = (
    DETECTOR_CONSENSUS_DIR / "manual_detector_results.csv"
)
DETECTOR_CONSENSUS_REPORT_PATH = DETECTOR_CONSENSUS_DIR / "proxy_label_report.json"
HELDOUT_AI_SIMILARITY_SAMPLES_PATH = REPO_ROOT / "heldout_ai_similarity_samples.txt"
CANDIDATE_HUMAN_SIMILARITY_SAMPLES_PATH = (
    REPO_ROOT / "candidate_human_similarity_samples.txt"
)
CANDIDATE_HUMAN_RAW_SAMPLES_PATH = (
    REPO_ROOT / "candidate_human_similarity_samples_raw.txt"
)
CANDIDATE_HUMAN_TOKEN_RAW_SAMPLES_PATH = (
    REPO_ROOT / "candidate_human_token_similarity_samples_raw.txt"
)
HELDOUT_AI_RAW_SAMPLES_PATH = REPO_ROOT / "heldout_ai_similarity_samples_raw.txt"
ENV_PATH = REPO_ROOT / ".env"

# --- refs_v2 bank (parallel to the v1 bank; v1 paths above are never written by v2) ---
AI_SOLUTIONS_V2_DIR = DATA_DIR / "ai_solutions_v2"
PROGRESS_LOG_V2_PATH = AI_SOLUTIONS_V2_DIR / "_progress.jsonl"
REFERENCE_INDEX_V2_DIR = OUTPUTS_DIR / "reference_index_v2"
REFERENCE_INDEX_V2_MANIFEST_PATH = OUTPUTS_DIR / "reference_index_v2_manifest.json"
REFS_V2_REPORT_PATH = OUTPUTS_DIR / "refs_v2_generation_report.json"
REFS_V2_EVAL_DIR = OUTPUTS_DIR / "reference_index_v2_eval"
AI_SOLUTIONS_HUMANLIKE_DIR = DATA_DIR / "ai_solutions_v2_humanlike"
PROGRESS_LOG_HUMANLIKE_PATH = AI_SOLUTIONS_HUMANLIKE_DIR / "_progress.jsonl"
EVAL_BARE_GPT55_DIR = DATA_DIR / "eval_bare_gpt55"
# D11 = D10 + one "hardened" gpt-5.5 ref per cluster. D10 is never written by this.
AI_SOLUTIONS_D11_HARDENED_DIR = DATA_DIR / "ai_solutions_d11_hardened"
REFERENCE_INDEX_D11_DIR = OUTPUTS_DIR / "reference_index_d11"
REFERENCE_INDEX_D11_REFERENCES_PATH = REFERENCE_INDEX_D11_DIR / "reference_texts.json"
D11_COMPARISON_REPORT = OUTPUTS_DIR / "d11_comparison.json"
# D10-swap: D10 minus v1:complete_function, plus the d11 hardened ref. Self-contained.
REFERENCE_INDEX_D10SWAP_DIR = OUTPUTS_DIR / "reference_index_d10_swap"
# --- D10: the production reference bank (self-contained; no solution dir needed at serve time) ---
# D10: pinned explicitly so it survives as the fallback even when the production
# pointer moves. These are the SOURCE paths that builders read from.
REFERENCE_INDEX_D10_DIR = OUTPUTS_DIR / "reference_index_d10"
REFERENCE_INDEX_D10_MANIFEST_PATH = REFERENCE_INDEX_D10_DIR / "bank_manifest.json"
REFERENCE_INDEX_D10_HASHES_PATH = REFERENCE_INDEX_D10_DIR / "reference_hashes.json"
REFERENCE_INDEX_D10_REFERENCES_PATH = REFERENCE_INDEX_D10_DIR / "reference_texts.json"
ARCHIVE_DIR = REPO_ROOT / "archive"
# Pre-2022 human solutions, sourced manually, for false-positive validation.
HUMAN_VALIDATION_DIR = DATA_DIR / "human_validation"
HUMAN_VALIDATION_MANIFEST = OUTPUTS_DIR / "human_validation_manifest.json"
HUMAN_VALIDATION_REPORT = OUTPUTS_DIR / "human_validation_report.json"
# Parallel voyage-code-4 bank, for the embedding-model comparison. The code-3 D10
# bank above is never written by this path.
VOYAGE_CODE_4_MODEL = "voyage-code-4"
REFERENCE_INDEX_D10_CODE4_DIR = OUTPUTS_DIR / "reference_index_d10_code4"
CODE4_COMPARISON_REPORT = OUTPUTS_DIR / "voyage_code4_comparison.json"
# Parallel OpenAI text-embedding index (1536-d), for the embedder comparison.
OPENAI_EMBED_MODEL = "openai/text-embedding-3-small"
OPENAI_EMBED_CACHE_DIR = DATA_DIR / "embedding_cache_openai"
REFERENCE_INDEX_D10_OPENAI_DIR = OUTPUTS_DIR / "reference_index_d10_openai"
OPENAI_EMBED_COMPARISON_REPORT = OUTPUTS_DIR / "openai_embed_comparison.json"


@dataclass(frozen=True)
class OpenRouterSettings:
    api_key: str
    base_url: str
    gemini_model: str
    deepseek_model: str
    openai_model: str
    concurrency: int
    timeout_seconds: int
    anthropic_model: str = ""
    openai_model_v2: str = ""


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
        anthropic_model=os.getenv("MODEL_ANTHROPIC", "").strip().strip('"'),
        openai_model_v2=os.getenv("MODEL_OPENAI_V2", "").strip().strip('"'),
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


def model_slugs_v2(settings: OpenRouterSettings) -> tuple[str, ...]:
    """v2 swaps DeepSeek for Anthropic and uses the free-tier-representative OpenAI
    slug; Gemini is held constant from v1 as the control arm."""
    if not settings.anthropic_model:
        raise ValueError("Missing required environment variable MODEL_ANTHROPIC")
    if not settings.openai_model_v2:
        raise ValueError("Missing required environment variable MODEL_OPENAI_V2")
    return (settings.gemini_model, settings.anthropic_model, settings.openai_model_v2)


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

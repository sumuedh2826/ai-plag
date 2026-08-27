from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from collections.abc import Sequence

import numpy as np
import voyageai
from voyageai.error import RateLimitError, ServiceUnavailableError, Timeout
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_random_exponential

from nw_ai_code_detector.config import EMBEDDING_CACHE_DIR, VoyageSettings
from nw_ai_code_detector.constants import (
    VOYAGE_CODE_3_MODEL,
    VOYAGE_CODE_3_USD_PER_MILLION_TOKENS,
    VOYAGE_EMBED_INPUT_TYPE,
    VOYAGE_EMBED_RETRY_ATTEMPTS,
)


@dataclass(frozen=True)
class EmbeddingBatch:
    vectors: tuple[tuple[float, ...], ...]
    billed_tokens: int
    cache_hits: int
    cache_misses: int
    cost_usd: float


class VoyageEmbedder:
    def __init__(self, settings: VoyageSettings) -> None:
        self._settings = settings
        self._client = voyageai.Client(
            api_key=settings.api_key,
            max_retries=VOYAGE_EMBED_RETRY_ATTEMPTS,
            timeout=settings.timeout_seconds,
        )
        EMBEDDING_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def embed_texts(self, texts: Sequence[str]) -> EmbeddingBatch:
        vectors_by_index: dict[int, tuple[float, ...]] = {}
        pending_indexes: list[int] = []
        pending_texts: list[str] = []
        for index, text in enumerate(texts):
            cached = _read_cached_vector(_cache_path(self._settings.model, text))
            if cached is None:
                pending_indexes.append(index)
                pending_texts.append(text)
                continue
            vectors_by_index[index] = cached

        billed_tokens = 0
        chunks = _split_batches(pending_indexes, pending_texts, self._settings.batch_size)
        if chunks:
            billed_tokens = _embed_chunks(
                self._client,
                self._settings,
                chunks,
                vectors_by_index,
            )
        ordered = tuple(vectors_by_index[index] for index in range(len(texts)))
        miss_count = len(pending_texts)
        cost_usd = billed_tokens * VOYAGE_CODE_3_USD_PER_MILLION_TOKENS / 1_000_000
        return EmbeddingBatch(
            vectors=ordered,
            billed_tokens=billed_tokens,
            cache_hits=len(texts) - miss_count,
            cache_misses=miss_count,
            cost_usd=cost_usd,
        )


def cached_vector_for_text(
    text: str,
    model: str = VOYAGE_CODE_3_MODEL,
) -> tuple[float, ...] | None:
    return _read_cached_vector(_cache_path(model, text))


def embedding_cache_key(
    text: str,
    model: str = VOYAGE_CODE_3_MODEL,
) -> str:
    return _cache_path(model, text).stem


def l2_normalize(vector: Sequence[float]) -> tuple[float, ...]:
    return _l2_normalize(vector)


def _split_batches(
    indexes: Sequence[int],
    texts: Sequence[str],
    batch_size: int,
) -> list[tuple[tuple[int, ...], tuple[str, ...]]]:
    batches: list[tuple[tuple[int, ...], tuple[str, ...]]] = []
    for start in range(0, len(texts), batch_size):
        end = start + batch_size
        batches.append((tuple(indexes[start:end]), tuple(texts[start:end])))
    return batches


def _embed_chunks(
    client: voyageai.Client,
    settings: VoyageSettings,
    chunks: Sequence[tuple[tuple[int, ...], tuple[str, ...]]],
    vectors_by_index: dict[int, tuple[float, ...]],
) -> int:
    billed_tokens = 0
    with ThreadPoolExecutor(max_workers=settings.concurrency) as pool:
        futures = [
            pool.submit(_embed_and_cache_chunk, client, settings.model, chunk)
            for chunk in chunks
        ]
        for future in as_completed(futures):
            chunk_indexes, chunk_vectors, tokens = future.result()
            billed_tokens += tokens
            for index, vector in zip(chunk_indexes, chunk_vectors):
                vectors_by_index[index] = vector
    return billed_tokens


@retry(
    retry=retry_if_exception_type((RateLimitError, ServiceUnavailableError, Timeout)),
    wait=wait_random_exponential(multiplier=1, max=30),
    stop=stop_after_attempt(VOYAGE_EMBED_RETRY_ATTEMPTS),
    reraise=True,
)
def _embed_and_cache_chunk(
    client: voyageai.Client,
    model: str,
    chunk: tuple[tuple[int, ...], tuple[str, ...]],
) -> tuple[tuple[int, ...], tuple[tuple[float, ...], ...], int]:
    indexes, texts = chunk
    response = client.embed(
        list(texts),
        model=model,
        input_type=VOYAGE_EMBED_INPUT_TYPE,
    )
    vectors = tuple(_l2_normalize(vector) for vector in response.embeddings)
    for text, vector in zip(texts, vectors):
        _write_cached_vector(_cache_path(model, text), vector)
    return indexes, vectors, int(response.total_tokens)


def _l2_normalize(vector: Sequence[float]) -> tuple[float, ...]:
    array = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(array))
    if norm == 0:
        return tuple(float(value) for value in array)
    normalized = array / norm
    return tuple(float(value) for value in normalized)


def _cache_path(model: str, text: str) -> Path:
    digest = sha256(f"{model}|{VOYAGE_EMBED_INPUT_TYPE}|{text}".encode("utf-8")).hexdigest()
    return EMBEDDING_CACHE_DIR / f"{digest}.json"


def _read_cached_vector(path: Path) -> tuple[float, ...] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    vector = payload.get("vector")
    if not isinstance(vector, list) or not vector:
        return None
    return tuple(float(value) for value in vector)


def _write_cached_vector(path: Path, vector: Sequence[float]) -> None:
    payload = {"vector": list(vector)}
    temp_path = path.with_suffix(".json.tmp")
    temp_path.write_text(json.dumps(payload), encoding="utf-8")
    temp_path.replace(path)

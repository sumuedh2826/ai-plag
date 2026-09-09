from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from threading import Lock
from collections.abc import Mapping, Sequence

import numpy as np
import voyageai

from nw_ai_code_detector.config import (
    AI_SOLUTIONS_DIR,
    REFERENCE_INDEX_DIR,
    VoyageSettings,
)
from nw_ai_code_detector.constants import (
    VOYAGE_CODE_3_USD_PER_MILLION_TOKENS,
    VOYAGE_EMBED_INPUT_TYPE,
    VOYAGE_EMBED_RETRY_ATTEMPTS,
)
from nw_ai_code_detector.data_load import QuestionRecord, load_questions
from nw_ai_code_detector.embedder import (
    EmbeddingBatch,
    VoyageEmbedder,
    l2_normalize,
)
from nw_ai_code_detector.eligibility_data import LoadedSolution, load_solution_records
from nw_ai_code_detector.build_reference_index_d10 import load_reference_hashes_d10
from nw_ai_code_detector.similarity_explanation import bank_reference_texts
from nw_ai_code_detector.index import ClusterKey, ReferenceIndex
from nw_ai_code_detector.score_query import (
    DetectionResult,
    SubmissionScoreRequest,
    score_submission,
)
from nw_ai_code_detector.stripper import strip_solution_body

QUESTIONS_PATH_ENV = "DEMO_QUESTIONS_PATH"
# The published file carries only the ~500 questions the bank covers, with test
# cases stripped. The full local dataset is used when present (pipelines need it).
QUESTIONS_PATH_DEMO = Path("data/questions_demo.json")
QUESTIONS_PATH_FULL = Path("data/questions.json")
QUESTIONS_PATH_DEFAULT = (
    QUESTIONS_PATH_DEMO if QUESTIONS_PATH_DEMO.is_file() else QUESTIONS_PATH_FULL
)


@dataclass(frozen=True)
class DemoScore:
    result: DetectionResult
    pasted_body: str
    language: str
    completed_code: str
    stripped_code: str
    nearest_reference: LoadedSolution | None
    cache_hit: bool


@dataclass(frozen=True)
class DemoScoreRequest:
    raw_code: str
    question: QuestionRecord
    language: str


class InMemoryVoyageEmbedder:
    """Embed submissions without persisting pasted-code vectors."""

    def __init__(self, settings: VoyageSettings) -> None:
        self._settings = settings
        self._client = voyageai.Client(
            api_key=settings.api_key,
            max_retries=VOYAGE_EMBED_RETRY_ATTEMPTS,
            timeout=settings.timeout_seconds,
        )
        self._vectors: dict[str, tuple[float, ...]] = {}
        self._lock = Lock()
        self.last_cache_hit = False

    def embed_texts(self, texts: Sequence[str]) -> EmbeddingBatch:
        vectors = []
        cache_hits = 0
        billed_tokens = 0
        for text in texts:
            key = _memory_cache_key(self._settings.model, text)
            with self._lock:
                vector = self._vectors.get(key)
            if vector is None:
                vector, tokens = self._embed(text)
                billed_tokens += tokens
                with self._lock:
                    self._vectors[key] = vector
            else:
                cache_hits += 1
            vectors.append(vector)
        self.last_cache_hit = cache_hits == len(texts)
        cost = billed_tokens * VOYAGE_CODE_3_USD_PER_MILLION_TOKENS / 1_000_000
        return EmbeddingBatch(
            tuple(vectors),
            billed_tokens,
            cache_hits,
            len(texts) - cache_hits,
            cost,
        )

    def vector_for_text(self, text: str) -> tuple[float, ...] | None:
        key = _memory_cache_key(self._settings.model, text)
        with self._lock:
            return self._vectors.get(key)

    def _embed(self, text: str) -> tuple[tuple[float, ...], int]:
        response = self._client.embed(
            [text],
            model=self._settings.model,
            input_type=VOYAGE_EMBED_INPUT_TYPE,
        )
        return l2_normalize(response.embeddings[0]), int(response.total_tokens)


def load_demo_questions(path: Path = QUESTIONS_PATH_DEFAULT) -> dict[str, QuestionRecord]:
    return load_questions(path)


def load_reference_index(path: Path = REFERENCE_INDEX_DIR) -> ReferenceIndex:
    return ReferenceIndex.load(path)


def questions_for_index(
    questions: Mapping[str, QuestionRecord],
    index: ReferenceIndex,
) -> dict[str, QuestionRecord]:
    question_ids = {token.split(":", 1)[0] for token in index.cluster_tokens()}
    selected = {
        question_id: question
        for question_id, question in questions.items()
        if question_id in question_ids
    }
    if len(selected) != len(question_ids):
        raise RuntimeError("Question metadata does not cover the reference bank")
    return selected


def load_reference_hashes(
    root: Path | None = None,
) -> dict[tuple[str, str], set[str]]:
    """Exact-match hashes for the production bank.

    D10 ships its own hashes, so serving needs no solution directory. Passing an
    explicit `root` still walks a solution tree (used by the v1 baseline path)."""
    if root is None:
        return load_reference_hashes_d10()
    hashes: dict[tuple[str, str], set[str]] = {}
    for record in load_solution_records(root, "mixed_v1"):
        if not record.parse_ok or not record.stripped_code:
            continue
        pair = (record.question_id, record.language)
        digest = sha256(record.stripped_code.encode("utf-8")).hexdigest()
        hashes.setdefault(pair, set()).add(digest)
    return hashes


def score_demo_submission(
    request: DemoScoreRequest,
    index: ReferenceIndex,
    embedder: InMemoryVoyageEmbedder,
    reference_hashes: Mapping[tuple[str, str], set[str]],
) -> DemoScore:
    if isinstance(embedder, VoyageEmbedder):
        raise TypeError("Demo embedding must not use the disk-backed VoyageEmbedder")
    boilerplate = request.question.boilerplates.get(request.language, "")
    if not boilerplate:
        raise ValueError("Selected question has no boilerplate for this language")
    if not request.raw_code.strip():
        raise ValueError("Submission code cannot be empty")
    result = score_submission(
        SubmissionScoreRequest(
            request.raw_code,
            boilerplate,
            request.question.question_id,
            request.language,
        ),
        index,
        embedder,
        reference_hashes,
    )
    stripped = strip_solution_body(
        request.raw_code,
        boilerplate,
        request.language,
    )
    query = embedder.vector_for_text(stripped)
    nearest = _nearest_reference(
        request.question.question_id,
        request.language,
        query,
        index,
    )
    return DemoScore(
        result,
        request.raw_code,
        request.language,
        request.raw_code,
        stripped,
        nearest,
        embedder.last_cache_hit,
    )


def available_languages(question: QuestionRecord) -> tuple[str, ...]:
    return tuple(
        language
        for language in ("CPP", "PYTHON")
        if question.boilerplates.get(language)
    )

def _nearest_reference(
    question_id: str,
    language: str,
    query: tuple[float, ...] | None,
    index: ReferenceIndex,
) -> LoadedSolution | None:
    if query is None:
        return None
    bank = bank_reference_texts()
    token = f"{question_id}:{language}"
    if bank is not None and token in bank:
        records = bank[token]
    else:
        cluster = AI_SOLUTIONS_DIR / question_id / language
        records = [
            record
            for record in load_solution_records(cluster, "mixed_v1")
            if record.parse_ok and record.stripped_code
        ]
    vectors = index.get_cluster(ClusterKey(question_id, language)).vectors
    if len(records) != int(vectors.shape[0]):
        raise RuntimeError("Generated reference records do not align with index vectors")
    query_vector = np.asarray(query, dtype=np.float32)
    similarities = np.asarray(vectors @ query_vector, dtype=np.float32)
    return records[int(np.argmax(similarities))]


def _memory_cache_key(model: str, text: str) -> str:
    payload = f"{model}|{VOYAGE_EMBED_INPUT_TYPE}|{text}"
    return sha256(payload.encode("utf-8")).hexdigest()

from __future__ import annotations

import difflib
from dataclasses import dataclass
from collections.abc import Sequence

import numpy as np

from nw_ai_code_detector.config import AI_SOLUTIONS_DIR
from nw_ai_code_detector.constants import (
    HIGH_AI_SCORE_FLOOR,
)
from nw_ai_code_detector.explanation_card import (
    ExplanationCardRequest,
    build_explanation_card,
    format_explanation_card,
)
from nw_ai_code_detector.discount_layer import CanonicalityAssessment
import json
from functools import lru_cache

from nw_ai_code_detector.config import REFERENCE_INDEX_D10_REFERENCES_PATH
from nw_ai_code_detector.eligibility_data import LoadedSolution, load_solution_records
from nw_ai_code_detector.embedder import cached_vector_for_text
from nw_ai_code_detector.index import ClusterKey


@dataclass(frozen=True)
class NearestReferenceMatch:
    persona: str | None
    model: str | None
    cosine: float
    reference_line_start: int | None
    reference_line_end: int | None
    matched_reference_lines: tuple[str, ...]


def find_nearest_generated_reference(
    question_id: str,
    language: str,
    stripped_code: str,
) -> NearestReferenceMatch | None:
    query = cached_vector_for_text(stripped_code)
    if query is None:
        return None
    return find_nearest_generated_reference_from_vector(
        question_id,
        language,
        stripped_code,
        query,
    )


def find_nearest_generated_reference_from_vector(
    question_id: str,
    language: str,
    stripped_code: str,
    query: tuple[float, ...],
) -> NearestReferenceMatch | None:
    cluster_dir = AI_SOLUTIONS_DIR / question_id / language
    if not cluster_dir.is_dir():
        return None
    best: tuple[float, LoadedSolution] | None = None
    query_vector = np.asarray(query, dtype=np.float32)
    for record in load_solution_records(cluster_dir, "mixed_v1"):
        if not record.parse_ok or not record.stripped_code:
            continue
        vector = cached_vector_for_text(record.stripped_code)
        if vector is None:
            continue
        cosine = float(np.asarray(vector, dtype=np.float32) @ query_vector)
        if best is None or cosine > best[0]:
            best = (cosine, record)
    if best is None:
        return None
    cosine, record = best
    start, end, lines = _matching_reference_span(stripped_code, record.stripped_code or "")
    return NearestReferenceMatch(
        record.persona,
        record.generator,
        cosine,
        start,
        end,
        lines,
    )


@lru_cache(maxsize=1)
def bank_reference_texts() -> dict[str, list[LoadedSolution]] | None:
    """Reference texts bundled with the production bank, in vector-row order."""
    if not REFERENCE_INDEX_D10_REFERENCES_PATH.is_file():
        return None
    payload = json.loads(
        REFERENCE_INDEX_D10_REFERENCES_PATH.read_text(encoding="utf-8")
    )
    out: dict[str, list[LoadedSolution]] = {}
    for token, refs in payload.items():
        question_id, language = token.split(":", 1)
        out[token] = [
            LoadedSolution(
                question_id=question_id,
                language=language,
                raw_code=None,
                stripped_code=ref["stripped_code"],
                parse_ok=True,
                generator=ref.get("model"),
                persona=ref.get("persona"),
                relative_path=f"{token}#{position}",
                source="d10",
            )
            for position, ref in enumerate(refs)
        ]
    return out


def find_nearest_generated_reference_from_cluster(
    key: ClusterKey,
    stripped_code: str,
    query: Sequence[float],
    reference_vectors: np.ndarray,
) -> NearestReferenceMatch | None:
    bank = bank_reference_texts()
    if bank is not None and key.token in bank:
        records = bank[key.token]
    else:
        cluster_dir = AI_SOLUTIONS_DIR / key.question_id / key.language
        records = [
            record
            for record in load_solution_records(cluster_dir, "mixed_v1")
            if record.parse_ok and record.stripped_code
        ]
    if not records:
        return None
    if len(records) != int(reference_vectors.shape[0]):
        raise RuntimeError("Generated reference records do not align with index vectors")
    query_vector = np.asarray(query, dtype=np.float32)
    similarities = np.asarray(reference_vectors @ query_vector, dtype=np.float32)
    nearest_index = int(np.argmax(similarities))
    record = records[nearest_index]
    start, end, lines = _matching_reference_span(
        stripped_code,
        record.stripped_code or "",
    )
    return NearestReferenceMatch(
        record.persona,
        record.generator,
        float(similarities[nearest_index]),
        start,
        end,
        lines,
    )


def build_reference_similarity_summary(
    assessment: CanonicalityAssessment,
    match: NearestReferenceMatch | None,
    language: str,
) -> str:
    """State reference-similarity only; never claim authorship."""
    card = build_explanation_card(
        ExplanationCardRequest(
            assessment,
            match,
            language,
            assessment.cluster_diversity,
        )
    )
    return format_explanation_card(card)


def illustrative_unlocked_band(score: float | None) -> str:
    if score is None:
        return "not_scored"
    if score >= HIGH_AI_SCORE_FLOOR:
        return "high_ai_range_unlocked"
    if score >= 0.90:
        return "review_range_unlocked"
    return "lower_similarity_unlocked"


def _matching_reference_span(
    query_stripped: str,
    reference_stripped: str,
) -> tuple[int | None, int | None, tuple[str, ...]]:
    query_lines = _content_lines(query_stripped)
    ref_lines = _content_lines(reference_stripped)
    if not query_lines or not ref_lines:
        return None, None, ()
    query_text = [line[1] for line in query_lines]
    ref_text = [line[1] for line in ref_lines]
    matcher = difflib.SequenceMatcher(a=ref_text, b=query_text, autojunk=False)
    blocks = sorted(matcher.get_matching_blocks(), key=lambda item: item.size, reverse=True)
    for block in blocks:
        if block.size < 2:
            continue
        start_index = ref_lines[block.a][0]
        end_index = ref_lines[block.a + block.size - 1][0]
        matched = tuple(ref_text[block.a : block.a + block.size][:8])
        return start_index, end_index, matched
    return None, None, ()


def _content_lines(source: str) -> list[tuple[int, str]]:
    rows = []
    for index, raw in enumerate(source.splitlines(), start=1):
        line = raw.strip()
        if not line or _is_wrapper_line(line):
            continue
        rows.append((index, line))
    return rows


def _is_wrapper_line(line: str) -> bool:
    compact = line.replace(" ", "")
    return compact in {
        "{",
        "}",
        "};",
        "public:",
        "classsolution:",
        "classsolution{",
        "classsolution",
    }

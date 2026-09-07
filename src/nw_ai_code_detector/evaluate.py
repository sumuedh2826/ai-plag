from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from collections.abc import Mapping, Sequence

import numpy as np

from nw_ai_code_detector.config import (
    AI_SOLUTIONS_DIR,
    DATA_DIR,
    EVAL_AI_SOLUTIONS_DIR,
    EVAL_SCORES_PATH,
    SELECTED_500_PATH,
    load_voyage_settings,
)
from nw_ai_code_detector.constants import (
    DEFAULT_GENERATION_CONCURRENCY,
    GENERATION_LANGUAGES,
    GROUPS_KEY,
    HUMAN_NEGATIVES_PER_LANGUAGE,
)
from nw_ai_code_detector.data_load import Dataset, load_dataset
from nw_ai_code_detector.embedder import EmbeddingBatch, VoyageEmbedder
from nw_ai_code_detector.generate_ai_refs import _limit_questions, _load_selected_questions
from nw_ai_code_detector.generate_heldout import generate_held_out_refs
from nw_ai_code_detector.index import ClusterKey, ClusterVectors, ReferenceIndex
from nw_ai_code_detector.scorer import (
    LanguageMetrics,
    ScoredItem,
    histogram_lines,
    metrics_for_items,
    score_item,
)
from nw_ai_code_detector.select_500 import EligibleQuestion
from nw_ai_code_detector.stripper import Language, SourceParseError, strip_solution_body


@dataclass(frozen=True)
class TextRecord:
    question_id: str
    language: str
    role: str
    source: str
    text: str


def main() -> int:
    options = _parse_options()
    dataset = load_dataset()
    questions = _limit_questions(_load_selected_questions(SELECTED_500_PATH), options.limit)
    started = time.perf_counter()
    generate_held_out_refs(dataset, questions, options.gen_concurrency)
    records = _collect_records(dataset, questions)
    embedder = VoyageEmbedder(load_voyage_settings())
    batch = embedder.embed_texts([record.text for record in records])
    index = _build_reference_index(records, batch.vectors)
    index.save()
    scored, excluded = _score_eval_items(records, batch.vectors, index)
    _write_scores(scored, excluded, batch)
    _print_results(scored, excluded, batch, time.perf_counter() - started, options.limit)
    return 0


def _parse_options() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--gen-concurrency", type=int, default=DEFAULT_GENERATION_CONCURRENCY)
    return parser.parse_args()


def _collect_records(
    dataset: Dataset,
    questions: Sequence[EligibleQuestion],
) -> list[TextRecord]:
    selected_ids = {item.question_id for item in questions}
    records: list[TextRecord] = []
    records.extend(_load_json_records(AI_SOLUTIONS_DIR, selected_ids, "reference"))
    records.extend(_load_json_records(EVAL_AI_SOLUTIONS_DIR, selected_ids, "held_out"))
    records.extend(_load_human_records(dataset, selected_ids))
    return records


def _load_json_records(
    root: Path,
    selected_ids: set[str],
    role: str,
) -> list[TextRecord]:
    records: list[TextRecord] = []
    if not root.is_dir():
        return records
    for path in sorted(root.rglob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("qid") not in selected_ids:
            continue
        if payload.get("parse_ok") is not True:
            continue
        text = payload.get("stripped_code") or ""
        if not text.strip():
            continue
        records.append(
            TextRecord(
                question_id=str(payload["qid"]),
                language=str(payload["language"]),
                role=role,
                source=str(payload.get("persona") or path.name),
                text=text,
            )
        )
    return records


def _load_human_records(dataset: Dataset, selected_ids: set[str]) -> list[TextRecord]:
    groups = json.loads((DATA_DIR / "scored_submissions.json").read_text(encoding="utf-8"))
    mapping = groups.get(GROUPS_KEY) if isinstance(groups, dict) else {}
    if not isinstance(mapping, dict):
        return []
    records: list[TextRecord] = []
    for question_id in sorted(selected_ids):
        for language in GENERATION_LANGUAGES:
            records.extend(
                _humans_for_pair(dataset, mapping, question_id, language)
            )
    return records


def _humans_for_pair(
    dataset: Dataset,
    mapping: Mapping[str, object],
    question_id: str,
    language: Language,
) -> list[TextRecord]:
    group_records = mapping.get(f"{question_id}:{language.value}")
    if not isinstance(group_records, list):
        return []
    question = dataset.questions[question_id]
    boilerplate = question.boilerplates.get(language.value, "")
    accepted: list[TextRecord] = []
    for index, source_record in enumerate(group_records):
        if len(accepted) >= HUMAN_NEGATIVES_PER_LANGUAGE:
            break
        stripped = _strip_human(source_record, boilerplate, language)
        if stripped is None:
            continue
        accepted.append(
            TextRecord(
                question_id=question_id,
                language=language.value,
                role="human",
                source=f"human_{index}",
                text=stripped,
            )
        )
    return accepted


def _strip_human(
    source_record: object,
    boilerplate: str,
    language: Language,
) -> str | None:
    if not isinstance(source_record, dict):
        return None
    raw_code = source_record.get("raw_code")
    if not isinstance(raw_code, str) or not raw_code.strip():
        return None
    try:
        stripped = strip_solution_body(raw_code, boilerplate, language)
    except (SourceParseError, ValueError):
        return None
    if not stripped.strip():
        return None
    return stripped


def _build_reference_index(
    records: Sequence[TextRecord],
    vectors: Sequence[Sequence[float]],
) -> ReferenceIndex:
    grouped: dict[str, list[tuple[int, Sequence[float]]]] = {}
    for index, record in enumerate(records):
        if record.role != "reference":
            continue
        token = f"{record.question_id}:{record.language}"
        grouped.setdefault(token, []).append((index, vectors[index]))
    clusters: list[ClusterVectors] = []
    for token, items in grouped.items():
        question_id, language = token.split(":", 1)
        matrix = np.asarray([item[1] for item in items], dtype=np.float32)
        clusters.append(
            ClusterVectors(
                key=ClusterKey(question_id, language),
                vector_ids=tuple(item[0] for item in items),
                vectors=matrix,
            )
        )
    return ReferenceIndex.from_clusters(clusters)


def _score_eval_items(
    records: Sequence[TextRecord],
    vectors: Sequence[Sequence[float]],
    index: ReferenceIndex,
) -> tuple[list[ScoredItem], dict[str, int]]:
    human_pairs = {
        (record.question_id, record.language)
        for record in records
        if record.role == "human"
    }
    excluded = _excluded_pair_counts(records, human_pairs)
    scored: list[ScoredItem] = []
    for record, vector in zip(records, vectors):
        if record.role == "reference":
            continue
        pair = (record.question_id, record.language)
        if pair not in human_pairs:
            continue
        nn_score = score_item(
            index,
            ClusterKey(record.question_id, record.language),
            vector,
        )
        scored.append(
            ScoredItem(
                question_id=record.question_id,
                language=record.language,
                label="ai" if record.role == "held_out" else "human",
                source=record.source,
                nn_score=nn_score,
            )
        )
    return scored, excluded


def _excluded_pair_counts(
    records: Sequence[TextRecord],
    human_pairs: set[tuple[str, str]],
) -> dict[str, int]:
    candidate_pairs = {
        (record.question_id, record.language)
        for record in records
        if record.role in {"held_out", "human", "reference"}
    }
    excluded = {"CPP": 0, "PYTHON": 0}
    for question_id, language in candidate_pairs:
        if (question_id, language) not in human_pairs:
            excluded[language] += 1
    return excluded


def _write_scores(
    scored: Sequence[ScoredItem],
    excluded: Mapping[str, int],
    batch: EmbeddingBatch,
) -> None:
    EVAL_SCORES_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "excluded_pairs": dict(excluded),
        "embedding_tokens": batch.billed_tokens,
        "embedding_cost_usd": batch.cost_usd,
        "cache_hits": batch.cache_hits,
        "cache_misses": batch.cache_misses,
        "items": [asdict(item) for item in scored],
    }
    temp_path = EVAL_SCORES_PATH.with_suffix(".json.tmp")
    temp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temp_path.replace(EVAL_SCORES_PATH)


def _print_results(
    scored: Sequence[ScoredItem],
    excluded: Mapping[str, int],
    batch: EmbeddingBatch,
    elapsed: float,
    limit: int | None,
) -> None:
    rows = [
        metrics_for_items(scored, "CPP", excluded["CPP"]),
        metrics_for_items(scored, "PYTHON", excluded["PYTHON"]),
        metrics_for_items(scored, None, excluded["CPP"] + excluded["PYTHON"]),
    ]
    print("\n=== embedding + eval summary ===")
    print(
        f"embed billed_tokens={batch.billed_tokens} cost_usd={batch.cost_usd:.6f} "
        f"cache_hits={batch.cache_hits} cache_misses={batch.cache_misses}"
    )
    print(f"wall_clock_s={elapsed:.1f}")
    print(
        f"{'lang':<10} {'n_pos':>6} {'n_neg':>6} {'excl':>5} "
        f"{'AUROC_nn':>9} {'R@1%':>7} {'R@5%':>7} "
        f"{'pos_mean':>9} {'neg_mean':>9}"
    )
    for row in rows:
        print(_format_metrics_row(row))
    _print_histograms(scored)
    if limit is not None:
        _print_projection(elapsed, batch, limit)


def _format_metrics_row(row: LanguageMetrics) -> str:
    return (
        f"{row.language:<10} {row.n_positives:6d} {row.n_negatives:6d} "
        f"{row.excluded_pairs:5d} {_fmt(row.auroc_nn):>9} "
        f"{_fmt(row.recall_at_fpr.get('1%')):>7} "
        f"{_fmt(row.recall_at_fpr.get('5%')):>7} {_fmt(row.positive_nn_mean):>9} "
        f"{_fmt(row.negative_nn_mean):>9}"
    )


def _fmt(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def _print_histograms(scored: Sequence[ScoredItem]) -> None:
    print("\n=== nn score distributions ===")
    for language in ("CPP", "PYTHON", None):
        scoped = [
            item
            for item in scored
            if language is None or item.language == language
        ]
        label = language or "COMBINED"
        pos = [item.nn_score for item in scoped if item.label == "ai"]
        neg = [item.nn_score for item in scoped if item.label == "human"]
        for line in histogram_lines(pos, f"{label} held-out AI"):
            print(line)
        for line in histogram_lines(neg, f"{label} human"):
            print(line)


def _print_projection(elapsed: float, batch: EmbeddingBatch, limit: int) -> None:
    scale = 500 / limit
    print("\n=== full-run projection from --limit smoke ===")
    print(f"scale={scale:.1f}x  projected_wall_s={elapsed * scale:.0f} (~{elapsed * scale / 60:.1f} min)")
    print(
        f"projected_embed_tokens={int(batch.billed_tokens * scale)} "
        f"projected_embed_cost_usd={batch.cost_usd * scale:.4f}"
    )
    print("Waiting for go before the full 500-question run.")


if __name__ == "__main__":
    raise SystemExit(main())

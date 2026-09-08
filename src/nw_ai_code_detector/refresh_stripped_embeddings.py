from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Mapping, Sequence

import numpy as np

from nw_ai_code_detector.build_model_dataset_v2 import (
    EXPECTED_MIXED_CLUSTER_COUNT,
    EXPECTED_MIXED_REFERENCES_PER_CLUSTER,
    GPT_HEAVY_CANDIDATES_DIR,
)
from nw_ai_code_detector.config import (
    AI_SOLUTIONS_DIR,
    DATA_DIR,
    EVAL_AI_SOLUTIONS_DIR,
    REFERENCE_INDEX_V1_DIR,
    load_voyage_settings,
)
from nw_ai_code_detector.constants import (
    GROUPS_KEY,
    HUMAN_STRIPPED_CODE_FIELD,
    RAW_CODE_FIELD,
    RAW_OUTPUT_FIELD,
    STRIPPED_CODE_FIELD,
)
from nw_ai_code_detector.data_load import Dataset, load_dataset
from nw_ai_code_detector.embedder import EmbeddingBatch, VoyageEmbedder, cached_vector_for_text
from nw_ai_code_detector.index import ClusterKey, ClusterVectors, ReferenceIndex
from nw_ai_code_detector.stripper import SourceParseError, strip_solution_body


@dataclass(frozen=True)
class CorpusChange:
    source: str
    language: str
    identifier: str
    question_id: str
    stripped: str
    is_reference: bool


@dataclass(frozen=True)
class AiRestripTarget:
    path: Path
    root: Path
    source: str
    is_reference: bool


@dataclass(frozen=True)
class RefreshReport:
    changed_by_source: dict[str, dict[str, int]]
    total_changed: int
    unique_changed_texts: int
    billed_tokens: int
    cost_usd: float
    cache_hits: int
    cache_misses: int
    clusters_rebuilt: int
    cluster_count: int
    references_per_cluster: int


def main() -> int:
    dataset = load_dataset()
    changes = _collect_and_write_changes(dataset)
    batch = _embed_changed(changes)
    rebuilt = _rebuild_affected_clusters(changes)
    report = _build_report(changes, batch, rebuilt)
    _print_report(report)
    _assert_cluster_invariants()
    return 0


def _collect_and_write_changes(dataset: Dataset) -> list[CorpusChange]:
    changes = []
    changes.extend(_restrip_ai_tree(AI_SOLUTIONS_DIR, dataset, "ai_solutions", True))
    changes.extend(
        _restrip_ai_tree(EVAL_AI_SOLUTIONS_DIR, dataset, "eval_ai_solutions", False)
    )
    changes.extend(
        _restrip_ai_tree(GPT_HEAVY_CANDIDATES_DIR, dataset, "gpt_heavy_v2", False)
    )
    changes.extend(_restrip_humans(dataset))
    return changes


def _restrip_ai_tree(
    root: Path,
    dataset: Dataset,
    source: str,
    is_reference: bool,
) -> list[CorpusChange]:
    if not root.is_dir():
        return []
    changes: list[CorpusChange] = []
    for path in sorted(root.rglob("*.json")):
        target = AiRestripTarget(path, root, source, is_reference)
        payload = json.loads(path.read_text(encoding="utf-8"))
        change = _apply_ai_restrip(target, payload, dataset)
        if change is not None:
            changes.append(change)
    return changes


def _apply_ai_restrip(
    target: AiRestripTarget,
    payload: Mapping[str, object],
    dataset: Dataset,
) -> CorpusChange | None:
    language = str(payload.get("language") or "")
    question_id = str(payload.get("qid") or payload.get("question_id") or "")
    boilerplate = _boilerplate(dataset, question_id, language)
    stripped = _strip_or_none(_ai_raw_code(payload), boilerplate, language)
    previous = payload.get(STRIPPED_CODE_FIELD)
    previous_text = previous if isinstance(previous, str) else ""
    if stripped is None or previous_text == stripped:
        return None
    updated = dict(payload)
    updated[STRIPPED_CODE_FIELD] = stripped
    _write_json(target.path, updated)
    relative = target.path.relative_to(target.root).as_posix()
    return CorpusChange(
        source=target.source,
        language=language,
        identifier=relative,
        question_id=question_id,
        stripped=stripped,
        is_reference=target.is_reference,
    )


def _restrip_humans(dataset: Dataset) -> list[CorpusChange]:
    path = DATA_DIR / "scored_submissions.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    groups = payload.get(GROUPS_KEY)
    if not isinstance(groups, dict):
        raise RuntimeError("scored_submissions.json groups are unavailable")
    changes: list[CorpusChange] = []
    for group_key, records in groups.items():
        if not isinstance(records, list):
            continue
        question_id, language = _split_group_key(group_key)
        boilerplate = _boilerplate(dataset, question_id, language)
        for index, record in enumerate(records):
            change = _apply_human_restrip(
                record,
                question_id,
                language,
                boilerplate,
            )
            if change is None:
                continue
            records[index] = change[0]
            changes.append(change[1])
    if changes:
        _write_json(path, payload)
    return changes


def _apply_human_restrip(
    record: object,
    question_id: str,
    language: str,
    boilerplate: str,
) -> tuple[dict[str, object], CorpusChange] | None:
    if not isinstance(record, dict):
        return None
    raw_code = record.get(RAW_CODE_FIELD)
    if not isinstance(raw_code, str):
        return None
    stripped = _strip_or_none(raw_code, boilerplate, language)
    previous = record.get(HUMAN_STRIPPED_CODE_FIELD)
    previous_text = previous if isinstance(previous, str) else ""
    if stripped is None or previous_text == stripped:
        return None
    updated = dict(record)
    updated[HUMAN_STRIPPED_CODE_FIELD] = stripped
    identifier = f"{question_id}:{language}:{record.get('user_id', '')}"
    change = CorpusChange(
        source="candidate_human",
        language=language,
        identifier=identifier,
        question_id=question_id,
        stripped=stripped,
        is_reference=False,
    )
    return updated, change


def _embed_changed(changes: Sequence[CorpusChange]) -> EmbeddingBatch:
    unique_texts = list(dict.fromkeys(item.stripped for item in changes))
    if not unique_texts:
        return EmbeddingBatch(vectors=(), billed_tokens=0, cache_hits=0, cache_misses=0, cost_usd=0.0)
    embedder = VoyageEmbedder(load_voyage_settings())
    return embedder.embed_texts(unique_texts)


def _rebuild_affected_clusters(changes: Sequence[CorpusChange]) -> int:
    affected = {
        ClusterKey(item.question_id, item.language)
        for item in changes
        if item.is_reference
    }
    if not affected:
        return 0
    grouped = _reference_stripped_by_cluster()
    rebuilt = 0
    for key in sorted(affected, key=lambda item: item.token):
        texts = grouped[key]
        if len(texts) != EXPECTED_MIXED_REFERENCES_PER_CLUSTER:
            raise RuntimeError(f"Cluster {key.token} has {len(texts)} references")
        vectors = _cached_matrix(texts)
        _persist_cluster(key, vectors)
        rebuilt += 1
    return rebuilt


def _reference_stripped_by_cluster() -> dict[ClusterKey, list[str]]:
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
    return grouped


def _cached_matrix(texts: Sequence[str]) -> np.ndarray:
    vectors = []
    for text in texts:
        vector = cached_vector_for_text(text)
        if vector is None:
            raise RuntimeError("Changed reference is missing from the embedding cache")
        vectors.append(vector)
    return np.asarray(vectors, dtype=np.float32)


def _persist_cluster(key: ClusterKey, vectors: np.ndarray) -> None:
    manifest = json.loads((REFERENCE_INDEX_V1_DIR / "manifest.json").read_text(encoding="utf-8"))
    entry = manifest["clusters"][key.token]
    vector_ids = tuple(int(value) for value in entry["vector_ids"])
    cluster = ClusterVectors(key, vector_ids, vectors)
    index = ReferenceIndex.from_clusters((cluster,))
    index.write_cluster(key, REFERENCE_INDEX_V1_DIR)


def _assert_cluster_invariants() -> None:
    manifest = json.loads((REFERENCE_INDEX_V1_DIR / "manifest.json").read_text(encoding="utf-8"))
    clusters = manifest["clusters"]
    if len(clusters) != EXPECTED_MIXED_CLUSTER_COUNT:
        raise RuntimeError(f"Cluster count changed: {len(clusters)}")
    for token, entry in clusters.items():
        count = int(entry["count"])
        if count != EXPECTED_MIXED_REFERENCES_PER_CLUSTER:
            raise RuntimeError(f"Cluster {token} count changed: {count}")
        question_id = str(entry["question_id"])
        language = str(entry["language"])
        if token != f"{question_id}:{language}":
            raise RuntimeError(f"Cluster routing token mismatch: {token}")
        stem = token.replace(":", "__")
        vectors = np.load(REFERENCE_INDEX_V1_DIR / f"{stem}.npy")
        if vectors.shape[0] != EXPECTED_MIXED_REFERENCES_PER_CLUSTER:
            raise RuntimeError(f"Cluster {token} vector rows changed")


def _build_report(
    changes: Sequence[CorpusChange],
    batch: EmbeddingBatch,
    rebuilt: int,
) -> RefreshReport:
    by_source: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for item in changes:
        by_source[item.source][item.language] += 1
        by_source[item.source]["total"] += 1
    return RefreshReport(
        changed_by_source={source: dict(counts) for source, counts in by_source.items()},
        total_changed=len(changes),
        unique_changed_texts=len({item.stripped for item in changes}),
        billed_tokens=batch.billed_tokens,
        cost_usd=batch.cost_usd,
        cache_hits=batch.cache_hits,
        cache_misses=batch.cache_misses,
        clusters_rebuilt=rebuilt,
        cluster_count=EXPECTED_MIXED_CLUSTER_COUNT,
        references_per_cluster=EXPECTED_MIXED_REFERENCES_PER_CLUSTER,
    )


def _print_report(report: RefreshReport) -> None:
    print(json.dumps({
        "changed_by_source": report.changed_by_source,
        "total_changed": report.total_changed,
        "unique_changed_texts": report.unique_changed_texts,
        "billed_tokens": report.billed_tokens,
        "cost_usd": report.cost_usd,
        "cache_hits": report.cache_hits,
        "cache_misses": report.cache_misses,
        "clusters_rebuilt": report.clusters_rebuilt,
        "cluster_count": report.cluster_count,
        "references_per_cluster": report.references_per_cluster,
    }, indent=2))


def _boilerplate(dataset: Dataset, question_id: str, language: str) -> str:
    question = dataset.questions.get(question_id)
    if question is None:
        return ""
    return question.boilerplates.get(language, "")


def _ai_raw_code(payload: Mapping[str, object]) -> str:
    raw_output = payload.get(RAW_OUTPUT_FIELD)
    if isinstance(raw_output, str) and raw_output.strip():
        return raw_output
    raw_code = payload.get(RAW_CODE_FIELD)
    if isinstance(raw_code, str):
        return raw_code
    return ""


def _strip_or_none(raw_code: str, boilerplate: str, language: str) -> str | None:
    if not raw_code.strip() or not language:
        return None
    try:
        return strip_solution_body(raw_code, boilerplate, language)
    except (SourceParseError, ValueError, KeyError):
        return None


def _split_group_key(group_key: str) -> tuple[str, str]:
    question_id, language = str(group_key).rsplit(":", 1)
    return question_id, language


def _write_json(path: Path, payload: Mapping[str, object] | dict[str, object]) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temp_path.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())

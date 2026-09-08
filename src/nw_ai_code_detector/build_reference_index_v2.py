from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

import numpy as np

from nw_ai_code_detector.config import (
    AI_SOLUTIONS_DIR,
    AI_SOLUTIONS_V2_DIR,
    REFERENCE_INDEX_V1_DIR,
    REFERENCE_INDEX_V2_DIR,
    REFERENCE_INDEX_V2_MANIFEST_PATH,
    load_voyage_settings,
)
from nw_ai_code_detector.constants import CANONICALITY_EMBEDDING_DIMENSION
from nw_ai_code_detector.embedder import VoyageEmbedder, cached_vector_for_text, l2_normalize
from nw_ai_code_detector.index import ClusterKey, ClusterVectors, ReferenceIndex
from nw_ai_code_detector.personas_v2 import PERSONA_V2_ORDER, REFS_V2_BANK_VERSION
from nw_ai_code_detector.refs_v2_guard import ensure_v1_untouched

EXPECTED_CLUSTERS = 1000
EXPECTED_REFS_PER_CLUSTER = len(PERSONA_V2_ORDER)
UNIT_NORM_TOLERANCE = 1e-3


@dataclass(frozen=True)
class RefRecord:
    question_id: str
    language: str
    persona: str
    model: str
    stripped_code: str
    path: Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the refs_v2 reference index.")
    parser.add_argument("--stage", choices=("embed", "build"), required=True)
    args = parser.parse_args()
    ensure_v1_untouched()
    records = _load_records()
    print(f"loaded {len(records)} refs across {len({(r.question_id, r.language) for r in records})} clusters")
    result = _stage_embed(records) if args.stage == "embed" else _stage_build(records)
    ensure_v1_untouched()
    return result


def _load_records() -> list[RefRecord]:
    records: list[RefRecord] = []
    for path in sorted(AI_SOLUTIONS_V2_DIR.rglob("*.json")):
        if path.name.startswith("_"):
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("parse_ok") is not True:
            continue
        text = payload.get("stripped_code") or ""
        if not text.strip():
            continue
        records.append(
            RefRecord(
                question_id=str(payload["qid"]),
                language=str(payload["language"]),
                persona=str(payload["persona"]),
                model=str(payload["model"]),
                stripped_code=text,
                path=path,
            )
        )
    return records


def _stage_embed(records: Sequence[RefRecord]) -> int:
    embedder = VoyageEmbedder(load_voyage_settings())
    batch = embedder.embed_texts([record.stripped_code for record in records])
    print(
        f"embedded {len(records)} texts | billed_tokens={batch.billed_tokens} "
        f"cost=${batch.cost_usd:.4f} cache_hits={batch.cache_hits} "
        f"cache_misses={batch.cache_misses}"
    )
    return 0


def _stage_build(records: Sequence[RefRecord]) -> int:
    _assert_target_is_not_v1()
    grouped: dict[tuple[str, str], list[RefRecord]] = {}
    for record in records:
        grouped.setdefault((record.question_id, record.language), []).append(record)

    if len(grouped) != EXPECTED_CLUSTERS:
        raise RuntimeError(f"Expected {EXPECTED_CLUSTERS} clusters, found {len(grouped)}")

    clusters: list[ClusterVectors] = []
    manifest: dict[str, object] = {}
    for (question_id, language), group in sorted(grouped.items()):
        ordered = sorted(group, key=lambda r: r.persona)
        if len(ordered) != EXPECTED_REFS_PER_CLUSTER:
            raise RuntimeError(
                f"{question_id}:{language} has {len(ordered)} refs, "
                f"expected {EXPECTED_REFS_PER_CLUSTER}"
            )
        vectors = []
        for record in ordered:
            cached = cached_vector_for_text(record.stripped_code)
            if cached is None:
                raise RuntimeError(
                    f"Missing embedding for {question_id}:{language}:{record.persona}; "
                    "run --stage embed first"
                )
            vectors.append(l2_normalize(cached))
        matrix = np.asarray(vectors, dtype=np.float32)
        _validate_matrix(matrix, question_id, language)
        key = ClusterKey(question_id, language)
        clusters.append(
            ClusterVectors(
                key=key,
                vector_ids=tuple(range(len(ordered))),
                vectors=matrix,
            )
        )
        manifest[key.token] = {
            "question_id": question_id,
            "language": language,
            "count": len(ordered),
            "personas": [r.persona for r in ordered],
            "models": [r.model for r in ordered],
            "hashes": [_content_hash(r.stripped_code) for r in ordered],
            "unique_hashes": len({_content_hash(r.stripped_code) for r in ordered}),
            "sources": [str(r.path) for r in ordered],
        }

    index = ReferenceIndex.from_clusters(clusters)
    REFERENCE_INDEX_V2_DIR.mkdir(parents=True, exist_ok=True)
    index.save(REFERENCE_INDEX_V2_DIR)
    payload = {
        "bank_version": REFS_V2_BANK_VERSION,
        "dimension": int(clusters[0].vectors.shape[1]),
        "cluster_count": len(clusters),
        "refs_per_cluster": EXPECTED_REFS_PER_CLUSTER,
        "personas": [p.value for p in PERSONA_V2_ORDER],
        "solutions_dir": str(AI_SOLUTIONS_V2_DIR),
        "v1_index_dir": str(REFERENCE_INDEX_V1_DIR),
        "built_at": datetime.now(timezone.utc).isoformat(),
        "clusters": manifest,
    }
    REFERENCE_INDEX_V2_MANIFEST_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"saved {len(clusters)} clusters to {REFERENCE_INDEX_V2_DIR}")
    print(f"manifest -> {REFERENCE_INDEX_V2_MANIFEST_PATH}")
    uniq = [int(v["unique_hashes"]) for v in manifest.values()]
    print(f"unique refs per cluster: mean={np.mean(uniq):.2f} min={min(uniq)} max={max(uniq)}")
    return 0


def _assert_target_is_not_v1() -> None:
    target = REFERENCE_INDEX_V2_DIR.resolve()
    if target == REFERENCE_INDEX_V1_DIR.resolve():
        raise RuntimeError("refs_v2 refused to write into the v1 index directory")
    if AI_SOLUTIONS_V2_DIR.resolve() == AI_SOLUTIONS_DIR.resolve():
        raise RuntimeError("refs_v2 solutions dir collides with the v1 bank")


def _validate_matrix(matrix: np.ndarray, question_id: str, language: str) -> None:
    if matrix.shape != (EXPECTED_REFS_PER_CLUSTER, CANONICALITY_EMBEDDING_DIMENSION):
        raise RuntimeError(f"{question_id}:{language} shape {matrix.shape} unexpected")
    if matrix.dtype != np.float32 or not np.isfinite(matrix).all():
        raise RuntimeError(f"{question_id}:{language} has non-finite or wrong-dtype vectors")
    norms = np.linalg.norm(matrix, axis=1)
    if not np.allclose(norms, 1.0, atol=UNIT_NORM_TOLERANCE):
        raise RuntimeError(f"{question_id}:{language} vectors are not unit-norm")


def _content_hash(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())

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
    ACTIVE_BANK_HASHES_PATH,
    REFERENCE_INDEX_D10_DIR,
    REFERENCE_INDEX_D10_HASHES_PATH,
    REFERENCE_INDEX_D10_MANIFEST_PATH,
    REFERENCE_INDEX_D10_REFERENCES_PATH,
    REFERENCE_INDEX_V1_DIR,
    REFERENCE_INDEX_V2_DIR,
    load_voyage_settings,
)
from nw_ai_code_detector.constants import CANONICALITY_EMBEDDING_DIMENSION
from nw_ai_code_detector.embedder import VoyageEmbedder, cached_vector_for_text, l2_normalize
from nw_ai_code_detector.index import ClusterKey, ClusterVectors, ReferenceIndex
from nw_ai_code_detector.refs_v2_guard import ensure_v1_untouched

BANK_VERSION = "d10"
# v2's six, plus the four v1 personas with no v2 string equivalent. v1's
# optimal_explained is excluded (string-identical to v2's most_efficient) and
# v1's naive_dump is excluded (measured to reduce cluster diversity).
V2_PERSONAS = ("bare", "short_names", "evade_detection",
               "most_efficient", "descriptive_names", "less_obvious")
V1_PERSONAS = ("complete_function", "struggling", "anti_detector", "terse")
EXPECTED_CLUSTERS = 1000
MAX_REFS = len(V2_PERSONAS) + len(V1_PERSONAS)
UNIT_NORM_TOLERANCE = 1e-3


@dataclass(frozen=True)
class Ref:
    persona: str
    model: str
    code: str
    source: str


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the D10 production reference bank.")
    parser.add_argument("--stage", choices=("embed", "build"), required=True)
    args = parser.parse_args()
    ensure_v1_untouched()
    groups = _collect()
    print(f"collected {sum(len(v) for v in groups.values())} refs "
          f"across {len(groups)} clusters (pre-dedup)")
    result = _embed(groups) if args.stage == "embed" else _build(groups)
    ensure_v1_untouched()
    return result


def _assert_sources_present() -> None:
    """Both source banks must exist. Building from a partially-archived tree would
    silently emit a degraded bank instead of failing."""
    for root, label in ((AI_SOLUTIONS_DIR, "v1"), (AI_SOLUTIONS_V2_DIR, "v2")):
        if not root.is_dir():
            raise RuntimeError(
                f"{label} solution bank missing at {root}; D10 must be built from both "
                "source banks (restore it from archive/ before rebuilding)"
            )


def _collect() -> dict[tuple[str, str], list[Ref]]:
    _assert_sources_present()
    wanted = {f"v2:{p}" for p in V2_PERSONAS} | {f"v1:{p}" for p in V1_PERSONAS}
    groups: dict[tuple[str, str], dict[str, Ref]] = {}
    for root, tag in ((AI_SOLUTIONS_DIR, "v1"), (AI_SOLUTIONS_V2_DIR, "v2")):
        for path in sorted(root.rglob("*.json")):
            if path.name.startswith("_"):
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("parse_ok") is not True:
                continue
            key = f"{tag}:{payload['persona']}"
            if key not in wanted:
                continue
            code = payload.get("stripped_code") or ""
            if not code.strip():
                continue
            pair = (str(payload["qid"]), str(payload["language"]))
            groups.setdefault(pair, {})[key] = Ref(
                persona=key, model=str(payload["model"]), code=code, source=str(path)
            )
    ordered = [f"v2:{p}" for p in V2_PERSONAS] + [f"v1:{p}" for p in V1_PERSONAS]
    out = {
        pair: [refs[k] for k in ordered if k in refs]
        for pair, refs in groups.items()
    }
    found = {r.persona for refs in out.values() for r in refs}
    missing = set(ordered) - found
    if missing:
        raise RuntimeError(f"D10 sources incomplete; no refs found for: {sorted(missing)}")
    return out


def _dedup(refs: Sequence[Ref]) -> list[Ref]:
    seen, out = set(), []
    for ref in refs:
        if ref.code in seen:
            continue
        seen.add(ref.code); out.append(ref)
    return out


def _embed(groups) -> int:
    texts = [r.code for refs in groups.values() for r in _dedup(refs)]
    batch = VoyageEmbedder(load_voyage_settings()).embed_texts(texts)
    print(f"embedded {len(texts)} | billed={batch.billed_tokens} cost=${batch.cost_usd:.4f} "
          f"hits={batch.cache_hits} misses={batch.cache_misses}")
    return 0


def _build(groups) -> int:
    _assert_target_is_fresh()
    if len(groups) != EXPECTED_CLUSTERS:
        raise RuntimeError(f"Expected {EXPECTED_CLUSTERS} clusters, found {len(groups)}")
    clusters: list[ClusterVectors] = []
    manifest: dict[str, object] = {}
    hashes_out: dict[str, list[str]] = {}
    texts_out: dict[str, list[dict]] = {}
    sizes: list[int] = []
    for (question_id, language), raw in sorted(groups.items()):
        refs = _dedup(raw)
        if not 2 <= len(refs) <= MAX_REFS:
            raise RuntimeError(f"{question_id}:{language} has {len(refs)} refs after dedup")
        vectors = []
        for ref in refs:
            cached = cached_vector_for_text(ref.code)
            if cached is None:
                raise RuntimeError(
                    f"Missing embedding for {question_id}:{language}:{ref.persona}; "
                    "run --stage embed first"
                )
            vectors.append(l2_normalize(cached))
        matrix = np.asarray(vectors, dtype=np.float32)
        _validate(matrix, question_id, language)
        key = ClusterKey(question_id, language)
        clusters.append(ClusterVectors(key, tuple(range(len(refs))), matrix))
        digests = [_hash(r.code) for r in refs]
        hashes_out[key.token] = digests
        # Row order matches the vectors, so the explanation card can name the
        # nearest reference without any solution directory present.
        texts_out[key.token] = [
            {"persona": r.persona, "model": r.model, "stripped_code": r.code} for r in refs
        ]
        manifest[key.token] = {
            "question_id": question_id, "language": language,
            "count": len(refs), "personas": [r.persona for r in refs],
            "models": [r.model for r in refs], "hashes": digests,
            "sources": [r.source for r in refs],
        }
        sizes.append(len(refs))

    index = ReferenceIndex.from_clusters(clusters)
    REFERENCE_INDEX_D10_DIR.mkdir(parents=True, exist_ok=True)
    index.save(REFERENCE_INDEX_D10_DIR)
    REFERENCE_INDEX_D10_MANIFEST_PATH.write_text(json.dumps({
        "bank_version": BANK_VERSION,
        "dimension": int(clusters[0].vectors.shape[1]),
        "cluster_count": len(clusters),
        "v2_personas": list(V2_PERSONAS), "v1_personas": list(V1_PERSONAS),
        "refs_per_cluster": {"mean": float(np.mean(sizes)),
                             "min": int(min(sizes)), "max": int(max(sizes))},
        "self_contained": True,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "clusters": manifest,
    }, indent=2), encoding="utf-8")
    # Exact-match hashes travel with the bank so serving needs no solution dir.
    REFERENCE_INDEX_D10_HASHES_PATH.write_text(json.dumps(hashes_out, indent=2), encoding="utf-8")
    REFERENCE_INDEX_D10_REFERENCES_PATH.write_text(json.dumps(texts_out), encoding="utf-8")
    print(f"saved {len(clusters)} clusters -> {REFERENCE_INDEX_D10_DIR}")
    print(f"refs per cluster: mean={np.mean(sizes):.2f} min={min(sizes)} max={max(sizes)}")
    print(f"manifest -> {REFERENCE_INDEX_D10_MANIFEST_PATH.name}, "
          f"hashes -> {REFERENCE_INDEX_D10_HASHES_PATH.name}, "
          f"texts -> {REFERENCE_INDEX_D10_REFERENCES_PATH.name}")
    return 0


def load_reference_hashes_d10(
    path: Path | None = None,
) -> dict[tuple[str, str], set[str]]:
    """Serve-time exact-match hashes, read from whichever bank is live."""
    if path is None:
        path = ACTIVE_BANK_HASHES_PATH
    payload = json.loads(path.read_text(encoding="utf-8"))
    out: dict[tuple[str, str], set[str]] = {}
    for token, digests in payload.items():
        question_id, language = token.split(":", 1)
        out[(question_id, language)] = set(digests)
    return out


def _assert_target_is_fresh() -> None:
    target = REFERENCE_INDEX_D10_DIR.resolve()
    for protected in (REFERENCE_INDEX_V1_DIR, REFERENCE_INDEX_V2_DIR):
        if target == protected.resolve():
            raise RuntimeError(f"D10 refused to write into {protected}")


def _validate(matrix: np.ndarray, question_id: str, language: str) -> None:
    if matrix.shape[1] != CANONICALITY_EMBEDDING_DIMENSION:
        raise RuntimeError(f"{question_id}:{language} dim {matrix.shape[1]} unexpected")
    if matrix.dtype != np.float32 or not np.isfinite(matrix).all():
        raise RuntimeError(f"{question_id}:{language} non-finite or wrong dtype")
    if not np.allclose(np.linalg.norm(matrix, axis=1), 1.0, atol=UNIT_NORM_TOLERANCE):
        raise RuntimeError(f"{question_id}:{language} vectors not unit-norm")


def _hash(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())

"""D10-swap = D10 refs minus v1:complete_function, plus the d11 hardened ref.

Self-contained like D10: own vectors, own reference_texts.json, reference_hashes.json
and bank_manifest.json, so serving needs no solution directory. D10 is read-only.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from hashlib import sha256

import numpy as np

from nw_ai_code_detector.config import (
    AI_SOLUTIONS_D11_HARDENED_DIR,
    REFERENCE_INDEX_D10SWAP_DIR,
    REFERENCE_INDEX_D10_DIR,
    REFERENCE_INDEX_D10_REFERENCES_PATH,
    REFERENCE_INDEX_V1_DIR,
)
from nw_ai_code_detector.constants import CANONICALITY_EMBEDDING_DIMENSION
from nw_ai_code_detector.embedder import cached_vector_for_text, l2_normalize
from nw_ai_code_detector.index import ClusterKey, ClusterVectors, ReferenceIndex

DROP_PERSONA = "v1:complete_function"
ADD_PERSONA = "d11:hardened"
EXPECTED_CLUSTERS = 1000
UNIT_NORM_TOLERANCE = 1e-3


def _hardened() -> dict[str, dict]:
    out = {}
    for path in sorted(AI_SOLUTIONS_D11_HARDENED_DIR.rglob("*.json")):
        if path.name.startswith("_"):
            continue
        p = json.loads(path.read_text(encoding="utf-8"))
        if p.get("parse_ok") and (p.get("stripped_code") or "").strip():
            out[f"{p['qid']}:{p['language']}"] = {
                "persona": ADD_PERSONA, "model": p["model"],
                "stripped_code": p["stripped_code"],
            }
    return out


def main() -> int:
    argparse.ArgumentParser(description=__doc__).parse_args()
    for protected in (REFERENCE_INDEX_D10_DIR, REFERENCE_INDEX_V1_DIR):
        if REFERENCE_INDEX_D10SWAP_DIR.resolve() == protected.resolve():
            raise RuntimeError(f"refused to overwrite {protected}")

    base = json.loads(REFERENCE_INDEX_D10_REFERENCES_PATH.read_text(encoding="utf-8"))
    if len(base) != EXPECTED_CLUSTERS:
        raise RuntimeError(f"D10 has {len(base)} clusters, expected {EXPECTED_CLUSTERS}")
    hardened = _hardened()
    print(f"source: D10 {len(base)} clusters | hardened refs {len(hardened)}")

    clusters, texts_out, hashes_out, manifest = [], {}, {}, {}
    sizes, dropped, added = [], 0, 0
    for token, group in sorted(base.items()):
        refs = []
        for ref in group:
            if ref["persona"] == DROP_PERSONA:
                dropped += 1
                continue
            refs.append(ref)
        candidate = hardened.get(token)
        if candidate and candidate["stripped_code"] not in {r["stripped_code"] for r in refs}:
            refs.append(candidate); added += 1
        if len(refs) < 2:
            raise RuntimeError(f"{token} would have {len(refs)} refs")
        vectors = []
        for ref in refs:
            cached = cached_vector_for_text(ref["stripped_code"])
            if cached is None:
                raise RuntimeError(f"missing embedding for {token}:{ref['persona']}")
            vectors.append(l2_normalize(cached))
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.shape[1] != CANONICALITY_EMBEDDING_DIMENSION or matrix.dtype != np.float32:
            raise RuntimeError(f"{token} bad shape/dtype {matrix.shape} {matrix.dtype}")
        if not np.isfinite(matrix).all():
            raise RuntimeError(f"{token} non-finite vectors")
        if not np.allclose(np.linalg.norm(matrix, axis=1), 1.0, atol=UNIT_NORM_TOLERANCE):
            raise RuntimeError(f"{token} vectors not unit-norm")
        qid, language = token.split(":", 1)
        key = ClusterKey(qid, language)
        clusters.append(ClusterVectors(key, tuple(range(len(refs))), matrix))
        texts_out[token] = refs
        digests = [sha256(r["stripped_code"].encode("utf-8")).hexdigest() for r in refs]
        hashes_out[token] = digests
        manifest[token] = {
            "question_id": qid, "language": language, "count": len(refs),
            "personas": [r["persona"] for r in refs],
            "models": [r["model"] for r in refs], "hashes": digests,
        }
        sizes.append(len(refs))

    index = ReferenceIndex.from_clusters(clusters)
    REFERENCE_INDEX_D10SWAP_DIR.mkdir(parents=True, exist_ok=True)
    index.save(REFERENCE_INDEX_D10SWAP_DIR)
    (REFERENCE_INDEX_D10SWAP_DIR / "reference_texts.json").write_text(
        json.dumps(texts_out), encoding="utf-8")
    (REFERENCE_INDEX_D10SWAP_DIR / "reference_hashes.json").write_text(
        json.dumps(hashes_out, indent=2), encoding="utf-8")
    (REFERENCE_INDEX_D10SWAP_DIR / "bank_manifest.json").write_text(json.dumps({
        "bank_version": "d10_swap",
        "derived_from": str(REFERENCE_INDEX_D10_DIR),
        "dropped_persona": DROP_PERSONA, "added_persona": ADD_PERSONA,
        "dimension": int(clusters[0].vectors.shape[1]),
        "cluster_count": len(clusters),
        "refs_per_cluster": {"mean": float(np.mean(sizes)),
                             "min": int(min(sizes)), "max": int(max(sizes))},
        "self_contained": True,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "clusters": manifest,
    }, indent=2), encoding="utf-8")
    print(f"saved {len(clusters)} clusters -> {REFERENCE_INDEX_D10SWAP_DIR}")
    print(f"  dropped {dropped} {DROP_PERSONA} refs, added {added} {ADD_PERSONA} refs")
    print(f"  refs/cluster mean {np.mean(sizes):.2f} min {min(sizes)} max {max(sizes)}")
    print(f"  artifacts: index + reference_texts.json + reference_hashes.json + bank_manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

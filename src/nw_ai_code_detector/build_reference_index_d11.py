"""D11 = D10 + one 'hardened' gpt-5.5 reference per cluster.

D10 is read-only here: its bundled reference_texts.json is the source of the first
9-10 refs, and the hardened ref is appended. Written to a fresh directory so D10
remains the deployable fallback.
"""
from __future__ import annotations

import argparse
import json
from hashlib import sha256

import numpy as np

from nw_ai_code_detector.config import (
    AI_SOLUTIONS_D11_HARDENED_DIR,
    REFERENCE_INDEX_D10_REFERENCES_PATH,
    REFERENCE_INDEX_D11_DIR,
    REFERENCE_INDEX_D11_REFERENCES_PATH,
    REFERENCE_INDEX_DIR,
    load_voyage_settings,
)
from nw_ai_code_detector.embedder import VoyageEmbedder, cached_vector_for_text, l2_normalize
from nw_ai_code_detector.index import ClusterKey, ClusterVectors, ReferenceIndex

HARDENED_PERSONA = "hardened"


def _hardened() -> dict[str, dict]:
    out = {}
    for path in sorted(AI_SOLUTIONS_D11_HARDENED_DIR.rglob("*.json")):
        if path.name.startswith("_"):
            continue
        p = json.loads(path.read_text(encoding="utf-8"))
        if p.get("parse_ok") and (p.get("stripped_code") or "").strip():
            out[f"{p['qid']}:{p['language']}"] = {
                "persona": f"d11:{HARDENED_PERSONA}", "model": p["model"],
                "stripped_code": p["stripped_code"],
            }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("embed", "build"), required=True)
    args = parser.parse_args()
    base = json.loads(REFERENCE_INDEX_D10_REFERENCES_PATH.read_text(encoding="utf-8"))
    extra = _hardened()
    print(f"D10 clusters {len(base)} | hardened refs available {len(extra)}")

    if args.stage == "embed":
        texts = [r["stripped_code"] for r in extra.values()]
        batch = VoyageEmbedder(load_voyage_settings()).embed_texts(texts)
        print(f"  embedded {len(texts)} | cost ${batch.cost_usd:.4f} "
              f"hits={batch.cache_hits} misses={batch.cache_misses}")
        return 0

    if REFERENCE_INDEX_D11_DIR.resolve() == REFERENCE_INDEX_DIR.resolve():
        raise RuntimeError("D11 refused to overwrite the D10 bank")
    clusters, texts_out, sizes, added = [], {}, [], 0
    for token, group in sorted(base.items()):
        refs = list(group)
        candidate = extra.get(token)
        if candidate and candidate["stripped_code"] not in {r["stripped_code"] for r in refs}:
            refs.append(candidate); added += 1
        vectors = []
        for ref in refs:
            v = cached_vector_for_text(ref["stripped_code"])
            if v is None:
                raise RuntimeError(f"missing embedding for {token}; run --stage embed")
            vectors.append(l2_normalize(v))
        qid, language = token.split(":", 1)
        clusters.append(ClusterVectors(ClusterKey(qid, language), tuple(range(len(refs))),
                                       np.asarray(vectors, dtype=np.float32)))
        texts_out[token] = refs
        sizes.append(len(refs))
    index = ReferenceIndex.from_clusters(clusters)
    REFERENCE_INDEX_D11_DIR.mkdir(parents=True, exist_ok=True)
    index.save(REFERENCE_INDEX_D11_DIR)
    REFERENCE_INDEX_D11_REFERENCES_PATH.write_text(json.dumps(texts_out), encoding="utf-8")
    print(f"saved {len(clusters)} clusters -> {REFERENCE_INDEX_D11_DIR}")
    print(f"  hardened ref added to {added} clusters "
          f"({len(base)-added} were byte-identical duplicates)")
    print(f"  refs/cluster mean {np.mean(sizes):.2f} (D10 was 9.42), min {min(sizes)}, max {max(sizes)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

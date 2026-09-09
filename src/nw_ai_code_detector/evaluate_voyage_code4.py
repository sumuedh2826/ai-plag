"""voyage-code-4 vs voyage-code-3, on the clean evaluation.

Builds a parallel code-4 index with the SAME D10 cluster composition, then scores
the same held-out sets and the same verified humans with the same zero-FP recipe.
The code-3 D10 bank is never written.

  --stage embed     re-embed every text with voyage-code-4 (cache-backed)
  --stage build     assemble the parallel code-4 index
  --stage evaluate  recall + human placement, with diffs vs code-3
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

from nw_ai_code_detector.config import (
    CODE4_COMPARISON_REPORT,
    REFERENCE_INDEX_D10_CODE4_DIR,
    REFERENCE_INDEX_D10_REFERENCES_PATH,
    REFERENCE_INDEX_DIR,
    VOYAGE_CODE_4_MODEL,
    VoyageSettings,
    load_voyage_settings,
)
from nw_ai_code_detector.constants import (
    CLUSTER_LOW_DIVERSITY_DISTANCE as FLOOR,
    SIGNIFICANT_TOKEN_THRESHOLDS_BY_LANGUAGE as TOKEN_FLOOR,
)
from nw_ai_code_detector.discount_layer import mean_pairwise_cosine_distance
from nw_ai_code_detector.embedder import VoyageEmbedder, cached_vector_for_text, l2_normalize
from nw_ai_code_detector.evaluate_reference_index_v2 import (
    _load_heldout_ai, _load_labeled_humans,
)
from nw_ai_code_detector.index import ClusterKey, ClusterVectors, ReferenceIndex
from nw_ai_code_detector.significant_code_tokens import (
    SignificantCodeTokenizationError, significant_code_token_count,
)

LANGUAGES = ("CPP", "PYTHON")
EMBED_BATCH = 128
EMBED_CONCURRENCY = 24
MISLABEL = "d286afd6"
# code-3 numbers this run is compared against.
CODE3 = {
    ("CPP", "existing"): 0.8020, ("CPP", "bare"): 0.8801,
    ("PYTHON", "existing"): 0.9033, ("PYTHON", "bare"): 0.9371,
}


def _settings() -> VoyageSettings:
    base = load_voyage_settings()
    return VoyageSettings(api_key=base.api_key, model=VOYAGE_CODE_4_MODEL,
                          batch_size=EMBED_BATCH, concurrency=EMBED_CONCURRENCY,
                          timeout_seconds=base.timeout_seconds)


def _bare_items() -> list[tuple[str, str, str]]:
    out = []
    for path in sorted(glob.glob("data/eval_bare_gpt55/*/*/*.json")):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("parse_ok"):
            out.append((payload["qid"], payload["language"], payload["stripped_code"]))
    return out


def _all_texts() -> list[str]:
    refs = json.loads(REFERENCE_INDEX_D10_REFERENCES_PATH.read_text(encoding="utf-8"))
    texts = [r["stripped_code"] for group in refs.values() for r in group]
    texts += [i.code for i in _load_heldout_ai()]
    texts += [code for _q, _l, code in _bare_items()]
    texts += [i.code for i in _load_labeled_humans()]
    seen, unique = set(), []
    for t in texts:
        if t not in seen:
            seen.add(t); unique.append(t)
    return unique


def stage_embed() -> int:
    texts = _all_texts()
    print(f"embedding {len(texts)} unique texts with {VOYAGE_CODE_4_MODEL} "
          f"(batch {EMBED_BATCH}, concurrency {EMBED_CONCURRENCY})")
    batch = VoyageEmbedder(_settings()).embed_texts(texts)
    print(f"  billed_tokens={batch.billed_tokens:,} cost=${batch.cost_usd:.4f} "
          f"cache_hits={batch.cache_hits} misses={batch.cache_misses}")
    return 0


def _vec(text: str) -> np.ndarray | None:
    cached = cached_vector_for_text(text, VOYAGE_CODE_4_MODEL)
    if cached is None:
        return None
    return np.asarray(l2_normalize(cached), dtype=np.float32)


def stage_build() -> int:
    if REFERENCE_INDEX_D10_CODE4_DIR.resolve() == REFERENCE_INDEX_DIR.resolve():
        raise RuntimeError("refused to overwrite the code-3 bank")
    refs = json.loads(REFERENCE_INDEX_D10_REFERENCES_PATH.read_text(encoding="utf-8"))
    clusters = []
    for token, group in sorted(refs.items()):
        question_id, language = token.split(":", 1)
        vectors = []
        for ref in group:
            v = _vec(ref["stripped_code"])
            if v is None:
                raise RuntimeError(f"missing code-4 embedding for {token}; run --stage embed")
            vectors.append(v)
        matrix = np.asarray(vectors, dtype=np.float32)
        clusters.append(ClusterVectors(ClusterKey(question_id, language),
                                       tuple(range(len(group))), matrix))
    index = ReferenceIndex.from_clusters(clusters)
    REFERENCE_INDEX_D10_CODE4_DIR.mkdir(parents=True, exist_ok=True)
    index.save(REFERENCE_INDEX_D10_CODE4_DIR)
    sizes = [c.vectors.shape[0] for c in clusters]
    print(f"saved {len(clusters)} clusters to {REFERENCE_INDEX_D10_CODE4_DIR} "
          f"(refs/cluster mean {np.mean(sizes):.2f}, dim {clusters[0].vectors.shape[1]})")
    return 0


def _load_bank(directory: Path, model: str):
    index = ReferenceIndex.load(directory)
    clusters, diverse = {}, set()
    for token in index.cluster_tokens():
        q, l = token.split(":", 1)
        m = index.get_cluster(ClusterKey(q, l)).vectors
        clusters[(q, l)] = m
        if mean_pairwise_cosine_distance(m) >= FLOOR:
            diverse.add((q, l))
    return clusters, diverse


def _score(items, clusters, diverse, model, language):
    out = []
    for qid, lang, code, tokens in items:
        if lang != language:
            continue
        key = (qid, lang)
        if key not in clusters or key not in diverse or tokens < TOKEN_FLOOR[lang]:
            continue
        v = cached_vector_for_text(code, model)
        if v is None:
            continue
        q = np.asarray(l2_normalize(v), dtype=np.float32)
        out.append((qid, float(np.max(clusters[key] @ q))))
    return out


def _tokenised(triples):
    out = []
    for qid, lang, code in triples:
        try:
            out.append((qid, lang, code, significant_code_token_count(code, lang)))
        except SignificantCodeTokenizationError:
            continue
    return out


def stage_evaluate() -> int:
    from nw_ai_code_detector.constants import VOYAGE_CODE_3_MODEL
    banks = {
        "code-3": (_load_bank(REFERENCE_INDEX_DIR, VOYAGE_CODE_3_MODEL), VOYAGE_CODE_3_MODEL),
        "code-4": (_load_bank(REFERENCE_INDEX_D10_CODE4_DIR, VOYAGE_CODE_4_MODEL), VOYAGE_CODE_4_MODEL),
    }
    humans = _tokenised([(i.question_id, i.language, i.code)
                         for i in _load_labeled_humans()
                         if not i.question_id.startswith(MISLABEL)])
    existing = _tokenised([(i.question_id, i.language, i.code) for i in _load_heldout_ai()])
    bare = _tokenised(_bare_items())

    report = {}
    for name, ((clusters, diverse), model) in banks.items():
        print("=" * 96); print(f"{name}"); print("=" * 96)
        report[name] = {}
        for language in LANGUAGES:
            neg = _score(humans, clusters, diverse, model, language)
            if not neg:
                continue
            threshold = float(np.nextafter(max(s for _q, s in neg), np.inf))
            row = {"threshold": threshold, "n_humans": len(neg),
                   "max_human": max(s for _q, s in neg)}
            print(f"  {language}: threshold {threshold:.7f}  ({len(neg)} verified humans, "
                  f"max {max(s for _q,s in neg):.6f})")
            for label, items in (("existing", existing), ("bare", bare)):
                pos = _score(items, clusters, diverse, model, language)
                caught = sum(1 for _q, s in pos if s >= threshold)
                recall = caught / len(pos) if pos else 0.0
                row[label] = {"n": len(pos), "caught": caught, "recall": recall}
                base = CODE3[(language, label)]
                diff = f"{recall-base:+.4f}" if name == "code-4" else ""
                print(f"     {label:9s} n={len(pos):4d} caught={caught:4d} "
                      f"recall={recall:.4f} {diff}")
            report[name][language] = row
        print()
    CODE4_COMPARISON_REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {CODE4_COMPARISON_REPORT}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("embed", "build", "evaluate"), required=True)
    args = parser.parse_args()
    return {"embed": stage_embed, "build": stage_build, "evaluate": stage_evaluate}[args.stage]()


if __name__ == "__main__":
    raise SystemExit(main())

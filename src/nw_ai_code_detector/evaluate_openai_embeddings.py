"""openai/text-embedding-3-small vs voyage-code-3, on the clean evaluation.

Parallel 1536-d index with the SAME D10 cluster composition. The voyage-code-3 D10
bank is never written. Embeddings go through OpenRouter (which does serve
/v1/embeddings even though it lists no embedding models) into their own cache dir.

  --stage embed / build / evaluate
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from nw_ai_code_detector.config import (
    ENV_PATH,
    OPENAI_EMBED_CACHE_DIR,
    OPENAI_EMBED_COMPARISON_REPORT,
    OPENAI_EMBED_MODEL,
    REFERENCE_INDEX_D10_OPENAI_DIR,
    REFERENCE_INDEX_D10_REFERENCES_PATH,
    REFERENCE_INDEX_DIR,
)
from nw_ai_code_detector.constants import (
    CLUSTER_LOW_DIVERSITY_DISTANCE as FLOOR3,
    SIGNIFICANT_TOKEN_THRESHOLDS_BY_LANGUAGE as TOKEN_FLOOR,
    VOYAGE_CODE_3_MODEL,
)
from nw_ai_code_detector.discount_layer import mean_pairwise_cosine_distance
from nw_ai_code_detector.embedder import cached_vector_for_text, l2_normalize
from nw_ai_code_detector.evaluate_reference_index_v2 import (
    _load_heldout_ai, _load_labeled_humans,
)
from nw_ai_code_detector.index import ClusterKey, ClusterVectors, ReferenceIndex
from nw_ai_code_detector.significant_code_tokens import (
    SignificantCodeTokenizationError, significant_code_token_count,
)

LANGUAGES = ("CPP", "PYTHON")
BATCH = 128
WORKERS = 16
ENDPOINT = "https://openrouter.ai/api/v1/embeddings"
VOYAGE3 = {("CPP", "existing"): 0.8020, ("CPP", "bare"): 0.8801,
           ("PYTHON", "existing"): 0.9033, ("PYTHON", "bare"): 0.9371}


def _path(text: str) -> Path:
    digest = sha256(f"{OPENAI_EMBED_MODEL}|{text}".encode("utf-8")).hexdigest()
    return OPENAI_EMBED_CACHE_DIR / f"{digest}.json"


def openai_vector(text: str) -> np.ndarray | None:
    path = _path(text)
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return np.asarray(l2_normalize(payload["vector"]), dtype=np.float32)


def _embed_batch(texts: list[str], key: str) -> tuple[list[list[float]], float, int]:
    request = urllib.request.Request(
        ENDPOINT,
        data=json.dumps({"model": OPENAI_EMBED_MODEL, "input": texts}).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    for attempt in range(5):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                payload = json.loads(response.read())
            usage = payload.get("usage") or {}
            return ([d["embedding"] for d in payload["data"]],
                    float(usage.get("cost") or 0.0), int(usage.get("total_tokens") or 0))
        except Exception:
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


def _bare_items():
    out = []
    for path in sorted(glob.glob("data/eval_bare_gpt55/*/*/*.json")):
        p = json.loads(Path(path).read_text(encoding="utf-8"))
        if p.get("parse_ok"):
            out.append((p["qid"], p["language"], p["stripped_code"]))
    return out


def _all_texts() -> list[str]:
    refs = json.loads(REFERENCE_INDEX_D10_REFERENCES_PATH.read_text(encoding="utf-8"))
    texts = [r["stripped_code"] for g in refs.values() for r in g]
    texts += [i.code for i in _load_heldout_ai()]
    texts += [c for _q, _l, c in _bare_items()]
    texts += [i.code for i in _load_labeled_humans()]
    seen, unique = set(), []
    for t in texts:
        if t not in seen:
            seen.add(t); unique.append(t)
    return unique


def stage_embed() -> int:
    load_dotenv(ENV_PATH)
    key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY missing")
    OPENAI_EMBED_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    texts = [t for t in _all_texts() if not _path(t).is_file()]
    print(f"{OPENAI_EMBED_MODEL}: {len(texts)} texts to embed "
          f"(batch {BATCH}, {WORKERS} workers)")
    if not texts:
        print("  all cached already"); return 0
    batches = [texts[i:i + BATCH] for i in range(0, len(texts), BATCH)]
    started = time.perf_counter()
    total_cost = total_tokens = 0.0
    done = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for batch, (vectors, cost, tokens) in zip(
            batches, pool.map(lambda b: _embed_batch(b, key), batches)
        ):
            for text, vector in zip(batch, vectors):
                _path(text).write_text(json.dumps({"vector": vector}), encoding="utf-8")
            total_cost += cost; total_tokens += tokens; done += len(batch)
            if done % 2048 < BATCH:
                print(f"  {done}/{len(texts)}  ${total_cost:.4f}  "
                      f"{time.perf_counter()-started:.0f}s", flush=True)
    elapsed = time.perf_counter() - started
    print(f"\n  embedded {done} texts | tokens {int(total_tokens):,} | "
          f"cost ${total_cost:.4f} | {elapsed:.0f}s")
    return 0


def stage_build() -> int:
    if REFERENCE_INDEX_D10_OPENAI_DIR.resolve() == REFERENCE_INDEX_DIR.resolve():
        raise RuntimeError("refused to overwrite the voyage bank")
    refs = json.loads(REFERENCE_INDEX_D10_REFERENCES_PATH.read_text(encoding="utf-8"))
    clusters = []
    for token, group in sorted(refs.items()):
        qid, language = token.split(":", 1)
        vectors = []
        for ref in group:
            v = openai_vector(ref["stripped_code"])
            if v is None:
                raise RuntimeError(f"missing embedding for {token}; run --stage embed")
            vectors.append(v)
        clusters.append(ClusterVectors(ClusterKey(qid, language),
                                       tuple(range(len(group))),
                                       np.asarray(vectors, dtype=np.float32)))
    index = ReferenceIndex.from_clusters(clusters)
    REFERENCE_INDEX_D10_OPENAI_DIR.mkdir(parents=True, exist_ok=True)
    index.save(REFERENCE_INDEX_D10_OPENAI_DIR)
    sizes = [c.vectors.shape[0] for c in clusters]
    print(f"saved {len(clusters)} clusters -> {REFERENCE_INDEX_D10_OPENAI_DIR} "
          f"(mean {np.mean(sizes):.2f} refs, dim {clusters[0].vectors.shape[1]})")
    return 0


def _bank(directory: Path):
    index = ReferenceIndex.load(directory)
    clusters, divs = {}, {}
    for token in index.cluster_tokens():
        q, l = token.split(":", 1)
        m = index.get_cluster(ClusterKey(q, l)).vectors
        clusters[(q, l)] = m
        divs[(q, l)] = mean_pairwise_cosine_distance(m)
    return clusters, divs


def _tokenised(triples):
    out = []
    for qid, lang, code in triples:
        try:
            out.append((qid, lang, code, significant_code_token_count(code, lang)))
        except SignificantCodeTokenizationError:
            continue
    return out


def _score(items, clusters, diverse, getter, language):
    out = []
    for qid, lang, code, tokens in items:
        key = (qid, lang)
        if lang != language or key not in clusters or key not in diverse:
            continue
        if tokens < TOKEN_FLOOR[lang]:
            continue
        v = getter(code)
        if v is None:
            continue
        out.append((qid, float(np.max(clusters[key] @ v))))
    return out


def stage_evaluate() -> int:
    v_clusters, v_divs = _bank(REFERENCE_INDEX_DIR)
    o_clusters, o_divs = _bank(REFERENCE_INDEX_D10_OPENAI_DIR)
    keep = float(np.mean([d >= FLOOR3 for d in v_divs.values()]))
    floor_openai = float(np.quantile(list(o_divs.values()), 1 - keep))
    print(f"voyage-code-3 floor {FLOOR3} keeps {100*keep:.1f}% of clusters")
    print(f"scale-matched OpenAI floor: {floor_openai:.4f}\n")

    def voyage_vec(code):
        v = cached_vector_for_text(code, VOYAGE_CODE_3_MODEL)
        return None if v is None else np.asarray(l2_normalize(v), dtype=np.float32)

    humans = _tokenised([(i.question_id, i.language, i.code)
                         for i in _load_labeled_humans()
                         if not i.question_id.startswith("d286afd6")])
    existing = _tokenised([(i.question_id, i.language, i.code) for i in _load_heldout_ai()])
    bare = _tokenised(_bare_items())

    banks = {
        "voyage-code-3 (1024d)": (v_clusters, {k for k, d in v_divs.items() if d >= FLOOR3},
                                  voyage_vec, FLOOR3),
        "openai-3-small (1536d)": (o_clusters, {k for k, d in o_divs.items() if d >= floor_openai},
                                   openai_vector, floor_openai),
    }
    report = {}
    print(f"{'bank':24s} {'lang':7s} {'floor':>7s} {'humans':>7s} {'threshold':>11s} "
          f"{'exist rec':>10s} {'bare rec':>9s} {'CAUGHT':>7s}")
    for name, (clusters, diverse, getter, floor) in banks.items():
        report[name] = {}
        for language in LANGUAGES:
            neg = _score(humans, clusters, diverse, getter, language)
            if not neg:
                continue
            threshold = float(np.nextafter(max(s for _q, s in neg), np.inf))
            row = {"threshold": threshold, "n_humans": len(neg), "floor": floor}
            e = _score(existing, clusters, diverse, getter, language)
            b = _score(bare, clusters, diverse, getter, language)
            ec = sum(1 for _q, s in e if s >= threshold)
            bc = sum(1 for _q, s in b if s >= threshold)
            row["existing"] = {"n": len(e), "caught": ec, "recall": ec / len(e)}
            row["bare"] = {"n": len(b), "caught": bc, "recall": bc / len(b)}
            report[name][language] = row
            print(f"{name:24s} {language:7s} {floor:7.4f} {len(neg):7d} {threshold:11.7f} "
                  f"{ec/len(e):10.4f} {bc/len(b):9.4f} {ec+bc:7d}")
    print()
    for language in LANGUAGES:
        for which in ("existing", "bare"):
            base = VOYAGE3[(language, which)]
            new = report["openai-3-small (1536d)"][language][which]["recall"]
            print(f"  {language:7s} {which:9s} voyage {base:.4f} -> openai {new:.4f}  "
                  f"({new-base:+.4f})")
    OPENAI_EMBED_COMPARISON_REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {OPENAI_EMBED_COMPARISON_REPORT}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("embed", "build", "evaluate"), required=True)
    args = parser.parse_args()
    return {"embed": stage_embed, "build": stage_build,
            "evaluate": stage_evaluate}[args.stage]()


if __name__ == "__main__":
    raise SystemExit(main())

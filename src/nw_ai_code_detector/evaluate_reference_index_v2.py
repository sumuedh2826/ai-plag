from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from nw_ai_code_detector.config import (
    DATA_DIR,
    EVAL_AI_SOLUTIONS_DIR,
    REFERENCE_INDEX_V1_DIR,
    REFERENCE_INDEX_V2_DIR,
    REFS_V2_EVAL_DIR,
)
from nw_ai_code_detector.constants import (
    CLUSTER_LOW_DIVERSITY_DISTANCE,
    GROUPS_KEY,
    HUMAN_STRIPPED_CODE_FIELD,
    SIGNIFICANT_TOKEN_THRESHOLDS_BY_LANGUAGE,
)
from nw_ai_code_detector.discount_layer import mean_pairwise_cosine_distance
from nw_ai_code_detector.embedder import cached_vector_for_text, l2_normalize
from nw_ai_code_detector.index import ClusterKey, ReferenceIndex
from nw_ai_code_detector.refs_v2_guard import ensure_v1_untouched
from nw_ai_code_detector.significant_code_tokens import (
    SignificantCodeTokenizationError,
    significant_code_token_count,
)

MANUAL_LABELS_PATH = Path("outputs/manual_labels.jsonl")
LANGUAGES = ("CPP", "PYTHON")


@dataclass(frozen=True)
class Item:
    question_id: str
    language: str
    label: str          # "human" | "ai"
    code: str
    token_count: int


@dataclass(frozen=True)
class Scored:
    item: Item
    nn_max: float
    scoreable: bool


def main() -> int:
    ensure_v1_untouched()
    humans = _load_labeled_humans()
    ai = _load_heldout_ai()
    print(f"labeled HUMAN negatives: {len(humans)}  |  held-out AI positives: {len(ai)}")

    banks = {
        "v1": ReferenceIndex.load(REFERENCE_INDEX_V1_DIR),
        "v2": ReferenceIndex.load(REFERENCE_INDEX_V2_DIR),
    }
    scored = {name: _score_all(index, humans + ai) for name, index in banks.items()}

    # Clusters scoreable (diverse enough) in BOTH banks -> apples-to-apples population.
    both = _intersection_clusters(banks)
    print(f"clusters scoreable in BOTH banks: {len(both)} of {len(banks['v1'].cluster_tokens())}")

    report: dict[str, object] = {"intersection_cluster_count": len(both), "results": {}}
    print("\n" + "=" * 96)
    print("PRIMARY - recall on the INTERSECTION (clusters scoreable in both banks)")
    print("=" * 96)
    _print_table(scored, both, report, "intersection")

    print("\n" + "=" * 96)
    print("SECONDARY - each bank on its OWN full scoreable population")
    print("=" * 96)
    _print_table(scored, None, report, "full_scoreable")

    REFS_V2_EVAL_DIR.mkdir(parents=True, exist_ok=True)
    (REFS_V2_EVAL_DIR / "comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {REFS_V2_EVAL_DIR / 'comparison.json'}")
    ensure_v1_untouched()
    return 0


def _print_table(
    scored: Mapping[str, Sequence[Scored]],
    restrict: set[tuple[str, str]] | None,
    report: dict,
    scope: str,
) -> None:
    print(
        f"{'lang':7s} {'bank':5s} {'T_zero_fp':>12s} {'humans':>7s} {'flagged':>8s} "
        f"{'n_ai':>6s} {'caught':>7s} {'recall':>8s}"
    )
    section: dict[str, object] = {}
    for language in LANGUAGES:
        row: dict[str, object] = {}
        for bank in ("v1", "v2"):
            rows = [
                s for s in scored[bank]
                if s.item.language == language and s.scoreable
                and (restrict is None or (s.item.question_id, s.item.language) in restrict)
            ]
            neg = [s.nn_max for s in rows if s.item.label == "human"]
            pos = [s.nn_max for s in rows if s.item.label == "ai"]
            if not neg or not pos:
                print(f"{language:7s} {bank:5s} {'n/a':>12s} {len(neg):7d} {'-':>8s} {len(pos):6d}")
                continue
            threshold = float(np.nextafter(max(neg), np.inf))
            flagged = sum(1 for v in neg if v >= threshold)
            caught = sum(1 for v in pos if v >= threshold)
            recall = caught / len(pos)
            print(
                f"{language:7s} {bank:5s} {threshold:12.7f} {len(neg):7d} {flagged:8d} "
                f"{len(pos):6d} {caught:7d} {recall:8.4f}"
            )
            row[bank] = {
                "T_zero_fp": threshold,
                "n_humans": len(neg),
                "flagged_humans": flagged,
                "n_ai": len(pos),
                "caught": caught,
                "recall": recall,
            }
        if "v1" in row and "v2" in row:
            delta = row["v2"]["recall"] - row["v1"]["recall"]
            verdict = "BETTER" if delta > 0.02 else ("WORSE" if delta < -0.02 else "SAME")
            print(f"{'':7s} {'':5s} {'delta':>12s} {'':7s} {'':8s} {'':6s} {'':7s} "
                  f"{delta:+8.4f}   -> {verdict}")
            row["delta_recall"] = delta
            row["verdict"] = verdict
        section[language] = row
    report["results"][scope] = section


def _intersection_clusters(banks: Mapping[str, ReferenceIndex]) -> set[tuple[str, str]]:
    per_bank = []
    for index in banks.values():
        ok = set()
        for token in index.cluster_tokens():
            cluster = index.get_cluster(ClusterKey(*token.split(":", 1)))
            if mean_pairwise_cosine_distance(cluster.vectors) >= CLUSTER_LOW_DIVERSITY_DISTANCE:
                ok.add((cluster.key.question_id, cluster.key.language))
        per_bank.append(ok)
    return set.intersection(*per_bank)


def _score_all(index: ReferenceIndex, items: Sequence[Item]) -> list[Scored]:
    out: list[Scored] = []
    for item in items:
        key = ClusterKey(item.question_id, item.language)
        try:
            cluster = index.get_cluster(key)
        except KeyError:
            continue
        if cluster.key.question_id != item.question_id:
            raise RuntimeError("Cross-question reference routing")
        if cluster.key.language != item.language:
            raise RuntimeError("Cross-language reference routing")
        vector = cached_vector_for_text(item.code)
        if vector is None:
            continue
        query = np.asarray(l2_normalize(vector), dtype=np.float32)
        nn_max = float(np.max(cluster.vectors @ query))
        diverse = mean_pairwise_cosine_distance(cluster.vectors) >= CLUSTER_LOW_DIVERSITY_DISTANCE
        enough = item.token_count >= SIGNIFICANT_TOKEN_THRESHOLDS_BY_LANGUAGE[item.language]
        out.append(Scored(item=item, nn_max=nn_max, scoreable=bool(diverse and enough)))
    return out


def _load_labeled_humans() -> list[Item]:
    labels = {}
    for line in MANUAL_LABELS_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            payload = json.loads(line)
            labels[payload["record_id"]] = payload
    groups = json.loads((DATA_DIR / "scored_submissions.json").read_text(encoding="utf-8"))
    mapping = groups.get(GROUPS_KEY) or {}
    items: list[Item] = []
    for record_id, payload in labels.items():
        if payload.get("my_label") != "HUMAN":
            continue
        parts = record_id.split("|")
        if len(parts) < 4:
            continue
        qid, language, group_index = parts[1], parts[2], int(parts[3])
        records = mapping.get(f"{qid}:{language}")
        if not isinstance(records, list) or group_index >= len(records):
            continue
        code = records[group_index].get(HUMAN_STRIPPED_CODE_FIELD) or ""
        if not code.strip():
            continue
        try:
            tokens = significant_code_token_count(code, language)
        except SignificantCodeTokenizationError:
            continue
        items.append(Item(qid, language, "human", code, tokens))
    return items


def _load_heldout_ai() -> list[Item]:
    items: list[Item] = []
    for path in sorted(EVAL_AI_SOLUTIONS_DIR.rglob("*.json")):
        if path.name.startswith("_"):
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("parse_ok") is not True:
            continue
        code = payload.get("stripped_code") or ""
        if not code.strip():
            continue
        language = str(payload["language"])
        try:
            tokens = significant_code_token_count(code, language)
        except SignificantCodeTokenizationError:
            continue
        items.append(Item(str(payload["qid"]), language, "ai", code, tokens))
    return items


if __name__ == "__main__":
    raise SystemExit(main())

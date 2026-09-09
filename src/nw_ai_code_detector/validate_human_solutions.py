"""False-positive harness for manually sourced, pre-2022 human solutions.

This module NEVER writes a solution. It scaffolds empty paste files, applies a
mechanical rename so a sourced solution fits our boilerplate signature, and scores
the result against the production bank. Every rename is reported as a diff so the
adaptation can be audited: if anything but identifiers changed, the run is rejected.

  --scaffold   create the paste files + manifest
  --score      score whatever has been filled in
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

from nw_ai_code_detector.config import (
    HUMAN_VALIDATION_DIR,
    HUMAN_VALIDATION_MANIFEST,
    HUMAN_VALIDATION_REPORT,
    REFERENCE_INDEX_DIR,
    load_voyage_settings,
)
from nw_ai_code_detector.constants import (
    CLUSTER_LOW_DIVERSITY_DISTANCE,
    SIGNIFICANT_TOKEN_THRESHOLDS_BY_LANGUAGE,
)
from nw_ai_code_detector.data_load import load_dataset
from nw_ai_code_detector.discount_layer import mean_pairwise_cosine_distance
from nw_ai_code_detector.embedder import VoyageEmbedder, l2_normalize
from nw_ai_code_detector.index import ClusterKey, ReferenceIndex
from nw_ai_code_detector.significant_code_tokens import (
    SignificantCodeTokenizationError,
    significant_code_token_count,
)
from nw_ai_code_detector.stripper import Language, SourceParseError, strip_solution_body

PLACEHOLDER = "PASTE THE SOURCED PRE-2022 SOLUTION BELOW THIS LINE"
LANGUAGES = ("CPP", "PYTHON")
EXTENSION = {"CPP": "cpp", "PYTHON": "py"}
# LeetCode's own type names, mapped onto whatever our boilerplate uses.
NODE_TYPES = ("ListNode", "TreeNode")


@dataclass
class Target:
    qid: str
    lc: int
    lcname: str
    language: str
    function: str
    our_params: list[str]
    our_signature: str
    path: str


# --------------------------------------------------------------------------
# signature parsing (our boilerplate, and the pasted solution)
# --------------------------------------------------------------------------

def _cpp_signature(source: str, function: str) -> tuple[str, list[str]] | None:
    m = re.search(rf"^[^\n]*\b{re.escape(function)}\s*\(([^)]*)\)", source, re.M)
    if not m:
        return None
    return m.group(0).strip(), _cpp_params(m.group(1))


def _cpp_params(raw: str) -> list[str]:
    names = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        ident = re.findall(r"[A-Za-z_]\w*", part)
        if ident:
            names.append(ident[-1])
    return names


def _py_signature(source: str, function: str) -> tuple[str, list[str]] | None:
    m = re.search(rf"^\s*def\s+{re.escape(function)}\s*\(([^)]*)\)", source, re.M)
    if not m:
        return None
    params = [
        p.strip().split(":")[0].split("=")[0].strip()
        for p in m.group(1).split(",")
        if p.strip() and p.strip() != "self"
    ]
    return m.group(0).strip(), params


def signature_of(source: str, function: str, language: str):
    return (_cpp_signature if language == "CPP" else _py_signature)(source, function)


def _candidate_methods(source: str, language: str) -> list[tuple[str, list[str]]]:
    """Every method the pasted solution defines, as (name, params)."""
    out: list[tuple[str, list[str]]] = []
    if language == "PYTHON":
        for m in re.finditer(r"^\s*def\s+(\w+)\s*\(([^)]*)\)", source, re.M):
            if m.group(1).startswith("__"):
                continue
            params = [
                p.strip().split(":")[0].split("=")[0].strip()
                for p in m.group(2).split(",")
                if p.strip() and p.strip() != "self"
            ]
            out.append((m.group(1), params))
        return out
    for m in re.finditer(r"^[ \t]*[\w:<>,\s\*&]+?\b(\w+)\s*\(([^)]*)\)\s*\{", source, re.M):
        name = m.group(1)
        if name in {"if", "for", "while", "switch", "return", "main"}:
            continue
        out.append((name, _cpp_params(m.group(2))))
    return out


def _select_method(source: str, target: "Target") -> tuple[str, list[str]]:
    """Pick the entry point. A sourced solution often carries private helpers, so
    take the one whose parameter count matches our boilerplate. Ambiguity is
    refused rather than guessed - a wrong pick would rename the wrong function."""
    candidates = _candidate_methods(source, target.language)
    if not candidates:
        raise ValueError("Could not find a function definition in the pasted solution")
    exact = [c for c in candidates if len(c[1]) == len(target.our_params)]
    if len(exact) == 1:
        return exact[0]
    listing = "; ".join(f"{n}({', '.join(p)})" for n, p in candidates)
    if not exact:
        raise ValueError(
            f"No method matches our {len(target.our_params)}-parameter signature "
            f"{target.our_signature}. Found: {listing}"
        )
    raise ValueError(
        f"Ambiguous: {len(exact)} methods match our parameter count. Found: {listing}. "
        "Rename the entry point by hand so the intended one is unambiguous."
    )


# --------------------------------------------------------------------------
# mechanical adaptation - identifiers only, never logic
# --------------------------------------------------------------------------

def adapt(pasted: str, target: Target, node_type: str | None) -> tuple[str, list[str]]:
    """Rename class, method and parameters to fit our boilerplate. Returns
    (adapted source, list of renames applied)."""
    their_name, their_params = _select_method(pasted, target)
    renames: list[tuple[str, str]] = []

    if their_name != target.function:
        renames.append((their_name, target.function))
    if len(their_params) != len(target.our_params):
        raise ValueError(
            f"Parameter count differs: pasted has {len(their_params)} "
            f"{their_params}, our boilerplate expects {len(target.our_params)} "
            f"{target.our_params}. Adapt the signature by hand or skip this one."
        )
    for theirs, ours in zip(their_params, target.our_params):
        if theirs != ours:
            renames.append((theirs, ours))
    for lc_type in NODE_TYPES:
        if node_type and lc_type != node_type and re.search(rf"\b{lc_type}\b", pasted):
            renames.append((lc_type, node_type))
    if re.search(r"\bclass\s+Solution\b", pasted):
        renames.append(("Solution", "solution"))

    adapted = pasted
    for old, new in renames:
        adapted = re.sub(rf"\b{re.escape(old)}\b", new, adapted)
    return adapted, [f"{o} -> {n}" for o, n in renames]


def rename_only_diff(before: str, after: str, renames: list[str]) -> list[str]:
    """Changed lines, so the adaptation can be eyeballed. Also verifies that the
    only differences are the declared identifier substitutions."""
    mapping = dict(r.split(" -> ") for r in renames)
    canonical_before = before
    for old, new in mapping.items():
        canonical_before = re.sub(rf"\b{re.escape(old)}\b", new, canonical_before)
    if canonical_before != after:
        raise ValueError("Adaptation changed more than the declared identifiers")
    return [
        line for line in difflib.unified_diff(
            before.splitlines(), after.splitlines(), lineterm="", n=0
        )
        if line.startswith(("+", "-")) and not line.startswith(("+++", "---"))
    ]


# --------------------------------------------------------------------------

def _boilerplate(question, language: str) -> str:
    keys = ("CPP",) if language == "CPP" else ("PYTHON", "PYTHON39")
    for key in keys:
        entry = question.boilerplates.get(key)
        if isinstance(entry, str) and entry.strip():
            return entry
    return ""


def build_targets() -> list[Target]:
    candidates = [
        c for c in json.loads(Path("outputs/leetcode_candidates.json").read_text())
        if c["tier"] == "A"
    ]
    dataset = load_dataset()
    targets: list[Target] = []
    for candidate in candidates:
        question = dataset.questions[candidate["qid"]]
        for language in LANGUAGES:
            boiler = _boilerplate(question, language)
            if not boiler.strip():
                continue
            parsed = signature_of(boiler, candidate["fn"], language)
            if parsed is None:
                continue
            signature, params = parsed
            targets.append(Target(
                qid=candidate["qid"], lc=candidate["lc"], lcname=candidate["lcname"],
                language=language, function=candidate["fn"], our_params=params,
                our_signature=signature,
                path=str(HUMAN_VALIDATION_DIR / language /
                         f"LC{candidate['lc']}_{candidate['qid'][:8]}.{EXTENSION[language]}"),
            ))
    return targets


def scaffold() -> int:
    targets = build_targets()
    created = skipped = 0
    for target in targets:
        path = Path(target.path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_file():
            skipped += 1
            continue
        comment = "//" if target.language == "CPP" else "#"
        path.write_text("\n".join([
            f"{comment} LeetCode {target.lc}: {target.lcname}",
            f"{comment} our question id : {target.qid}",
            f"{comment} our signature   : {target.our_signature}",
            f"{comment} our parameters  : {', '.join(target.our_params) or '(none)'}",
            f"{comment}",
            f"{comment} Paste a REAL pre-2022 human solution (LeetCode discuss / old GitHub).",
            f"{comment} Paste it as-you-found-it; the harness renames the class, method and",
            f"{comment} parameters to fit our boilerplate and prints the diff. Do not rewrite logic.",
            f"{comment} Optional provenance, kept out of scoring:",
            f"{comment} SOURCE: <url>",
            f"{comment} DATE:   <YYYY-MM-DD, must be before 2022>",
            f"{comment}",
            f"{comment} {PLACEHOLDER}",
            "",
        ]), encoding="utf-8")
        created += 1
    HUMAN_VALIDATION_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    HUMAN_VALIDATION_MANIFEST.write_text(
        json.dumps([asdict(t) for t in targets], indent=2), encoding="utf-8")
    print(f"scaffolded {created} paste files ({skipped} already existed) "
          f"across {len({t.qid for t in targets})} questions")
    print(f"  files    -> {HUMAN_VALIDATION_DIR}")
    print(f"  manifest -> {HUMAN_VALIDATION_MANIFEST}")
    return 0


def _pasted_body(path: Path) -> str | None:
    text = path.read_text(encoding="utf-8")
    if PLACEHOLDER not in text:
        return text.strip() or None
    body = text.split(PLACEHOLDER, 1)[1]
    body = "\n".join(
        line for line in body.splitlines()
        if not re.match(r"^\s*(//|#)\s*(SOURCE|DATE):", line)
    )
    return body.strip() or None


def score() -> int:
    targets = build_targets()
    index = ReferenceIndex.load(REFERENCE_INDEX_DIR)
    dataset = load_dataset()
    embedder = VoyageEmbedder(load_voyage_settings())

    pending = []
    for target in targets:
        body = _pasted_body(Path(target.path))
        if body:
            pending.append((target, body))
    if not pending:
        print("No solutions pasted yet. Fill some files, then re-run --score.")
        return 0
    print(f"found {len(pending)} pasted solutions\n")

    rows, texts = [], []
    for target, body in pending:
        question = dataset.questions[target.qid]
        boiler = _boilerplate(question, target.language)
        node_type = "Node" if re.search(r"\bNode\b", boiler) else None
        try:
            adapted, renames = adapt(body, target, node_type)
            diff = rename_only_diff(body, adapted, renames)
            stripped = strip_solution_body(adapted, boiler, Language(target.language))
            tokens = significant_code_token_count(stripped, target.language)
        except (ValueError, SourceParseError, SignificantCodeTokenizationError) as exc:
            rows.append({**asdict(target), "outcome": "adaptation_failed",
                         "detail": str(exc)})
            continue
        rows.append({**asdict(target), "outcome": "pending", "renames": renames,
                     "diff_lines": len(diff), "token_count": tokens,
                     "stripped": stripped})
        texts.append(stripped)

    if texts:
        batch = embedder.embed_texts(texts)
        vectors = iter(batch.vectors)
        print(f"embedded {len(texts)} | cost ${batch.cost_usd:.4f} "
              f"hits={batch.cache_hits} misses={batch.cache_misses}\n")
        for row in rows:
            if row["outcome"] != "pending":
                continue
            vector = np.asarray(l2_normalize(next(vectors)), dtype=np.float32)
            key = ClusterKey(row["qid"], row["language"])
            try:
                cluster = index.get_cluster(key)
            except KeyError:
                row["outcome"] = "no_cluster"
                continue
            if cluster.key.question_id != row["qid"]:
                raise RuntimeError("Cross-question reference routing")
            if cluster.key.language != row["language"]:
                raise RuntimeError("Cross-language reference routing")
            row["nn_max"] = float(np.max(cluster.vectors @ vector))
            row["cluster_diversity"] = mean_pairwise_cosine_distance(cluster.vectors)
            if row["token_count"] < SIGNIFICANT_TOKEN_THRESHOLDS_BY_LANGUAGE[row["language"]]:
                row["outcome"] = "not_scored_too_short"
            elif row["cluster_diversity"] < CLUSTER_LOW_DIVERSITY_DISTANCE:
                row["outcome"] = "not_scored_tight_cluster"
            else:
                row["outcome"] = "scored"
    _report(rows)
    return 0


def _report(rows: list[dict]) -> None:
    from nw_ai_code_detector.evaluate_reference_index_v2 import _load_labeled_humans
    # Threshold = the production zero-FP point, derived exactly as the evaluator does.
    index = ReferenceIndex.load(REFERENCE_INDEX_DIR)
    thresholds: dict[str, float] = {}
    for language in LANGUAGES:
        negatives = []
        for human in _load_labeled_humans():
            if human.language != language or human.question_id.startswith("d286afd6"):
                continue
            try:
                cluster = index.get_cluster(ClusterKey(human.question_id, human.language))
            except KeyError:
                continue
            from nw_ai_code_detector.embedder import cached_vector_for_text
            cached = cached_vector_for_text(human.code)
            if cached is None:
                continue
            if (human.token_count < SIGNIFICANT_TOKEN_THRESHOLDS_BY_LANGUAGE[language]
                    or mean_pairwise_cosine_distance(cluster.vectors)
                    < CLUSTER_LOW_DIVERSITY_DISTANCE):
                continue
            negatives.append(float(np.max(
                cluster.vectors @ np.asarray(l2_normalize(cached), dtype=np.float32))))
        if negatives:
            thresholds[language] = float(np.nextafter(max(negatives), np.inf))

    print(f"{'LC':>6s} {'qid':10s} {'lang':7s} {'tokens':>6s} {'nn_max':>9s} "
          f"{'threshold':>10s} {'outcome':24s} {'VERDICT'}")
    counts = {"scored": 0, "flagged": 0}
    for row in sorted(rows, key=lambda r: (r["language"], r["lc"])):
        threshold = thresholds.get(row["language"])
        verdict = ""
        if row["outcome"] == "scored" and threshold is not None:
            counts["scored"] += 1
            flagged = row["nn_max"] >= threshold
            row["flagged"] = flagged
            counts["flagged"] += int(flagged)
            verdict = "FALSE POSITIVE" if flagged else "ok (not flagged)"
        nn = f"{row['nn_max']:.6f}" if "nn_max" in row else "-"
        th = f"{threshold:.6f}" if threshold else "-"
        print(f"{row['lc']:>6d} {row['qid'][:8]:10s} {row['language']:7s} "
              f"{row.get('token_count','-'):>6} {nn:>9s} {th:>10s} "
              f"{row['outcome']:24s} {verdict}")
        if row["outcome"] == "adaptation_failed":
            print(f"          ! {row['detail']}")
    print()
    if counts["scored"]:
        rate = counts["flagged"] / counts["scored"]
        print(f"GUARANTEED-HUMAN FALSE-POSITIVE RATE: {counts['flagged']}/{counts['scored']}"
              f" = {100*rate:.1f}%")
    else:
        print("Nothing scoreable yet.")
    for row in rows:
        row.pop("stripped", None)
    HUMAN_VALIDATION_REPORT.write_text(json.dumps(
        {"thresholds": thresholds, "rows": rows}, indent=2), encoding="utf-8")
    print(f"wrote {HUMAN_VALIDATION_REPORT}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--scaffold", action="store_true")
    group.add_argument("--score", action="store_true")
    args = parser.parse_args()
    return scaffold() if args.scaffold else score()


if __name__ == "__main__":
    raise SystemExit(main())

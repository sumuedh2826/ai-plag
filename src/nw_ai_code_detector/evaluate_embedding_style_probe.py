from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from collections.abc import Mapping, Sequence

import numpy as np

from nw_ai_code_detector.build_canonicality_eligibility_tokens import (
    load_candidate_human_records,
)
from nw_ai_code_detector.build_model_dataset_v2 import load_question_assignments
from nw_ai_code_detector.config import (
    DATA_DIR,
    EVAL_AI_SOLUTIONS_DIR,
    OUTPUTS_DIR,
    REFERENCE_INDEX_DIR,
)
from nw_ai_code_detector.constants import (
    GROUPS_KEY,
    HELDOUT_AI_SOURCE,
    NAMING_SHORT_NAME_MAX_LENGTH,
)
from nw_ai_code_detector.eligibility_data import load_solution_records
from nw_ai_code_detector.embedder import cached_vector_for_text
from nw_ai_code_detector.evaluate_style_signals import _human_raw
from nw_ai_code_detector.index import ClusterKey
from nw_ai_code_detector.scorer import _auroc
from nw_ai_code_detector.stripper import Language
from nw_ai_code_detector.style_signals import (
    _author_comment_bodies,
    _is_descriptive_name,
    _node_text,
    _parse_tree,
    _should_skip_identifier,
)

UNSURE_LOW = 0.90
UNSURE_HIGH = 0.97
EXAMPLE_COUNT = 15
REPORT_PATH = OUTPUTS_DIR / "embedding_style_probe_report.txt"
PYTHON_IDIOM_PATTERNS = (
    re.compile(r"\benumerate\s*\("),
    re.compile(r"\bzip\s*\("),
    re.compile(r"\bdefaultdict\s*\("),
    re.compile(r"\bfor\s+\w+\s+in\s+"),
    re.compile(r"\[.+\s+for\s+.+\s+in\s+"),
    re.compile(r"\bwith\s+"),
)
CPP_IDIOM_PATTERNS = (
    re.compile(r"#\s*include\s*<bits/stdc\+\+\.h>"),
    re.compile(r"using\s+namespace\s+std"),
    re.compile(r"\bvector\s*<"),
    re.compile(r"\bauto\s+"),
    re.compile(r"\bnullptr\b"),
    re.compile(r"for\s*\(\s*auto\b"),
)


@dataclass(frozen=True)
class ProbeItem:
    source: str
    question_id: str
    language: str
    difficulty: str
    raw_code: str
    max_cosine: float
    mean_cosine: float
    nearest_distance: float
    cluster_z: float | None
    features: Mapping[str, float]


def main() -> int:
    clusters = _load_live_clusters()
    humans = _score_humans(clusters)
    heldout = _score_heldout(clusters)
    humans = _zscore_humans(humans)
    heldout = _zscore_against_humans(heldout, humans)
    report = _build_report(humans, heldout)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(report)
    print(f"Wrote {REPORT_PATH}")
    return 0


def _load_live_clusters() -> dict[ClusterKey, np.ndarray]:
    manifest = json.loads((REFERENCE_INDEX_DIR / "manifest.json").read_text(encoding="utf-8"))
    clusters = {}
    for token, item in manifest["clusters"].items():
        key = ClusterKey(str(item["question_id"]), str(item["language"]))
        path = REFERENCE_INDEX_DIR / f"{token.replace(':', '__')}.npy"
        clusters[key] = np.asarray(np.load(path), dtype=np.float32)
    return clusters


def _score_humans(clusters: Mapping[ClusterKey, np.ndarray]) -> list[ProbeItem]:
    groups = json.loads((DATA_DIR / "scored_submissions.json").read_text(encoding="utf-8"))
    mapping = groups.get(GROUPS_KEY)
    items = []
    for candidate in load_candidate_human_records(load_question_assignments()):
        if not candidate.valid or not candidate.stripped_code:
            continue
        raw = _human_raw(mapping, candidate.question_id, candidate.language, candidate.group_index)
        item = _score_one(
            "candidate_human",
            candidate.question_id,
            candidate.language,
            candidate.difficulty,
            raw,
            candidate.stripped_code,
            clusters,
        )
        if item is not None:
            items.append(item)
    return items


def _score_heldout(clusters: Mapping[ClusterKey, np.ndarray]) -> list[ProbeItem]:
    difficulty_by_qid = {
        row.question_id: row.difficulty for row in load_question_assignments()
    }
    items = []
    for record in load_solution_records(EVAL_AI_SOLUTIONS_DIR, HELDOUT_AI_SOURCE):
        if not record.parse_ok or not record.stripped_code:
            continue
        item = _score_one(
            "heldout_ai",
            record.question_id,
            record.language,
            str(difficulty_by_qid.get(record.question_id) or "UNKNOWN"),
            record.raw_code or "",
            record.stripped_code,
            clusters,
        )
        if item is not None:
            items.append(item)
    return items


def _score_one(
    source: str,
    question_id: str,
    language: str,
    difficulty: str,
    raw_code: str,
    stripped_code: str,
    clusters: Mapping[ClusterKey, np.ndarray],
) -> ProbeItem | None:
    matrix = clusters.get(ClusterKey(question_id, language))
    vector = cached_vector_for_text(stripped_code)
    if matrix is None or vector is None:
        return None
    sims = matrix @ np.asarray(vector, dtype=np.float32)
    max_cosine = float(np.max(sims))
    return ProbeItem(
        source=source,
        question_id=question_id,
        language=language,
        difficulty=difficulty,
        raw_code=raw_code,
        max_cosine=max_cosine,
        mean_cosine=float(np.mean(sims)),
        nearest_distance=1.0 - max_cosine,
        cluster_z=None,
        features=_stylometric_features(raw_code, stripped_code, language),
    )


def _zscore_humans(rows: Sequence[ProbeItem]) -> list[ProbeItem]:
    grouped: dict[tuple[str, str], list[ProbeItem]] = defaultdict(list)
    for row in rows:
        grouped[(row.question_id, row.language)].append(row)
    updated = []
    for members in grouped.values():
        values = [item.max_cosine for item in members]
        for index, item in enumerate(members):
            updated.append(_with_z(item, _leave_one_out_z(values, index)))
    return updated


def _zscore_against_humans(
    rows: Sequence[ProbeItem],
    humans: Sequence[ProbeItem],
) -> list[ProbeItem]:
    groups: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in humans:
        groups[(row.question_id, row.language)].append(row.max_cosine)
    return [
        _with_z(row, _z_against(groups.get((row.question_id, row.language), ()), row.max_cosine))
        for row in rows
    ]


def _leave_one_out_z(values: Sequence[float], index: int) -> float | None:
    others = [value for i, value in enumerate(values) if i != index]
    return _z_against(others, values[index])


def _z_against(baseline: Sequence[float], value: float) -> float | None:
    if len(baseline) < 2:
        return None
    mean = float(np.mean(baseline))
    std = float(np.std(baseline, ddof=1))
    if std < 1e-8:
        std = 1e-8
    return (value - mean) / std


def _with_z(item: ProbeItem, z_score: float | None) -> ProbeItem:
    return ProbeItem(
        source=item.source,
        question_id=item.question_id,
        language=item.language,
        difficulty=item.difficulty,
        raw_code=item.raw_code,
        max_cosine=item.max_cosine,
        mean_cosine=item.mean_cosine,
        nearest_distance=item.nearest_distance,
        cluster_z=z_score,
        features=item.features,
    )


def _stylometric_features(raw_code: str, stripped_code: str, language: str) -> dict[str, float]:
    naming = _naming_stats(_identifier_counts(stripped_code, language))
    formatting = _formatting_stats(raw_code)
    comments = _comment_stats(raw_code, language)
    idioms = _idiom_stats(stripped_code, language)
    return {**naming, **formatting, **comments, **idioms}


def _identifier_counts(stripped_code: str, language: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    if not stripped_code.strip():
        return counts
    source_language = Language(language)
    tree = _parse_tree(stripped_code, source_language)
    _collect_name_counts(tree.root_node, stripped_code, counts, source_language)
    return counts


def _collect_name_counts(node, source: str, counts: Counter[str], language: Language) -> None:
    if node.type == "identifier":
        if not _should_skip_identifier(node, source):
            counts[_node_text(node, source)] += 1
        return
    if language is Language.CPP and node.type == "type_identifier":
        return
    for child in node.named_children:
        _collect_name_counts(child, source, counts, language)


def _naming_stats(counts: Counter[str]) -> dict[str, float]:
    unique = list(counts)
    total = max(len(unique), 1)
    return {
        "naming_frac_single_letter": sum(1 for name in unique if len(name) == 1) / total,
        "naming_frac_short": sum(1 for name in unique if len(name) <= NAMING_SHORT_NAME_MAX_LENGTH) / total,
        "naming_frac_descriptive": sum(1 for name in unique if _is_descriptive_name(name)) / total,
        "naming_entropy": _shannon(list(counts.values())),
        "naming_entropy_normalized": _normalized_entropy(list(counts.values())),
        "naming_unique_count": float(len(unique)),
    }


def _shannon(values: Sequence[int]) -> float:
    total = float(sum(values))
    if total <= 0:
        return 0.0
    entropy = 0.0
    for value in values:
        if value <= 0:
            continue
        probability = value / total
        entropy -= probability * math.log2(probability)
    return entropy


def _normalized_entropy(values: Sequence[int]) -> float:
    if len(values) <= 1:
        return 0.0
    return _shannon(values) / math.log2(len(values))


def _formatting_stats(raw_code: str) -> dict[str, float]:
    lines = raw_code.splitlines() or [""]
    indents = [_leading_spaces(line) for line in lines if line.strip()]
    tabs = sum(line.startswith("\t") for line in lines)
    blanks = sum(1 for line in lines if not line.strip())
    lengths = [len(line) for line in lines]
    return {
        "fmt_blank_frac": blanks / len(lines),
        "fmt_mean_line_len": float(np.mean(lengths)) if lengths else 0.0,
        "fmt_indent_std": float(np.std(indents)) if len(indents) > 1 else 0.0,
        "fmt_tab_frac": tabs / len(lines),
    }


def _leading_spaces(line: str) -> int:
    stripped = line.lstrip(" \t")
    return len(line) - len(stripped)


def _comment_stats(raw_code: str, language: str) -> dict[str, float]:
    try:
        bodies = _author_comment_bodies(raw_code, "", Language(language))
    except Exception:
        bodies = ()
    comment_chars = sum(len(body) for body in bodies)
    return {
        "comment_present": 1.0 if bodies else 0.0,
        "comment_char_frac": comment_chars / max(len(raw_code), 1),
    }


def _idiom_stats(stripped_code: str, language: str) -> dict[str, float]:
    patterns = PYTHON_IDIOM_PATTERNS if language == "PYTHON" else CPP_IDIOM_PATTERNS
    hits = sum(1 for pattern in patterns if pattern.search(stripped_code))
    return {"idiom_hit_rate": hits / len(patterns)}


def _build_report(humans: Sequence[ProbeItem], heldout: Sequence[ProbeItem]) -> str:
    lines = [
        "EMBEDDING + STYLOMETRIC PROBE (report only, not wired)",
        "Positives = held-out AI. Comparison set = candidate-human pool.",
        "Absolute FPR is unreliable (confident-human set ~40-50). Relative KEEP/DROP only.",
        "max_cosine is inner product of L2-normalized Voyage vectors = existing ai_nn_max.",
        "cluster_z = (max_cosine - mean) / std of candidate-humans on the same (qid, lang).",
        "Humans: leave-one-out. Held-out AI: full human cluster stats.",
        "",
    ]
    lines.extend(_part1(humans, heldout))
    lines.extend(_part2(humans, heldout))
    lines.append("EXAMPLES (weighted to embedding-unsure max_cosine 0.90-0.97)")
    for item in _examples(humans, heldout):
        lines.append("=" * 80)
        lines.append(_example_header(item))
        lines.append(item.raw_code.rstrip() or "(empty raw)")
        lines.append("")
    return "\n".join(lines) + "\n"


def _part1(humans: Sequence[ProbeItem], heldout: Sequence[ProbeItem]) -> list[str]:
    lines = ["PART 1 — EMBEDDING SIMILARITY (raw vs per-cluster normalized)", "-" * 80]
    for language in ("CPP", "PYTHON"):
        h_lang = [row for row in humans if row.language == language]
        a_lang = [row for row in heldout if row.language == language]
        lines.append(f"=== {language} ===")
        lines.append(
            f"n_human={len(h_lang)} n_heldout_ai={len(a_lang)} "
            f"raw_max_cosine_AUROC={_auroc_attr(a_lang, h_lang, 'max_cosine')} "
            f"(this IS canonicality ai_nn_max)"
        )
        lines.append(
            f"mean_cosine_AUROC={_auroc_attr(a_lang, h_lang, 'mean_cosine')} "
            f"nearest_distance_AUROC={_auroc_attr(a_lang, h_lang, 'nearest_distance')}"
        )
        lines.append(
            f"cluster_z_AUROC={_auroc_attr(a_lang, h_lang, 'cluster_z')} "
            f"n_with_z human={_count_z(h_lang)} ai={_count_z(a_lang)}"
        )
        lines.append(f"{'diff':<8} {'nH':>5} {'nAI':>5} {'raw_AUROC':>10} {'z_AUROC':>10} {'delta':>8}")
        for difficulty in ("EASY", "MEDIUM", "HARD"):
            h_d = [row for row in h_lang if row.difficulty == difficulty]
            a_d = [row for row in a_lang if row.difficulty == difficulty]
            raw = _auroc_attr(a_d, h_d, "max_cosine")
            normalized = _auroc_attr(a_d, h_d, "cluster_z")
            lines.append(
                f"{difficulty:<8} {len(h_d):>5} {len(a_d):>5} {raw:>10} {normalized:>10} "
                f"{_delta(normalized, raw):>8}"
            )
        lines.append("")
    lines.append("KEY QUESTION: does cluster_z beat raw max-cosine on EASY?")
    for language in ("CPP", "PYTHON"):
        h_e = [row for row in humans if row.language == language and row.difficulty == "EASY"]
        a_e = [row for row in heldout if row.language == language and row.difficulty == "EASY"]
        raw = _auroc_attr(a_e, h_e, "max_cosine")
        normalized = _auroc_attr(a_e, h_e, "cluster_z")
        lines.append(f"  {language} EASY raw={raw} z={normalized} -> {_beat_label(normalized, raw)}")
    lines.append(_part1_verdict(humans, heldout))
    lines.append("")
    return lines


def _part1_verdict(humans: Sequence[ProbeItem], heldout: Sequence[ProbeItem]) -> str:
    gains = []
    for language in ("CPP", "PYTHON"):
        h_e = [row for row in humans if row.language == language and row.difficulty == "EASY"]
        a_e = [row for row in heldout if row.language == language and row.difficulty == "EASY"]
        gains.append(
            _numeric(_auroc_attr(a_e, h_e, "cluster_z"))
            - _numeric(_auroc_attr(a_e, h_e, "max_cosine"))
        )
    overall_raw = _numeric(_auroc_attr(heldout, humans, "max_cosine"))
    overall_z = _numeric(_auroc_attr(heldout, humans, "cluster_z"))
    easy_gain = float(np.mean(gains))
    if easy_gain >= 0.02 and overall_z >= overall_raw:
        return "PART 1 VERDICT: KEEP cluster_z as a scoring variant (beats raw, including EASY)."
    if easy_gain >= 0.01:
        return "PART 1 VERDICT: WEAK KEEP cluster_z (small EASY gain only). Do not replace raw canonicality."
    return "PART 1 VERDICT: DROP cluster_z as a replacement. Keep raw max-cosine as the embedding baseline."


def _part2(humans: Sequence[ProbeItem], heldout: Sequence[ProbeItem]) -> list[str]:
    feature_names = list(humans[0].features) if humans else []
    lines = [
        "PART 2 — NAMING + STYLOMETRIC FEATURES (independence vs embedding)",
        "-" * 80,
        "KEEP only if the feature separates in the embedding-unsure slice and is not an embedding echo.",
        "dir + means AI scored higher on the raw feature; - means we flipped it for AUROC.",
        "",
    ]
    verdicts = []
    for language in ("CPP", "PYTHON"):
        h_lang = [row for row in humans if row.language == language]
        a_lang = [row for row in heldout if row.language == language]
        lines.append(f"=== {language} ===")
        lines.append(
            f"{'feature':<28} {'AUROC':>7} {'dir':>4} {'corr_nn':>8} "
            f"{'unsureH':>8} {'unsureAI':>8} {'unsureAUC':>10} {'verdict':<10}"
        )
        for name in feature_names:
            standalone, direction = _oriented_auroc(a_lang, h_lang, name)
            corr = _feature_corr(list(h_lang) + list(a_lang), name)
            h_u = [row for row in h_lang if _unsure(row.max_cosine)]
            a_u = [row for row in a_lang if _unsure(row.max_cosine)]
            unsure, _ = _oriented_auroc(a_u, h_u, name)
            verdict = _feature_verdict(name, standalone, corr, unsure, language)
            verdicts.append((language, name, verdict, standalone, unsure, corr))
            lines.append(
                f"{name:<28} {_fmt(standalone):>7} {direction:>4} {_fmt(corr):>8} "
                f"{len(h_u):>8} {len(a_u):>8} {_fmt(unsure):>10} {verdict:<10}"
            )
        lines.append("")
    lines.append(_naming_verdict(verdicts))
    lines.append(_feature_summary(verdicts))
    lines.append("")
    return lines


def _oriented_auroc(
    positives: Sequence[ProbeItem],
    negatives: Sequence[ProbeItem],
    name: str,
) -> tuple[float | None, str]:
    pos = _values(positives, name)
    neg = _values(negatives, name)
    if len(pos) < 2 or len(neg) < 2:
        return None, "?"
    forward = _auroc(pos, neg)
    flipped = _auroc([-value for value in pos], [-value for value in neg])
    if forward >= flipped:
        return forward, "+"
    return flipped, "-"


def _values(rows: Sequence[ProbeItem], name: str) -> list[float]:
    if name in {"max_cosine", "mean_cosine", "nearest_distance", "cluster_z"}:
        return [float(value) for value in (getattr(row, name) for row in rows) if value is not None]
    return [float(row.features[name]) for row in rows]


def _auroc_attr(positives: Sequence[ProbeItem], negatives: Sequence[ProbeItem], name: str) -> str:
    pos = _values(positives, name)
    neg = _values(negatives, name)
    if len(pos) < 2 or len(neg) < 2:
        return "n/a"
    return f"{_auroc(pos, neg):.4f}"


def _feature_corr(rows: Sequence[ProbeItem], name: str) -> float | None:
    xs = [float(row.features[name]) for row in rows]
    ys = [row.max_cosine for row in rows]
    if len(xs) < 3:
        return None
    value = float(np.corrcoef(np.asarray(xs), np.asarray(ys))[0, 1])
    if np.isnan(value):
        return None
    return value


def _feature_verdict(
    name: str,
    standalone: float | None,
    corr: float | None,
    unsure: float | None,
    language: str,
) -> str:
    if name.startswith("comment"):
        return "DROP-art"
    standalone_value = standalone or 0.5
    unsure_value = unsure or 0.5
    corr_value = abs(corr or 0.0)
    python_bar = 0.58 if language == "PYTHON" else 0.60
    if unsure_value >= python_bar and corr_value < 0.40 and standalone_value >= 0.55:
        return "KEEP"
    if unsure_value >= 0.57 and corr_value < 0.30:
        return "WEAK-KEEP"
    return "DROP"


def _naming_verdict(
    verdicts: Sequence[tuple[str, str, str, float | None, float | None, float | None]],
) -> str:
    naming = [row for row in verdicts if row[1].startswith("naming_")]
    python = [row for row in naming if row[0] == "PYTHON"]
    best_python = max((row[4] or 0.5) for row in python) if python else 0.5
    best_any = max((row[3] or 0.5) for row in naming) if naming else 0.5
    if best_any < 0.56 and best_python < 0.56:
        return (
            "NAMING VERDICT: DEAD — richer ratios/entropy do not separate, same as uniform-naming. "
            "DROP all naming features."
        )
    if best_python >= 0.60:
        return "NAMING VERDICT: KEEP naming feature(s) with PYTHON unsure-slice AUROC>=0.60."
    return (
        f"NAMING VERDICT: still weak (best standalone={best_any:.3f}, "
        f"best PYTHON unsure={best_python:.3f}). DROP for scoring."
    )


def _feature_summary(
    verdicts: Sequence[tuple[str, str, str, float | None, float | None, float | None]],
) -> str:
    keep = [f"{lang}:{name}" for lang, name, verdict, *_ in verdicts if verdict in {"KEEP", "WEAK-KEEP"}]
    if not keep:
        return "FEATURE SUMMARY: KEEP none. DROP all stylometric/naming features for scoring."
    return "FEATURE SUMMARY: " + ", ".join(keep)


def _examples(humans: Sequence[ProbeItem], heldout: Sequence[ProbeItem]) -> list[ProbeItem]:
    unsure_h = [row for row in humans if _unsure(row.max_cosine)]
    unsure_a = [row for row in heldout if _unsure(row.max_cosine)]
    sure_h = [row for row in humans if not _unsure(row.max_cosine)]
    sure_a = [row for row in heldout if not _unsure(row.max_cosine)]
    picked = _mix(unsure_a, 4) + _mix(unsure_h, 7) + _mix(sure_a, 2) + _mix(sure_h, 2)
    unique = []
    seen = set()
    for item in picked:
        key = (item.source, item.question_id, item.language, item.max_cosine)
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
        if len(unique) >= EXAMPLE_COUNT:
            break
    return unique


def _mix(rows: Sequence[ProbeItem], limit: int) -> list[ProbeItem]:
    by_lang = {
        "PYTHON": [row for row in rows if row.language == "PYTHON"],
        "CPP": [row for row in rows if row.language == "CPP"],
    }
    for members in by_lang.values():
        members.sort(key=lambda row: abs(row.max_cosine - 0.935))
    out = []
    for index in range(max(len(by_lang["PYTHON"]), len(by_lang["CPP"]))):
        if index < len(by_lang["PYTHON"]):
            out.append(by_lang["PYTHON"][index])
        if index < len(by_lang["CPP"]):
            out.append(by_lang["CPP"][index])
        if len(out) >= limit:
            return out[:limit]
    return out[:limit]


def _example_header(item: ProbeItem) -> str:
    feature_bits = " ".join(f"{key}={value:.3f}" for key, value in item.features.items())
    z_text = "n/a" if item.cluster_z is None else f"{item.cluster_z:.3f}"
    return (
        f"source={item.source} lang={item.language} diff={item.difficulty} qid={item.question_id} "
        f"max_cosine={item.max_cosine:.4f} mean_cosine={item.mean_cosine:.4f} "
        f"nearest_dist={item.nearest_distance:.4f} cluster_z={z_text} {feature_bits}"
    )


def _unsure(value: float) -> bool:
    return UNSURE_LOW <= value <= UNSURE_HIGH


def _count_z(rows: Sequence[ProbeItem]) -> int:
    return sum(1 for row in rows if row.cluster_z is not None)


def _delta(left: str, right: str) -> str:
    try:
        return f"{float(left) - float(right):+.4f}"
    except ValueError:
        return "n/a"


def _beat_label(normalized: str, raw: str) -> str:
    try:
        gain = float(normalized) - float(raw)
    except ValueError:
        return "n/a"
    if gain >= 0.02:
        return "YES, z beats raw"
    if gain >= 0.005:
        return "tiny gain"
    if gain > -0.005:
        return "tie"
    return "NO, raw wins"


def _numeric(value: str) -> float:
    try:
        return float(value)
    except ValueError:
        return 0.5


def _fmt(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


if __name__ == "__main__":
    raise SystemExit(main())

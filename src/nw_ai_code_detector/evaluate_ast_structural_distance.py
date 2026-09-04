from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Mapping, Sequence

import numpy as np

from nw_ai_code_detector.ast_structural_features import (
    extract_structural_ast_features,
    structural_feature_vector,
)
from nw_ai_code_detector.build_canonicality_eligibility_tokens import (
    load_candidate_human_records,
)
from nw_ai_code_detector.build_model_dataset_v2 import load_question_assignments
from nw_ai_code_detector.config import (
    AI_SOLUTIONS_DIR,
    DATA_DIR,
    EVAL_AI_SOLUTIONS_DIR,
    OUTPUTS_DIR,
    REFERENCE_INDEX_DIR,
)
from nw_ai_code_detector.constants import GROUPS_KEY, HELDOUT_AI_SOURCE
from nw_ai_code_detector.eligibility_data import load_solution_records
from nw_ai_code_detector.embedder import cached_vector_for_text
from nw_ai_code_detector.evaluate_style_signals import _human_raw
from nw_ai_code_detector.index import ClusterKey
from nw_ai_code_detector.scorer import _auroc

UNSURE_NN_LOW = 0.90
UNSURE_NN_HIGH = 0.97
EXAMPLE_COUNT = 15
REPORT_PATH = OUTPUTS_DIR / "ast_structural_distance_report.txt"


@dataclass(frozen=True)
class StructuralScoredItem:
    source: str
    question_id: str
    language: str
    raw_code: str
    stripped_code: str
    ai_nn_max: float
    structural_distance: float


@dataclass(frozen=True)
class ClusterStats:
    mean: np.ndarray
    std: np.ndarray


def main() -> int:
    stats = _cluster_stats_from_mixed_v1()
    clusters = _load_live_cluster_vectors()
    humans = _score_humans(stats, clusters)
    heldout = _score_heldout(stats, clusters)
    report = _build_report(humans, heldout)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(report)
    print(f"Wrote {REPORT_PATH}")
    return 0


def _cluster_stats_from_mixed_v1() -> dict[ClusterKey, ClusterStats]:
    grouped: dict[ClusterKey, list[tuple[float, ...]]] = {}
    for path in sorted(AI_SOLUTIONS_DIR.rglob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("parse_ok") is not True:
            continue
        stripped = payload.get("stripped_code") or ""
        if not isinstance(stripped, str) or not stripped.strip():
            continue
        features = extract_structural_ast_features(stripped, str(payload.get("language") or ""))
        if features is None:
            continue
        key = ClusterKey(str(payload.get("qid") or ""), str(payload.get("language") or ""))
        grouped.setdefault(key, []).append(structural_feature_vector(features))
    return {key: _stats_for_matrix(np.asarray(rows, dtype=np.float64)) for key, rows in grouped.items()}


def _stats_for_matrix(matrix: np.ndarray) -> ClusterStats:
    mean = np.mean(matrix, axis=0)
    std = np.std(matrix, axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return ClusterStats(mean, std)


def _load_live_cluster_vectors() -> dict[ClusterKey, np.ndarray]:
    manifest = json.loads((REFERENCE_INDEX_DIR / "manifest.json").read_text(encoding="utf-8"))
    clusters: dict[ClusterKey, np.ndarray] = {}
    for token, item in manifest["clusters"].items():
        key = ClusterKey(str(item["question_id"]), str(item["language"]))
        path = REFERENCE_INDEX_DIR / f"{token.replace(':', '__')}.npy"
        clusters[key] = np.asarray(np.load(path), dtype=np.float32)
    return clusters


def _score_humans(
    stats: Mapping[ClusterKey, ClusterStats],
    clusters: Mapping[ClusterKey, np.ndarray],
) -> list[StructuralScoredItem]:
    groups = json.loads((DATA_DIR / "scored_submissions.json").read_text(encoding="utf-8"))
    mapping = groups.get(GROUPS_KEY)
    items = []
    for candidate in load_candidate_human_records(load_question_assignments()):
        if not candidate.valid or not candidate.stripped_code:
            continue
        raw = _human_raw(mapping, candidate.question_id, candidate.language, candidate.group_index)
        item = _score_item(
            "candidate_human",
            candidate.question_id,
            candidate.language,
            raw,
            candidate.stripped_code,
            stats,
            clusters,
        )
        if item is not None:
            items.append(item)
    return items


def _score_heldout(
    stats: Mapping[ClusterKey, ClusterStats],
    clusters: Mapping[ClusterKey, np.ndarray],
) -> list[StructuralScoredItem]:
    items = []
    for record in load_solution_records(EVAL_AI_SOLUTIONS_DIR, HELDOUT_AI_SOURCE):
        if not record.parse_ok or not record.stripped_code:
            continue
        item = _score_item(
            "heldout_ai",
            record.question_id,
            record.language,
            record.raw_code or "",
            record.stripped_code,
            stats,
            clusters,
        )
        if item is not None:
            items.append(item)
    return items


def _score_item(
    source: str,
    question_id: str,
    language: str,
    raw_code: str,
    stripped_code: str,
    stats: Mapping[ClusterKey, ClusterStats],
    clusters: Mapping[ClusterKey, np.ndarray],
) -> StructuralScoredItem | None:
    key = ClusterKey(question_id, language)
    cluster_stats = stats.get(key)
    cluster_vectors = clusters.get(key)
    if cluster_stats is None or cluster_vectors is None:
        return None
    features = extract_structural_ast_features(stripped_code, language)
    if features is None:
        return None
    nn_score = _nn_max(stripped_code, cluster_vectors)
    if nn_score is None:
        return None
    z_scored = (np.asarray(structural_feature_vector(features)) - cluster_stats.mean) / cluster_stats.std
    distance = float(np.linalg.norm(z_scored))
    return StructuralScoredItem(
        source=source,
        question_id=question_id,
        language=language,
        raw_code=raw_code,
        stripped_code=stripped_code,
        ai_nn_max=nn_score,
        structural_distance=distance,
    )


def _nn_max(stripped_code: str, cluster_vectors: np.ndarray) -> float | None:
    vector = cached_vector_for_text(stripped_code)
    if vector is None:
        return None
    query = np.asarray(vector, dtype=np.float32)
    return float(np.max(cluster_vectors @ query))


def _build_report(
    humans: Sequence[StructuralScoredItem],
    heldout: Sequence[StructuralScoredItem],
) -> str:
    lines = [
        "AST STRUCTURAL DISTANCE — USEFULNESS + INDEPENDENCE (report only, not wired)",
        "Distance = L2 of z-scored AST features vs the mixed-v1 AI cluster mean.",
        "Z-score stats come from that cluster's 6 AI refs. Held-out AI and humans are queries.",
        "AUROC uses score = -distance (higher = closer to AI structure), positives = held-out AI.",
        "",
    ]
    for language in ("CPP", "PYTHON"):
        lines.extend(_language_section(language, humans, heldout))
    lines.append("VERDICT")
    lines.append(_verdict(humans, heldout))
    lines.append("")
    lines.append("EXAMPLES (weighted to embedding-unsure slice 0.90-0.97)")
    for item in _select_examples(humans, heldout):
        lines.append("=" * 80)
        lines.append(
            f"source={item.source} lang={item.language} qid={item.question_id} "
            f"ai_nn_max={item.ai_nn_max:.4f} structural_distance={item.structural_distance:.4f}"
        )
        lines.append(item.raw_code.rstrip() or "(empty raw)")
        lines.append("")
    return "\n".join(lines) + "\n"


def _language_section(
    language: str,
    humans: Sequence[StructuralScoredItem],
    heldout: Sequence[StructuralScoredItem],
) -> list[str]:
    human_rows = [row for row in humans if row.language == language]
    ai_rows = [row for row in heldout if row.language == language]
    lines = [f"=== {language} ==="]
    lines.extend(_separation_lines(human_rows, ai_rows))
    lines.extend(_independence_lines(human_rows, ai_rows))
    lines.extend(_unsure_lines(human_rows, ai_rows))
    lines.append("")
    return lines


def _separation_lines(
    humans: Sequence[StructuralScoredItem],
    ai_rows: Sequence[StructuralScoredItem],
) -> list[str]:
    human_d = [row.structural_distance for row in humans]
    ai_d = [row.structural_distance for row in ai_rows]
    auroc = _safe_auroc([-value for value in ai_d], [-value for value in human_d])
    return [
        f"(a) SEPARATION  n_human={len(humans)} n_heldout_ai={len(ai_rows)}",
        f"    mean_distance human={_mean(human_d):.4f}  heldout_ai={_mean(ai_d):.4f}",
        f"    AUROC(-distance, AI positive)={_fmt(auroc)}",
        "    0.5 = no separation; >>0.5 = AI closer to AI-cluster structure than humans.",
    ]


def _independence_lines(
    humans: Sequence[StructuralScoredItem],
    ai_rows: Sequence[StructuralScoredItem],
) -> list[str]:
    combined = list(humans) + list(ai_rows)
    corr = _pearson(
        [row.structural_distance for row in combined],
        [row.ai_nn_max for row in combined],
    )
    return [
        "(b) INDEPENDENCE  Pearson corr(structural_distance, ai_nn_max)",
        f"    r={_fmt(corr)}  (high |r| = redundant with embedding; near 0 = extra axis)",
    ]


def _unsure_lines(
    humans: Sequence[StructuralScoredItem],
    ai_rows: Sequence[StructuralScoredItem],
) -> list[str]:
    human_u = [row for row in humans if _unsure(row.ai_nn_max)]
    ai_u = [row for row in ai_rows if _unsure(row.ai_nn_max)]
    auroc = _safe_auroc(
        [-row.structural_distance for row in ai_u],
        [-row.structural_distance for row in human_u],
    )
    return [
        f"(c) UNSURE SLICE ai_nn_max in [{UNSURE_NN_LOW:.2f}, {UNSURE_NN_HIGH:.2f}]",
        f"    n_human={len(human_u)} n_heldout_ai={len(ai_u)}",
        f"    mean_distance human={_mean([row.structural_distance for row in human_u]):.4f} "
        f"heldout_ai={_mean([row.structural_distance for row in ai_u]):.4f}",
        f"    AUROC(-distance, AI positive)={_fmt(auroc)}",
    ]


def _verdict(
    humans: Sequence[StructuralScoredItem],
    heldout: Sequence[StructuralScoredItem],
) -> str:
    parts = []
    for language in ("CPP", "PYTHON"):
        human_rows = [row for row in humans if row.language == language]
        ai_rows = [row for row in heldout if row.language == language]
        overall = _safe_auroc(
            [-row.structural_distance for row in ai_rows],
            [-row.structural_distance for row in human_rows],
        )
        human_u = [row for row in human_rows if _unsure(row.ai_nn_max)]
        ai_u = [row for row in ai_rows if _unsure(row.ai_nn_max)]
        slice_auroc = _safe_auroc(
            [-row.structural_distance for row in ai_u],
            [-row.structural_distance for row in human_u],
        )
        corr = _pearson(
            [row.structural_distance for row in human_rows + ai_rows],
            [row.ai_nn_max for row in human_rows + ai_rows],
        )
        parts.append(
            f"{language}: overall_AUROC={_fmt(overall)} unsure_AUROC={_fmt(slice_auroc)} "
            f"corr(dist,nn)={_fmt(corr)} -> {_keep_or_drop(overall, slice_auroc, corr)}"
        )
    return "\n".join(parts)


def _keep_or_drop(overall: float | None, slice_auroc: float | None, corr: float | None) -> str:
    overall_value = overall or 0.5
    slice_value = slice_auroc or 0.5
    corr_value = abs(corr or 0.0)
    if slice_value >= 0.70 and corr_value < 0.35:
        return "KEEP as second signal (unsure slice separates and is not an embedding echo)"
    if overall_value >= 0.70 and slice_value < 0.60:
        return "DROP for scoring (separates overall but not in the embedding-unsure region)"
    if corr_value >= 0.40:
        return "DROP for scoring (mostly echoes embedding nn_max)"
    return "DROP for v0 (does not clearly add signal beyond embedding)"


def _select_examples(
    humans: Sequence[StructuralScoredItem],
    heldout: Sequence[StructuralScoredItem],
) -> list[StructuralScoredItem]:
    unsure_h = [row for row in humans if _unsure(row.ai_nn_max)]
    unsure_a = [row for row in heldout if _unsure(row.ai_nn_max)]
    sure_h = [row for row in humans if not _unsure(row.ai_nn_max)]
    sure_a = [row for row in heldout if not _unsure(row.ai_nn_max)]
    picked: list[StructuralScoredItem] = []
    picked.extend(_mix_langs(unsure_a, 4))
    picked.extend(_mix_langs(unsure_h, 7))
    picked.extend(_mix_langs(sure_a, 2))
    picked.extend(_mix_langs(sure_h, 2))
    unique = []
    seen = set()
    for item in picked:
        key = (item.source, item.question_id, item.language, item.stripped_code[:80])
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
        if len(unique) >= EXAMPLE_COUNT:
            break
    return unique


def _mix_langs(rows: Sequence[StructuralScoredItem], limit: int) -> list[StructuralScoredItem]:
    cpp = [row for row in rows if row.language == "CPP"]
    python = [row for row in rows if row.language == "PYTHON"]
    cpp.sort(key=lambda row: abs(row.ai_nn_max - 0.935))
    python.sort(key=lambda row: abs(row.ai_nn_max - 0.935))
    out = []
    for index in range(max(len(cpp), len(python))):
        if index < len(python):
            out.append(python[index])
        if index < len(cpp):
            out.append(cpp[index])
        if len(out) >= limit:
            return out[:limit]
    return out[:limit]


def _unsure(nn_score: float) -> bool:
    return UNSURE_NN_LOW <= nn_score <= UNSURE_NN_HIGH


def _safe_auroc(positives: Sequence[float], negatives: Sequence[float]) -> float | None:
    if len(positives) < 2 or len(negatives) < 2:
        return None
    return _auroc(positives, negatives)


def _pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) < 3:
        return None
    matrix = np.corrcoef(np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64))
    value = float(matrix[0, 1])
    if np.isnan(value):
        return None
    return value


def _mean(values: Sequence[float]) -> float:
    if not values:
        return float("nan")
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _fmt(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


if __name__ == "__main__":
    raise SystemExit(main())

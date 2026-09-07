from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from collections.abc import Mapping, Sequence

import numpy as np

from nw_ai_code_detector.build_canonicality_eligibility_tokens import (
    load_candidate_human_records,
)
from nw_ai_code_detector.build_model_dataset_v2 import load_question_assignments
from nw_ai_code_detector.config import (
    DATA_DIR,
    DISCOUNT_LAYER_DIR,
    EVAL_AI_SOLUTIONS_DIR,
)
from nw_ai_code_detector.constants import (
    CANDIDATE_HUMAN_SOURCE,
    CLUSTER_LOW_DIVERSITY_DISTANCE,
    COMMENTED_OUT_CODE_DISCOUNT,
    DESCRIPTIVE_RAISE_FLOOR,
    GROUPS_KEY,
    HAS_CANONICALITY_SCORE_STATUSES,
    HELDOUT_AI_SOURCE,
    INSUFFICIENT_EVIDENCE_STATUS,
    LOW_CONFIDENCE_SHORT_STATUS,
    LOW_CONFIDENCE_STATUS,
    RAW_CODE_FIELD,
    SHORT_LOW_CONFIDENCE_MAX_TOKENS_BY_LANGUAGE,
    SIGNIFICANT_TOKEN_THRESHOLDS_BY_LANGUAGE,
)
from nw_ai_code_detector.data_load import load_dataset
from nw_ai_code_detector.discount_layer import (
    CanonicalityAssessment,
    CanonicalityDiscountRequest,
    assess_canonicality,
    mean_pairwise_cosine_distance,
)
from nw_ai_code_detector.eligibility_data import load_solution_records
from nw_ai_code_detector.embedder import cached_vector_for_text
from nw_ai_code_detector.evaluation.evaluate_ai_reference_scores_v2 import (
    load_reference_clusters,
)
from nw_ai_code_detector.evaluation.experiment_metrics import pooled_auroc, score_statistics
from nw_ai_code_detector.index import ClusterKey
from nw_ai_code_detector.significant_code_tokens import (
    SignificantCodeTokenizationError,
    significant_code_token_count,
)
from nw_ai_code_detector.similarity_explanation import illustrative_unlocked_band
from nw_ai_code_detector.style_signals import has_commented_out_code, naming_fractions
from tools.labeling.constants import MANUAL_LABELS_PATH, REVIEW_QUEUE_PATH
from tools.labeling.labels_store import load_labels
from tools.labeling.review_data import load_queue


@dataclass(frozen=True)
class DiscountScoreParts:
    source: str
    question_id: str
    language: str
    raw_code: str
    stripped_code: str
    boilerplate: str
    token_count: int | None


@dataclass(frozen=True)
class DiscountEvalRow:
    source: str
    question_id: str
    language: str
    raw_code: str
    reading: CanonicalityAssessment


def main() -> int:
    dataset = load_dataset()
    assignments = load_question_assignments()
    clusters = load_reference_clusters()
    diversity = _cluster_diversity_map(clusters)
    humans = _score_humans(dataset, assignments, clusters, diversity)
    heldout = _score_heldout(dataset, clusters, diversity)
    report = _build_report(humans, heldout, diversity)
    examples = _select_examples(humans, heldout)
    _write_outputs(report, examples, humans, heldout)
    print(json.dumps(report, indent=2))
    print(f"Wrote {DISCOUNT_LAYER_DIR}")
    return 0


def _cluster_diversity_map(clusters: Mapping[ClusterKey, object]) -> dict[str, float]:
    return {
        key.token: mean_pairwise_cosine_distance(cluster.vectors)
        for key, cluster in clusters.items()
    }


def _score_humans(dataset, assignments, clusters, diversity) -> list[DiscountEvalRow]:
    groups = json.loads((DATA_DIR / "scored_submissions.json").read_text(encoding="utf-8"))
    mapping = groups.get(GROUPS_KEY)
    rows: list[DiscountEvalRow] = []
    for candidate in load_candidate_human_records(assignments):
        if not candidate.valid or not candidate.stripped_code:
            continue
        raw = _human_raw(mapping, candidate.question_id, candidate.language, candidate.group_index)
        boilerplate = dataset.questions[candidate.question_id].boilerplates.get(
            candidate.language,
            "",
        )
        row = _score_row(
            DiscountScoreParts(
                CANDIDATE_HUMAN_SOURCE,
                candidate.question_id,
                candidate.language,
                raw,
                candidate.stripped_code,
                boilerplate,
                candidate.significant_code_token_count,
            ),
            clusters,
            diversity,
        )
        if row is not None:
            rows.append(row)
    return rows


def _score_heldout(dataset, clusters, diversity) -> list[DiscountEvalRow]:
    rows: list[DiscountEvalRow] = []
    for record in load_solution_records(EVAL_AI_SOLUTIONS_DIR, HELDOUT_AI_SOURCE):
        if not record.parse_ok or not record.stripped_code:
            continue
        question = dataset.questions.get(record.question_id)
        boilerplate = question.boilerplates.get(record.language, "") if question else ""
        row = _score_row(
            DiscountScoreParts(
                HELDOUT_AI_SOURCE,
                record.question_id,
                record.language,
                record.raw_code or "",
                record.stripped_code,
                boilerplate,
                _safe_token_count(record.stripped_code, record.language),
            ),
            clusters,
            diversity,
        )
        if row is not None:
            rows.append(row)
    return rows


def _score_row(
    parts: DiscountScoreParts,
    clusters,
    diversity: Mapping[str, float],
) -> DiscountEvalRow | None:
    if parts.token_count is None:
        return None
    pair = _canonicality_pair(
        parts.question_id,
        parts.language,
        parts.stripped_code,
        clusters,
    )
    if pair is None:
        return None
    key = ClusterKey(parts.question_id, parts.language)
    cluster_diversity = diversity.get(key.token)
    if cluster_diversity is None:
        return None
    commented = has_commented_out_code(parts.raw_code, parts.boilerplate, parts.language)
    naming = naming_fractions(parts.stripped_code, parts.language)
    assessment = assess_canonicality(
        CanonicalityDiscountRequest(
            pair[0],
            pair[1],
            parts.token_count,
            parts.language,
            cluster_diversity,
            commented,
            naming.frac_descriptive,
            naming.convention_frac,
        )
    )
    return DiscountEvalRow(
        parts.source,
        parts.question_id,
        parts.language,
        parts.raw_code,
        assessment,
    )


def _canonicality_pair(question_id: str, language: str, stripped: str, clusters):
    vector = cached_vector_for_text(stripped)
    if vector is None:
        return None
    cluster = clusters.get(ClusterKey(question_id, language))
    if cluster is None:
        return None
    query = np.asarray(vector, dtype=np.float32)
    similarities = np.sort(np.asarray(cluster.vectors @ query, dtype=np.float64))[::-1]
    maximum = float(similarities[0])
    top3 = float(np.mean(similarities[: min(3, len(similarities))]))
    return maximum, top3


def _safe_token_count(stripped: str, language: str) -> int | None:
    try:
        return significant_code_token_count(stripped, language)
    except SignificantCodeTokenizationError:
        return None


def _human_raw(mapping: object, question_id: str, language: str, index: int) -> str:
    if not isinstance(mapping, dict):
        return ""
    records = mapping.get(f"{question_id}:{language}")
    if not isinstance(records, list) or index >= len(records):
        return ""
    record = records[index]
    if not isinstance(record, dict):
        return ""
    raw = record.get(RAW_CODE_FIELD)
    return raw if isinstance(raw, str) else ""


def _build_report(
    humans: Sequence[DiscountEvalRow],
    heldout: Sequence[DiscountEvalRow],
    diversity: Mapping[str, float],
) -> dict[str, object]:
    combined = (*humans, *heldout)
    return {
        "formula": {
            "short_code": (
                "below language floor -> insufficient_evidence; "
                "floor through language short ceiling "
                f"{dict(SHORT_LOW_CONFIDENCE_MAX_TOKENS_BY_LANGUAGE)} -> "
                "low_confidence_short (scored, flag only, no extra discount); "
                "not a score multiplier"
            ),
            "low_cluster_diversity": "route low_confidence; not a score multiplier",
            "scoreable_score": (
                "raw_canonicality * (1 - commented_out); naming stays "
                "confidence-only and never changes the score"
            ),
            "token_floors": dict(SIGNIFICANT_TOKEN_THRESHOLDS_BY_LANGUAGE),
            "cluster_low_diversity_distance": CLUSTER_LOW_DIVERSITY_DISTANCE,
            "commented_out_code_discount": COMMENTED_OUT_CODE_DISCOUNT,
            "descriptive_raise": "removed_from_score",
            "descriptive_raise_floor": DESCRIPTIVE_RAISE_FLOOR,
            "naming_discount": "removed",
            "auroc_label": (
                "vs contaminated candidate-human pool, pessimistic floor"
            ),
        },
        "n_candidate_human": len(humans),
        "n_heldout_ai": len(heldout),
        "cluster_diversity": _diversity_distribution(diversity),
        "descriptive_raise": {
            "provisional_floor": DESCRIPTIVE_RAISE_FLOOR,
            "candidate_human": _descriptive_trigger_rates(humans),
            "heldout_ai": _descriptive_trigger_rates(heldout),
            "by_language": {
                "CPP": {
                    "candidate_human": _descriptive_trigger_rates(
                        [row for row in humans if row.language == "CPP"]
                    ),
                    "heldout_ai": _descriptive_trigger_rates(
                        [row for row in heldout if row.language == "CPP"]
                    ),
                },
                "PYTHON": {
                    "candidate_human": _descriptive_trigger_rates(
                        [row for row in humans if row.language == "PYTHON"]
                    ),
                    "heldout_ai": _descriptive_trigger_rates(
                        [row for row in heldout if row.language == "PYTHON"]
                    ),
                },
            },
        },
        "labeled_confident_humans_scoreable": _labeled_scoreable_humans(humans),
        "by_language": {
            "CPP": _language_report(humans, heldout, "CPP"),
            "PYTHON": _language_report(humans, heldout, "PYTHON"),
        },
        "routing": {
            "candidate_human": _routing(humans),
            "heldout_ai": _routing(heldout),
        },
    }


def _diversity_distribution(diversity: Mapping[str, float]) -> dict[str, object]:
    values = list(diversity.values())
    low = sum(1 for value in values if value < CLUSTER_LOW_DIVERSITY_DISTANCE)
    return {
        "n_clusters": len(values),
        "mean_pairwise_cosine_distance": _stats_payload(score_statistics(values)),
        "clusters_low_confidence": low,
        "note": (
            "With only 6 refs this is a coarse spread estimate. "
            "Tight clusters are routed to low_confidence, not discounted."
        ),
    }


def _language_report(
    humans: Sequence[DiscountEvalRow],
    heldout: Sequence[DiscountEvalRow],
    language: str,
) -> dict[str, object]:
    human_rows = [row for row in humans if row.language == language]
    ai_rows = [row for row in heldout if row.language == language]
    scoreable_h = [row for row in human_rows if _is_scored(row)]
    scoreable_a = [row for row in ai_rows if _is_scored(row)]
    full_raw_h = [row.reading.raw_canonicality for row in human_rows]
    full_raw_a = [row.reading.raw_canonicality for row in ai_rows]
    raw_h = [row.reading.raw_canonicality for row in scoreable_h]
    raw_a = [row.reading.raw_canonicality for row in scoreable_a]
    score_h = [row.reading.score for row in scoreable_h if row.reading.score is not None]
    score_a = [row.reading.score for row in scoreable_a if row.reading.score is not None]
    return {
        "n_candidate_human": len(human_rows),
        "n_heldout_ai": len(ai_rows),
        "routing": {
            "candidate_human": _routing(human_rows),
            "heldout_ai": _routing(ai_rows),
        },
        "auroc_label": "vs contaminated candidate-human pool, pessimistic floor",
        "auroc_raw_all_embedded": pooled_auroc(full_raw_a, full_raw_h),
        "auroc_raw_scoreable": pooled_auroc(raw_a, raw_h),
        "auroc_scoreable_locked": pooled_auroc(score_a, score_h),
        "descriptive_raise": {
            "candidate_human": _descriptive_trigger_rates(human_rows),
            "heldout_ai": _descriptive_trigger_rates(ai_rows),
        },
        "scoreable_raw_canonicality": {
            "candidate_human": _stats_payload(score_statistics(raw_h)),
            "heldout_ai": _stats_payload(score_statistics(raw_a)),
        },
        "scoreable_score": {
            "candidate_human": _stats_payload(score_statistics(score_h)),
            "heldout_ai": _stats_payload(score_statistics(score_a)),
        },
        "mean_drop_scoreable": {
            "candidate_human": _mean_drop(scoreable_h),
            "heldout_ai": _mean_drop(scoreable_a),
        },
    }


def _descriptive_trigger_rates(rows: Sequence[DiscountEvalRow]) -> dict[str, object]:
    total = len(rows)
    denom = total or 1
    high = sum(1 for row in rows if row.reading.descriptive_raise.high)
    scored = [row for row in rows if _is_scored(row)]
    scored_high = sum(1 for row in scored if row.reading.descriptive_raise.high)
    scored_n = len(scored) or 1
    return {
        "n": total,
        "frac_descriptive_high": {"n": high, "rate": round(high / denom, 4)},
        "frac_descriptive_high_among_scoreable": {
            "n": scored_high,
            "rate": round(scored_high / scored_n, 4),
            "scoreable_n": len(scored),
        },
        "frac_descriptive_mean": _mean(
            [row.reading.descriptive_raise.frac_descriptive for row in rows]
        ),
    }


def _labeled_scoreable_humans(humans: Sequence[DiscountEvalRow]) -> dict[str, object]:
    if not MANUAL_LABELS_PATH.is_file() or not REVIEW_QUEUE_PATH.is_file():
        return {"n_scoreable": 0, "rows": [], "note": "labels or queue missing"}
    labels = load_labels()
    queue = {item.record_id: item for item in load_queue()}
    groups = json.loads((DATA_DIR / "scored_submissions.json").read_text(encoding="utf-8"))
    mapping = groups.get(GROUPS_KEY)
    rows: list[dict[str, object]] = []
    pushed = 0
    for payload in labels.values():
        if payload.get("my_label") != "HUMAN":
            continue
        item = _queue_item_for_label(payload, queue)
        if item is None:
            continue
        raw = _human_raw(mapping, item.question_id, item.language, item.group_index)
        match = _match_human_row(humans, item.question_id, item.language, raw)
        if match is None or not _is_scored(match):
            continue
        payload_row, raised = _labeled_human_payload(item, match.reading)
        if raised:
            pushed += 1
        rows.append(payload_row)
    ordered = sorted(rows, key=lambda row: int(row["order_index"]))
    return {
        "n_scoreable": len(ordered),
        "n_pushed_toward_ai": pushed,
        "rows": ordered,
    }


def _queue_item_for_label(payload: Mapping[str, object], queue: Mapping[str, object]):
    item = queue.get(str(payload["record_id"]))
    if item is not None:
        return item
    qid = str(payload.get("qid", ""))
    language = str(payload.get("language", ""))
    matches = [
        candidate
        for candidate in queue.values()
        if candidate.question_id == qid and candidate.language == language
    ]
    if len(matches) == 1:
        return matches[0]
    return None


def _labeled_human_payload(item, reading: CanonicalityAssessment) -> tuple[bool, dict[str, object]]:
    production = reading.score
    payload = {
        "order_index": item.order_index,
        "question_id": item.question_id,
        "language": item.language,
        "raw_canonicality": reading.raw_canonicality,
        "commented_out_discount": reading.commented_out_code.discount,
        "frac_descriptive": reading.descriptive_raise.frac_descriptive,
        "descriptive_high": reading.descriptive_raise.high,
        "descriptive_raise": reading.descriptive_raise.amount,
        "confidence": reading.confidence,
        "score": production,
        "band": illustrative_unlocked_band(production),
        "pushed_toward_ai": False,
    }
    return payload, False


def _match_human_row(
    humans: Sequence[DiscountEvalRow],
    question_id: str,
    language: str,
    raw: str,
) -> DiscountEvalRow | None:
    for row in humans:
        if row.question_id != question_id or row.language != language:
            continue
        if row.raw_code == raw:
            return row
    return None


def _mean_drop(rows: Sequence[DiscountEvalRow]) -> float | None:
    scored = [
        row.reading.raw_canonicality - row.reading.score
        for row in rows
        if row.reading.score is not None
    ]
    if not scored:
        return None
    return float(np.mean(scored))


def _is_scored(row: DiscountEvalRow) -> bool:
    return row.reading.routing.status in HAS_CANONICALITY_SCORE_STATUSES


def _routing(rows: Sequence[DiscountEvalRow]) -> dict[str, object]:
    total = len(rows) or 1
    insufficient = sum(
        1 for row in rows if row.reading.routing.status == INSUFFICIENT_EVIDENCE_STATUS
    )
    low = sum(1 for row in rows if row.reading.routing.status == LOW_CONFIDENCE_STATUS)
    short_flag = sum(
        1 for row in rows if row.reading.routing.status == LOW_CONFIDENCE_SHORT_STATUS
    )
    scored = sum(1 for row in rows if _is_scored(row))
    commented = sum(
        1 for row in rows if _is_scored(row) and row.reading.commented_out_code.present
    )
    return {
        "n": len(rows),
        "insufficient_evidence": {"n": insufficient, "rate": round(insufficient / total, 4)},
        "low_confidence": {"n": low, "rate": round(low / total, 4)},
        "low_confidence_short": {
            "n": short_flag,
            "rate": round(short_flag / total, 4),
        },
        "scored": {"n": scored, "rate": round(scored / total, 4)},
        "commented_out_discount_on_scored": {
            "n": commented,
            "rate": round(commented / max(scored, 1), 4),
        },
        "descriptive_raise": _descriptive_trigger_rates(rows),
    }


def _mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(values))


def _stats_payload(stats) -> dict[str, object]:
    return {
        "count": stats.count,
        "mean": stats.mean,
        "median": stats.median,
        "std": stats.std,
        "quantiles": stats.quantiles,
    }


def _select_examples(
    humans: Sequence[DiscountEvalRow],
    heldout: Sequence[DiscountEvalRow],
) -> list[dict[str, object]]:
    combined = (*humans, *heldout)
    selected: list[tuple[str, DiscountEvalRow]] = []
    selected.extend(
        ("insufficient_evidence", row) for row in _take_status(combined, INSUFFICIENT_EVIDENCE_STATUS, 5)
    )
    selected.extend(
        ("low_confidence", row) for row in _take_status(combined, LOW_CONFIDENCE_STATUS, 5)
    )
    commented = [
        row
        for row in combined
        if _is_scored(row) and row.reading.commented_out_code.present
    ]
    selected.extend(("commented_out_code", row) for row in _take_lang_mix(commented, 5))
    unique = _dedupe_rows(selected)
    return [_example_payload(row, reason) for reason, row in unique]


def _take_status(
    rows: Sequence[DiscountEvalRow],
    status: str,
    limit: int,
) -> list[DiscountEvalRow]:
    matched = [row for row in rows if row.reading.routing.status == status]
    return _take_lang_mix(matched, limit)


def _take_lang_mix(rows: Sequence[DiscountEvalRow], limit: int) -> list[DiscountEvalRow]:
    cpp = [row for row in rows if row.language == "CPP"][: limit // 2]
    python = [row for row in rows if row.language == "PYTHON"][: limit - len(cpp)]
    return cpp + python


def _dedupe_rows(
    rows: Sequence[tuple[str, DiscountEvalRow]],
) -> list[tuple[str, DiscountEvalRow]]:
    seen: set[tuple[str, str, str, str]] = set()
    unique: list[tuple[str, DiscountEvalRow]] = []
    for kind, row in rows:
        key = (row.source, row.question_id, row.language, row.raw_code[:80])
        if key in seen:
            continue
        seen.add(key)
        unique.append((kind, row))
    return unique


def _example_payload(row: DiscountEvalRow, reason: str) -> dict[str, object]:
    reading = row.reading
    return {
        "shown": reason,
        "source": row.source,
        "question_id": row.question_id,
        "language": row.language,
        "status": reading.routing.status,
        "reason": reading.routing.reason,
        "raw_canonicality": reading.raw_canonicality,
        "top3_mean": reading.top3_mean,
        "token_count": reading.token_count,
        "cluster_diversity": reading.cluster_diversity,
        "short_code_below_floor": reading.routing.short_code_below_floor,
        "low_cluster_diversity": reading.routing.low_cluster_diversity,
        "commented_out_code_present": reading.commented_out_code.present,
        "human_discount": reading.commented_out_code.discount,
        "frac_descriptive": reading.descriptive_raise.frac_descriptive,
        "descriptive_raise": reading.descriptive_raise.amount,
        "remaining_factor": reading.remaining_factor,
        "score": reading.score,
        "raw_excerpt": "\n".join(row.raw_code.splitlines()[:16]),
    }


def _write_outputs(
    report: Mapping[str, object],
    examples: Sequence[Mapping[str, object]],
    humans: Sequence[DiscountEvalRow],
    heldout: Sequence[DiscountEvalRow],
) -> None:
    DISCOUNT_LAYER_DIR.mkdir(parents=True, exist_ok=True)
    _write_json(DISCOUNT_LAYER_DIR / "report.json", report)
    _write_json(DISCOUNT_LAYER_DIR / "examples.json", list(examples))
    lines = [_row_json(row) for row in (*humans, *heldout)]
    (DISCOUNT_LAYER_DIR / "readings.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _row_json(row: DiscountEvalRow) -> str:
    payload = {
        "source": row.source,
        "question_id": row.question_id,
        "language": row.language,
        **asdict(row.reading),
    }
    return json.dumps(payload)


def _write_json(path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())

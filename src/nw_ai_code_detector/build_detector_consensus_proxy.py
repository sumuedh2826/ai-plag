from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from collections.abc import Mapping, Sequence

from nw_ai_code_detector.config import (
    AI_SOLUTIONS_DIR,
    DATA_DIR,
    DETECTOR_CONSENSUS_CACHE_DIR,
    DETECTOR_CONSENSUS_DIR,
    DETECTOR_CONSENSUS_LABELS_PATH,
    DETECTOR_CONSENSUS_MANUAL_BATCH_DIR,
    DETECTOR_CONSENSUS_MANUAL_RESULTS_CSV,
    DETECTOR_CONSENSUS_PROGRESS_PATH,
    DETECTOR_CONSENSUS_REPORT_PATH,
    EVAL_AI_SOLUTIONS_DIR,
    SIGNIFICANT_TOKEN_ELIGIBILITY_DIR,
)
from nw_ai_code_detector.constants import (
    CANDIDATE_HUMAN_SOURCE,
    DETECTOR_CONSENSUS_BATCH_SEED,
    DETECTOR_CONSENSUS_MANUAL_BATCH_SIZE,
    DETECTOR_CONSENSUS_MIN_RESPONDING,
    DETECTOR_CONSENSUS_MIN_SIGNIFICANT_TOKENS,
    DetectorConsensusLabel,
    DetectorConsensusSkipReason,
    DetectorVerdict,
    ExternalDetectorName,
    GROUPS_KEY,
    RAW_CODE_FIELD,
    STRIPPED_CODE_FIELD,
)
from nw_ai_code_detector.detector_consensus import (
    DetectorConsensusDecision,
    DetectorVote,
    decide_detector_consensus,
    live_detector_keys_configured,
)

MANUAL_CSV_FIELDS = (
    "opaque_id",
    "record_id",
    "detector_name",
    "verdict",
    "human_probability",
    "notes",
)
IDENTITY_OUTPUT_KEYS = frozenset({"user_id", "email", "ground_truth"})


@dataclass(frozen=True)
class ScoredCandidateRow:
    record_id: str
    question_id: str
    language: str
    difficulty: str
    significant_code_token_count: int
    stripped_hash: str
    group_index: int
    ai_nn_max_raw: float


def main() -> int:
    options = _parse_cli_options()
    configured = live_detector_keys_configured(dict(os.environ))
    rows = _load_scored_rows()
    groups = _load_submission_groups()
    csv_votes = _load_manual_votes(options.manual_csv)
    if options.run_apis and len(configured) < DETECTOR_CONSENSUS_MIN_RESPONDING:
        raise SystemExit(
            "Fewer than 2 live detector APIs are configured; refusing to call "
            "the network. Use the manual CSV path."
        )
    _persist_votes_to_cache(csv_votes)
    labeled = _label_all_rows(rows, groups)
    _write_jsonl(DETECTOR_CONSENSUS_LABELS_PATH, labeled)
    _append_progress_lines(labeled)
    exact_ids = _exact_match_record_ids(rows)
    batch_ids = _write_manual_batch(rows, groups, exact_ids)
    report = _build_report(rows, labeled, configured, batch_ids)
    _write_json(DETECTOR_CONSENSUS_REPORT_PATH, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _parse_cli_options() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-apis", action="store_true")
    parser.add_argument(
        "--manual-csv",
        type=Path,
        default=DETECTOR_CONSENSUS_MANUAL_RESULTS_CSV,
    )
    return parser.parse_args()


def _load_scored_rows() -> list[ScoredCandidateRow]:
    path = SIGNIFICANT_TOKEN_ELIGIBILITY_DIR / "candidate_human_scores.jsonl"
    rows: list[ScoredCandidateRow] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        payload = json.loads(line)
        parsed = _parse_record_id(str(payload["record_id"]))
        rows.append(
            ScoredCandidateRow(
                record_id=str(payload["record_id"]),
                question_id=str(payload["question_id"]),
                language=str(payload["language"]),
                difficulty=str(payload["difficulty"]),
                significant_code_token_count=int(
                    payload["significant_code_token_count"]
                ),
                stripped_hash=str(payload["stripped_hash"]),
                group_index=parsed[2],
                ai_nn_max_raw=float(payload["ai_nn_max_raw"]),
            )
        )
    return rows


def _parse_record_id(record_id: str) -> tuple[str, str, int]:
    prefix = f"{CANDIDATE_HUMAN_SOURCE}|"
    if not record_id.startswith(prefix):
        raise ValueError(f"Unexpected record_id: {record_id}")
    remainder = record_id[len(prefix) :]
    question_id, language, index_text, _hash = remainder.split("|", 3)
    return question_id, language, int(index_text)


def _load_submission_groups() -> Mapping[str, Sequence[Mapping[str, object]]]:
    payload = json.loads(
        (DATA_DIR / "scored_submissions.json").read_text(encoding="utf-8")
    )
    groups = payload.get(GROUPS_KEY)
    if not isinstance(groups, dict):
        raise RuntimeError("scored_submissions.json groups are unavailable")
    return groups


def _raw_code_for_row(
    groups: Mapping[str, Sequence[Mapping[str, object]]],
    row: ScoredCandidateRow,
) -> str | None:
    key = f"{row.question_id}:{row.language}"
    items = groups.get(key)
    if not isinstance(items, list) or row.group_index >= len(items):
        return None
    record = items[row.group_index]
    if not isinstance(record, dict):
        return None
    raw_code = record.get(RAW_CODE_FIELD)
    if not isinstance(raw_code, str) or not raw_code.strip():
        return None
    return raw_code


def _load_manual_votes(path: Path) -> dict[str, list[DetectorVote]]:
    votes: dict[str, list[DetectorVote]] = defaultdict(list)
    if not path.is_file():
        return votes
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for item in reader:
            vote = _vote_from_csv_row(item)
            if vote is None:
                continue
            votes[str(item["record_id"]).strip()].append(vote)
    return votes


def _vote_from_csv_row(item: Mapping[str, str]) -> DetectorVote | None:
    record_id = str(item.get("record_id") or "").strip()
    detector_name = str(item.get("detector_name") or "").strip().lower()
    verdict_text = str(item.get("verdict") or "").strip().lower()
    if not record_id or not detector_name or not verdict_text:
        return None
    try:
        verdict = DetectorVerdict(verdict_text)
    except ValueError:
        return None
    probability = _parse_probability(item.get("human_probability"))
    return DetectorVote(
        detector_name=_normalize_detector_name(detector_name),
        verdict=verdict,
        human_probability=probability,
    )


def _normalize_detector_name(detector_name: str) -> str:
    try:
        return ExternalDetectorName(detector_name).value
    except ValueError:
        return ExternalDetectorName.OTHER.value


def _parse_probability(raw: object) -> float | None:
    if raw is None or str(raw).strip() == "":
        return None
    value = float(raw)
    if value < 0.0 or value > 100.0:
        raise ValueError(f"human_probability out of range: {raw}")
    return value


def _persist_votes_to_cache(
    csv_votes: Mapping[str, Sequence[DetectorVote]],
) -> None:
    DETECTOR_CONSENSUS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for record_id, votes in csv_votes.items():
        directory = DETECTOR_CONSENSUS_CACHE_DIR / _safe_record_stem(record_id)
        directory.mkdir(parents=True, exist_ok=True)
        for vote in votes:
            path = directory / f"{vote.detector_name}.json"
            path.write_text(
                json.dumps(_vote_manifest(vote), indent=2, sort_keys=True),
                encoding="utf-8",
            )


def _label_all_rows(
    rows: Sequence[ScoredCandidateRow],
    groups: Mapping[str, Sequence[Mapping[str, object]]],
) -> list[dict[str, object]]:
    DETECTOR_CONSENSUS_DIR.mkdir(parents=True, exist_ok=True)
    labeled_at = datetime.now(timezone.utc).isoformat()
    labeled: list[dict[str, object]] = []
    for row in rows:
        raw_code = _raw_code_for_row(groups, row)
        votes = _cached_votes(row.record_id)
        decision = decide_detector_consensus(
            row.significant_code_token_count,
            votes,
            raw_code is not None,
        )
        payload = _label_payload(row, votes, decision, labeled_at)
        _reject_identity_keys(payload)
        labeled.append(payload)
    return labeled


def _cached_votes(record_id: str) -> list[DetectorVote]:
    directory = DETECTOR_CONSENSUS_CACHE_DIR / _safe_record_stem(record_id)
    if not directory.is_dir():
        return []
    votes: list[DetectorVote] = []
    for path in sorted(directory.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        votes.append(
            DetectorVote(
                detector_name=str(payload["detector_name"]),
                verdict=DetectorVerdict(str(payload["verdict"])),
                human_probability=_parse_probability(payload.get("human_probability")),
            )
        )
    return votes


def _safe_record_stem(record_id: str) -> str:
    return sha256(record_id.encode("utf-8")).hexdigest()


def _label_payload(
    row: ScoredCandidateRow,
    votes: Sequence[DetectorVote],
    decision: DetectorConsensusDecision,
    labeled_at: str,
) -> dict[str, object]:
    skip_reason = None
    if decision.skip_reason is not None:
        skip_reason = decision.skip_reason.value
    return {
        "record_id": row.record_id,
        "question_id": row.question_id,
        "language": row.language,
        "difficulty": row.difficulty,
        "significant_code_token_count": row.significant_code_token_count,
        "detector_consensus": decision.detector_consensus.value,
        "skip_reason": skip_reason,
        "responding_detector_count": decision.responding_detector_count,
        "responding_detector_names": list(decision.responding_detector_names),
        "detector_scores": [_vote_manifest(vote) for vote in votes],
        "labeled_at": labeled_at,
    }


def _vote_manifest(vote: DetectorVote) -> dict[str, object]:
    return {
        "detector_name": vote.detector_name,
        "verdict": vote.verdict.value,
        "human_probability": vote.human_probability,
    }


def _reject_identity_keys(payload: Mapping[str, object]) -> None:
    forbidden = IDENTITY_OUTPUT_KEYS.intersection(payload)
    if forbidden:
        raise RuntimeError(f"Forbidden identity keys: {sorted(forbidden)}")


def _write_manual_batch(
    rows: Sequence[ScoredCandidateRow],
    groups: Mapping[str, Sequence[Mapping[str, object]]],
    priority_ids: set[str],
) -> list[str]:
    eligible = [
        row
        for row in rows
        if row.significant_code_token_count
        >= DETECTOR_CONSENSUS_MIN_SIGNIFICANT_TOKENS
    ]
    selected = _select_manual_batch(eligible, priority_ids)
    files_dir = DETECTOR_CONSENSUS_MANUAL_BATCH_DIR / "files"
    files_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = DETECTOR_CONSENSUS_MANUAL_BATCH_DIR / "manifest.csv"
    template_path = DETECTOR_CONSENSUS_MANUAL_BATCH_DIR / (
        "manual_detector_results.template.csv"
    )
    _write_batch_files(files_dir, selected, groups)
    _write_manifest(manifest_path, selected)
    _write_csv_template(template_path)
    return [row.record_id for row in selected]


def _select_manual_batch(
    eligible: Sequence[ScoredCandidateRow],
    priority_ids: set[str],
) -> list[ScoredCandidateRow]:
    rng = random.Random(DETECTOR_CONSENSUS_BATCH_SEED)
    selected: list[ScoredCandidateRow] = []
    selected_ids: set[str] = set()
    for row in eligible:
        if row.record_id in priority_ids and row.record_id not in selected_ids:
            selected.append(row)
            selected_ids.add(row.record_id)
    by_stratum: dict[tuple[str, str, str], list[ScoredCandidateRow]] = defaultdict(
        list
    )
    for row in eligible:
        if row.record_id not in selected_ids:
            by_stratum[_batch_stratum(row)].append(row)
    for bucket in by_stratum.values():
        rng.shuffle(bucket)
        choice = bucket[0]
        selected.append(choice)
        selected_ids.add(choice.record_id)
    remainder = [row for row in eligible if row.record_id not in selected_ids]
    rng.shuffle(remainder)
    needed = DETECTOR_CONSENSUS_MANUAL_BATCH_SIZE - len(selected)
    if needed > 0:
        selected.extend(remainder[:needed])
    return selected[:DETECTOR_CONSENSUS_MANUAL_BATCH_SIZE]


def _batch_stratum(row: ScoredCandidateRow) -> tuple[str, str, str]:
    return (row.language, row.difficulty, _token_band(row.significant_code_token_count))


def _token_band(token_count: int) -> str:
    if token_count <= 80:
        return "60-80"
    if token_count <= 150:
        return "80-150"
    return ">150"


def _write_batch_files(
    files_dir: Path,
    selected: Sequence[ScoredCandidateRow],
    groups: Mapping[str, Sequence[Mapping[str, object]]],
) -> None:
    for index, row in enumerate(selected, start=1):
        raw_code = _raw_code_for_row(groups, row)
        if raw_code is None:
            raise RuntimeError(f"Missing raw_code for batch row {row.record_id}")
        path = files_dir / f"s{index:04d}.txt"
        path.write_text(raw_code, encoding="utf-8")


def _write_manifest(path: Path, selected: Sequence[ScoredCandidateRow]) -> None:
    rows = []
    for index, row in enumerate(selected, start=1):
        rows.append(
            {
                "opaque_id": f"s{index:04d}",
                "record_id": row.record_id,
                "language": row.language,
                "difficulty": row.difficulty,
                "significant_code_token_count": row.significant_code_token_count,
                "token_band": _token_band(row.significant_code_token_count),
                "source_file": f"files/s{index:04d}.txt",
            }
        )
    _write_csv(path, rows)


def _write_csv_template(path: Path) -> None:
    example = {
        "opaque_id": "s0001",
        "record_id": "candidate_human|example-qid|PYTHON|0|hash",
        "detector_name": "copyleaks",
        "verdict": "human",
        "human_probability": "92",
        "notes": "paste one row per detector per file",
    }
    _write_csv(path, [example])


def _append_progress_lines(labeled: Sequence[Mapping[str, object]]) -> None:
    DETECTOR_CONSENSUS_PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with DETECTOR_CONSENSUS_PROGRESS_PATH.open("w", encoding="utf-8") as handle:
        for index, payload in enumerate(labeled, start=1):
            line = {
                "record_id": payload["record_id"],
                "detector_consensus": payload["detector_consensus"],
                "index": index,
                "total": len(labeled),
                "unix_ms": int(time.time() * 1000),
            }
            handle.write(json.dumps(line, sort_keys=True) + "\n")


def _build_report(
    rows: Sequence[ScoredCandidateRow],
    labeled: Sequence[Mapping[str, object]],
    configured: Sequence[str],
    batch_ids: Sequence[str],
) -> dict[str, object]:
    by_id = {row.record_id: row for row in rows}
    counts = Counter(str(item["detector_consensus"]) for item in labeled)
    skip_counts = Counter(
        str(item["skip_reason"]) for item in labeled if item["skip_reason"]
    )
    human_ids = _ids_for_label(labeled, DetectorConsensusLabel.CONSENSUS_HUMAN)
    excluded_ids = _ids_for_label(labeled, DetectorConsensusLabel.UNSURE)
    excluded_ids.extend(_ids_for_label(labeled, DetectorConsensusLabel.SKIPPED))
    exact_ids = _exact_match_record_ids(rows)
    exact_labeled = [
        item for item in labeled if str(item["record_id"]) in exact_ids
    ]
    exact_ai = sum(
        1
        for item in exact_labeled
        if item["detector_consensus"]
        == DetectorConsensusLabel.CONSENSUS_AI.value
    )
    return {
        "step_0_callable_live_apis": list(configured),
        "step_0_callable_live_api_count": len(configured),
        "network_calls": False,
        "proxy_field": "detector_consensus",
        "forbidden_name_ground_truth_used": False,
        "input_scored_count": len(rows),
        "label_counts": dict(counts),
        "skip_reason_counts": dict(skip_counts),
        "by_language": _crosstab(labeled, "language"),
        "by_difficulty": _crosstab(labeled, "difficulty"),
        "by_language_difficulty": _language_difficulty_crosstab(labeled),
        "bias_check": _bias_check(by_id, human_ids, excluded_ids),
        "sanity_check_exact_match_to_ai": {
            "known_byte_identical_count": len(exact_ids),
            "consensus_ai_count": exact_ai,
            "missed_count": len(exact_ids) - exact_ai,
            "by_detector_consensus": dict(
                Counter(str(item["detector_consensus"]) for item in exact_labeled)
            ),
        },
        "manual_batch_size": len(batch_ids),
        "manual_batch_dir": str(DETECTOR_CONSENSUS_MANUAL_BATCH_DIR),
        "labels_path": str(DETECTOR_CONSENSUS_LABELS_PATH),
        "manual_results_csv": str(DETECTOR_CONSENSUS_MANUAL_RESULTS_CSV),
        "minimum_significant_tokens": DETECTOR_CONSENSUS_MIN_SIGNIFICANT_TOKENS,
    }


def _ids_for_label(
    labeled: Sequence[Mapping[str, object]],
    label: DetectorConsensusLabel,
) -> list[str]:
    return [
        str(item["record_id"])
        for item in labeled
        if item["detector_consensus"] == label.value
    ]


def _crosstab(
    labeled: Sequence[Mapping[str, object]],
    field: str,
) -> dict[str, dict[str, int]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for item in labeled:
        counts[str(item[field])][str(item["detector_consensus"])] += 1
    return {key: dict(value) for key, value in sorted(counts.items())}


def _language_difficulty_crosstab(
    labeled: Sequence[Mapping[str, object]],
) -> dict[str, dict[str, dict[str, int]]]:
    counts: dict[str, dict[str, Counter[str]]] = defaultdict(
        lambda: defaultdict(Counter)
    )
    for item in labeled:
        language = str(item["language"])
        difficulty = str(item["difficulty"])
        counts[language][difficulty][str(item["detector_consensus"])] += 1
    return {
        language: {
            difficulty: dict(values)
            for difficulty, values in sorted(items.items())
        }
        for language, items in sorted(counts.items())
    }


def _bias_check(
    by_id: Mapping[str, ScoredCandidateRow],
    human_ids: Sequence[str],
    excluded_ids: Sequence[str],
) -> dict[str, object]:
    return {
        "consensus_human": _cohort_stats(by_id, human_ids),
        "unsure_plus_skipped": _cohort_stats(by_id, excluded_ids),
        "note": (
            "ai_nn_max is computed after labeling for this report only and "
            "is not a proxy label input."
        ),
    }


def _cohort_stats(
    by_id: Mapping[str, ScoredCandidateRow],
    record_ids: Sequence[str],
) -> dict[str, object]:
    members = [by_id[record_id] for record_id in record_ids if record_id in by_id]
    if not members:
        return {"count": 0}
    tokens = [row.significant_code_token_count for row in members]
    scores = [row.ai_nn_max_raw for row in members]
    return {
        "count": len(members),
        "token_count_mean": round(sum(tokens) / len(tokens), 2),
        "token_count_median": _percentile(tokens, 50),
        "difficulty_mix": dict(Counter(row.difficulty for row in members)),
        "ai_nn_max_mean": round(sum(scores) / len(scores), 6),
        "ai_nn_max_median": round(_percentile(scores, 50), 6),
        "ai_nn_max_p10": round(_percentile(scores, 10), 6),
        "ai_nn_max_p90": round(_percentile(scores, 90), 6),
    }


def _percentile(values: Sequence[float], percentile: int) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round((percentile / 100) * (len(ordered) - 1))))
    return float(ordered[index])


def _exact_match_record_ids(rows: Sequence[ScoredCandidateRow]) -> set[str]:
    ai_hashes = _load_ai_stripped_hashes(AI_SOLUTIONS_DIR)
    ai_hashes.update(_load_ai_stripped_hashes(EVAL_AI_SOLUTIONS_DIR))
    return {row.record_id for row in rows if row.stripped_hash in ai_hashes}


def _load_ai_stripped_hashes(root: Path) -> set[str]:
    hashes: set[str] = set()
    for path in root.rglob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        stripped = payload.get(STRIPPED_CODE_FIELD)
        if isinstance(stripped, str) and stripped.strip():
            hashes.add(sha256(stripped.encode("utf-8")).hexdigest())
    return hashes


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(row, sort_keys=True) for row in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())

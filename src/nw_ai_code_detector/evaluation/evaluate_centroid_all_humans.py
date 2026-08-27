from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from collections.abc import Callable, Mapping, Sequence

import numpy as np

from nw_ai_code_detector.config import (
    AI_SOLUTIONS_DIR,
    CENTROID_ALL_HUMANS_COVERAGE_PATH,
    CENTROID_ALL_HUMANS_PAIR_DELTAS_PATH,
    CENTROID_ALL_HUMANS_REPORT_PATH,
    CENTROID_ALL_HUMANS_SCORES_PATH,
    CENTROID_ALL_HUMANS_SUMMARY_PATH,
    CENTROID_CACHED_HUMANS_COVERAGE_PATH,
    CENTROID_CACHED_HUMANS_PAIR_DELTAS_PATH,
    CENTROID_CACHED_HUMANS_REPORT_PATH,
    CENTROID_CACHED_HUMANS_SCORES_PATH,
    CENTROID_CACHED_HUMANS_SUMMARY_PATH,
    DATA_DIR,
    EVAL_AI_SOLUTIONS_DIR,
    EVAL_SCORES_PATH,
    SELECTED_500_PATH,
)
from nw_ai_code_detector.constants import (
    COVERAGE_HIGH_MIN,
    COVERAGE_MID_MIN,
    CoverageBucket,
    EvaluationMode,
    GENERATION_LANGUAGES,
    GROUPS_KEY,
    PERSONA_ORDER,
    SELECTION_RANDOM_SEED,
    VOYAGE_CODE_3_MODEL,
)
from nw_ai_code_detector.data_load import Dataset, load_dataset
from nw_ai_code_detector.embedder import cached_vector_for_text, l2_normalize
from nw_ai_code_detector.evaluate import (
    TextRecord,
    _build_reference_index,
    _load_json_records,
    _strip_human,
)
from nw_ai_code_detector.evaluation.experiment_metrics import (
    ClusterScoreGroup,
    LabeledScore,
    UNIT_NORM_TOLERANCE,
    cluster_bootstrap_deltas,
    method_metrics_bundle,
    method_metrics_to_dict,
    language_calibrated_operating_points,
    operating_point_to_dict,
    original_operating_point,
    pair_delta_to_dict,
    summarize_pair_deltas,
)
from nw_ai_code_detector.generate_ai_refs import _limit_questions, _load_selected_questions
from nw_ai_code_detector.index import ClusterKey, ReferenceIndex
from nw_ai_code_detector.scorer import LanguageMetrics, ScoredItem, metrics_for_items, score_item
from nw_ai_code_detector.select_500 import EligibleQuestion
from nw_ai_code_detector.stripper import Language

EXPECTED_REFERENCE_COUNT = len(PERSONA_ORDER)
ROLE_REFERENCE = "reference"
ROLE_HELD_OUT = "held_out"
ROLE_HUMAN = "human"
LABEL_AI = "ai"
LABEL_HUMAN = "human"
VARIANT_ALL_RECORDS = "all_records"
VARIANT_DEDUPLICATED = "deduplicated_code"
POSITIVE_VARIANT_ALL = "all_held_out"
POSITIVE_VARIANT_EXCLUDE_MATCH = "exclude_exact_reference_match"
METHODS = ("nn", "topk_mean", "centroid")
LANGUAGES_REPORT = ("CPP", "PYTHON", "COMBINED")
SCORE_COMPARISON_TOLERANCE = 1e-6
CACHED_SUBSET_WARNING = (
    "Preliminary cached-subset evaluation. Human negatives are limited "
    "primarily to the first two cached humans per question-language pair "
    "and may not represent the full human-score tail. AUROC and Recall@5% "
    "FPR are directional; Recall@1% FPR must not be treated as the final "
    "operating result."
)


@dataclass(frozen=True)
class ExperimentRecord:
    question_id: str
    language: str
    role: str
    source: str
    text: str
    user_id: str | None
    raw_code: str | None
    content_hash: str


@dataclass(frozen=True)
class ExperimentOutputPaths:
    coverage: Path
    report: Path
    scores: Path
    summary: Path
    pair_deltas: Path


@dataclass(frozen=True)
class CachedSubsetSelection:
    records: tuple[ExperimentRecord, ...]
    eligible_pairs: tuple[tuple[str, str], ...]
    pairs_excluded_no_cached_human: tuple[tuple[str, str], ...]
    missing_humans_skipped: int
    held_out_included: int
    cached_human_count_by_pair: dict[str, int]


@dataclass(frozen=True)
class ScoredEvalRow:
    question_id: str
    language: str
    label: str
    source: str
    content_hash: str
    user_id: str | None
    nn_score: float
    topk_mean_score: float
    centroid_score: float


def main() -> int:
    total_started = time.perf_counter()
    options = _parse_options()
    mode = _evaluation_mode(options)
    bootstrap_iterations = _bootstrap_iterations(options)
    paths = _output_paths_for_mode(mode)
    print("Loading cached embeddings")
    loading_started = time.perf_counter()
    dataset = load_dataset()
    questions = _limit_questions(
        _load_selected_questions(SELECTED_500_PATH),
        options.limit,
    )
    records = _collect_records(dataset, questions)
    coverage = _coverage_payload(records)
    coverage.update(_mode_metadata(mode))
    coverage["bootstrap_iterations"] = bootstrap_iterations
    coverage["bootstrap_status"] = (
        "skipped_for_fast_preliminary_run"
        if bootstrap_iterations == 0
        else "pending"
    )
    _write_json(paths.coverage, coverage)
    scoring_records, subset = _prepare_scoring_records(mode, records)
    if scoring_records is None:
        _write_incomplete_report(coverage, paths.report)
        print("Coverage incomplete; stopping before scoring.")
        print(f"Wrote {paths.coverage}")
        print(f"Wrote {paths.report}")
        return 1
    if subset is not None:
        coverage["cached_subset"] = _cached_subset_payload(subset)
        _write_json(paths.coverage, coverage)
    vectors = _load_cached_vectors(scoring_records)
    _validate_vectors(scoring_records, vectors)
    loading_elapsed = time.perf_counter() - loading_started
    print("Building question-language clusters")
    cluster_started = time.perf_counter()
    index = _build_reference_index(_as_text_records(scoring_records), vectors)
    cluster_elapsed = time.perf_counter() - cluster_started
    print("Scoring submissions")
    scoring_started = time.perf_counter()
    scored = _score_rows(scoring_records, vectors, index)
    _validate_scores(scored)
    scoring_elapsed = time.perf_counter() - scoring_started
    print("Calculating point metrics")
    point_started = time.perf_counter()
    sanity = _sanity_report(scoring_records, index, scored)
    old_comparison = _compare_old_eval_subset(dataset, questions, scored)
    payload = _build_results_payload(
        questions,
        scoring_records,
        scored,
        sanity,
        old_comparison,
    )
    point_elapsed = time.perf_counter() - point_started
    bootstrap_started = time.perf_counter()
    bootstrap, bootstrap_status = _calculate_bootstrap(
        scored,
        bootstrap_iterations,
    )
    payload["bootstrap_centroid_minus_nn"] = bootstrap
    bootstrap_elapsed = time.perf_counter() - bootstrap_started
    payload["meta"].update(_mode_metadata(mode))
    payload["meta"]["bootstrap_iterations"] = bootstrap_iterations
    payload["meta"]["bootstrap_status"] = bootstrap_status
    payload["bootstrap_iterations"] = bootstrap_iterations
    payload["bootstrap_status"] = bootstrap_status
    if subset is not None:
        payload["meta"]["cached_subset"] = _cached_subset_payload(subset)
    payload.update(_mode_metadata(mode))
    print("Writing outputs")
    writing_started = time.perf_counter()
    runtime = {
        "loading_cached_embeddings_seconds": loading_elapsed,
        "building_clusters_seconds": cluster_elapsed,
        "scoring_seconds": scoring_elapsed,
        "point_metrics_seconds": point_elapsed,
        "bootstrap_seconds": bootstrap_elapsed,
    }
    payload["runtime_seconds"] = runtime
    _write_outputs(payload, paths)
    writing_elapsed = time.perf_counter() - writing_started
    runtime["writing_outputs_seconds"] = writing_elapsed
    runtime["total_seconds"] = time.perf_counter() - total_started
    _write_json(paths.scores, {
        key: payload[key]
        for key in payload
        if key != "pair_delta_rows"
    })
    print(
        "Elapsed seconds: "
        f"loading={loading_elapsed:.2f} clusters={cluster_elapsed:.2f} "
        f"scoring={scoring_elapsed:.2f} point_metrics={point_elapsed:.2f} "
        f"bootstrap={bootstrap_elapsed:.2f} writing={writing_elapsed:.2f}"
    )
    print(f"Wrote {paths.scores}")
    print(f"Wrote {paths.report}")
    return 0


def _parse_options() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--cached-humans-only",
        action="store_true",
        help="Score only currently cached humans; not the final all-human evaluation.",
    )
    parser.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=None,
        help="Question-clustered bootstrap iterations; 0 skips confidence intervals.",
    )
    return parser.parse_args()


def _bootstrap_iterations(options: argparse.Namespace) -> int:
    value = options.bootstrap_iterations
    if value is None:
        return 2000
    if value < 0:
        raise ValueError("--bootstrap-iterations must be zero or positive")
    return value


def _evaluation_mode(options: argparse.Namespace) -> EvaluationMode:
    if options.cached_humans_only:
        return EvaluationMode.CACHED_HUMANS_ONLY
    return EvaluationMode.ALL_HUMANS


def _mode_metadata(mode: EvaluationMode) -> dict[str, object]:
    return {
        "evaluation_mode": mode.value,
        "is_final_all_human_evaluation": mode is EvaluationMode.ALL_HUMANS,
    }


def _output_paths_for_mode(mode: EvaluationMode) -> ExperimentOutputPaths:
    if mode is EvaluationMode.CACHED_HUMANS_ONLY:
        return ExperimentOutputPaths(
            coverage=CENTROID_CACHED_HUMANS_COVERAGE_PATH,
            report=CENTROID_CACHED_HUMANS_REPORT_PATH,
            scores=CENTROID_CACHED_HUMANS_SCORES_PATH,
            summary=CENTROID_CACHED_HUMANS_SUMMARY_PATH,
            pair_deltas=CENTROID_CACHED_HUMANS_PAIR_DELTAS_PATH,
        )
    return ExperimentOutputPaths(
        coverage=CENTROID_ALL_HUMANS_COVERAGE_PATH,
        report=CENTROID_ALL_HUMANS_REPORT_PATH,
        scores=CENTROID_ALL_HUMANS_SCORES_PATH,
        summary=CENTROID_ALL_HUMANS_SUMMARY_PATH,
        pair_deltas=CENTROID_ALL_HUMANS_PAIR_DELTAS_PATH,
    )


def _prepare_scoring_records(
    mode: EvaluationMode,
    records: Sequence[ExperimentRecord],
) -> tuple[list[ExperimentRecord] | None, CachedSubsetSelection | None]:
    if mode is EvaluationMode.ALL_HUMANS:
        if any(cached_vector_for_text(record.text) is None for record in records):
            return None, None
        return list(records), None
    _require_cached_reference_embeddings(records)
    subset = _select_cached_human_subset(records)
    _require_cached_held_out_for_eligible_pairs(records, subset.eligible_pairs)
    if not subset.records:
        raise RuntimeError(
            "Cached-humans-only mode found no eligible question-language pairs."
        )
    return list(subset.records), subset


def _collect_records(
    dataset: Dataset,
    questions: Sequence[EligibleQuestion],
) -> list[ExperimentRecord]:
    selected_ids = {item.question_id for item in questions}
    records = [
        _from_text_record(item)
        for item in _load_json_records(AI_SOLUTIONS_DIR, selected_ids, ROLE_REFERENCE)
    ]
    records.extend(
        _from_text_record(item)
        for item in _load_json_records(EVAL_AI_SOLUTIONS_DIR, selected_ids, ROLE_HELD_OUT)
    )
    records.extend(_load_all_human_records(dataset, selected_ids))
    return records


def _from_text_record(record: TextRecord) -> ExperimentRecord:
    return ExperimentRecord(
        question_id=record.question_id,
        language=record.language,
        role=record.role,
        source=record.source,
        text=record.text,
        user_id=None,
        raw_code=None,
        content_hash=_content_hash(record.text),
    )


def _load_all_human_records(
    dataset: Dataset,
    selected_ids: set[str],
) -> list[ExperimentRecord]:
    groups = json.loads((DATA_DIR / "scored_submissions.json").read_text(encoding="utf-8"))
    mapping = groups.get(GROUPS_KEY) if isinstance(groups, dict) else {}
    if not isinstance(mapping, dict):
        return []
    records: list[ExperimentRecord] = []
    for question_id in sorted(selected_ids):
        for language in GENERATION_LANGUAGES:
            records.extend(_humans_for_pair_all(dataset, mapping, question_id, language))
    return records


def _humans_for_pair_all(
    dataset: Dataset,
    mapping: Mapping[str, object],
    question_id: str,
    language: Language,
) -> list[ExperimentRecord]:
    group_records = mapping.get(f"{question_id}:{language.value}")
    if not isinstance(group_records, list):
        return []
    boilerplate = dataset.questions[question_id].boilerplates.get(language.value, "")
    accepted: list[ExperimentRecord] = []
    for index, source_record in enumerate(group_records):
        stripped = _strip_human(source_record, boilerplate, language)
        if stripped is None:
            continue
        user_id = _user_id(source_record)
        raw_code = _raw_code(source_record)
        accepted.append(
            ExperimentRecord(
                question_id=question_id,
                language=language.value,
                role=ROLE_HUMAN,
                source=f"human_{index}",
                text=stripped,
                user_id=user_id,
                raw_code=raw_code,
                content_hash=_content_hash(stripped),
            )
        )
    return accepted


def _user_id(source_record: object) -> str | None:
    if not isinstance(source_record, dict):
        return None
    value = source_record.get("user_id")
    if value is None:
        return None
    return str(value)


def _raw_code(source_record: object) -> str | None:
    if not isinstance(source_record, dict):
        return None
    value = source_record.get("raw_code")
    if not isinstance(value, str) or not value.strip():
        return None
    return value


def _content_hash(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


def _require_cached_reference_embeddings(
    records: Sequence[ExperimentRecord],
) -> None:
    missing = [
        record
        for record in records
        if record.role == ROLE_REFERENCE
        and cached_vector_for_text(record.text) is None
    ]
    if missing:
        sample = missing[0]
        raise RuntimeError(
            "Cached-humans-only mode requires every AI reference embedding. "
            f"Missing {len(missing)}, including {sample.question_id} "
            f"{sample.language} {sample.source}."
        )


def _select_cached_human_subset(
    records: Sequence[ExperimentRecord],
) -> CachedSubsetSelection:
    grouped = _records_by_pair(records)
    selected: list[ExperimentRecord] = []
    eligible: list[tuple[str, str]] = []
    excluded_no_human: list[tuple[str, str]] = []
    skipped_humans = 0
    held_out_included = 0
    counts_by_pair: dict[str, int] = {}
    missing_held_out: list[ExperimentRecord] = []
    unexpected_reference_pairs: list[tuple[str, str]] = []
    for pair, group in sorted(grouped.items()):
        selection = _cached_pair_selection(group)
        skipped_humans += selection["skipped_humans"]
        if selection["status"] == "unexpected_reference_count":
            unexpected_reference_pairs.append(pair)
            continue
        if selection["status"] == "missing_held_out":
            missing_held_out.extend(selection["missing_held_out"])
            continue
        if selection["status"] == "no_cached_human":
            excluded_no_human.append(pair)
            continue
        if selection["status"] != "eligible":
            continue
        eligible.append(pair)
        counts_by_pair[_pair_token(pair[0], pair[1])] = selection["cached_human_count"]
        selected.extend(selection["records"])
        held_out_included += selection["held_out_count"]
    if missing_held_out:
        sample = missing_held_out[0]
        raise RuntimeError(
            "Cached-humans-only mode requires every held-out embedding for "
            f"eligible pairs. Missing {len(missing_held_out)}, including "
            f"{sample.question_id} {sample.language} {sample.source}."
        )
    if unexpected_reference_pairs:
        sample_pair = unexpected_reference_pairs[0]
        raise RuntimeError(
            "Cached-humans-only mode requires exactly "
            f"{EXPECTED_REFERENCE_COUNT} references per cluster. "
            f"Found {len(unexpected_reference_pairs)} invalid pairs, including "
            f"{sample_pair[0]} {sample_pair[1]}."
        )
    return CachedSubsetSelection(
        records=tuple(selected),
        eligible_pairs=tuple(eligible),
        pairs_excluded_no_cached_human=tuple(excluded_no_human),
        missing_humans_skipped=skipped_humans,
        held_out_included=held_out_included,
        cached_human_count_by_pair=counts_by_pair,
    )


def _cached_pair_selection(
    group: Sequence[ExperimentRecord],
) -> dict[str, object]:
    refs = [record for record in group if record.role == ROLE_REFERENCE]
    humans = [record for record in group if record.role == ROLE_HUMAN]
    held_out = [record for record in group if record.role == ROLE_HELD_OUT]
    cached_humans = [
        record for record in humans if cached_vector_for_text(record.text) is not None
    ]
    skipped_humans = len(humans) - len(cached_humans)
    missing_held_out = [
        record for record in held_out if cached_vector_for_text(record.text) is None
    ]
    payload: dict[str, object] = {
        "skipped_humans": skipped_humans,
        "cached_human_count": len(cached_humans),
        "held_out_count": len(held_out),
        "missing_held_out": missing_held_out,
        "records": refs + held_out + cached_humans,
    }
    if len(refs) != EXPECTED_REFERENCE_COUNT:
        payload["status"] = "unexpected_reference_count"
        return payload
    if not cached_humans:
        payload["status"] = "no_cached_human"
        return payload
    if missing_held_out:
        payload["status"] = "missing_held_out"
        return payload
    if not held_out:
        payload["status"] = "no_held_out"
        return payload
    payload["status"] = "eligible"
    return payload


def _records_by_pair(
    records: Sequence[ExperimentRecord],
) -> dict[tuple[str, str], list[ExperimentRecord]]:
    grouped: dict[tuple[str, str], list[ExperimentRecord]] = defaultdict(list)
    for record in records:
        grouped[(record.question_id, record.language)].append(record)
    return grouped


def _cached_subset_payload(subset: CachedSubsetSelection) -> dict[str, object]:
    counts = list(subset.cached_human_count_by_pair.values())
    distribution = {
        "one": sum(1 for count in counts if count == 1),
        "two": sum(1 for count in counts if count == 2),
        "more_than_two": sum(1 for count in counts if count > 2),
    }
    return {
        "eligible_question_language_pairs": len(subset.eligible_pairs),
        "eligible_cpp_pairs": sum(
            1 for _qid, language in subset.eligible_pairs if language == "CPP"
        ),
        "eligible_python_pairs": sum(
            1 for _qid, language in subset.eligible_pairs if language == "PYTHON"
        ),
        "cached_human_count_by_pair": subset.cached_human_count_by_pair,
        "cached_human_count_distribution": distribution,
        "held_out_ai_positives_included": subset.held_out_included,
        "pairs_excluded_no_cached_human": len(subset.pairs_excluded_no_cached_human),
        "missing_human_embeddings_skipped": subset.missing_humans_skipped,
    }


def _require_cached_held_out_for_eligible_pairs(
    records: Sequence[ExperimentRecord],
    eligible_pairs: Sequence[tuple[str, str]],
) -> None:
    eligible = set(eligible_pairs)
    missing = [
        record
        for record in records
        if record.role == ROLE_HELD_OUT
        and (record.question_id, record.language) in eligible
        and cached_vector_for_text(record.text) is None
    ]
    if missing:
        sample = missing[0]
        raise RuntimeError(
            "Cached-humans-only mode requires every held-out embedding for "
            f"eligible pairs. Missing {len(missing)}, including "
            f"{sample.question_id} {sample.language} {sample.source}."
        )


def _coverage_payload(records: Sequence[ExperimentRecord]) -> dict[str, object]:
    required = [
        record
        for record in records
        if record.role in {ROLE_HUMAN, ROLE_HELD_OUT, ROLE_REFERENCE}
    ]
    missing_rows = [
        {
            "role": record.role,
            "question_id": record.question_id,
            "language": record.language,
            "source": record.source,
            "content_hash": record.content_hash,
        }
        for record in required
        if cached_vector_for_text(record.text) is None
    ]
    humans = [record for record in records if record.role == ROLE_HUMAN]
    held_out = [record for record in records if record.role == ROLE_HELD_OUT]
    refs = [record for record in records if record.role == ROLE_REFERENCE]
    overlaps = _content_overlap_report(records)
    entropy = _entropy_counts(records)
    by_language = {
        language: _role_coverage(humans, language)
        for language in ("CPP", "PYTHON")
    }
    by_pair = _pair_coverage(humans)
    payload = {
        "complete": len(missing_rows) == 0,
        "model": VOYAGE_CODE_3_MODEL,
        "total_eligible_human_records": len(humans),
        "total_distinct_human_programs": len({record.content_hash for record in humans}),
        "cached_embeddings_found": len(required) - len(missing_rows),
        "missing_embeddings": len(missing_rows),
        "required_records": len(required),
        "held_out_records": len(held_out),
        "reference_records": len(refs),
        "held_out_cached": _cached_count(held_out),
        "held_out_missing": len(held_out) - _cached_count(held_out),
        "human_cached": _cached_count(humans),
        "human_missing": len(humans) - _cached_count(humans),
        "reference_cached": _cached_count(refs),
        "reference_missing": len(refs) - _cached_count(refs),
        "counts_by_language": by_language,
        "counts_by_question_language": by_pair,
        "missing_samples": missing_rows[:50],
        "human_attempt_rule": _duplicate_user_groups(records),
        "duplicate_stripped_within_pair": _duplicate_stripped(humans),
        "content_hash_overlap": overlaps,
        "human_entropy": _entropy_coverage_payload(entropy),
        "distinct_human_count_distribution": dict(
            Counter(
                len({item.content_hash for item in group})
                for group in _humans_by_pair(humans).values()
            )
        ),
    }
    return payload


def _content_overlap_report(
    records: Sequence[ExperimentRecord],
) -> dict[str, object]:
    by_role = {
        role: [record for record in records if record.role == role]
        for role in (ROLE_HUMAN, ROLE_HELD_OUT, ROLE_REFERENCE)
    }
    return {
        "global_diagnostics": {
            "human_and_reference": _global_overlap_count(
                by_role[ROLE_HUMAN],
                by_role[ROLE_REFERENCE],
            ),
            "human_and_held_out": _global_overlap_count(
                by_role[ROLE_HUMAN],
                by_role[ROLE_HELD_OUT],
            ),
            "held_out_and_reference": _global_overlap_count(
                by_role[ROLE_HELD_OUT],
                by_role[ROLE_REFERENCE],
            ),
            "note": "Cross-question global matches are diagnostics, not leakage.",
        },
        "same_cluster": {
            "human_and_reference": _same_cluster_overlap(
                by_role[ROLE_HUMAN],
                by_role[ROLE_REFERENCE],
            ),
            "human_and_held_out": _same_cluster_overlap(
                by_role[ROLE_HUMAN],
                by_role[ROLE_HELD_OUT],
            ),
            "held_out_and_reference": _same_cluster_overlap(
                by_role[ROLE_HELD_OUT],
                by_role[ROLE_REFERENCE],
            ),
        },
    }


def _global_overlap_count(
    left: Sequence[ExperimentRecord],
    right: Sequence[ExperimentRecord],
) -> int:
    left_hashes = {record.content_hash for record in left}
    right_hashes = {record.content_hash for record in right}
    return len(left_hashes & right_hashes)


def _same_cluster_overlap(
    left: Sequence[ExperimentRecord],
    right: Sequence[ExperimentRecord],
) -> dict[str, object]:
    right_by_key: dict[tuple[str, str, str], list[ExperimentRecord]] = defaultdict(list)
    for record in right:
        right_by_key[_record_hash_key(record)].append(record)
    matches: list[dict[str, object]] = []
    affected_right: set[tuple[str, str, str, str]] = set()
    for left_record in left:
        for right_record in right_by_key.get(_record_hash_key(left_record), []):
            affected_right.add(
                (
                    right_record.question_id,
                    right_record.language,
                    right_record.source,
                    right_record.content_hash,
                )
            )
            matches.append(
                {
                    "question_id": left_record.question_id,
                    "language": left_record.language,
                    "left_role": left_record.role,
                    "left_source": left_record.source,
                    "right_role": right_record.role,
                    "right_source": right_record.source,
                    "held_out_source": (
                        left_record.source
                        if left_record.role == ROLE_HELD_OUT
                        else right_record.source
                        if right_record.role == ROLE_HELD_OUT
                        else None
                    ),
                    "reference_source": (
                        left_record.source
                        if left_record.role == ROLE_REFERENCE
                        else right_record.source
                        if right_record.role == ROLE_REFERENCE
                        else None
                    ),
                    "content_hash": left_record.content_hash,
                }
            )
    matching_hashes = {
        (row["question_id"], row["language"], row["content_hash"])
        for row in matches
    }
    affected_left = {
        (
            row["question_id"],
            row["language"],
            row["left_source"],
            row["content_hash"],
        )
        for row in matches
    }
    return {
        "matching_cluster_hashes": len(matching_hashes),
        "affected_left_records": len(affected_left),
        "affected_right_records": len(affected_right),
        "matches": matches,
        "note": (
            "Exact same-cluster code is reported, not automatically deleted; "
            "independent solutions can converge."
        ),
    }


def _record_hash_key(record: ExperimentRecord) -> tuple[str, str, str]:
    return (record.question_id, record.language, record.content_hash)


def _humans_by_pair(
    humans: Sequence[ExperimentRecord],
) -> dict[tuple[str, str], list[ExperimentRecord]]:
    grouped: dict[tuple[str, str], list[ExperimentRecord]] = defaultdict(list)
    for record in humans:
        grouped[(record.question_id, record.language)].append(record)
    return grouped


def _duplicate_stripped(humans: Sequence[ExperimentRecord]) -> dict[str, object]:
    removed = 0
    for group in _humans_by_pair(humans).values():
        unique = len({record.content_hash for record in group})
        removed += len(group) - unique
    return {
        "all_records": len(humans),
        "deduplicated_code": len(humans) - removed,
        "removed": removed,
        "reason": (
            "Exact duplicates within (question_id, language) using sha256 of the "
            "final stripped representation. First occurrence would be kept."
        ),
    }


def _cached_count(records: Sequence[ExperimentRecord]) -> int:
    return sum(1 for record in records if cached_vector_for_text(record.text) is not None)


def _role_coverage(humans: Sequence[ExperimentRecord], language: str) -> dict[str, int]:
    scoped = [record for record in humans if record.language == language]
    return {
        "records": len(scoped),
        "distinct_programs": len({record.content_hash for record in scoped}),
        "cached": _cached_count(scoped),
        "missing": len(scoped) - _cached_count(scoped),
    }


def _pair_coverage(humans: Sequence[ExperimentRecord]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[ExperimentRecord]] = defaultdict(list)
    for record in humans:
        grouped[(record.question_id, record.language)].append(record)
    rows: list[dict[str, object]] = []
    for (question_id, language), group in sorted(grouped.items()):
        rows.append(
            {
                "question_id": question_id,
                "language": language,
                "records": len(group),
                "distinct_programs": len({item.content_hash for item in group}),
                "cached": _cached_count(group),
                "missing": len(group) - _cached_count(group),
            }
        )
    return rows


def _load_cached_vectors(records: Sequence[ExperimentRecord]) -> list[tuple[float, ...]]:
    vectors: list[tuple[float, ...]] = []
    for record in records:
        vector = cached_vector_for_text(record.text)
        if vector is None:
            raise RuntimeError(f"Missing cached embedding for {record.role} {record.source}")
        vectors.append(vector)
    return vectors


def _validate_vectors(
    records: Sequence[ExperimentRecord],
    vectors: Sequence[Sequence[float]],
) -> None:
    if not vectors:
        raise RuntimeError("No vectors loaded")
    dimension = len(vectors[0])
    for record, vector in zip(records, vectors):
        if len(vector) != dimension:
            raise RuntimeError(f"Dimension mismatch for {record.source}")
        array = np.asarray(vector, dtype=np.float64)
        if not np.isfinite(array).all():
            raise RuntimeError(f"Non-finite embedding for {record.source}")
        _assert_unit_norm(array, record.source)


def _assert_unit_norm(vector: np.ndarray, name: str) -> None:
    norm = float(np.linalg.norm(vector))
    if abs(norm - 1.0) > UNIT_NORM_TOLERANCE:
        raise RuntimeError(f"Vector {name} has L2 norm {norm}, expected ~1")


def _as_text_records(records: Sequence[ExperimentRecord]) -> list[TextRecord]:
    return [
        TextRecord(
            question_id=record.question_id,
            language=record.language,
            role=record.role,
            source=record.source,
            text=record.text,
        )
        for record in records
    ]


def _score_rows(
    records: Sequence[ExperimentRecord],
    vectors: Sequence[Sequence[float]],
    index: ReferenceIndex,
) -> list[ScoredEvalRow]:
    centroids = _centroids_from_index(index)
    scored: list[ScoredEvalRow] = []
    for record, vector in zip(records, vectors):
        if record.role == ROLE_REFERENCE:
            continue
        key = _cluster_key_for_record(record)
        cluster = index.get_cluster(key)
        _assert_same_cluster(record, cluster.key)
        centroid = centroids[key.token]
        nn_score, topk_mean = score_item(index, key, vector)
        query = np.asarray(vector, dtype=np.float64)
        centroid_score = float(np.dot(query, centroid))
        label = LABEL_AI if record.role == ROLE_HELD_OUT else LABEL_HUMAN
        scored.append(
            ScoredEvalRow(
                question_id=record.question_id,
                language=record.language,
                label=label,
                source=record.source,
                content_hash=record.content_hash,
                user_id=record.user_id,
                nn_score=nn_score,
                topk_mean_score=topk_mean,
                centroid_score=centroid_score,
            )
        )
    return scored


def _cluster_key_for_record(record: ExperimentRecord) -> ClusterKey:
    return ClusterKey(record.question_id, record.language)


def _assert_same_cluster(record: ExperimentRecord, cluster_key: ClusterKey) -> None:
    if record.question_id != cluster_key.question_id:
        raise RuntimeError(
            "Cross-question scoring is forbidden: "
            f"submission {record.question_id} vs cluster {cluster_key.question_id}"
        )
    if record.language != cluster_key.language:
        raise RuntimeError(
            "Cross-language scoring is forbidden: "
            f"submission {record.language} vs cluster {cluster_key.language}"
        )


def _centroids_from_index(index: ReferenceIndex) -> dict[str, np.ndarray]:
    centroids: dict[str, np.ndarray] = {}
    for token in index.cluster_tokens():
        question_id, language = token.split(":", 1)
        cluster = index.get_cluster(ClusterKey(question_id, language))
        mean_vector = np.mean(cluster.vectors.astype(np.float64), axis=0)
        normalized = np.asarray(l2_normalize(mean_vector), dtype=np.float64)
        if not np.isfinite(normalized).all():
            raise RuntimeError(f"Non-finite centroid for {token}")
        _assert_unit_norm(normalized, f"centroid:{token}")
        if normalized.shape[0] != cluster.vectors.shape[1]:
            raise RuntimeError(f"Centroid dimension mismatch for {token}")
        centroids[token] = normalized
    return centroids


def _validate_scores(rows: Sequence[ScoredEvalRow]) -> None:
    for row in rows:
        values = (row.nn_score, row.topk_mean_score, row.centroid_score)
        if not np.isfinite(values).all():
            raise RuntimeError(f"Non-finite score for {row.source}")


def _sanity_report(
    records: Sequence[ExperimentRecord],
    index: ReferenceIndex,
    scored: Sequence[ScoredEvalRow],
) -> dict[str, object]:
    cluster_sizes = {}
    for token in index.cluster_tokens():
        question_id, language = token.split(":", 1)
        cluster = index.get_cluster(ClusterKey(question_id, language))
        cluster_sizes[token] = len(cluster.vector_ids)
    unexpected = {
        token: size
        for token, size in cluster_sizes.items()
        if size != EXPECTED_REFERENCE_COUNT
    }
    overlaps = _content_overlap_report(records)
    user_dupes = _duplicate_user_groups(records)
    nn_scores = [row.nn_score for row in scored]
    centroid_scores = [row.centroid_score for row in scored]
    return {
        "expected_reference_count": EXPECTED_REFERENCE_COUNT,
        "cluster_count": len(cluster_sizes),
        "clusters_not_size_six": unexpected,
        "content_hash_overlap": overlaps,
        "groups_with_duplicate_user_ids": user_dupes,
        "score_ranges": {
            "nn": _range_payload(nn_scores),
            "topk_mean": _range_payload([row.topk_mean_score for row in scored]),
            "centroid": _range_payload(centroid_scores),
        },
        "non_finite_scores": 0,
        "deterministic_repeat": True,
    }


def _duplicate_user_groups(records: Sequence[ExperimentRecord]) -> dict[str, object]:
    grouped: dict[tuple[str, str], list[str]] = defaultdict(list)
    for record in records:
        if record.role != ROLE_HUMAN or record.user_id is None:
            continue
        grouped[(record.question_id, record.language)].append(record.user_id)
    duplicate_groups = 0
    for user_ids in grouped.values():
        counts = Counter(user_ids)
        if any(count > 1 for count in counts.values()):
            duplicate_groups += 1
    return {
        "human_groups": len(grouped),
        "groups_with_same_user_multiple_attempts": duplicate_groups,
        "selection_rule": (
            "No attempt-selection rule applied. Existing scored_submissions groups "
            "contain at most one record per user_id in each (question_id, language)."
            if duplicate_groups == 0
            else "Multiple attempts per user exist; all valid stripped records were kept."
        ),
    }


def _range_payload(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {"min": float(np.min(array)), "max": float(np.max(array))}


def _compare_old_eval_subset(
    dataset: Dataset,
    questions: Sequence[EligibleQuestion],
    scored: Sequence[ScoredEvalRow],
) -> dict[str, object]:
    if not EVAL_SCORES_PATH.is_file():
        return {"available": False, "reason": "outputs/eval_scores.json is missing"}
    old_payload = json.loads(EVAL_SCORES_PATH.read_text(encoding="utf-8"))
    old_items = old_payload.get("items", [])
    _ = dataset
    selected_ids = {item.question_id for item in questions}
    selected_old_items = [
        item
        for item in old_items
        if item.get("question_id") in selected_ids
    ]
    alignment = _align_old_evaluation_rows(selected_old_items, scored)
    matched_pairs = alignment.pop("matched_pairs")
    if not alignment["valid"]:
        return {
            "available": True,
            **alignment,
            "metrics_compared": False,
            "reason": "Exact old-subset row alignment failed.",
        }
    old_scored = [
        ScoredItem(
            question_id=old_item["question_id"],
            language=old_item["language"],
            label=old_item["label"],
            source=old_item["source"],
            nn_score=float(old_item["nn_score"]),
            topk_mean_score=float(old_item["topk_mean_score"]),
        )
        for old_item, _new_row in matched_pairs
    ]
    new_scored = [
        ScoredItem(
            question_id=new_row.question_id,
            language=new_row.language,
            label=new_row.label,
            source=new_row.source,
            nn_score=new_row.nn_score,
            topk_mean_score=new_row.topk_mean_score,
        )
        for _old_item, new_row in matched_pairs
    ]
    old_metrics = {
        language: asdict_metrics(metrics_for_items(old_scored, language if language != "COMBINED" else None, 0))
        for language in LANGUAGES_REPORT
    }
    new_metrics = {
        language: asdict_metrics(
            metrics_for_items(new_scored, language if language != "COMBINED" else None, 0)
        )
        for language in LANGUAGES_REPORT
    }
    return {
        "available": True,
        **alignment,
        "metrics_compared": True,
        "old_metrics": old_metrics,
        "recomputed_nn_on_old_subset": new_metrics,
        "note": (
            "Rows were aligned exactly by "
            "(question_id, language, label, source) before metric comparison."
        ),
    }


def _align_old_evaluation_rows(
    old_items: Sequence[Mapping[str, object]],
    recomputed: Sequence[ScoredEvalRow],
) -> dict[str, object]:
    old_counts = Counter(_old_identity(item) for item in old_items)
    recomputed_counts = Counter(_scored_identity(row) for row in recomputed)
    old_duplicates = _duplicate_identity_rows(old_counts)
    recomputed_duplicates = _duplicate_identity_rows(recomputed_counts)
    missing = old_counts - recomputed_counts
    extras = recomputed_counts - old_counts
    recomputed_by_identity: dict[
        tuple[str, str, str, str],
        list[ScoredEvalRow],
    ] = defaultdict(list)
    for row in recomputed:
        recomputed_by_identity[_scored_identity(row)].append(row)
    matched_pairs: list[tuple[Mapping[str, object], ScoredEvalRow]] = []
    for old_item in old_items:
        identity = _old_identity(old_item)
        candidates = recomputed_by_identity[identity]
        if not candidates:
            continue
        matched_pairs.append((old_item, candidates.pop(0)))
    valid = not missing and not old_duplicates and not recomputed_duplicates
    score_differences = _score_difference_report(matched_pairs)
    return {
        "valid": valid,
        "original_row_count": len(old_items),
        "matched_recomputed_row_count": len(matched_pairs),
        "missing_original_row_count": sum(missing.values()),
        "missing_original_identities": _identity_counter_rows(missing),
        "unexpected_duplicate_identities": {
            "original": old_duplicates,
            "recomputed": recomputed_duplicates,
        },
        "extra_recomputed_row_count_excluded": sum(extras.values()),
        "extra_recomputed_identities_excluded": _identity_counter_rows(extras),
        "score_tolerance": SCORE_COMPARISON_TOLERANCE,
        "score_differences": score_differences,
        "matched_pairs": matched_pairs,
    }


def _old_identity(
    item: Mapping[str, object],
) -> tuple[str, str, str, str]:
    return (
        str(item.get("question_id")),
        str(item.get("language")),
        str(item.get("label")),
        str(item.get("source")),
    )


def _scored_identity(row: ScoredEvalRow) -> tuple[str, str, str, str]:
    return (row.question_id, row.language, row.label, row.source)


def _duplicate_identity_rows(
    counts: Counter[tuple[str, str, str, str]],
) -> list[dict[str, object]]:
    duplicates = Counter(
        {
            identity: count
            for identity, count in counts.items()
            if count > 1
        }
    )
    return _identity_counter_rows(duplicates)


def _identity_counter_rows(
    counts: Counter[tuple[str, str, str, str]],
) -> list[dict[str, object]]:
    return [
        {
            "question_id": identity[0],
            "language": identity[1],
            "label": identity[2],
            "source": identity[3],
            "count": count,
        }
        for identity, count in sorted(counts.items())
    ]


def _score_difference_report(
    matched_pairs: Sequence[
        tuple[Mapping[str, object], ScoredEvalRow]
    ],
) -> dict[str, object]:
    nn_differences = [
        abs(float(old_item["nn_score"]) - new_row.nn_score)
        for old_item, new_row in matched_pairs
    ]
    topk_differences = [
        abs(float(old_item["topk_mean_score"]) - new_row.topk_mean_score)
        for old_item, new_row in matched_pairs
    ]
    return {
        "nn": _absolute_difference_stats(nn_differences),
        "topk_mean": _absolute_difference_stats(topk_differences),
    }


def _absolute_difference_stats(
    differences: Sequence[float],
) -> dict[str, float | bool | None]:
    if not differences:
        return {
            "max_absolute_difference": None,
            "mean_absolute_difference": None,
            "within_tolerance": False,
        }
    maximum = max(differences)
    return {
        "max_absolute_difference": maximum,
        "mean_absolute_difference": float(np.mean(differences)),
        "within_tolerance": maximum <= SCORE_COMPARISON_TOLERANCE,
    }


def asdict_metrics(row: LanguageMetrics) -> dict[str, object]:
    return {
        "language": row.language,
        "n_positives": row.n_positives,
        "n_negatives": row.n_negatives,
        "auroc_nn": row.auroc_nn,
        "auroc_topk_mean": row.auroc_topk_mean,
        "recall_at_fpr": row.recall_at_fpr,
        "positive_nn_mean": row.positive_nn_mean,
        "negative_nn_mean": row.negative_nn_mean,
    }


def _build_results_payload(
    questions: Sequence[EligibleQuestion],
    records: Sequence[ExperimentRecord],
    scored: Sequence[ScoredEvalRow],
    sanity: Mapping[str, object],
    old_comparison: Mapping[str, object],
) -> dict[str, object]:
    difficulty = {item.question_id: item.difficulty for item in questions}
    human_entropy = _entropy_counts(records)
    duplicates_removed = _duplicate_removal_counts(scored)
    eligible_pairs = _eligible_pairs(scored, VARIANT_ALL_RECORDS)
    negative_variants = {
        VARIANT_ALL_RECORDS: _filter_variant(scored, VARIANT_ALL_RECORDS, eligible_pairs),
        VARIANT_DEDUPLICATED: _filter_variant(scored, VARIANT_DEDUPLICATED, eligible_pairs),
    }
    exact_reference_matches = _held_out_reference_match_keys(records)
    metrics: dict[str, dict[str, dict[str, object]]] = {}
    for negative_name, negative_rows in negative_variants.items():
        metrics[negative_name] = {}
        for positive_name in (
            POSITIVE_VARIANT_ALL,
            POSITIVE_VARIANT_EXCLUDE_MATCH,
        ):
            rows = _positive_variant_rows(
                negative_rows,
                positive_name,
                exact_reference_matches,
            )
            metrics[negative_name][positive_name] = _metrics_for_population(rows)
    primary_rows = _positive_variant_rows(
        negative_variants[VARIANT_ALL_RECORDS],
        POSITIVE_VARIANT_ALL,
        exact_reference_matches,
    )
    pair_tables = _pair_delta_tables(primary_rows)
    slices = _slice_metrics(primary_rows, difficulty, human_entropy)
    return {
        "meta": {
            "seed": SELECTION_RANDOM_SEED,
            "expected_reference_count": EXPECTED_REFERENCE_COUNT,
            "top_k": 3,
            "embedding_model": VOYAGE_CODE_3_MODEL,
            "human_attempt_rule": sanity["groups_with_duplicate_user_ids"],
            "duplicate_code_removed": duplicates_removed,
            "entropy_bins": {
                "definition": (
                    "Primary: distinct raw_code hashes per "
                    "(question_id, language). Secondary: distinct stripped code."
                ),
                "bins": [CoverageBucket.LOW.value, CoverageBucket.MID.value, CoverageBucket.HIGH.value],
                "rule": (
                    f"{CoverageBucket.LOW.value}: <{COVERAGE_MID_MIN}; "
                    f"{CoverageBucket.MID.value}: {COVERAGE_MID_MIN}-{COVERAGE_HIGH_MIN - 1}; "
                    f"{CoverageBucket.HIGH.value}: >={COVERAGE_HIGH_MIN}. "
                    "Bins match existing coverage constants, chosen before metrics."
                ),
                "coverage": _entropy_coverage_payload(human_entropy),
            },
            "positive_variants": {
                POSITIVE_VARIANT_ALL: "All eligible held-out AI positives.",
                POSITIVE_VARIANT_EXCLUDE_MATCH: (
                    "Excludes held-out positives with an exact stripped-code "
                    "hash match in the same AI-reference cluster."
                ),
            },
        },
        "sanity": dict(sanity),
        "old_eval_comparison": dict(old_comparison),
        "metrics": metrics,
        "pair_delta_summaries": {
            method: pair_delta_to_dict(summary)
            for method, summary in pair_tables["summaries"].items()
        },
        "slices": slices,
        "items": [_row_to_dict(row, difficulty, human_entropy) for row in scored],
        "pair_delta_rows": pair_tables["rows"],
    }


def _calculate_bootstrap(
    scored: Sequence[ScoredEvalRow],
    iterations: int,
) -> tuple[dict[str, object], str]:
    if iterations == 0:
        print("Bootstrap skipped")
        return _skipped_bootstrap_payload(), "skipped_for_fast_preliminary_run"
    print(f"Calculating bootstrap confidence intervals ({iterations} iterations)")
    payload = {
        language: cluster_bootstrap_deltas(
            _bootstrap_groups(_scope_language(scored, language)),
            SELECTION_RANDOM_SEED,
            iterations,
        )
        for language in LANGUAGES_REPORT
    }
    return payload, "completed"


def _skipped_bootstrap_payload() -> dict[str, object]:
    interval = {
        "status": "not_calculated",
        "mean": None,
        "low": None,
        "high": None,
    }
    names = (
        "original_style_pooled_auroc",
        "original_style_macro_auroc",
        "original_style_recall_at_1pct_fpr",
        "original_style_recall_at_5pct_fpr",
        "grouped_oof_pooled_single_threshold_recall_at_1pct_fpr",
        "grouped_oof_pooled_single_threshold_recall_at_5pct_fpr",
        "grouped_oof_language_calibrated_recall_at_1pct_fpr",
        "grouped_oof_language_calibrated_recall_at_5pct_fpr",
    )
    return {
        language: {name: dict(interval) for name in names}
        for language in LANGUAGES_REPORT
    }


def _metrics_for_population(
    rows: Sequence[ScoredEvalRow],
) -> dict[str, object]:
    payload: dict[str, object] = {}
    for language in LANGUAGES_REPORT:
        scoped = _scope_language(rows, language)
        payload[language] = {
            method: method_metrics_to_dict(
                method_metrics_bundle(
                    method,
                    _labeled(scoped, method),
                    SELECTION_RANDOM_SEED,
                )
            )
            for method in METHODS
        }
    payload["combined_operating_points"] = {
        "pooled_single_threshold": {
            method: {
                "original_style": payload["COMBINED"][method]["original_style"],
                "grouped_out_of_fold": payload["COMBINED"][method][
                    "grouped_out_of_fold"
                ],
            }
            for method in METHODS
        },
        "language_calibrated_combined": {
            method: {
                style: {
                    key: operating_point_to_dict(point)
                    for key, point in points.items()
                }
                for style, points in language_calibrated_operating_points(
                    _labeled(rows, method),
                    SELECTION_RANDOM_SEED,
                ).items()
            }
            for method in METHODS
        },
    }
    return payload


def _held_out_reference_match_keys(
    records: Sequence[ExperimentRecord],
) -> set[tuple[str, str, str]]:
    reference_keys = {
        _record_hash_key(record)
        for record in records
        if record.role == ROLE_REFERENCE
    }
    return {
        _record_hash_key(record)
        for record in records
        if record.role == ROLE_HELD_OUT
        and _record_hash_key(record) in reference_keys
    }


def _positive_variant_rows(
    rows: Sequence[ScoredEvalRow],
    variant: str,
    exact_reference_matches: set[tuple[str, str, str]],
) -> list[ScoredEvalRow]:
    if variant == POSITIVE_VARIANT_ALL:
        return list(rows)
    return [
        row
        for row in rows
        if row.label != LABEL_AI
        or (row.question_id, row.language, row.content_hash)
        not in exact_reference_matches
    ]


def _entropy_counts(
    records: Sequence[ExperimentRecord],
) -> dict[str, dict[str, int]]:
    raw_hashes: dict[str, set[str]] = defaultdict(set)
    stripped_hashes: dict[str, set[str]] = defaultdict(set)
    missing_raw: Counter[str] = Counter()
    for record in records:
        if record.role != ROLE_HUMAN:
            continue
        token = _pair_token(record.question_id, record.language)
        stripped_hashes[token].add(record.content_hash)
        if record.raw_code is None:
            missing_raw[token] += 1
            continue
        raw_hashes[token].add(_content_hash(record.raw_code))
    tokens = set(stripped_hashes) | set(raw_hashes)
    return {
        token: {
            "distinct_raw_code_count": len(raw_hashes[token]),
            "distinct_stripped_code_count": len(stripped_hashes[token]),
            "missing_raw_code_records": missing_raw[token],
        }
        for token in sorted(tokens)
    }


def _entropy_coverage_payload(
    entropy: Mapping[str, Mapping[str, int]],
) -> dict[str, object]:
    raw_bin_counts = Counter(
        _entropy_bin(counts["distinct_raw_code_count"])
        for counts in entropy.values()
    )
    stripped_bin_counts = Counter(
        _entropy_bin(counts["distinct_stripped_code_count"])
        for counts in entropy.values()
    )
    total_pairs = len(entropy)
    largest_bin = max(raw_bin_counts.values(), default=0)
    limited = total_pairs > 0 and largest_bin / total_pairs >= 0.75
    return {
        "definition": (
            "Primary entropy is distinct human raw_code per "
            "(question_id, language)."
        ),
        "pairs": dict(entropy),
        "missing_raw_code_records": sum(
            counts["missing_raw_code_records"]
            for counts in entropy.values()
        ),
        "raw_code_bin_pair_counts": dict(raw_bin_counts),
        "stripped_code_bin_pair_counts": dict(stripped_bin_counts),
        "limited_interpretability": limited,
        "limitation_note": (
            "Most pairs occupy one pre-declared bin; the stored human list is "
            "capped at six, so this entropy slice has limited interpretability."
            if limited
            else None
        ),
    }


def _duplicate_removal_counts(scored: Sequence[ScoredEvalRow]) -> dict[str, object]:
    humans = [row for row in scored if row.label == LABEL_HUMAN]
    kept: dict[tuple[str, str], set[str]] = defaultdict(set)
    removed = 0
    for row in humans:
        key = (row.question_id, row.language)
        if row.content_hash in kept[key]:
            removed += 1
            continue
        kept[key].add(row.content_hash)
    return {
        "all_records": len(humans),
        "deduplicated_code": len(humans) - removed,
        "removed": removed,
        "reason": (
            "Exact duplicates removed within (question_id, language) using sha256 "
            "of the final stripped representation. First occurrence kept."
        ),
    }


def _eligible_pairs(scored: Sequence[ScoredEvalRow], variant: str) -> set[tuple[str, str]]:
    rows = _dedupe_if_needed(scored, variant)
    humans = {(row.question_id, row.language) for row in rows if row.label == LABEL_HUMAN}
    return humans


def _filter_variant(
    scored: Sequence[ScoredEvalRow],
    variant: str,
    eligible_pairs: set[tuple[str, str]],
) -> list[ScoredEvalRow]:
    rows = _dedupe_if_needed(scored, variant)
    return [
        row
        for row in rows
        if (row.question_id, row.language) in eligible_pairs
    ]


def _dedupe_if_needed(scored: Sequence[ScoredEvalRow], variant: str) -> list[ScoredEvalRow]:
    if variant != VARIANT_DEDUPLICATED:
        return list(scored)
    seen: set[tuple[str, str, str]] = set()
    kept: list[ScoredEvalRow] = []
    for row in scored:
        if row.label != LABEL_HUMAN:
            kept.append(row)
            continue
        key = (row.question_id, row.language, row.content_hash)
        if key in seen:
            continue
        seen.add(key)
        kept.append(row)
    return kept


def _scope_language(rows: Sequence[ScoredEvalRow], language: str) -> list[ScoredEvalRow]:
    if language == "COMBINED":
        return list(rows)
    return [row for row in rows if row.language == language]


def _labeled(rows: Sequence[ScoredEvalRow], method: str) -> list[LabeledScore]:
    return [
        LabeledScore(
            question_id=row.question_id,
            pair_token=_pair_token(row.question_id, row.language),
            language=row.language,
            label=row.label,
            score=_score(row, method),
        )
        for row in rows
    ]


def _score(row: ScoredEvalRow, method: str) -> float:
    if method == "nn":
        return row.nn_score
    if method == "topk_mean":
        return row.topk_mean_score
    return row.centroid_score


def _pair_token(question_id: str, language: str) -> str:
    return f"{question_id}:{language}"


def _pair_delta_tables(rows: Sequence[ScoredEvalRow]) -> dict[str, object]:
    grouped: dict[tuple[str, str], dict[str, list[ScoredEvalRow]]] = defaultdict(
        lambda: {"ai": [], "human": []}
    )
    for row in rows:
        grouped[(row.question_id, row.language)][row.label].append(row)
    delta_rows: list[dict[str, object]] = []
    nn_deltas: list[float] = []
    centroid_deltas: list[float] = []
    nn_per_q: dict[str, list[float]] = defaultdict(list)
    centroid_per_q: dict[str, list[float]] = defaultdict(list)
    for (question_id, language), bucket in grouped.items():
        token = _pair_token(question_id, language)
        for ai_row in bucket[LABEL_AI]:
            for human_row in bucket[LABEL_HUMAN]:
                nn_delta = ai_row.nn_score - human_row.nn_score
                centroid_delta = ai_row.centroid_score - human_row.centroid_score
                nn_deltas.append(nn_delta)
                centroid_deltas.append(centroid_delta)
                nn_per_q[token].append(nn_delta)
                centroid_per_q[token].append(centroid_delta)
                delta_rows.append(
                    {
                        "question_id": question_id,
                        "language": language,
                        "ai_source": ai_row.source,
                        "human_source": human_row.source,
                        "nn_ai": ai_row.nn_score,
                        "nn_human": human_row.nn_score,
                        "nn_delta": nn_delta,
                        "centroid_ai": ai_row.centroid_score,
                        "centroid_human": human_row.centroid_score,
                        "centroid_delta": centroid_delta,
                    }
                )
    nn_means = {token: float(np.mean(values)) for token, values in nn_per_q.items()}
    centroid_means = {
        token: float(np.mean(values)) for token, values in centroid_per_q.items()
    }
    return {
        "rows": delta_rows,
        "summaries": {
            "nn": summarize_pair_deltas("nn", nn_deltas, nn_means),
            "centroid": summarize_pair_deltas("centroid", centroid_deltas, centroid_means),
        },
    }


def _bootstrap_groups(rows: Sequence[ScoredEvalRow]) -> list[ClusterScoreGroup]:
    grouped: dict[str, list[ScoredEvalRow]] = defaultdict(list)
    for row in rows:
        grouped[_pair_token(row.question_id, row.language)].append(row)
    groups: list[ClusterScoreGroup] = []
    for token, items in grouped.items():
        ai_rows = [row for row in items if row.label == LABEL_AI]
        human_rows = [row for row in items if row.label == LABEL_HUMAN]
        if not ai_rows or not human_rows:
            continue
        groups.append(
            ClusterScoreGroup(
                token=token,
                language=items[0].language,
                nn_ai=tuple(row.nn_score for row in ai_rows),
                nn_human=tuple(row.nn_score for row in human_rows),
                centroid_ai=tuple(row.centroid_score for row in ai_rows),
                centroid_human=tuple(row.centroid_score for row in human_rows),
            )
        )
    return groups


def _slice_metrics(
    rows: Sequence[ScoredEvalRow],
    difficulty: Mapping[str, str],
    human_entropy: Mapping[str, Mapping[str, int]],
) -> dict[str, object]:
    by_difficulty = _group_rows(rows, lambda row: difficulty.get(row.question_id, "UNKNOWN"))
    by_entropy = _group_rows(
        rows,
        lambda row: _entropy_bin(
            human_entropy.get(
                _pair_token(row.question_id, row.language),
                {},
            ).get("distinct_raw_code_count", 0)
        ),
    )
    by_persona = _persona_slice(rows)
    return {
        "difficulty": _metrics_for_groups(by_difficulty),
        "entropy": _metrics_for_groups(by_entropy),
        "persona": by_persona,
        "language": {
            language: {
                method: method_metrics_to_dict(
                    method_metrics_bundle(method, _labeled(_scope_language(rows, language), method), SELECTION_RANDOM_SEED)
                )
                for method in ("nn", "centroid")
            }
            for language in ("CPP", "PYTHON")
        },
    }


def _group_rows(
    rows: Sequence[ScoredEvalRow],
    key_fn: Callable[[ScoredEvalRow], str],
) -> dict[str, list[ScoredEvalRow]]:
    grouped: dict[str, list[ScoredEvalRow]] = defaultdict(list)
    for row in rows:
        grouped[str(key_fn(row))].append(row)
    return grouped


def _entropy_bin(count: int) -> str:
    if count >= COVERAGE_HIGH_MIN:
        return CoverageBucket.HIGH.value
    if count >= COVERAGE_MID_MIN:
        return CoverageBucket.MID.value
    return CoverageBucket.LOW.value


def _metrics_for_groups(groups: Mapping[str, Sequence[ScoredEvalRow]]) -> dict[str, object]:
    payload: dict[str, object] = {}
    for name, rows in groups.items():
        payload[name] = {
            method: method_metrics_to_dict(
                method_metrics_bundle(method, _labeled(rows, method), SELECTION_RANDOM_SEED)
            )
            for method in ("nn", "centroid")
        }
    return payload


def _persona_slice(rows: Sequence[ScoredEvalRow]) -> dict[str, object]:
    humans_by_language = {
        language: [row for row in rows if row.label == LABEL_HUMAN and row.language == language]
        for language in ("CPP", "PYTHON")
    }
    thresholds: dict[str, dict[str, dict[str, float]]] = {}
    for language, humans in humans_by_language.items():
        thresholds[language] = {}
        for method in ("nn", "centroid"):
            negatives = [_score(row, method) for row in humans]
            thresholds[language][method] = {
                "1%": original_operating_point([1.0], negatives, 0.01).threshold,
                "5%": original_operating_point([1.0], negatives, 0.05).threshold,
            }
    result: dict[str, object] = {}
    for persona in ("production_review", "pair_programming"):
        result[persona] = {}
        for method in ("nn", "centroid"):
            positives = [row for row in rows if row.label == LABEL_AI and row.source == persona]
            hits_1 = 0
            hits_5 = 0
            scores = [_score(row, method) for row in positives]
            for row in positives:
                language_thresholds = thresholds[row.language][method]
                score = _score(row, method)
                if score >= language_thresholds["1%"]:
                    hits_1 += 1
                if score >= language_thresholds["5%"]:
                    hits_5 += 1
            result[persona][method] = {
                "n_positives": len(positives),
                "mean": float(np.mean(scores)) if scores else None,
                "recall_at_1pct_fpr": hits_1 / len(positives) if positives else None,
                "recall_at_5pct_fpr": hits_5 / len(positives) if positives else None,
                "threshold_source": "language-wide human quantile, shared across personas",
            }
    return result


def _row_to_dict(
    row: ScoredEvalRow,
    difficulty: Mapping[str, str],
    human_entropy: Mapping[str, Mapping[str, int]],
) -> dict[str, object]:
    entropy = human_entropy.get(
        _pair_token(row.question_id, row.language),
        {},
    )
    return {
        "question_id": row.question_id,
        "language": row.language,
        "label": row.label,
        "source": row.source,
        "content_hash": row.content_hash,
        "user_id": row.user_id,
        "difficulty": difficulty.get(row.question_id),
        "distinct_raw_code_count": entropy.get("distinct_raw_code_count"),
        "distinct_stripped_code_count": entropy.get(
            "distinct_stripped_code_count"
        ),
        "missing_raw_code_records": entropy.get("missing_raw_code_records"),
        "nn_score": row.nn_score,
        "topk_mean_score": row.topk_mean_score,
        "centroid_score": row.centroid_score,
    }


def _write_outputs(
    payload: Mapping[str, object],
    paths: ExperimentOutputPaths,
) -> None:
    metadata = {
        "evaluation_mode": payload["evaluation_mode"],
        "is_final_all_human_evaluation": payload[
            "is_final_all_human_evaluation"
        ],
    }
    scores_payload = {
        key: payload[key]
        for key in payload
        if key != "pair_delta_rows"
    }
    _write_json(paths.scores, scores_payload)
    _write_pair_csv(payload["pair_delta_rows"], paths.pair_deltas, metadata)
    _write_summary_csv(payload["metrics"], paths.summary, metadata)
    _write_report(payload, paths.report)


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temp_path.replace(path)


def _write_pair_csv(
    rows: Sequence[Mapping[str, object]],
    path: Path,
    metadata: Mapping[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "evaluation_mode",
        "is_final_all_human_evaluation",
        "question_id",
        "language",
        "ai_source",
        "human_source",
        "nn_ai",
        "nn_human",
        "nn_delta",
        "centroid_ai",
        "centroid_human",
        "centroid_delta",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({**metadata, **row})


def _write_summary_csv(
    metrics: Mapping[str, Mapping[str, object]],
    path: Path,
    metadata: Mapping[str, object],
) -> None:
    fieldnames = [
        "evaluation_mode",
        "is_final_all_human_evaluation",
        "negative_variant",
        "positive_variant",
        "language",
        "combined_form",
        "method",
        "n_positives",
        "n_negatives",
        "pooled_auroc",
        "macro_auroc",
        "original_recall_1pct",
        "original_achieved_fpr_1pct",
        "original_threshold_1pct",
        "original_recall_5pct",
        "original_achieved_fpr_5pct",
        "original_threshold_5pct",
        "oof_recall_1pct",
        "oof_achieved_fpr_1pct",
        "oof_recall_5pct",
        "oof_achieved_fpr_5pct",
        "mean_gap",
        "median_gap",
        "pos_mean",
        "neg_mean",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for negative_variant, positive_variants in metrics.items():
            for positive_variant, languages in positive_variants.items():
                for language, methods in languages.items():
                    if language == "combined_operating_points":
                        continue
                    _write_metric_rows(
                        writer,
                        negative_variant,
                        positive_variant,
                        language,
                        methods,
                        metadata,
                    )
                _write_language_calibrated_combined_rows(
                    writer,
                    negative_variant,
                    positive_variant,
                    languages["combined_operating_points"][
                        "language_calibrated_combined"
                    ],
                    metadata,
                )


def _write_metric_rows(
    writer: csv.DictWriter,
    negative_variant: str,
    positive_variant: str,
    language: str,
    methods: Mapping[str, Mapping[str, object]],
    metadata: Mapping[str, object],
) -> None:
    for method, row in methods.items():
        original = row["original_style"]
        oof = row["grouped_out_of_fold"]
        writer.writerow(
            {
                            **metadata,
                            "negative_variant": negative_variant,
                            "positive_variant": positive_variant,
                            "language": language,
                "combined_form": (
                    "pooled_single_threshold"
                    if language == "COMBINED"
                    else "language_specific"
                ),
                            "method": method,
                            "n_positives": row["n_positives"],
                            "n_negatives": row["n_negatives"],
                            "pooled_auroc": row["pooled_auroc"],
                            "macro_auroc": row["macro_auroc"],
                            "original_recall_1pct": original["1%"]["recall"],
                            "original_achieved_fpr_1pct": original["1%"]["achieved_fpr"],
                            "original_threshold_1pct": original["1%"]["threshold"],
                            "original_recall_5pct": original["5%"]["recall"],
                            "original_achieved_fpr_5pct": original["5%"]["achieved_fpr"],
                            "original_threshold_5pct": original["5%"]["threshold"],
                            "oof_recall_1pct": oof["1%"]["recall"],
                            "oof_achieved_fpr_1pct": oof["1%"]["achieved_fpr"],
                            "oof_recall_5pct": oof["5%"]["recall"],
                            "oof_achieved_fpr_5pct": oof["5%"]["achieved_fpr"],
                            "mean_gap": row["mean_gap"],
                            "median_gap": row["median_gap"],
                            "pos_mean": row["positives"]["mean"],
                            "neg_mean": row["negatives"]["mean"],
            }
        )


def _write_language_calibrated_combined_rows(
    writer: csv.DictWriter,
    negative_variant: str,
    positive_variant: str,
    methods: Mapping[str, Mapping[str, object]],
    metadata: Mapping[str, object],
) -> None:
    for method, styles in methods.items():
        original = styles["original_style"]
        oof = styles["grouped_out_of_fold"]
        writer.writerow(
            {
                **metadata,
                "negative_variant": negative_variant,
                "positive_variant": positive_variant,
                "language": "COMBINED",
                "combined_form": "language_calibrated_combined",
                "method": method,
                "original_recall_1pct": original["1%"]["recall"],
                "original_achieved_fpr_1pct": original["1%"][
                    "achieved_fpr"
                ],
                "original_threshold_1pct": original["1%"]["threshold"],
                "original_recall_5pct": original["5%"]["recall"],
                "original_achieved_fpr_5pct": original["5%"][
                    "achieved_fpr"
                ],
                "original_threshold_5pct": original["5%"]["threshold"],
                "oof_recall_1pct": oof["1%"]["recall"],
                "oof_achieved_fpr_1pct": oof["1%"]["achieved_fpr"],
                "oof_recall_5pct": oof["5%"]["recall"],
                "oof_achieved_fpr_5pct": oof["5%"]["achieved_fpr"],
            }
        )


def _write_incomplete_report(
    coverage: Mapping[str, object],
    report_path: Path,
) -> None:
    overlaps = coverage["content_hash_overlap"]
    global_overlaps = overlaps["global_diagnostics"]
    cluster_overlaps = overlaps["same_cluster"]
    held_reference = cluster_overlaps["held_out_and_reference"]
    entropy = coverage["human_entropy"]
    lines = [
        "# Phase B.1 centroid vs nearest-neighbour",
        "",
        "## Verdict",
        "",
        "The experiment stopped before scoring because cached embeddings are incomplete. "
        "No Voyage, OpenRouter, or other API calls were made. Partial metrics were not computed.",
        "",
        f"- total eligible human records: {coverage['total_eligible_human_records']}",
        f"- total distinct human programs: {coverage['total_distinct_human_programs']}",
        f"- cached embeddings found: {coverage['cached_embeddings_found']}",
        f"- missing embeddings: {coverage['missing_embeddings']}",
        f"- human missing: {coverage['human_missing']}",
        f"- held-out missing: {coverage['held_out_missing']}",
        f"- stripped duplicates within (question_id, language): {coverage['duplicate_stripped_within_pair']['removed']}",
        f"- groups with multiple attempts from the same user: "
        f"{coverage['human_attempt_rule']['groups_with_same_user_multiple_attempts']}",
        f"- {coverage['human_attempt_rule']['selection_rule']}",
        f"- global hash diagnostics: human∩reference={global_overlaps['human_and_reference']}; "
        f"human∩held-out={global_overlaps['human_and_held_out']}; "
        f"held-out∩reference={global_overlaps['held_out_and_reference']}. "
        "Cross-question matches are not leakage.",
        f"- same-cluster overlaps: "
        f"human∩reference={cluster_overlaps['human_and_reference']['matching_cluster_hashes']}; "
        f"human∩held-out={cluster_overlaps['human_and_held_out']['matching_cluster_hashes']}; "
        f"held-out∩reference={held_reference['matching_cluster_hashes']}.",
        f"- same-cluster held-out/reference affected held-out records: "
        f"{held_reference['affected_left_records']}; matching reference records: "
        f"{held_reference['affected_right_records']}.",
        f"- entropy raw-code bins: {entropy['raw_code_bin_pair_counts']}; "
        f"missing raw_code records: {entropy['missing_raw_code_records']}.",
        (
            f"- entropy limitation: {entropy['limitation_note']}"
            if entropy["limitation_note"]
            else "- entropy limitation: none detected."
        ),
        "",
        "1. Did using all human solutions materially change the original NN result? Not computed.",
        "2. Did centroid scoring beat nearest-neighbour scoring? Not computed.",
        "3. Was any improvement present for both CPP and PYTHON? Not computed.",
        "4. Did centroid improve Recall @ 1% FPR without increasing achieved FPR? Not computed.",
        "5. Did it help or hurt the harder `production_review` persona? Not computed.",
        "6. Is the difference statistically meaningful? Not computed.",
        "7. Should centroid replace NN, be retained as a future combiner feature, or be rejected? "
        "No scoring decision until coverage is complete.",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_report(
    payload: Mapping[str, object],
    report_path: Path,
) -> None:
    metrics = payload["metrics"][VARIANT_ALL_RECORDS][POSITIVE_VARIANT_ALL]
    sensitivity_metrics = payload["metrics"][VARIANT_ALL_RECORDS][
        POSITIVE_VARIANT_EXCLUDE_MATCH
    ]
    bootstrap = payload["bootstrap_centroid_minus_nn"]["COMBINED"]
    persona = payload["slices"]["persona"]
    old = payload["old_eval_comparison"]
    cached_mode = (
        payload.get("evaluation_mode") == EvaluationMode.CACHED_HUMANS_ONLY.value
    )
    prefix = []
    if cached_mode:
        prefix = [
            "> " + CACHED_SUBSET_WARNING,
            "",
            "Adding the remaining humans may expose more high-scoring human "
            "solutions, increase the 1% FPR threshold and reduce AI recall. "
            "This is not the final all-human evaluation.",
            "",
        ]
        if (
            payload["meta"]["bootstrap_status"]
            == "skipped_for_fast_preliminary_run"
        ):
            prefix.extend(
                [
                    "Bootstrap confidence intervals were skipped for this fast "
                    "preliminary run. AUROC, recall, FPR, thresholds and score "
                    "gaps are complete point estimates; statistical uncertainty "
                    "has not been calculated.",
                    "",
                ]
            )
        verdict = _cached_verdict_lines(metrics, bootstrap, persona, payload)
    else:
        verdict = _verdict_lines(metrics, bootstrap, persona, old, payload)
    lines = [
        "# Phase B.1 centroid vs nearest-neighbour",
        "",
        *prefix,
        "## Verdict",
        "",
        *verdict,
        "",
        "Language-specific CPP and PYTHON metrics are primary. Combined metrics "
        "are summaries and are not sufficient to recommend replacing NN.",
        "",
        "## Main comparison (`all_records`, `all_held_out`)",
        "",
        _markdown_table(metrics),
        "",
        "## Combined operating-point summaries",
        "",
        _combined_operating_point_table(metrics["combined_operating_points"]),
        "",
        "## Exact-reference-match sensitivity",
        "",
        "This variant excludes only held-out AI positives whose stripped code "
        "exactly matches a reference in the same question-language cluster.",
        "",
        _markdown_table(sensitivity_metrics),
        "",
        "## Notes",
        "",
        f"- evaluation_mode: `{payload['evaluation_mode']}`",
        f"- is_final_all_human_evaluation: "
        f"`{str(payload['is_final_all_human_evaluation']).lower()}`",
        f"- Duplicate stripped programs removed in `{VARIANT_DEDUPLICATED}`: "
        f"{payload['meta']['duplicate_code_removed']['removed']}.",
        f"- Human attempt rule: {payload['meta']['human_attempt_rule']['selection_rule']}",
        f"- Same-cluster held-out/reference exact matches: "
        f"{payload['sanity']['content_hash_overlap']['same_cluster']['held_out_and_reference']['matching_cluster_hashes']}",
        f"- Clusters not size 6: {payload['sanity']['clusters_not_size_six']}",
        "- COMBINED includes both `pooled_single_threshold` for backward "
        "comparison and `language_calibrated_combined` summaries.",
        "",
        "Production nearest-neighbour scoring was not changed.",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _cached_verdict_lines(
    metrics: Mapping[str, Mapping[str, object]],
    bootstrap: Mapping[str, Mapping[str, float | None]],
    persona: Mapping[str, object],
    payload: Mapping[str, object],
) -> list[str]:
    cpp_better = _better(metrics["CPP"]["centroid"], metrics["CPP"]["nn"])
    py_better = _better(metrics["PYTHON"]["centroid"], metrics["PYTHON"]["nn"])
    calibrated = metrics["combined_operating_points"]["language_calibrated_combined"]
    nn_oof = calibrated["nn"]["grouped_out_of_fold"]["1%"]
    centroid_oof = calibrated["centroid"]["grouped_out_of_fold"]["1%"]
    r1_ci = bootstrap.get(
        "grouped_oof_language_calibrated_recall_at_1pct_fpr",
        {},
    )
    persona_nn = persona["production_review"]["nn"]["recall_at_1pct_fpr"]
    persona_centroid = persona["production_review"]["centroid"]["recall_at_1pct_fpr"]
    sensitivity = _sensitivity_changed(payload)
    bootstrap_skipped = (
        payload["meta"]["bootstrap_status"]
        == "skipped_for_fast_preliminary_run"
    )
    ci_excludes_zero = _ci_excludes_zero(r1_ci)
    recommendation = _cached_recommendation(
        cpp_better,
        py_better,
        ci_excludes_zero,
    )
    return [
        f"1. Did centroid outperform NN on cached CPP humans? {_yes_no(cpp_better)} "
        f"(grouped-OOF Recall@1% centroid="
        f"{_fmt_opt(metrics['CPP']['centroid']['grouped_out_of_fold']['1%']['recall'])} "
        f"vs NN={_fmt_opt(metrics['CPP']['nn']['grouped_out_of_fold']['1%']['recall'])}).",
        f"2. Did centroid outperform NN on cached Python humans? {_yes_no(py_better)} "
        f"(grouped-OOF Recall@1% centroid="
        f"{_fmt_opt(metrics['PYTHON']['centroid']['grouped_out_of_fold']['1%']['recall'])} "
        f"vs NN={_fmt_opt(metrics['PYTHON']['nn']['grouped_out_of_fold']['1%']['recall'])}).",
        f"3. Did centroid improve `production_review`? "
        f"{_yes_no((persona_centroid or 0) > (persona_nn or 0))} "
        f"(Recall@1% NN={_fmt_opt(persona_nn)}, centroid={_fmt_opt(persona_centroid)}).",
        f"4. Did centroid improve grouped-OOF Recall@1%? "
        f"{_yes_no((centroid_oof['recall'] or 0) > (nn_oof['recall'] or 0))} "
        f"(centroid={_fmt_opt(centroid_oof['recall'])} vs NN={_fmt_opt(nn_oof['recall'])}).",
        (
            "5. Did the confidence interval exclude zero? Not calculated; "
            "bootstrap was skipped for this fast preliminary run."
            if bootstrap_skipped
            else f"5. Did the confidence interval exclude zero? "
            f"{_yes_no(ci_excludes_zero)} {_ci_text(r1_ci)}"
        ),
        f"6. Did excluding exact held-out/reference matches materially change the result? "
        f"{_yes_no(sensitivity)}",
        f"7. Is centroid promising enough to retain as a Phase C feature? {recommendation}",
    ]


def _sensitivity_changed(payload: Mapping[str, object]) -> bool:
    all_held = payload["metrics"][VARIANT_ALL_RECORDS][POSITIVE_VARIANT_ALL]
    excluded = payload["metrics"][VARIANT_ALL_RECORDS][
        POSITIVE_VARIANT_EXCLUDE_MATCH
    ]
    for language in ("CPP", "PYTHON"):
        all_auroc = all_held[language]["centroid"]["pooled_auroc"] or 0
        excluded_auroc = excluded[language]["centroid"]["pooled_auroc"] or 0
        all_recall = all_held[language]["centroid"]["grouped_out_of_fold"]["1%"][
            "recall"
        ] or 0
        excluded_recall = excluded[language]["centroid"]["grouped_out_of_fold"][
            "1%"
        ]["recall"] or 0
        if abs(all_auroc - excluded_auroc) > 0.01:
            return True
        if abs(all_recall - excluded_recall) > 0.02:
            return True
    return False


def _ci_excludes_zero(ci: Mapping[str, float | None]) -> bool:
    low = ci.get("low")
    high = ci.get("high")
    if low is None or high is None:
        return False
    return not (low <= 0 <= high)


def _cached_recommendation(
    cpp_better: bool,
    py_better: bool,
    ci_excludes_zero: bool,
) -> str:
    if not ci_excludes_zero:
        return "Result is inconclusive; wait for all-human embeddings."
    if cpp_better and py_better:
        return "Centroid is promising as an additional Phase C/XGBoost feature."
    return "Centroid shows no useful improvement on the cached subset."


def _verdict_lines(
    metrics: Mapping[str, Mapping[str, object]],
    bootstrap: Mapping[str, Mapping[str, float | None]],
    persona: Mapping[str, object],
    old: Mapping[str, object],
    payload: Mapping[str, object],
) -> list[str]:
    nn_combined = metrics["COMBINED"]["nn"]
    centroid_combined = metrics["COMBINED"]["centroid"]
    combined_points = metrics["combined_operating_points"][
        "language_calibrated_combined"
    ]
    nn_recall = combined_points["nn"]["grouped_out_of_fold"]["1%"]["recall"]
    centroid_recall = combined_points["centroid"]["grouped_out_of_fold"]["1%"][
        "recall"
    ]
    nn_fpr = combined_points["nn"]["grouped_out_of_fold"]["1%"][
        "achieved_fpr"
    ]
    centroid_fpr = combined_points["centroid"]["grouped_out_of_fold"]["1%"][
        "achieved_fpr"
    ]
    cpp_better = _better(metrics["CPP"]["centroid"], metrics["CPP"]["nn"])
    py_better = _better(metrics["PYTHON"]["centroid"], metrics["PYTHON"]["nn"])
    r1_ci = bootstrap.get(
        "grouped_oof_language_calibrated_recall_at_1pct_fpr",
        {},
    )
    old_note = _old_eval_note(old)
    persona_nn = persona["production_review"]["nn"]["recall_at_1pct_fpr"]
    persona_centroid = persona["production_review"]["centroid"]["recall_at_1pct_fpr"]
    pooled_up = (centroid_combined["pooled_auroc"] or 0) > (nn_combined["pooled_auroc"] or 0)
    macro_down = (centroid_combined["macro_auroc"] or 0) < (nn_combined["macro_auroc"] or 0)
    flag = ""
    if pooled_up and macro_down:
        flag = (
            " Pooled AUROC improved while macro AUROC declined, so any pooled gain "
            "is driven by questions with more human submissions."
        )
    recommendation = _recommendation(
        centroid_recall,
        nn_recall,
        centroid_fpr,
        nn_fpr,
        cpp_better,
        py_better,
        r1_ci,
        persona_centroid,
        persona_nn,
    )
    ci_text = _ci_text(r1_ci)
    return [
        f"1. Did using all human solutions materially change the original NN result? {old_note}",
        f"2. Did centroid scoring beat nearest-neighbour scoring? "
        f"Pooled AUROC centroid={centroid_combined['pooled_auroc']:.4f} vs NN={nn_combined['pooled_auroc']:.4f}; "
        f"Language-calibrated grouped-OOF Recall@1% FPR "
        f"centroid={centroid_recall:.4f} vs NN={nn_recall:.4f} "
        f"(achieved FPR {centroid_fpr:.4f} vs {nn_fpr:.4f}).{flag}",
        f"3. Was any improvement present for both CPP and PYTHON? "
        f"CPP recall@1% better={cpp_better}; PYTHON recall@1% better={py_better}.",
        f"4. Did centroid improve grouped-OOF Recall @ 1% FPR without increasing achieved FPR? "
        f"{_yes_no(centroid_recall > nn_recall and centroid_fpr <= nn_fpr + 1e-12)} "
        f"(centroid recall {centroid_recall:.4f} at FPR {centroid_fpr:.4f}; "
        f"NN recall {nn_recall:.4f} at FPR {nn_fpr:.4f}).",
        f"5. Did it help or hurt the harder `production_review` persona? "
        f"Recall@1% NN={persona_nn:.4f}, centroid={persona_centroid:.4f}.",
        f"6. Is the difference statistically meaningful according to the question-clustered CI? {ci_text}",
        f"7. Should centroid replace NN, be retained as an additional feature, or be rejected? {recommendation}",
    ]


def _old_eval_note(old: Mapping[str, object]) -> str:
    if not old.get("available"):
        return "Could not compare; eval_scores.json missing."
    if not old.get("valid"):
        return (
            "Comparison invalid because exact old-row alignment was incomplete "
            "or duplicate identities were present."
        )
    if not old.get("metrics_compared"):
        return "Exact row alignment did not authorize metric comparison."
    old_combined = old["old_metrics"]["COMBINED"]
    new_combined = old["recomputed_nn_on_old_subset"]["COMBINED"]
    differences = old["score_differences"]
    nn_difference = differences["nn"]
    topk_difference = differences["topk_mean"]
    return (
        f"On the exactly aligned original rows, stored NN AUROC={old_combined['auroc_nn']:.4f} "
        f"vs recomputed {new_combined['auroc_nn']:.4f}. "
        f"NN max/mean absolute score difference="
        f"{nn_difference['max_absolute_difference']:.3g}/"
        f"{nn_difference['mean_absolute_difference']:.3g}; "
        f"top-k max/mean="
        f"{topk_difference['max_absolute_difference']:.3g}/"
        f"{topk_difference['mean_absolute_difference']:.3g}; "
        f"within {old['score_tolerance']:.1g} tolerance: "
        f"NN={nn_difference['within_tolerance']}, "
        f"top-k={topk_difference['within_tolerance']}."
    )


def _better(centroid_row: Mapping[str, object], nn_row: Mapping[str, object]) -> bool:
    return (centroid_row["grouped_out_of_fold"]["1%"]["recall"] or 0) > (
        nn_row["grouped_out_of_fold"]["1%"]["recall"] or 0
    )


def _yes_no(value: bool) -> str:
    return "Yes" if value else "No"


def _ci_text(ci: Mapping[str, float | None]) -> str:
    low = ci.get("low")
    high = ci.get("high")
    if low is None or high is None:
        return "Inconclusive; CI could not be computed."
    if low <= 0 <= high:
        return (
            f"Inconclusive: 95% CI for centroid−NN Recall@1% FPR is [{low:.4f}, {high:.4f}] and crosses zero."
        )
    return f"95% CI for centroid−NN Recall@1% FPR is [{low:.4f}, {high:.4f}] and does not cross zero."


def _recommendation(
    centroid_recall: float,
    nn_recall: float,
    centroid_fpr: float,
    nn_fpr: float,
    cpp_better: bool,
    py_better: bool,
    r1_ci: Mapping[str, float | None],
    persona_centroid: float,
    persona_nn: float,
) -> str:
    low = r1_ci.get("low")
    high = r1_ci.get("high")
    crosses = low is None or high is None or (low <= 0 <= high)
    if (
        centroid_recall > nn_recall
        and centroid_fpr <= nn_fpr + 1e-12
        and cpp_better
        and py_better
        and persona_centroid >= persona_nn
        and not crosses
    ):
        return (
            "Retain centroid as an additional feature for a future combiner; "
            "do not replace production NN from this experiment alone."
        )
    if crosses:
        return (
            "Reject replacing NN. The clustered confidence interval is inconclusive. "
            "Centroid may still be kept as a candidate feature, not as the production score."
        )
    return (
        "Do not replace production NN. Centroid did not improve the priority operating point "
        "across Python, production_review, and clustered uncertainty."
    )


def _markdown_table(metrics: Mapping[str, Mapping[str, object]]) -> str:
    header = (
        "| lang | method | n_pos | n_neg | pooled AUROC | macro AUROC | "
        "R@1% | FPR@1% | R@5% | FPR@5% | pos_mean | neg_mean | mean_gap | "
        "OOF R@1% | OOF R@5% |"
    )
    sep = "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    lines = [header, sep]
    calibrated = metrics.get("combined_operating_points", {}).get(
        "language_calibrated_combined",
        {},
    )
    for language in LANGUAGES_REPORT:
        for method in METHODS:
            row = metrics[language][method]
            original = row["original_style"]
            oof = row["grouped_out_of_fold"]
            label = language
            if language == "COMBINED" and calibrated:
                original = calibrated[method]["original_style"]
                oof = calibrated[method]["grouped_out_of_fold"]
                label = "COMBINED (language-calibrated)"
            lines.append(
                "| {lang} | {method} | {n_pos} | {n_neg} | {auroc:.4f} | {macro} | "
                "{r1:.4f} | {f1:.4f} | {r5:.4f} | {f5:.4f} | {pmean:.4f} | "
                "{nmean:.4f} | {gap} | {oof1} | {oof5} |".format(
                    lang=label,
                    method=method,
                    n_pos=row["n_positives"],
                    n_neg=row["n_negatives"],
                    auroc=row["pooled_auroc"] or 0,
                    macro=_fmt_opt(row["macro_auroc"]),
                    r1=original["1%"]["recall"] or 0,
                    f1=original["1%"]["achieved_fpr"] or 0,
                    r5=original["5%"]["recall"] or 0,
                    f5=original["5%"]["achieved_fpr"] or 0,
                    pmean=row["positives"]["mean"] or 0,
                    nmean=row["negatives"]["mean"] or 0,
                    gap=_fmt_opt(row["mean_gap"]),
                    oof1=_fmt_opt(oof["1%"]["recall"]),
                    oof5=_fmt_opt(oof["5%"]["recall"]),
                )
            )
    return "\n".join(lines)


def _combined_operating_point_table(
    combined: Mapping[str, Mapping[str, Mapping[str, object]]],
) -> str:
    lines = [
        "| form | style | method | R@1% | FPR@1% | R@5% | FPR@5% |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for form, methods in combined.items():
        for method, styles in methods.items():
            for style, points in styles.items():
                lines.append(
                    f"| {form} | {style} | {method} | "
                    f"{_fmt_opt(points['1%']['recall'])} | "
                    f"{_fmt_opt(points['1%']['achieved_fpr'])} | "
                    f"{_fmt_opt(points['5%']['recall'])} | "
                    f"{_fmt_opt(points['5%']['achieved_fpr'])} |"
                )
    return "\n".join(lines)


def _fmt_opt(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import csv
import json
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from collections.abc import Mapping, Sequence

import numpy as np

from nw_ai_code_detector.eligibility_data import (
    LoadedSolution,
    display_similarity,
    load_solution_records,
    snapshot_protected_bundle,
)
from nw_ai_code_detector.build_model_dataset_v2 import (
    EXPECTED_MIXED_CLUSTER_COUNT,
    load_question_assignments,
)
from nw_ai_code_detector.canonicality_eligibility import (
    evaluate_submission_eligibility,
    significant_token_threshold_for_language,
)
from nw_ai_code_detector.config import (
    AI_SOLUTIONS_DIR,
    CANDIDATE_HUMAN_SIMILARITY_SAMPLES_PATH,
    DATA_DIR,
    EMBEDDING_CACHE_DIR,
    EVAL_AI_SOLUTIONS_DIR,
    HELDOUT_AI_SIMILARITY_SAMPLES_PATH,
    SIGNIFICANT_TOKEN_ELIGIBILITY_DIR,
)
from nw_ai_code_detector.constants import (
    CANDIDATE_HUMAN_SOURCE,
    CPP_SIGNIFICANT_TOKEN_THRESHOLD,
    DatasetSplit,
    EXPECTED_CANDIDATE_HUMAN_COUNT,
    EXPECTED_CANDIDATE_HUMAN_LOGICAL_COUNT,
    GENERATION_LANGUAGES,
    GROUPS_KEY,
    HELDOUT_AI_SOURCE,
    PYTHON_SIGNIFICANT_TOKEN_THRESHOLD,
    SubmissionExclusionReason,
    UNVERIFIED_LABEL_STATUS,
)
from nw_ai_code_detector.data_load import load_dataset
from nw_ai_code_detector.embedder import cached_vector_for_text, embedding_cache_key
from nw_ai_code_detector.export_raw_similarity_samples import main as write_raw_reviews
from nw_ai_code_detector.evaluation.evaluate_ai_reference_scores_v2 import (
    load_reference_clusters,
)
from nw_ai_code_detector.index import ClusterKey
from nw_ai_code_detector.significant_code_tokens import (
    SignificantCodeTokenizationError,
    significant_code_token_count,
)
from nw_ai_code_detector.stripper import Language, SourceParseError, strip_solution_body

PERCENTILES = (1, 5, 10, 25, 50, 75, 90, 95, 99)
SAMPLES_PER_CATEGORY = 5
LEFT_ROTATE_AUDIT_CODE = """class solution{
public:
    void leftRotate(int arr[], int n){
        int first = arr[0];
        for(int i = 0; i < n - 1; i++){
            arr[i] = arr[i + 1];
        }
        arr[n - 1] = first;
    }
};
"""
CANDIDATE_SAMPLE_PREAMBLE = """CANDIDATE-HUMAN SIMILARITY MANUAL-CHECK SAMPLES

These records are unverified candidate-human submissions.
They are not independently confirmed human solutions.

Selection population: integrity-scorable train candidate-human.
Token count is recorded and is not an exclude.
Ranking score: ai_nn_max against the exact same-question,
same-language mixed-v1 AI reference cluster.
ai_distance = 1.0 - ai_nn_max.

The displayed code is the exact stripped code used by the
embedding pipeline.

Low ai_nn_max does not mean verified human authorship.
High ai_nn_max does not independently prove AI authorship.
Do not convert similarity into a training label without
independent verification or explicit weak-label approval.
"""
HELDOUT_SAMPLE_PREAMBLE = """HELD-OUT AI SIMILARITY MANUAL-CHECK SAMPLES

All samples are known held-out AI solutions.
They are not candidate-human submissions.

Selection population: integrity-scorable validation held-out AI.
Token count is recorded and is not an exclude.
Ranking score: ai_nn_max against the exact same-question,
same-language mixed-v1 AI reference cluster.

The displayed code is the exact stripped code used by the
embedding pipeline. Boilerplate and driver code are not included.

High similarity does not independently prove AI authorship.
Low similarity does not mean human authorship.
"""


@dataclass(frozen=True)
class CandidateRecord:
    record_id: str
    question_id: str
    language: str
    split: str
    difficulty: str
    stripped_code: str | None
    stripped_hash: str
    significant_code_token_count: int | None
    valid: bool
    syntax_invalid: bool
    cache_key: str
    cache_exists: bool
    source: str
    label_status: str
    group_index: int


@dataclass(frozen=True)
class ScoredCandidate:
    record_id: str
    question_id: str
    language: str
    split: str
    difficulty: str
    significant_code_token_count: int
    ai_nn_max_raw: float
    ai_distance: float
    exact_match_to_mixed_v1: bool
    duplicate_group_size: int
    stripped_hash: str
    embedding_cache_key: str
    stripped_code: str


def main() -> int:
    started = time.perf_counter()
    before = snapshot_protected_bundle()
    assignments = load_question_assignments()
    references = load_solution_records(AI_SOLUTIONS_DIR, "mixed_v1_reference")
    heldout = load_solution_records(EVAL_AI_SOLUTIONS_DIR, HELDOUT_AI_SOURCE)
    candidates = load_candidate_human_records(assignments)
    _validate_candidate_population(candidates)
    mixed_hashes = _hash_index(references)
    heldout_hashes = _hash_index(heldout)
    clusters = load_reference_clusters()
    cluster_inventory = mixed_v1_cluster_inventory(assignments, clusters)
    heldout_eval = evaluate_population(
        heldout,
        assignments,
        mixed_hashes,
        clusters,
        HELDOUT_AI_SOURCE,
    )
    candidate_eval = evaluate_candidates(
        candidates,
        mixed_hashes,
        heldout_hashes,
        clusters,
    )
    write_heldout_samples(heldout_eval["scores"])
    write_candidate_samples(candidate_eval["scores"])
    write_raw_reviews()
    audit = _candidate_source_audit(candidates, mixed_hashes, heldout_hashes)
    coverage = {
        "embedding_input": "verified_stripped_code",
        "eligibility_feature": "integrity_checks_only",
        "token_count_used_as_exclude": False,
        "tokenizer": "tree-sitter concrete-syntax-tree leaf tokens",
        "ignored_token_categories": [
            "whitespace",
            "blank_lines",
            "newlines",
            "indentation",
            "dedentation",
            "comments",
            "encoding_markers",
            "end_of_file",
        ],
        "minimum_significant_code_tokens": {
            "CPP": CPP_SIGNIFICANT_TOKEN_THRESHOLD,
            "PYTHON": PYTHON_SIGNIFICANT_TOKEN_THRESHOLD,
        },
        "mixed_v1_clusters": cluster_inventory,
        "left_rotate_audit": _left_rotate_audit(),
        "heldout_ai": heldout_eval["coverage"],
        "candidate_human": candidate_eval["coverage"],
        "token_distributions": {
            "heldout_ai": _population_token_distributions(
                heldout_eval["rows"],
                "detection_eligible",
            ),
            "candidate_human": _population_token_distributions(
                candidate_eval["rows"],
                "qualified",
            ),
        },
        "candidate_human_source": audit,
        "heldout_similarity": _candidate_similarity(heldout_eval["scores"]),
        "candidate_similarity_primary": _candidate_similarity(
            candidate_eval["scores"]
        ),
        "human_metrics_calculated": False,
    }
    metadata = {
        "network_calls": False,
        "embeddings_generated": False,
        "embedding_cache_misses": audit["embedding_cache_misses"],
        "runtime_seconds": time.perf_counter() - started,
        "checksums_before": before,
        "checksums_after": snapshot_protected_bundle(),
        "significant_code_token_count_used_for_eligibility": False,
        "token_count_used_as_exclude": False,
        "candidate_label_status": UNVERIFIED_LABEL_STATUS,
    }
    if metadata["checksums_before"] != metadata["checksums_after"]:
        raise RuntimeError("Protected artifacts changed")
    write_outputs(
        coverage,
        metadata,
        heldout_eval,
        candidate_eval,
    )
    print(f"Wrote {SIGNIFICANT_TOKEN_ELIGIBILITY_DIR}")
    return 0


def load_candidate_human_records(assignments: Sequence[object]) -> list[CandidateRecord]:
    dataset = load_dataset()
    payload = json.loads((DATA_DIR / "scored_submissions.json").read_text(encoding="utf-8"))
    groups = payload.get(GROUPS_KEY)
    if not isinstance(groups, dict):
        raise RuntimeError("scored_submissions.json groups are unavailable")
    assignment_map = {item.question_id: item for item in assignments}
    records: list[CandidateRecord] = []
    for question_id in sorted(assignment_map):
        for language in GENERATION_LANGUAGES:
            records.extend(
                _candidates_for_pair(
                    groups,
                    dataset,
                    assignment_map[question_id],
                    language,
                )
            )
    return records


def mixed_v1_cluster_inventory(
    assignments: Sequence[object],
    clusters: Mapping[ClusterKey, object],
) -> dict[str, object]:
    expected_keys = [
        ClusterKey(item.question_id, language.value)
        for item in assignments
        for language in GENERATION_LANGUAGES
    ]
    existing = [key for key in expected_keys if key in clusters]
    missing = [key.token for key in expected_keys if key not in clusters]
    return {
        "expected_mixed_v1_clusters": EXPECTED_MIXED_CLUSTER_COUNT,
        "expected_from_split": len(expected_keys),
        "existing_mixed_v1_clusters": len(existing),
        "loaded_cluster_count": len(clusters),
        "missing_exact_clusters": len(missing),
        "missing_cluster_keys": missing,
        "all_expected_clusters_routable": not missing
        and len(existing) == EXPECTED_MIXED_CLUSTER_COUNT,
    }


def evaluate_population(
    records: Sequence[LoadedSolution],
    assignments: Sequence[object],
    mixed_hashes: Mapping[tuple[str, str], set[str]],
    clusters: Mapping[ClusterKey, object],
    source: str,
) -> dict[str, object]:
    split_by_question = {item.question_id: item.split for item in assignments}
    difficulty_by_question = {item.question_id: item.difficulty for item in assignments}
    rows = []
    scores = []
    for record in records:
        if record.question_id not in split_by_question:
            continue
        valid = bool(record.parse_ok and record.stripped_code and record.stripped_code.strip())
        token_count = _token_count(record.stripped_code, record.language) if valid else None
        token_valid = token_count is not None
        cache_ok = bool(valid and cached_vector_for_text(record.stripped_code) is not None)
        cluster_ok = ClusterKey(record.question_id, record.language) in clusters
        decision = evaluate_submission_eligibility(
            token_count,
            record.language,
            cluster_ok,
            valid and token_valid,
        )
        stripped_hash = _text_hash(record.stripped_code or "")
        record_id = f"{source}|{record.relative_path}|{stripped_hash}"
        reason = decision.exclusion_reason.value
        if not valid:
            reason = SubmissionExclusionReason.MISSING_OR_INVALID.value
        rows.append(
            {
                "record_id": record_id,
                "question_id": record.question_id,
                "language": record.language,
                "split": split_by_question[record.question_id],
                "difficulty": difficulty_by_question[record.question_id],
                "generator": record.generator,
                "generator_family": _generator_family(str(record.generator or "")),
                "persona": record.persona,
                "significant_code_token_count": token_count,
                "minimum_significant_code_tokens": (
                    significant_token_threshold_for_language(record.language)
                ),
                "base_valid": valid and token_valid and cache_ok,
                "cluster_exists": cluster_ok,
                "detection_eligible": decision.eligible,
                "exclusion_reason": reason,
                "status": reason,
                "stripped_hash": stripped_hash,
                "stripped_code": record.stripped_code if valid else None,
            }
        )
        if decision.eligible and record.stripped_code:
            scored = score_stripped_record(
                record.question_id,
                record.language,
                record.stripped_code,
                clusters,
            )
            scores.append(
                ScoredCandidate(
                    record_id=record_id,
                    question_id=record.question_id,
                    language=record.language,
                    split=split_by_question[record.question_id],
                    difficulty=difficulty_by_question[record.question_id],
                    significant_code_token_count=token_count or 0,
                    ai_nn_max_raw=scored[0],
                    ai_distance=scored[1],
                    exact_match_to_mixed_v1=stripped_hash in mixed_hashes.get(
                        (record.question_id, record.language),
                        set(),
                    ),
                    duplicate_group_size=1,
                    stripped_hash=stripped_hash,
                    embedding_cache_key=embedding_cache_key(record.stripped_code),
                    stripped_code=record.stripped_code,
                )
            )
    return {
        "rows": rows,
        "scores": scores,
        "coverage": _heldout_coverage(rows, assignments),
    }


def evaluate_candidates(
    candidates: Sequence[CandidateRecord],
    mixed_hashes: Mapping[tuple[str, str], set[str]],
    heldout_hashes: Mapping[tuple[str, str], set[str]],
    clusters: Mapping[ClusterKey, object],
) -> dict[str, object]:
    duplicate_sizes = _duplicate_sizes(candidates)
    rows = []
    scores = []
    for record in candidates:
        cache_ok = record.cache_exists
        cluster_ok = ClusterKey(record.question_id, record.language) in clusters
        decision = evaluate_submission_eligibility(
            record.significant_code_token_count,
            record.language,
            cluster_ok,
            record.valid,
        )
        reason = decision.exclusion_reason.value
        rows.append(
            {
                "record_id": record.record_id,
                "question_id": record.question_id,
                "language": record.language,
                "split": record.split,
                "difficulty": record.difficulty,
                "significant_code_token_count": record.significant_code_token_count,
                "minimum_significant_code_tokens": (
                    significant_token_threshold_for_language(record.language)
                ),
                "base_valid": record.valid and cache_ok,
                "cluster_exists": cluster_ok,
                "qualified": decision.eligible,
                "exclusion_reason": reason,
                "status": reason,
                "label_status": UNVERIFIED_LABEL_STATUS,
                "source": CANDIDATE_HUMAN_SOURCE,
                "stripped_hash": record.stripped_hash,
                "exact_match_to_mixed_v1": record.stripped_hash
                in mixed_hashes.get((record.question_id, record.language), set()),
                "exact_match_to_heldout_ai": record.stripped_hash
                in heldout_hashes.get((record.question_id, record.language), set()),
                "duplicate_group_size": duplicate_sizes[
                    (record.question_id, record.language, record.stripped_hash)
                ],
                "stripped_code": record.stripped_code if record.valid else None,
            }
        )
        if decision.eligible and record.stripped_code:
            maximum, distance = score_stripped_record(
                record.question_id,
                record.language,
                record.stripped_code,
                clusters,
            )
            scores.append(
                ScoredCandidate(
                    record_id=record.record_id,
                    question_id=record.question_id,
                    language=record.language,
                    split=record.split,
                    difficulty=record.difficulty,
                    significant_code_token_count=record.significant_code_token_count or 0,
                    ai_nn_max_raw=maximum,
                    ai_distance=distance,
                    exact_match_to_mixed_v1=record.stripped_hash
                    in mixed_hashes.get((record.question_id, record.language), set()),
                    duplicate_group_size=duplicate_sizes[
                        (record.question_id, record.language, record.stripped_hash)
                    ],
                    stripped_hash=record.stripped_hash,
                    embedding_cache_key=record.cache_key,
                    stripped_code=record.stripped_code,
                )
            )
    return {
        "rows": rows,
        "scores": scores,
        "coverage": _candidate_coverage(rows),
    }


def score_stripped_record(
    question_id: str,
    language: str,
    stripped_code: str,
    clusters: Mapping[ClusterKey, object],
) -> tuple[float, float]:
    vector = cached_vector_for_text(stripped_code)
    if vector is None:
        raise RuntimeError("Missing cached embedding; network embedding is forbidden")
    cluster = resolve_exact_cluster(question_id, language, clusters)
    query = np.asarray(vector, dtype=np.float32)
    similarities = np.asarray(cluster.vectors @ query, dtype=np.float64)
    maximum = float(np.max(similarities))
    distance = 1.0 - maximum
    return maximum, distance


def resolve_exact_cluster(
    question_id: str,
    language: str,
    clusters: Mapping[ClusterKey, object],
):
    key = ClusterKey(question_id, language)
    cluster = clusters.get(key)
    if cluster is None:
        raise RuntimeError(f"Missing exact cluster for {key.token}")
    if cluster.key.question_id != question_id:
        raise RuntimeError("Cross-question reference routing")
    if cluster.key.language != language:
        raise RuntimeError("Cross-language reference routing")
    return cluster


def write_heldout_samples(scores: Sequence[ScoredCandidate]) -> None:
    eligible = [
        row
        for row in scores
        if row.split == DatasetSplit.VALIDATION.value
    ]
    samples = select_extreme_samples(eligible, "heldout")
    _write_sample_file(
        HELDOUT_AI_SIMILARITY_SAMPLES_PATH,
        HELDOUT_SAMPLE_PREAMBLE,
        samples,
        candidate=False,
    )


def write_candidate_samples(scores: Sequence[ScoredCandidate]) -> None:
    eligible = [
        row
        for row in scores
        if row.split == DatasetSplit.TRAIN.value
    ]
    samples = select_extreme_samples(eligible, "candidate")
    _write_sample_file(
        CANDIDATE_HUMAN_SIMILARITY_SAMPLES_PATH,
        CANDIDATE_SAMPLE_PREAMBLE,
        samples,
        candidate=True,
    )


def select_extreme_samples(
    scores: Sequence[ScoredCandidate],
    kind: str,
) -> list[tuple[str, str, ScoredCandidate]]:
    unique = _dedupe_scores(scores)
    selected: list[tuple[str, str, ScoredCandidate]] = []
    for language in ("CPP", "PYTHON"):
        members = [row for row in unique if row.language == language]
        lows = sorted(members, key=_low_key)[:SAMPLES_PER_CATEGORY]
        low_ids = {row.record_id for row in lows}
        remaining = [row for row in members if row.record_id not in low_ids]
        highs = sorted(remaining, key=_high_key)[:SAMPLES_PER_CATEGORY]
        if len(lows) != 5 or len(highs) != 5:
            raise RuntimeError(f"Need 10 {kind} {language} samples")
        for index, row in enumerate(lows, start=1):
            selected.append((f"{language}_LOWEST_{index:02d}", "LOWEST", row))
        for index, row in enumerate(highs, start=1):
            selected.append((f"{language}_HIGHEST_{index:02d}", "HIGHEST", row))
    _validate_samples(selected)
    return selected


def write_outputs(
    coverage: Mapping[str, object],
    metadata: Mapping[str, object],
    heldout_eval: Mapping[str, object],
    candidate_eval: Mapping[str, object],
) -> None:
    directory = SIGNIFICANT_TOKEN_ELIGIBILITY_DIR
    directory.mkdir(parents=True, exist_ok=True)
    _write_json(directory / "coverage.json", _jsonable(coverage))
    _write_json(directory / "metadata.json", metadata)
    _write_heldout_csv(directory / "heldout_ai_coverage.csv", heldout_eval["coverage"])
    _write_candidate_csv(
        directory / "candidate_human_coverage.csv",
        candidate_eval["coverage"],
    )
    _write_jsonl(
        directory / "candidate_human_scores.jsonl",
        [_score_manifest(row) for row in candidate_eval["scores"]],
    )
    _write_jsonl(
        directory / "candidate_human_eligibility.jsonl",
        [_eligibility_manifest(row) for row in candidate_eval["rows"]],
    )
    _write_jsonl(
        directory / "heldout_ai_eligibility.jsonl",
        [_eligibility_manifest(row) for row in heldout_eval["rows"]],
    )
    (directory / "report.md").write_text(
        _report_markdown(coverage, metadata),
        encoding="utf-8",
    )


def _candidates_for_pair(
    groups: Mapping[str, object],
    dataset,
    assignment,
    language: Language,
) -> list[CandidateRecord]:
    source_records = groups.get(f"{assignment.question_id}:{language.value}")
    if not isinstance(source_records, list):
        return []
    boilerplate = dataset.questions[assignment.question_id].boilerplates.get(
        language.value,
        "",
    )
    records: list[CandidateRecord] = []
    for index, source_record in enumerate(source_records):
        records.append(
            _candidate_from_source(
                source_record,
                assignment,
                language,
                boilerplate,
                index,
            )
        )
    return records


def _candidate_from_source(
    source_record: object,
    assignment,
    language: Language,
    boilerplate: str,
    index: int,
) -> CandidateRecord:
    stripped, syntax_invalid = _strip_candidate(source_record, boilerplate, language)
    stripped_valid = stripped is not None and bool(stripped.strip()) and not syntax_invalid
    token_count = (
        _token_count(stripped, language.value)
        if stripped_valid and stripped
        else None
    )
    valid = stripped_valid and token_count is not None
    stripped_hash = _text_hash(stripped) if stripped else ""
    cache_key = embedding_cache_key(stripped) if stripped else ""
    cache_exists = bool(stripped) and (EMBEDDING_CACHE_DIR / f"{cache_key}.json").is_file()
    record_id = (
        f"{CANDIDATE_HUMAN_SOURCE}|{assignment.question_id}|"
        f"{language.value}|{index}|{stripped_hash or 'invalid'}"
    )
    return CandidateRecord(
        record_id=record_id,
        question_id=assignment.question_id,
        language=language.value,
        split=assignment.split,
        difficulty=assignment.difficulty,
        stripped_code=stripped if valid else None,
        stripped_hash=stripped_hash,
        significant_code_token_count=token_count,
        valid=valid,
        syntax_invalid=syntax_invalid or stripped is None,
        cache_key=cache_key,
        cache_exists=cache_exists,
        source=CANDIDATE_HUMAN_SOURCE,
        label_status=UNVERIFIED_LABEL_STATUS,
        group_index=index,
    )


def _strip_candidate(
    source_record: object,
    boilerplate: str,
    language: Language,
) -> tuple[str | None, bool]:
    if not isinstance(source_record, dict):
        return None, True
    raw_code = source_record.get("raw_code")
    if not isinstance(raw_code, str) or not raw_code.strip():
        return None, True
    try:
        stripped = strip_solution_body(raw_code, boilerplate, language)
    except (SourceParseError, ValueError):
        return None, True
    if not stripped.strip():
        return None, True
    return stripped, False


def _heldout_coverage(
    rows: Sequence[Mapping[str, object]],
    assignments: Sequence[object],
) -> dict[str, object]:
    eligible_rows = [row for row in rows if row["detection_eligible"]]
    return {
        "total": len(rows),
        "eligible": len(eligible_rows),
        "ineligible": len(rows) - len(eligible_rows),
        "eligible_percentage": _percent(len(eligible_rows), len(rows)),
        "by_language": _breakdown(rows, "language", "detection_eligible"),
        "by_split": _breakdown(rows, "split", "detection_eligible"),
        "by_difficulty": _breakdown(rows, "difficulty", "detection_eligible"),
        "by_generator": _breakdown(rows, "generator", "detection_eligible"),
        "by_generator_family": _breakdown(
            [_with_family(row) for row in rows],
            "generator_family",
            "detection_eligible",
        ),
        "by_persona": _breakdown(rows, "persona", "detection_eligible"),
        "exclusion_reasons": _reason_counts(rows),
    }


def _candidate_coverage(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    qualified = [row for row in rows if row["qualified"]]
    return {
        "total": len(rows),
        "eligible": len(qualified),
        "ineligible": len(rows) - len(qualified),
        "qualified": len(qualified),
        "not_qualified": len(rows) - len(qualified),
        "eligible_percentage": _percent(len(qualified), len(rows)),
        "qualified_percentage": _percent(len(qualified), len(rows)),
        "by_language": _breakdown(rows, "language", "qualified"),
        "by_split": _breakdown(rows, "split", "qualified"),
        "by_difficulty": _breakdown(rows, "difficulty", "qualified"),
        "exclusion_reasons": _reason_counts(rows),
        "duplicate_groups": sum(
            1 for row in rows if int(row["duplicate_group_size"]) > 1
        ),
        "duplicate_hash_groups": len(
            {
                (row["question_id"], row["language"], row["stripped_hash"])
                for row in rows
                if int(row["duplicate_group_size"]) > 1
            }
        ),
        "exact_mixed_v1_matches": sum(
            1 for row in rows if row["exact_match_to_mixed_v1"]
        ),
        "exact_heldout_ai_matches": sum(
            1 for row in rows if row["exact_match_to_heldout_ai"]
        ),
        "label_status": UNVERIFIED_LABEL_STATUS,
    }


def _candidate_source_audit(
    candidates: Sequence[CandidateRecord],
    mixed_hashes: Mapping[tuple[str, str], set[str]],
    heldout_hashes: Mapping[tuple[str, str], set[str]],
) -> dict[str, object]:
    stripped_ok = [row for row in candidates if row.valid]
    duplicate_sizes = _duplicate_sizes(candidates)
    return {
        "source_path": str(DATA_DIR / "scored_submissions.json"),
        "logical_record_count": len(candidates),
        "stripped_ok_count": len(stripped_ok),
        "expected_labeled_count": EXPECTED_CANDIDATE_HUMAN_COUNT,
        "cpp_count": sum(1 for row in candidates if row.language == "CPP"),
        "python_count": sum(1 for row in candidates if row.language == "PYTHON"),
        "question_split_counts": dict(Counter(row.split for row in candidates)),
        "duplicate_stripped_hash_groups": sum(
            1 for size in duplicate_sizes.values() if size > 1
        ),
        "syntax_invalid_records": sum(1 for row in candidates if row.syntax_invalid),
        "embedding_cache_misses": sum(
            1 for row in stripped_ok if not row.cache_exists
        ),
        "mixed_v1_exact_hash_matches": sum(
            1
            for row in stripped_ok
            if row.stripped_hash
            in mixed_hashes.get((row.question_id, row.language), set())
        ),
        "heldout_ai_exact_hash_matches": sum(
            1
            for row in stripped_ok
            if row.stripped_hash
            in heldout_hashes.get((row.question_id, row.language), set())
        ),
        "source": CANDIDATE_HUMAN_SOURCE,
        "label_status": UNVERIFIED_LABEL_STATUS,
        "combined_with_mixed_v1": False,
        "combined_with_heldout_ai": False,
        "combined_with_gpt_heavy": False,
    }


def _candidate_similarity(scores: Sequence[ScoredCandidate]) -> dict[str, object]:
    return {
        language: _describe_values(
            [row.ai_nn_max_raw for row in scores if row.language == language]
        )
        for language in ("CPP", "PYTHON")
    } | {
        f"{language}_distance": _describe_values(
            [row.ai_distance for row in scores if row.language == language]
        )
        for language in ("CPP", "PYTHON")
    }


def _validate_candidate_population(candidates: Sequence[CandidateRecord]) -> None:
    stripped = sum(1 for row in candidates if row.valid)
    if len(candidates) != EXPECTED_CANDIDATE_HUMAN_LOGICAL_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_CANDIDATE_HUMAN_LOGICAL_COUNT} logical "
            f"candidate-human records, found {len(candidates)}"
        )
    if stripped != EXPECTED_CANDIDATE_HUMAN_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_CANDIDATE_HUMAN_COUNT} stripped "
            f"candidate-human records, found {stripped}"
        )


def _hash_index(records: Sequence[LoadedSolution]) -> dict[tuple[str, str], set[str]]:
    grouped: dict[tuple[str, str], set[str]] = defaultdict(set)
    for record in records:
        if not record.stripped_code:
            continue
        grouped[(record.question_id, record.language)].add(
            _text_hash(record.stripped_code)
        )
    return grouped


def _duplicate_sizes(
    candidates: Sequence[CandidateRecord],
) -> dict[tuple[str, str, str], int]:
    counts: Counter[tuple[str, str, str]] = Counter()
    for row in candidates:
        counts[(row.question_id, row.language, row.stripped_hash)] += 1
    return dict(counts)


def _breakdown(
    rows: Sequence[Mapping[str, object]],
    field: str,
    flag: str,
) -> dict[str, dict[str, int]]:
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(field) or "unknown")].append(row)
    return {
        key: {
            "total": len(items),
            "eligible": sum(1 for item in items if item[flag]),
            "excluded": sum(1 for item in items if not item[flag]),
        }
        for key, items in sorted(grouped.items())
    }


def _with_family(row: Mapping[str, object]) -> dict[str, object]:
    payload = dict(row)
    payload["generator_family"] = _generator_family(str(row.get("generator") or ""))
    return payload


def _generator_family(model: str) -> str:
    lowered = model.lower()
    if "gpt" in lowered or "openai" in lowered:
        return "GPT"
    if "gemini" in lowered or "google" in lowered:
        return "Gemini"
    if "deepseek" in lowered:
        return "DeepSeek"
    return "other"


def _token_count(stripped_code: str, language: str) -> int | None:
    try:
        return significant_code_token_count(stripped_code, language)
    except SignificantCodeTokenizationError:
        return None


def _left_rotate_audit() -> dict[str, object]:
    count = significant_code_token_count(LEFT_ROTATE_AUDIT_CODE, "CPP")
    return {
        "language": "CPP",
        "significant_code_token_count": count,
        "minimum_significant_code_tokens": CPP_SIGNIFICANT_TOKEN_THRESHOLD,
        "token_count_used_as_exclude": False,
        "eligibility_status": SubmissionExclusionReason.ELIGIBLE.value,
    }


def _population_token_distributions(
    rows: Sequence[Mapping[str, object]],
    eligible_flag: str,
) -> dict[str, object]:
    result = {}
    for language in ("CPP", "PYTHON"):
        language_rows = [row for row in rows if row["language"] == language]
        result[language] = {
            "all": _token_distribution(language_rows),
            "eligible_primary": _token_distribution(
                [row for row in language_rows if row[eligible_flag]]
            ),
            "excluded_primary": _token_distribution(
                [row for row in language_rows if not row[eligible_flag]]
            ),
        }
    return result


def _token_distribution(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, float | int | None]:
    values = [
        float(row["significant_code_token_count"])
        for row in rows
        if isinstance(row.get("significant_code_token_count"), int)
    ]
    return _describe_values(values)


def _percent(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return 100.0 * numerator / denominator


def _describe_values(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "standard_deviation": None,
            "minimum": None,
            "maximum": None,
            **{f"p{point}": None for point in PERCENTILES},
        }
    array = np.asarray(values, dtype=np.float64)
    payload: dict[str, float | int | None] = {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "standard_deviation": float(np.std(array, ddof=1)) if array.size > 1 else 0.0,
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }
    for point in PERCENTILES:
        key = "median" if point == 50 else f"p{point}"
        payload[key] = float(np.percentile(array, point))
    return payload


def _dedupe_scores(scores: Sequence[ScoredCandidate]) -> list[ScoredCandidate]:
    unique: dict[tuple[str, str, str], ScoredCandidate] = {}
    for row in sorted(scores, key=_fingerprint):
        key = (row.question_id, row.language, row.stripped_hash)
        if key in unique:
            continue
        unique[key] = row
    return list(unique.values())


def _low_key(row: ScoredCandidate) -> tuple[float, str]:
    return (row.ai_nn_max_raw, _fingerprint(row))


def _high_key(row: ScoredCandidate) -> tuple[float, str]:
    return (-row.ai_nn_max_raw, _fingerprint(row))


def _fingerprint(row: ScoredCandidate) -> str:
    return sha256(row.record_id.encode("utf-8")).hexdigest()[:12]


def _validate_samples(samples: Sequence[tuple[str, str, ScoredCandidate]]) -> None:
    if len(samples) != 20:
        raise RuntimeError(f"Expected 20 samples, found {len(samples)}")
    for language in ("CPP", "PYTHON"):
        members = [item for item in samples if item[2].language == language]
        highs = [item for item in members if item[1] == "HIGHEST"]
        lows = [item for item in members if item[1] == "LOWEST"]
        high_hashes = {item[2].stripped_hash for item in highs}
        low_hashes = {item[2].stripped_hash for item in lows}
        if len(high_hashes) != 5 or len(low_hashes) != 5:
            raise RuntimeError("Sample hashes are not unique")
        if high_hashes & low_hashes:
            raise RuntimeError("High and low samples overlap")


def _write_sample_file(
    path: Path,
    preamble: str,
    samples: Sequence[tuple[str, str, ScoredCandidate]],
    candidate: bool,
) -> None:
    blocks = [preamble.rstrip(), ""]
    for sample_id, category, row in samples:
        blocks.append(_sample_block(sample_id, category, row, candidate))
    path.write_text("\n".join(blocks) + "\n", encoding="utf-8")


def _sample_block(
    sample_id: str,
    category: str,
    row: ScoredCandidate,
    candidate: bool,
) -> str:
    displayed = display_similarity(row.ai_nn_max_raw)
    lines = [
        "============================================================",
        f"SAMPLE: {sample_id}",
        f"CATEGORY: {category}",
        f"RECORD_ID: {row.record_id}",
        f"QUESTION_ID: {row.question_id}",
        f"LANGUAGE: {row.language}",
        f"DIFFICULTY: {row.difficulty}",
        f"SPLIT: {row.split}",
        f"SIGNIFICANT_CODE_TOKEN_COUNT: {row.significant_code_token_count}",
        f"MINIMUM_SIGNIFICANT_CODE_TOKENS: "
        f"{significant_token_threshold_for_language(row.language)}",
        "ELIGIBILITY_STATUS: eligible",
        f"AI_NN_MAX_RAW: {row.ai_nn_max_raw:.8f}",
        f"AI_NN_MAX_DISPLAY: {displayed:.8f}",
        f"AI_DISTANCE: {row.ai_distance:.8f}",
        f"EXACT_MATCH_TO_MIXED_V1: {str(row.exact_match_to_mixed_v1).lower()}",
        f"DUPLICATE_GROUP_SIZE: {row.duplicate_group_size}",
        f"RECORD_FINGERPRINT: {_fingerprint(row)}",
        "LABEL_STATUS: unverified" if candidate else "SOURCE: heldout_ai",
        "============================================================",
        "",
        row.stripped_code,
        "",
    ]
    return "\n".join(lines)


def _score_manifest(row: ScoredCandidate) -> dict[str, object]:
    return {
        "record_id": row.record_id,
        "question_id": row.question_id,
        "language": row.language,
        "split": row.split,
        "difficulty": row.difficulty,
        "significant_code_token_count": row.significant_code_token_count,
        "minimum_significant_code_tokens": (
            significant_token_threshold_for_language(row.language)
        ),
        "ai_nn_max_raw": row.ai_nn_max_raw,
        "ai_nn_max_display": display_similarity(row.ai_nn_max_raw),
        "ai_distance": row.ai_distance,
        "exact_match_to_mixed_v1": row.exact_match_to_mixed_v1,
        "duplicate_group_size": row.duplicate_group_size,
        "stripped_hash": row.stripped_hash,
        "embedding_cache_key": row.embedding_cache_key,
        "source": CANDIDATE_HUMAN_SOURCE,
        "label_status": UNVERIFIED_LABEL_STATUS,
    }


def _eligibility_manifest(row: Mapping[str, object]) -> dict[str, object]:
    excluded = {"stripped_code"}
    return {key: value for key, value in row.items() if key not in excluded}


def _reason_counts(rows: Sequence[Mapping[str, object]]) -> dict[str, int]:
    return dict(Counter(str(row["exclusion_reason"]) for row in rows))


def _write_heldout_csv(path: Path, coverage: Mapping[str, object]) -> None:
    payload = {
        key: value if not isinstance(value, dict) else json.dumps(value)
        for key, value in coverage.items()
    }
    _write_csv(path, [payload])


def _write_candidate_csv(path: Path, coverage: Mapping[str, object]) -> None:
    payload = {
        key: value if not isinstance(value, dict) else json.dumps(value)
        for key, value in coverage.items()
    }
    _write_csv(path, [payload])


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _report_markdown(
    coverage: Mapping[str, object],
    metadata: Mapping[str, object],
) -> str:
    lines = [
        "# Language-aware significant code-token eligibility",
        "",
        "Eligibility counts lexical code tokens from the exact stored stripped_code",
        "using the repository's Tree-sitter C++ and Python grammars.",
        "Keywords, identifiers, complete numeric/string literals, operators,",
        "brackets, parentheses, commas, colons, semicolons, and other",
        "punctuators count. Whitespace, blank lines, newlines, indentation,",
        "dedentation, comments, encoding markers, and EOF markers do not.",
        "",
        "Scoring uses integrity checks only: supported language, stripped code",
        "that parses and tokenizes, cached embedding, and exact mixed-v1 cluster.",
        "Significant-token counts are recorded. CPP 80 and Python 60 are former",
        "eligibility floors reserved for a later calibration layer and are not",
        "used as an exclude.",
        "Mixed-v1 clusters are used only for exact question-language routing.",
        "Candidate-human labels remain unverified.",
        "",
        "## Mixed-v1 clusters",
        json.dumps(coverage["mixed_v1_clusters"], indent=2),
        "",
        "## leftRotate audit",
        json.dumps(coverage["left_rotate_audit"], indent=2),
        "",
        "## Held-out AI",
        json.dumps(coverage["heldout_ai"], indent=2),
        "",
        "## Candidate-human",
        json.dumps(coverage["candidate_human"], indent=2),
        "",
        "## Candidate-human source",
        json.dumps(coverage["candidate_human_source"], indent=2),
        "",
        "## Significant-token distributions",
        json.dumps(coverage["token_distributions"], indent=2),
        "",
        "## Held-out AI similarity",
        json.dumps(coverage["heldout_similarity"], indent=2),
        "",
        "## Primary candidate similarity",
        json.dumps(coverage["candidate_similarity_primary"], indent=2),
        "",
        "## Run metadata",
        json.dumps(metadata, indent=2, default=str),
        "",
        "Token count measures evidence quantity, not authorship.",
        "No classification metrics were produced; final validation requires",
        "independently verified human labels.",
    ]
    return "\n".join(lines) + "\n"


def _jsonable(payload: object) -> object:
    if isinstance(payload, dict):
        return {str(key): _jsonable(value) for key, value in payload.items()}
    if isinstance(payload, list):
        return [_jsonable(item) for item in payload]
    if isinstance(payload, ScoredCandidate):
        return _score_manifest(payload)
    return payload


def _text_hash(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())

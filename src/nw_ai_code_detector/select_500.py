from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from collections.abc import Callable, Mapping, Sequence

from nw_ai_code_detector.config import SELECTED_500_PATH
from nw_ai_code_detector.constants import (
    COVERAGE_HIGH_MIN,
    COVERAGE_MID_MIN,
    CoverageBucket,
    Difficulty,
    HARD_DIFFICULTY_FLOOR,
    SELECTED_QUESTION_COUNT,
    SELECTION_RANDOM_SEED,
    TAG_MEMBERSHIP_FLOOR,
    TOPIC_TAG_PREFIX,
)
from nw_ai_code_detector.data_load import Dataset, QuestionRecord, load_dataset


@dataclass(frozen=True)
class EligibleQuestion:
    question_id: str
    difficulty: str
    tags: tuple[str, ...]
    primary_tag: str
    coverage_count: int
    coverage_bucket: str


def select_eligible_questions(dataset: Dataset) -> list[EligibleQuestion]:
    eligible: list[EligibleQuestion] = []
    for question in dataset.questions.values():
        coverage_count = dataset.human_submission_counts.get(question.question_id, 0)
        if not _is_eligible(question, coverage_count):
            continue
        eligible.append(_to_eligible(question, coverage_count))
    return eligible


def select_stratified_questions(
    eligible: Sequence[EligibleQuestion],
    selected_count: int,
    seed: int,
) -> list[EligibleQuestion]:
    if len(eligible) < selected_count:
        raise ValueError(
            f"Eligible pool has {len(eligible)} questions; need {selected_count}"
        )
    quotas = _stratum_quotas(eligible, selected_count)
    selected = _pick_by_quotas(eligible, quotas, seed)
    selected = _top_up_hard_questions(selected, eligible, seed)
    selected = _top_up_rare_tags(selected, eligible, seed)
    return sorted(selected, key=lambda item: item.question_id)


def write_selected_questions(
    selected: Sequence[EligibleQuestion],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(item) for item in selected]
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def print_distribution_table(
    selected: Sequence[EligibleQuestion],
    eligible: Sequence[EligibleQuestion],
) -> None:
    print("\n=== select_500 distribution ===")
    print(f"eligible pool: {len(eligible)}  selected: {len(selected)}")
    _print_counter_block("difficulty", _count_by(selected, lambda item: item.difficulty))
    _print_counter_block("primary tag", _count_by(selected, lambda item: item.primary_tag))
    _print_counter_block(
        "coverage bucket",
        _count_by(selected, lambda item: item.coverage_bucket),
    )
    _print_counter_block(
        "all-membership TOPIC tags",
        _count_membership_tags(selected),
    )
    print(
        f"Hard selected: {_count_by(selected, lambda item: item.difficulty)[Difficulty.HARD.value]} "
        f"(floor {HARD_DIFFICULTY_FLOOR})"
    )


def main() -> int:
    dataset = load_dataset()
    eligible = select_eligible_questions(dataset)
    selected = select_stratified_questions(
        eligible,
        SELECTED_QUESTION_COUNT,
        SELECTION_RANDOM_SEED,
    )
    write_selected_questions(selected, SELECTED_500_PATH)
    print_distribution_table(selected, eligible)
    print(f"\nWrote {SELECTED_500_PATH}")
    return 0


def _is_eligible(question: QuestionRecord, coverage_count: int) -> bool:
    return question.is_function_completion and coverage_count >= 1


def _to_eligible(question: QuestionRecord, coverage_count: int) -> EligibleQuestion:
    return EligibleQuestion(
        question_id=question.question_id,
        difficulty=question.difficulty,
        tags=question.tags,
        primary_tag=question.primary_tag,
        coverage_count=coverage_count,
        coverage_bucket=_coverage_bucket(coverage_count),
    )


def _coverage_bucket(coverage_count: int) -> str:
    if coverage_count >= COVERAGE_HIGH_MIN:
        return CoverageBucket.HIGH.value
    if coverage_count >= COVERAGE_MID_MIN:
        return CoverageBucket.MID.value
    return CoverageBucket.LOW.value


def _stratum_key(question: EligibleQuestion) -> tuple[str, str]:
    return (question.difficulty, question.primary_tag)


def _stratum_quotas(
    eligible: Sequence[EligibleQuestion],
    selected_count: int,
) -> dict[tuple[str, str], int]:
    grouped = _group_by_stratum(eligible)
    sizes = {key: len(items) for key, items in grouped.items()}
    total = sum(sizes.values())
    raw_shares = {
        key: selected_count * size / total for key, size in sizes.items()
    }
    quotas = {key: min(int(share), sizes[key]) for key, share in raw_shares.items()}
    remaining = selected_count - sum(quotas.values())
    remainders = sorted(
        raw_shares,
        key=lambda key: (raw_shares[key] - quotas[key], key),
        reverse=True,
    )
    for key in remainders:
        if remaining <= 0:
            break
        if quotas[key] >= sizes[key]:
            continue
        quotas[key] += 1
        remaining -= 1
    return quotas


def _pick_by_quotas(
    eligible: Sequence[EligibleQuestion],
    quotas: Mapping[tuple[str, str], int],
    seed: int,
) -> list[EligibleQuestion]:
    grouped = _group_by_stratum(eligible)
    selected: list[EligibleQuestion] = []
    for key, quota in quotas.items():
        ranked = _rank_for_selection(grouped[key], seed)
        selected.extend(ranked[:quota])
    return selected


def _top_up_hard_questions(
    selected: Sequence[EligibleQuestion],
    eligible: Sequence[EligibleQuestion],
    seed: int,
) -> list[EligibleQuestion]:
    selected_ids = {item.question_id for item in selected}
    hard_count = sum(1 for item in selected if item.difficulty == Difficulty.HARD.value)
    needed = HARD_DIFFICULTY_FLOOR - hard_count
    if needed <= 0:
        return list(selected)
    replacements = _rank_for_selection(
        [
            item
            for item in eligible
            if item.difficulty == Difficulty.HARD.value
            and item.question_id not in selected_ids
        ],
        seed,
    )
    return _replace_lowest_coverage(list(selected), replacements[:needed], seed)


def _top_up_rare_tags(
    selected: Sequence[EligibleQuestion],
    eligible: Sequence[EligibleQuestion],
    seed: int,
) -> list[EligibleQuestion]:
    updated = list(selected)
    membership_counts = _count_membership_tags(updated)
    pool_counts = _count_membership_tags(eligible)
    for tag, pool_count in sorted(pool_counts.items()):
        if pool_count < TAG_MEMBERSHIP_FLOOR:
            continue
        selected_count = membership_counts.get(tag, 0)
        needed = TAG_MEMBERSHIP_FLOOR - selected_count
        if needed <= 0:
            continue
        selected_ids = {item.question_id for item in updated}
        replacements = _rank_for_selection(
            [
                item
                for item in eligible
                if tag in item.tags and item.question_id not in selected_ids
            ],
            seed,
        )
        updated = _replace_lowest_coverage(updated, replacements[:needed], seed)
        membership_counts = _count_membership_tags(updated)
    return updated


def _replace_lowest_coverage(
    selected: list[EligibleQuestion],
    incoming: Sequence[EligibleQuestion],
    seed: int,
) -> list[EligibleQuestion]:
    if not incoming:
        return selected
    keepable = [
        item for item in selected if item.difficulty != Difficulty.HARD.value
    ]
    donors = list(reversed(_rank_for_selection(keepable, seed)))
    incoming_ids = {item.question_id for item in incoming}
    donor_ids: set[str] = set()
    for donor in donors:
        if len(donor_ids) == len(incoming):
            break
        donor_ids.add(donor.question_id)
    retained = [
        item
        for item in selected
        if item.question_id not in donor_ids
        and item.question_id not in incoming_ids
    ]
    added = list(incoming)[: len(donor_ids)]
    return retained + added


def _group_by_stratum(
    questions: Sequence[EligibleQuestion],
) -> dict[tuple[str, str], list[EligibleQuestion]]:
    grouped: dict[tuple[str, str], list[EligibleQuestion]] = defaultdict(list)
    for question in questions:
        grouped[_stratum_key(question)].append(question)
    return grouped


def _rank_for_selection(
    questions: Sequence[EligibleQuestion],
    seed: int,
) -> list[EligibleQuestion]:
    return sorted(
        questions,
        key=lambda item: (
            _coverage_rank(item.coverage_bucket),
            -item.coverage_count,
            _stable_rank(seed, item.question_id),
            item.question_id,
        ),
    )


def _coverage_rank(bucket: str) -> int:
    ranks = {
        CoverageBucket.HIGH.value: 0,
        CoverageBucket.MID.value: 1,
        CoverageBucket.LOW.value: 2,
    }
    return ranks[bucket]


def _stable_rank(seed: int, question_id: str) -> str:
    digest = sha256(f"{seed}:{question_id}".encode("utf-8")).hexdigest()
    return digest


def _count_by(
    questions: Sequence[EligibleQuestion],
    key_fn: Callable[[EligibleQuestion], str],
) -> Counter:
    return Counter(key_fn(item) for item in questions)


def _count_membership_tags(questions: Sequence[EligibleQuestion]) -> Counter:
    names: list[str] = []
    for question in questions:
        names.extend(
            tag for tag in question.tags if tag.startswith(TOPIC_TAG_PREFIX)
        )
    return Counter(names)


def _print_counter_block(title: str, counts: Counter) -> None:
    print(f"\n{title}:")
    for key, count in counts.most_common():
        print(f"  {key}: {count}")


if __name__ == "__main__":
    raise SystemExit(main())

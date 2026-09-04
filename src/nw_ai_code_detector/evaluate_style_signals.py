from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from collections.abc import Mapping, Sequence

from nw_ai_code_detector.build_canonicality_eligibility_tokens import (
    load_candidate_human_records,
    score_stripped_record,
)
from nw_ai_code_detector.build_model_dataset_v2 import load_question_assignments
from nw_ai_code_detector.config import (
    AI_SOLUTIONS_DIR,
    DATA_DIR,
    EVAL_AI_SOLUTIONS_DIR,
    STYLE_SIGNALS_DIR,
)
from nw_ai_code_detector.constants import (
    CANDIDATE_HUMAN_SOURCE,
    GROUPS_KEY,
    HELDOUT_AI_SOURCE,
    RAW_CODE_FIELD,
    StyleSignalName,
)
from nw_ai_code_detector.data_load import load_dataset
from nw_ai_code_detector.eligibility_data import load_solution_records
from nw_ai_code_detector.embedder import cached_vector_for_text
from nw_ai_code_detector.evaluation.evaluate_ai_reference_scores_v2 import (
    load_reference_clusters,
)
from nw_ai_code_detector.significant_code_tokens import (
    SignificantCodeTokenizationError,
    significant_code_token_count,
)
from nw_ai_code_detector.style_signals import StyleFlag, extract_style_flags


@dataclass(frozen=True)
class StyleEvalInputs:
    source: str
    question_id: str
    language: str
    raw_code: str
    stripped_code: str
    boilerplate: str
    token_count: int | None


@dataclass(frozen=True)
class StyleEvalRow:
    source: str
    question_id: str
    language: str
    token_count: int | None
    canonicality: float | None
    exact_match_flag: bool
    signals: tuple[StyleFlag, ...]
    raw_code: str
    explanation_bits: tuple[str, ...]


def main() -> int:
    dataset = load_dataset()
    assignments = load_question_assignments()
    hashes = _student_reference_hashes()
    mixed_only = _mixed_reference_hashes()
    clusters = load_reference_clusters()
    humans = _human_rows(dataset, assignments, hashes, clusters)
    heldout = _heldout_rows(dataset, mixed_only, clusters)
    rates = _firing_rates(humans, heldout)
    examples = _select_examples(humans, heldout)
    _write_outputs(rates, examples, humans, heldout)
    print(json.dumps(rates, indent=2))
    print(f"Wrote {STYLE_SIGNALS_DIR}")
    return 0


def _student_reference_hashes() -> dict[tuple[str, str], set[str]]:
    grouped = _hash_dir(AI_SOLUTIONS_DIR, "mixed_v1")
    extra = _hash_dir(EVAL_AI_SOLUTIONS_DIR, HELDOUT_AI_SOURCE)
    for key, values in extra.items():
        grouped.setdefault(key, set()).update(values)
    return grouped


def _mixed_reference_hashes() -> dict[tuple[str, str], set[str]]:
    return _hash_dir(AI_SOLUTIONS_DIR, "mixed_v1")


def _hash_dir(root, source: str) -> dict[tuple[str, str], set[str]]:
    grouped: dict[tuple[str, str], set[str]] = defaultdict(set)
    for record in load_solution_records(root, source):
        if not record.stripped_code:
            continue
        digest = sha256(record.stripped_code.encode("utf-8")).hexdigest()
        grouped[(record.question_id, record.language)].add(digest)
    return dict(grouped)


def _human_rows(
    dataset,
    assignments,
    hashes: Mapping[tuple[str, str], set[str]],
    clusters,
) -> list[StyleEvalRow]:
    candidates = load_candidate_human_records(assignments)
    groups = json.loads((DATA_DIR / "scored_submissions.json").read_text(encoding="utf-8"))
    mapping = groups.get(GROUPS_KEY)
    rows: list[StyleEvalRow] = []
    for candidate in candidates:
        if not candidate.valid or not candidate.stripped_code:
            continue
        raw = _human_raw(mapping, candidate.question_id, candidate.language, candidate.group_index)
        boilerplate = dataset.questions[candidate.question_id].boilerplates.get(
            candidate.language,
            "",
        )
        rows.append(
            _row_from_parts(
                StyleEvalInputs(
                    CANDIDATE_HUMAN_SOURCE,
                    candidate.question_id,
                    candidate.language,
                    raw,
                    candidate.stripped_code,
                    boilerplate,
                    candidate.significant_code_token_count,
                ),
                hashes,
                clusters,
            )
        )
    return rows


def _heldout_rows(
    dataset,
    hashes: Mapping[tuple[str, str], set[str]],
    clusters,
) -> list[StyleEvalRow]:
    rows: list[StyleEvalRow] = []
    for record in load_solution_records(EVAL_AI_SOLUTIONS_DIR, HELDOUT_AI_SOURCE):
        if not record.parse_ok or not record.stripped_code:
            continue
        question = dataset.questions.get(record.question_id)
        boilerplate = question.boilerplates.get(record.language, "") if question else ""
        raw = record.raw_code or ""
        rows.append(
            _row_from_parts(
                StyleEvalInputs(
                    HELDOUT_AI_SOURCE,
                    record.question_id,
                    record.language,
                    raw,
                    record.stripped_code,
                    boilerplate,
                    None,
                ),
                hashes,
                clusters,
            )
        )
    return rows


def _row_from_parts(
    inputs: StyleEvalInputs,
    hashes: Mapping[tuple[str, str], set[str]],
    clusters,
) -> StyleEvalRow:
    flags = extract_style_flags(
        inputs.raw_code,
        inputs.stripped_code,
        inputs.language,
        inputs.boilerplate,
    )
    digest = sha256(inputs.stripped_code.encode("utf-8")).hexdigest()
    exact = digest in hashes.get((inputs.question_id, inputs.language), set())
    canonicality = _canonicality(
        inputs.question_id,
        inputs.language,
        inputs.stripped_code,
        clusters,
    )
    counted = inputs.token_count
    if counted is None:
        counted = _safe_token_count(inputs.stripped_code, inputs.language)
    fired = tuple(flag.name for flag in flags if flag.fired)
    return StyleEvalRow(
        source=inputs.source,
        question_id=inputs.question_id,
        language=inputs.language,
        token_count=counted,
        canonicality=canonicality,
        exact_match_flag=exact,
        signals=flags,
        raw_code=inputs.raw_code,
        explanation_bits=fired,
    )


def _safe_token_count(stripped: str, language: str) -> int | None:
    try:
        return significant_code_token_count(stripped, language)
    except SignificantCodeTokenizationError:
        return None


def _canonicality(question_id: str, language: str, stripped: str, clusters) -> float | None:
    if cached_vector_for_text(stripped) is None:
        return None
    try:
        maximum, _distance = score_stripped_record(question_id, language, stripped, clusters)
    except RuntimeError:
        return None
    return maximum


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


def _firing_rates(
    humans: Sequence[StyleEvalRow],
    heldout: Sequence[StyleEvalRow],
) -> dict[str, object]:
    return {
        "candidate_human": _class_rates(humans),
        "heldout_ai": _class_rates(heldout),
        "student_harm": _student_harm(humans),
    }


def _class_rates(rows: Sequence[StyleEvalRow]) -> dict[str, object]:
    by_language = {"CPP": [row for row in rows if row.language == "CPP"], "PYTHON": [row for row in rows if row.language == "PYTHON"]}
    payload = {"n": len(rows), "by_language": {}}
    for language, members in by_language.items():
        payload["by_language"][language] = _language_rates(members)
    payload["overall"] = _language_rates(rows)
    return payload


def _language_rates(rows: Sequence[StyleEvalRow]) -> dict[str, object]:
    total = len(rows) or 1
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        for flag in row.signals:
            if flag.fired:
                counts[flag.name] += 1
    names = [item.value for item in StyleSignalName]
    return {
        "n": len(rows),
        "signals": {
            name: {
                "fired": counts[name],
                "rate": round(counts[name] / total, 4) if rows else 0.0,
            }
            for name in names
        },
        "exact_match": {
            "fired": sum(1 for row in rows if row.exact_match_flag),
            "rate": round(sum(1 for row in rows if row.exact_match_flag) / total, 4)
            if rows
            else 0.0,
        },
    }


def _student_harm(humans: Sequence[StyleEvalRow]) -> dict[str, object]:
    ai_names = {
        StyleSignalName.EXPLANATORY_COMMENTS.value,
        StyleSignalName.UNIFORM_VERBOSE_NAMING.value,
    }
    hits = [
        row
        for row in humans
        if any(flag.fired and flag.name in ai_names for flag in row.signals)
    ]
    by_signal = {
        name: sum(1 for row in humans if _has_fire(row, name))
        for name in ai_names
    }
    return {
        "candidate_humans": len(humans),
        "any_ai_leaning_flag": len(hits),
        "by_signal": by_signal,
        "by_language": dict(Counter(row.language for row in hits)),
    }


def _has_fire(row: StyleEvalRow, name: str) -> bool:
    return any(flag.fired and flag.name == name for flag in row.signals)


def _select_examples(
    humans: Sequence[StyleEvalRow],
    heldout: Sequence[StyleEvalRow],
) -> list[dict[str, object]]:
    selected: list[StyleEvalRow] = []
    selected.extend(_take_with_fire(humans, StyleSignalName.EXPLANATORY_COMMENTS.value, 6))
    selected.extend(_take_with_fire(humans, StyleSignalName.COMMENTED_OUT_CODE.value, 6))
    selected.extend(_take_with_fire(humans, StyleSignalName.UNUSED_LOCALS.value, 4))
    selected.extend(_take_with_fire(heldout, StyleSignalName.UNUSED_LOCALS.value, 4))
    selected.extend(_take_with_fire(heldout, StyleSignalName.UNIFORM_VERBOSE_NAMING.value, 2))
    unique = _dedupe_examples(selected)
    if len(unique) < 20:
        unique.extend(_fill_examples(humans, heldout, unique, 20 - len(unique)))
    return [_example_payload(row) for row in unique[:20]]


def _fill_examples(
    humans: Sequence[StyleEvalRow],
    heldout: Sequence[StyleEvalRow],
    already: Sequence[StyleEvalRow],
    needed: int,
) -> list[StyleEvalRow]:
    seen = {_example_key(row) for row in already}
    extras: list[StyleEvalRow] = []
    for row in (*heldout, *humans):
        key = _example_key(row)
        if key in seen:
            continue
        seen.add(key)
        extras.append(row)
        if len(extras) >= needed:
            break
    return extras


def _take_with_fire(rows: Sequence[StyleEvalRow], name: str, limit: int) -> list[StyleEvalRow]:
    matched = [row for row in rows if _has_fire(row, name)]
    cpp = [row for row in matched if row.language == "CPP"][: limit // 2]
    python = [row for row in matched if row.language == "PYTHON"][: limit - len(cpp)]
    return cpp + python


def _dedupe_examples(rows: Sequence[StyleEvalRow]) -> list[StyleEvalRow]:
    seen: set[tuple[str, str, str, str]] = set()
    unique: list[StyleEvalRow] = []
    for row in rows:
        key = _example_key(row)
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return unique


def _example_key(row: StyleEvalRow) -> tuple[str, str, str, str]:
    return (row.source, row.question_id, row.language, row.raw_code[:80])


def _example_payload(row: StyleEvalRow) -> dict[str, object]:
    raw = row.raw_code
    truncated = len(raw) > 2500
    display = raw[:2500]
    return {
        "source": row.source,
        "question_id": row.question_id,
        "language": row.language,
        "token_count": row.token_count,
        "canonicality": row.canonicality,
        "exact_match_flag": row.exact_match_flag,
        "signals": [asdict(flag) for flag in row.signals],
        "fired": list(row.explanation_bits),
        "raw_code_truncated": truncated,
        "raw_code": display,
    }


def _write_outputs(
    rates: Mapping[str, object],
    examples: Sequence[Mapping[str, object]],
    humans: Sequence[StyleEvalRow],
    heldout: Sequence[StyleEvalRow],
) -> None:
    STYLE_SIGNALS_DIR.mkdir(parents=True, exist_ok=True)
    _write_json(STYLE_SIGNALS_DIR / "firing_rates.json", rates)
    _write_json(STYLE_SIGNALS_DIR / "examples.json", list(examples))
    (STYLE_SIGNALS_DIR / "report.md").write_text(
        _report_markdown(rates, examples, len(humans), len(heldout)),
        encoding="utf-8",
    )


def _report_markdown(
    rates: Mapping[str, object],
    examples: Sequence[Mapping[str, object]],
    n_humans: int,
    n_heldout: int,
) -> str:
    harm = rates["student_harm"]
    lines = [
        "# Style signals v0 (rank-and-flag)",
        "",
        "No fused risk score. Canonicality ranks; flags are evidence.",
        f"Candidate-human n={n_humans}. Held-out AI n={n_heldout}.",
        "",
        "## Firing rates",
        "",
        "```json",
        json.dumps(rates, indent=2),
        "```",
        "",
        "## Student-harm (AI-leaning flags on candidate-humans)",
        "",
        json.dumps(harm, indent=2),
        "",
        f"## Examples ({len(examples)})",
        "",
        "See examples.json for raw_code and per-signal fired/not.",
    ]
    return "\n".join(lines) + "\n"


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())

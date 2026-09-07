from __future__ import annotations

import hmac
import os
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import streamlit as st

from demo.runtime import (
    DEMO_PASSWORD_ENV,
    QUESTIONS_PATH_DEFAULT,
    QUESTIONS_PATH_ENV,
    DemoScore,
    DemoScoreRequest,
    InMemoryVoyageEmbedder,
    available_languages,
    load_demo_questions,
    load_reference_hashes,
    load_reference_index,
    questions_for_index,
    score_demo_submission,
)
from nw_ai_code_detector.config import load_voyage_settings
from nw_ai_code_detector.constants import (
    DISPLAY_MATCH_PERCENT_MIDPOINT,
    INSUFFICIENT_EVIDENCE_STATUS,
    LOW_CONFIDENCE_SHORT_STATUS,
    LOW_CONFIDENCE_STATUS,
    MISSING_OR_INVALID_STATUS,
    SHORT_LOW_CONFIDENCE_MAX_TOKENS_BY_LANGUAGE,
    SIGNIFICANT_TOKEN_THRESHOLDS_BY_LANGUAGE,
)
from nw_ai_code_detector.data_load import QuestionRecord
from nw_ai_code_detector.explanation_card import ExplanationCard
from nw_ai_code_detector.score_query import DetectionResult

POC_BANNER = "POC / provisional — not validated"
NO_HIGHLIGHT_NOTE = "Line-level highlighting not yet available."
# Routing status is the authoritative confidence source; the detector's naming
# label is only an additive qualifier and must never replace a low-confidence route.
CONFIDENCE_LABEL_BY_STATUS = {
    LOW_CONFIDENCE_STATUS: "low — one obvious solution (tight cluster)",
    LOW_CONFIDENCE_SHORT_STATUS: "low — short code",
    INSUFFICIENT_EVIDENCE_STATUS: "not scored — below token floor",
    MISSING_OR_INVALID_STATUS: "not scored — missing or invalid",
}
STANDARD_CONFIDENCE_LABEL = "standard"


def main() -> None:
    st.set_page_config(page_title="AI Code Detector POC", layout="wide")
    _styles()
    st.error(POC_BANNER, icon="⚠️")
    st.title("AI Code Detector Demo")
    st.caption(
        "Measures resemblance to our generated AI reference solutions. "
        "It does not prove authorship or report a cheating percentage."
    )
    if not _password_gate():
        st.stop()
    try:
        questions, index, embedder, hashes = _resources()
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        st.error(f"Demo configuration error: {exc}")
        st.stop()
    question = _question_picker(questions)
    _question_statement(question)
    language = st.radio(
        "Language",
        available_languages(question),
        horizontal=True,
    )
    boilerplate = question.boilerplates.get(language, "")
    raw_code = _submission_editor(
        boilerplate,
        question.question_id,
        language,
    )
    ready = bool(raw_code.strip()) and raw_code != boilerplate
    if st.button("Score submission", type="primary", disabled=not ready):
        try:
            st.session_state.demo_score = score_demo_submission(
                DemoScoreRequest(raw_code, question, language),
                index,
                embedder,
                hashes,
            )
        except Exception as exc:
            st.error(f"Could not score this submission: {exc}")
            st.session_state.pop("demo_score", None)
    score = st.session_state.get("demo_score")
    if isinstance(score, DemoScore):
        _render_result(score)


@st.cache_resource
def _resources():
    configured_path = os.getenv(QUESTIONS_PATH_ENV, "").strip()
    path = Path(configured_path) if configured_path else QUESTIONS_PATH_DEFAULT
    index = load_reference_index()
    questions = questions_for_index(load_demo_questions(path), index)
    settings = load_voyage_settings()
    embedder = InMemoryVoyageEmbedder(settings)
    hashes = load_reference_hashes()
    return questions, index, embedder, hashes


def _password_gate() -> bool:
    expected = os.getenv(DEMO_PASSWORD_ENV, "")
    if not expected:
        st.error(f"Missing required environment variable: {DEMO_PASSWORD_ENV}")
        return False
    if st.session_state.get("demo_authenticated") is True:
        return True
    supplied = st.text_input("Shared password", type="password")
    if st.button("Unlock"):
        if hmac.compare_digest(supplied, expected):
            st.session_state.demo_authenticated = True
            st.rerun()
        st.error("Incorrect password")
    return False


def _question_picker(
    questions: dict[str, QuestionRecord],
) -> QuestionRecord:
    ordered = sorted(questions.values(), key=_question_label)
    selected = st.selectbox(
        "Question (type to search)",
        ordered,
        format_func=_question_label,
    )
    return selected


def _question_label(question: QuestionRecord) -> str:
    first_line = next(
        (line.strip() for line in question.statement_content.splitlines() if line.strip()),
        "Untitled question",
    )
    return f"{first_line[:90]} · {question.difficulty} · {question.question_id[:8]}"


def _question_statement(question: QuestionRecord) -> None:
    with st.expander("Full question statement"):
        st.markdown(question.statement_content)


def _submission_editor(
    boilerplate: str,
    question_id: str,
    language: str,
) -> str:
    code_language = "python" if language == "PYTHON" else "cpp"
    st.subheader("Production boilerplate")
    st.caption(
        "Fill the target function body in the submission editor below. "
        "Keep any helper functions you add; they are part of the submitted code."
    )
    st.code(boilerplate, language=code_language)
    return st.text_area(
        "Submission editor",
        value=boilerplate,
        height=360,
        key=f"submission_{question_id}_{language}",
        help=(
            "This starts with the platform boilerplate. Replace the fill marker/pass "
            "inside every target function and add helpers where the language permits."
        ),
    )


def _render_result(score: DemoScore) -> None:
    st.divider()
    _result_header(score)
    left, right = st.columns(2)
    code_language = "python" if score.language == "PYTHON" else "cpp"
    with left:
        st.subheader("Full submitted code")
        st.code(score.completed_code, language=code_language)
        with st.expander("Detector-stripped code"):
            st.code(score.stripped_code, language=code_language)
    with right:
        st.subheader("Nearest generated AI reference")
        st.caption("Our generated AI bank — not a student submission.")
        if score.nearest_reference is None:
            st.warning("Nearest reference code is unavailable.")
        else:
            reference_code = (
                score.nearest_reference.raw_code
                or score.nearest_reference.stripped_code
                or ""
            )
            st.code(reference_code, language=code_language)
        st.info(NO_HIGHLIGHT_NOTE)
    if score.result.explanation_card is not None:
        _render_explanation(score.result.explanation_card)


def _result_header(score: DemoScore) -> None:
    result = score.result
    percent = result.display_match_percent
    band = _band(percent)
    if percent is None:
        st.metric("AI-reference match", "Not scored")
    else:
        st.markdown(
            f'<div class="match-card {band}"><div>AI-reference match</div>'
            f'<strong>{percent}%</strong></div>',
            unsafe_allow_html=True,
        )
    flag = (
        "Flagged (provisional)"
        if percent is not None and percent >= DISPLAY_MATCH_PERCENT_MIDPOINT
        else "Not flagged"
    )
    first, second, third = st.columns(3)
    first.metric("Provisional flag", flag)
    second.metric("Scoring status", result.status)
    third.metric("Confidence", _confidence_label(result))
    if result.reason:
        st.caption(f"Detector reason: {result.reason}.")
    st.caption(_token_band_caption(result, score.language))
    st.caption(
        "≥50% = flagged (provisional). The 40–60% band is explicitly "
        "borderline/review, not a crisp human/AI split."
    )
    cache_text = "in-memory cache hit" if score.cache_hit else "live Voyage embedding"
    st.caption(f"Embedding: {cache_text}. Nothing from this submission was written to disk.")


def _render_explanation(card: ExplanationCard) -> None:
    st.subheader("Explanation")
    for section in card.sections:
        with st.container(border=True):
            st.markdown(f"**{section.heading}**")
            st.write(section.body)
    st.caption(card.footer)


def _band(percent: int | None) -> str:
    if percent is None:
        return "unscored"
    if 40 <= percent <= 60:
        return "borderline"
    if percent > 60:
        return "high"
    return "low"


def _confidence_label(result: DetectionResult) -> str:
    routed = CONFIDENCE_LABEL_BY_STATUS.get(result.status)
    if routed is None:
        return result.confidence or STANDARD_CONFIDENCE_LABEL
    if result.confidence:
        return f"{routed} · {result.confidence}"
    return routed


def _token_band_caption(result: DetectionResult, language: str) -> str:
    floor = SIGNIFICANT_TOKEN_THRESHOLDS_BY_LANGUAGE[language]
    ceiling = SHORT_LOW_CONFIDENCE_MAX_TOKENS_BY_LANGUAGE[language]
    counted = "not counted" if result.token_count is None else str(result.token_count)
    return (
        f"Significant tokens: {counted}. {language} floor {floor} "
        f"(below → insufficient_evidence); short-code low-confidence band "
        f"{floor}–{ceiling}."
    )


def _styles() -> None:
    st.markdown(
        """
        <style>
        .match-card { padding: 1rem; border-radius: .6rem; margin: .5rem 0 1rem; }
        .match-card strong { font-size: 2.4rem; }
        .match-card.low { background: #e8f4ea; color: #174d25; }
        .match-card.borderline { background: #fff0c2; color: #664d00; }
        .match-card.high { background: #fde2e2; color: #7f1d1d; }
        .match-card.unscored { background: #eceff3; color: #374151; }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()

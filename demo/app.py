from __future__ import annotations

import hmac
import logging
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
    StyleSignalName,
)
from demo.wording import (
    REVIEWER_QUESTION,
    explanation_facts,
    hardcoded_sentences,
    plain_confidence,
    plain_status,
    polish_facts,
)
from nw_ai_code_detector.data_load import QuestionRecord
from nw_ai_code_detector.discount_layer import mean_pairwise_cosine_distance
from nw_ai_code_detector.index import ClusterKey
from nw_ai_code_detector.score_query import DetectionResult

LOGGER = logging.getLogger(__name__)
POC_BANNER = "POC / provisional — not validated"
# TODO(security): the shared-password gate is DISABLED for local convenience.
# It MUST be re-enabled before any shared or public deploy - without it, anyone who
# reaches the URL can spend our Voyage API key. The mechanism below is intact; set
# DEMO_REQUIRE_PASSWORD=1 (and DEMO_SHARED_PASSWORD) to turn it back on.
REQUIRE_PASSWORD_ENV = "DEMO_REQUIRE_PASSWORD"


def main() -> None:
    st.set_page_config(page_title="AI Code Detector POC", layout="wide")
    _styles()
    st.error(POC_BANNER, icon="⚠️")
    st.title("AI Code Detector Demo")
    st.caption(
        "Measures resemblance to our generated AI reference solutions. "
        "It does not prove authorship or report a cheating percentage."
    )
    if _password_required():
        if not _password_gate():
            st.stop()
    else:
        st.warning(
            "Password gate is OFF (local mode). Re-enable with "
            f"{REQUIRE_PASSWORD_ENV}=1 before sharing this URL — it protects the "
            "Voyage API key.",
            icon="🔓",
        )
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
    # A form commits the editor's current text when the button is clicked, so a
    # single click scores the pasted code (no Ctrl+Enter step).
    with st.form("score_form", clear_on_submit=False):
        raw_code = _submission_editor(boilerplate, question.question_id, language)
        submitted = st.form_submit_button("Score submission", type="primary")
    if submitted:
        if not raw_code.strip() or raw_code == boilerplate:
            st.warning("Paste a submission first — the editor still holds the boilerplate.")
        else:
            try:
                st.session_state.demo_score = score_demo_submission(
                    DemoScoreRequest(raw_code, question, language),
                    index,
                    embedder,
                    hashes,
                )
                st.session_state.pop("polished", None)
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


def _password_required() -> bool:
    return os.getenv(REQUIRE_PASSWORD_ENV, "").strip() in {"1", "true", "True", "yes"}


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
    _side_by_side(score)
    _render_explanation(score)


def _side_by_side(score: DemoScore) -> None:
    """Stripped submission beside the nearest reference, plus guidance on what to look at.

    Deliberately no line-level highlighting: the detector scores whole-code similarity,
    so marking "matching lines" would fabricate precision we don't have and would mostly
    highlight boilerplate and brackets that everyone writes identically."""
    code_language = "python" if score.language == "PYTHON" else "cpp"
    reference = score.nearest_reference
    st.subheader("Submission vs nearest AI reference")
    st.caption(
        "Both shown as the detector sees them (boilerplate stripped). Compare the "
        "approach as a whole — the detector measures overall similarity, not "
        "line-by-line copying."
    )
    left, right = st.columns(2)
    with left:
        st.markdown("**Submission** (detector-stripped)")
        st.code(score.stripped_code, language=code_language)
        with st.expander("Full submitted code, as pasted"):
            st.code(score.completed_code, language=code_language)
    with right:
        st.markdown("**Nearest AI reference** (detector-stripped)")
        if reference is None:
            st.info("No nearest reference is available for this submission.")
        else:
            st.code(reference.stripped_code or "", language=code_language)
            st.caption(
                f"AI-generated reference, not a student submission — persona "
                f"`{reference.persona}`, model `{reference.generator}`."
            )
    _reviewer_guidance(score)


def _reviewer_guidance(score: DemoScore) -> None:
    facts = _facts_for(score)
    with st.container(border=True):
        st.markdown("**What to look at**")
        if facts.get("similarity"):
            st.write(facts["similarity"])
        st.write(facts["question_canonicality"])
        if facts.get("human_signals"):
            st.write(facts["human_signals"])
        st.info(REVIEWER_QUESTION, icon="🔎")


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
    headline, reason = plain_status(result.status)
    confidence = plain_confidence(result.status)
    flag = (
        "Flagged (provisional)"
        if percent is not None and percent >= DISPLAY_MATCH_PERCENT_MIDPOINT
        else "Not flagged"
    )
    first, second = st.columns(2)
    first.metric("Provisional flag", flag)
    second.metric("Scoring status", headline)
    if reason:
        st.info(reason)
    # Confidence appears only as a caveat. Showing "High confidence" on every normal
    # result trains reviewers to ignore the field.
    if confidence:
        st.warning(f"{confidence}.", icon="⚠️")
    # Internal routing strings stay in logs, not on screen.
    LOGGER.info(
        "score status=%s reason=%s confidence=%s tokens=%s",
        result.status, result.reason, result.confidence, result.token_count,
    )
    st.caption(
        "≥50% = flagged (provisional). The 40–60% band is explicitly "
        "borderline/review, not a crisp human/AI split."
    )
    cache_text = "in-memory cache hit" if score.cache_hit else "live Voyage embedding"
    st.caption(f"Embedding: {cache_text}. Nothing from this submission was written to disk.")


def _facts_for(score: DemoScore) -> dict:
    result = score.result
    return explanation_facts(
        status=result.status,
        match_level=_match_level(result.display_match_percent),
        cluster_diversity=_cluster_diversity(score),
        commented_out_code=_signal_fired(result, StyleSignalName.COMMENTED_OUT_CODE),
    )


def _render_explanation(score: DemoScore) -> None:
    card = score.result.explanation_card
    if card is None:
        return
    facts = _facts_for(score)
    st.subheader("Explanation")
    cache_key = f"polished_{id(score)}"
    if cache_key not in st.session_state:
        with st.spinner("Writing the explanation…"):
            st.session_state[cache_key] = polish_facts(facts)
    polished = st.session_state[cache_key]
    for heading, sentence in polished or hardcoded_sentences(facts):
        with st.container(border=True):
            st.markdown(f"**{heading}**")
            st.write(sentence)
    st.caption(card.footer)
    st.caption(
        "Wording written by a small model from the facts above — it never sees the "
        "code or the match percentage. The percentage and this disclaimer are not "
        "model-generated."
        if polished else
        "Standard wording (the rephrasing model was unavailable)."
    )


def _cluster_diversity(score: DemoScore) -> float | None:
    """Read-only lookup for wording; the scorer computes its own value independently."""
    reference = score.nearest_reference
    if reference is None:
        return None
    try:
        _questions, index, _embedder, _hashes = _resources()
        cluster = index.get_cluster(ClusterKey(reference.question_id, score.language))
        return mean_pairwise_cosine_distance(cluster.vectors)
    except Exception:
        return None


def _signal_fired(result: DetectionResult, name: StyleSignalName) -> bool:
    return any(flag.name == name.value and flag.fired for flag in result.signals)


def _match_level(percent: int | None) -> str | None:
    if percent is None:
        return None
    if percent > 60:
        return "high"
    if percent >= 40:
        return "medium"
    return "low"


def _band(percent: int | None) -> str:
    if percent is None:
        return "unscored"
    if 40 <= percent <= 60:
        return "borderline"
    if percent > 60:
        return "high"
    return "low"


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

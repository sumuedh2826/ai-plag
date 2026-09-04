from __future__ import annotations

from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import streamlit as st
import streamlit.components.v1 as components

from tools.labeling.constants import (
    BLIND_RELABEL_QUEUE_PATH,
    MANUAL_LABELS_PATH,
    REVIEW_QUEUE_PATH,
    SINGLE_RELABEL_QUEUE_PATH,
    ManualReviewLabel,
)
from tools.labeling.labels_store import (
    ManualLabelWrite,
    first_unlabeled_index,
    load_labels,
    load_relabel_audit,
    save_label,
    save_relabel,
)
from nw_ai_code_detector.display_match import display_match_percent

SHORTCUT_SCRIPT = """
<script>
const doc = window.parent.document;
doc.addEventListener("keydown", (event) => {
  const tag = (event.target && event.target.tagName) || "";
  if (tag === "TEXTAREA" || tag === "INPUT") {
    return;
  }
  const mapping = {h: "HUMAN", a: "AI", u: "UNSURE"};
  const label = mapping[event.key];
  if (!label) {
    return;
  }
  const buttons = Array.from(doc.querySelectorAll("button"));
  const match = buttons.find((button) => button.innerText.trim() === label);
  if (match) {
    match.click();
  }
});
</script>
"""


def main() -> None:
    st.set_page_config(page_title="Manual labeling", layout="wide")
    queue_path, blind_relabel = _active_queue()
    if not queue_path.is_file():
        st.error(f"Missing review queue: {queue_path}")
        st.stop()
    queue = load_queue(queue_path)
    labels = load_labels()
    completed_ids = _completed_ids(labels, blind_relabel)
    _init_index(queue, completed_ids, blind_relabel)
    index = int(st.session_state.review_index)
    index = min(max(index, 0), len(queue) - 1)
    st.session_state.review_index = index
    item = queue[index]
    page = load_review_page(item)
    _render_progress(index, queue, completed_ids, queue_path, blind_relabel)
    _render_hints(page.item, blind_relabel)
    _render_statement(page)
    _render_codes(page, blind_relabel)
    _render_actions(page, queue, labels, blind_relabel)
    components.html(SHORTCUT_SCRIPT, height=0)


def _active_queue() -> tuple[Path, bool]:
    if SINGLE_RELABEL_QUEUE_PATH.is_file():
        return SINGLE_RELABEL_QUEUE_PATH, True
    if BLIND_RELABEL_QUEUE_PATH.is_file():
        return BLIND_RELABEL_QUEUE_PATH, True
    return REVIEW_QUEUE_PATH, False


def _completed_ids(
    labels: dict[str, dict[str, object]],
    blind_relabel: bool,
) -> dict[str, object]:
    if not blind_relabel:
        return labels
    return {str(item["record_id"]): item for item in load_relabel_audit()}


def _init_index(
    queue: list[QueueItem],
    completed_ids: dict[str, object],
    blind_relabel: bool,
) -> None:
    session_key = _index_session_key(blind_relabel)
    if session_key not in st.session_state:
        record_ids = [item.record_id for item in queue]
        st.session_state[session_key] = first_unlabeled_index(record_ids, completed_ids)
    st.session_state.review_index = st.session_state[session_key]


def _render_progress(
    index: int,
    queue: list[QueueItem],
    completed_ids: dict[str, object],
    queue_path: Path,
    blind_relabel: bool,
) -> None:
    labeled_count = sum(1 for item in queue if item.record_id in completed_ids)
    st.caption(f"{MANUAL_LABELS_PATH}")
    if queue_path == SINGLE_RELABEL_QUEUE_PATH:
        st.warning(
            "Single-record re-label: other labels are unchanged. Saving writes "
            "HUMAN/AI/UNSURE with a timestamped audit of the old → new change."
        )
    elif blind_relabel:
        st.warning(
            "Blind Python re-review: detector score, band, similarity, token count, "
            "and previous label are hidden."
        )
    st.progress(labeled_count / len(queue))
    st.write(f"Reviewed **{labeled_count}/{len(queue)}** · viewing **{index + 1}/{len(queue)}**")
    if labeled_count == len(queue):
        if queue_path == SINGLE_RELABEL_QUEUE_PATH:
            st.success("Re-label saved. Remove the single-record queue when you are done.")
        else:
            st.success("Blind re-review complete. Scores remain hidden in this UI.")


def _render_hints(item: QueueItem, blind_relabel: bool) -> None:
    if blind_relabel:
        st.info(f"language `{item.language}` · blind code-to-code review")
        return
    match_text = "yes" if item.exact_match_to_ai else "no"
    match_percent = display_match_percent(item.ai_nn_max_raw, item.language)
    st.info(
        f"language `{item.language}` · difficulty `{item.difficulty}` · "
        f"tokens `{item.significant_code_token_count}` · "
        f"match `{match_percent}%` · exact_match `{match_text}`"
    )


def _render_statement(page: ReviewPage) -> None:
    st.subheader("Question")
    st.markdown(page.statement_content or "_no statement_")
    st.subheader("Boilerplate")
    st.code(page.boilerplate or "", language=_code_language(page.item.language))


def _render_codes(page: ReviewPage, blind_relabel: bool) -> None:
    left, right = st.columns(2)
    language = _code_language(page.item.language)
    with left:
        st.subheader("Candidate")
        st.markdown("**raw_code**")
        st.code(page.raw_code, language=language)
        st.markdown("**stripped_code**")
        if page.strip_error:
            st.error(page.strip_error)
        else:
            st.code(page.stripped_code or "", language=language)
    with right:
        st.subheader("Nearest mixed-v1 AI reference")
        if page.nearest_ai_error:
            st.error(page.nearest_ai_error)
            return
        nearest = page.nearest_ai
        if nearest is None:
            st.warning("No nearest reference")
            return
        if not blind_relabel:
            st.caption(
                f"{nearest.source_filename} · match "
                f"{display_match_percent(nearest.similarity, page.item.language)}%"
            )
        st.markdown("**stripped_code**")
        st.code(nearest.stripped_code, language=language)
        st.markdown("**raw_output**")
        if nearest.raw_output:
            st.code(nearest.raw_output, language=language)
        else:
            st.write("_not present on this JSON_")


def _render_actions(
    page: ReviewPage,
    queue: list[QueueItem],
    labels: dict[str, dict[str, object]],
    blind_relabel: bool,
) -> None:
    existing = labels.get(page.item.record_id, {})
    hide_previous = _hide_previous_label(blind_relabel)
    default_notes = "" if hide_previous else str(existing.get("notes") or "")
    notes = st.text_area("Notes (optional)", value=default_notes, key=page.item.record_id)
    if existing and not hide_previous:
        st.caption(f"Current label: {existing.get('my_label')}")
    previous, human, ai, unsure = st.columns(4)
    with previous:
        st.button("PREVIOUS", on_click=_go_previous, disabled=st.session_state.review_index <= 0)
    with human:
        st.button(
            ManualReviewLabel.HUMAN.value,
            on_click=_apply_label,
            args=(page, ManualReviewLabel.HUMAN, queue, blind_relabel),
        )
    with ai:
        st.button(
            ManualReviewLabel.AI.value,
            on_click=_apply_label,
            args=(page, ManualReviewLabel.AI, queue, blind_relabel),
        )
    with unsure:
        st.button(
            ManualReviewLabel.UNSURE.value,
            on_click=_apply_label,
            args=(page, ManualReviewLabel.UNSURE, queue, blind_relabel),
        )


def _apply_label(
    page: ReviewPage,
    label: ManualReviewLabel,
    queue: list[QueueItem],
    blind_relabel: bool,
) -> None:
    notes = str(st.session_state.get(page.item.record_id) or "")
    write = ManualLabelWrite(
        record_id=page.item.record_id,
        qid=page.item.question_id,
        language=page.item.language,
        label=label,
        notes=notes,
    )
    if blind_relabel:
        save_relabel(write)
    else:
        save_label(write)
    if st.session_state.review_index < len(queue) - 1:
        st.session_state.review_index += 1
    session_key = _index_session_key(blind_relabel)
    st.session_state[session_key] = st.session_state.review_index


def _go_previous() -> None:
    st.session_state.review_index = max(int(st.session_state.review_index) - 1, 0)


def _index_session_key(blind_relabel: bool) -> str:
    if SINGLE_RELABEL_QUEUE_PATH.is_file():
        return "single_relabel_index"
    if blind_relabel:
        return "blind_relabel_index"
    return "review_index"


def _hide_previous_label(blind_relabel: bool) -> bool:
    return blind_relabel and not SINGLE_RELABEL_QUEUE_PATH.is_file()


def _code_language(language: str) -> str:
    if language == "PYTHON":
        return "python"
    return "cpp"


if __name__ == "__main__":
    main()

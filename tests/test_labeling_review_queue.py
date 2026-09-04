import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.labeling.constants import ManualReviewLabel
from tools.labeling.labels_store import (
    ManualLabelWrite,
    first_unlabeled_index,
    save_relabel,
)
from tools.labeling.review_data import load_queue
from tools.labeling.select_review_set import ReviewPoolRecord, select_review_records
from tools.labeling.select_scoreable_batch import (
    ScoreableBatchSpec,
    ScoreableReviewRecord,
    select_scoreable_batch,
)


def _row(index, language, difficulty, exact=False):
    return ReviewPoolRecord(
        record_id=f"candidate_human|q{index}|{language}|0|h{index}",
        question_id=f"q{index}",
        language=language,
        difficulty=difficulty,
        group_index=0,
        significant_code_token_count=100,
        stripped_hash=f"h{index}",
        ai_nn_max_raw=0.5 + (index % 50) / 100,
        exact_match_to_ai=exact,
    )


class ManualReviewSelectionTests(unittest.TestCase):
    def test_force_includes_exact_matches(self):
        rows = self._pool()
        exact_ids = {row.record_id for row in rows if row.exact_match_to_ai}
        selected = select_review_records(rows, 80, 200, 4)
        selected_ids = {row.record_id for row in selected}
        self.assertTrue(exact_ids.issubset(selected_ids))
        self.assertEqual(len(selected), 80)

    def test_fills_language_difficulty_floor(self):
        rows = self._pool()
        selected = select_review_records(rows, 80, 200, 4)
        counts = Counter((row.language, row.difficulty) for row in selected)
        for language in ("CPP", "PYTHON"):
            for difficulty in ("EASY", "MEDIUM", "HARD"):
                self.assertGreaterEqual(counts[(language, difficulty)], 4)

    def test_same_seed_is_stable(self):
        rows = self._pool()
        first = [row.record_id for row in select_review_records(rows, 80, 200, 4)]
        second = [row.record_id for row in select_review_records(rows, 80, 200, 4)]
        self.assertEqual(first, second)

    def test_resume_starts_at_first_unlabeled(self):
        record_ids = ["a", "b", "c"]
        self.assertEqual(first_unlabeled_index(record_ids, {"a": {}}), 1)
        self.assertEqual(first_unlabeled_index(record_ids, {"a": {}, "b": {}, "c": {}}), 2)

    def test_blind_queue_does_not_require_detector_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "queue.json"
            path.write_text(
                json.dumps(
                    {
                        "items": [
                            {
                                "order_index": 0,
                                "record_id": "candidate_human|q1|PYTHON|2|hash",
                                "question_id": "q1",
                                "language": "PYTHON",
                                "difficulty": "EASY",
                                "group_index": 2,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            item = load_queue(path)[0]
        self.assertEqual(item.ai_nn_max_raw, 0.0)
        self.assertEqual(item.significant_code_token_count, 0)
        self.assertFalse(item.exact_match_to_ai)

    def test_relabel_updates_label_and_appends_old_to_new_audit(self):
        existing = {
            "record": {
                "record_id": "record",
                "qid": "q1",
                "language": "PYTHON",
                "my_label": "HUMAN",
                "notes": "old",
                "timestamp": "old-time",
            }
        }
        write = ManualLabelWrite(
            "record",
            "q1",
            "PYTHON",
            ManualReviewLabel.UNSURE,
            "rechecked",
        )
        with tempfile.TemporaryDirectory() as directory:
            labels_path = Path(directory) / "labels.jsonl"
            audit_path = Path(directory) / "audit.jsonl"
            with (
                patch("tools.labeling.labels_store.load_labels", return_value=existing),
                patch("tools.labeling.labels_store.MANUAL_LABELS_PATH", labels_path),
                patch("tools.labeling.labels_store.RELABEL_AUDIT_PATH", audit_path),
            ):
                save_relabel(write)
            updated = json.loads(labels_path.read_text(encoding="utf-8"))
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
        self.assertEqual(updated["my_label"], "UNSURE")
        self.assertEqual(audit["old_label"], "HUMAN")
        self.assertEqual(audit["new_label"], "UNSURE")

    def _pool(self):
        rows = []
        index = 0
        for language in ("CPP", "PYTHON"):
            for difficulty in ("EASY", "MEDIUM", "HARD"):
                for inner in range(20):
                    exact = (
                        language == "PYTHON"
                        and difficulty == "EASY"
                        and inner < 5
                    )
                    rows.append(_row(index, language, difficulty, exact))
                    index += 1
        return rows


class ScoreableBatchSelectionTests(unittest.TestCase):
    def test_excludes_labeled_record_ids_and_locators(self):
        rows = self._scoreable_pool()
        blocked = {rows[0].record_id}
        locators = {(rows[1].question_id, rows[1].language, rows[1].group_index)}
        selected = select_scoreable_batch(rows, self._spec(), blocked, locators)
        selected_ids = {row.record_id for row in selected}
        self.assertNotIn(rows[0].record_id, selected_ids)
        self.assertNotIn(rows[1].record_id, selected_ids)

    def test_balances_language_difficulty_and_descriptive(self):
        rows = self._scoreable_pool()
        selected = select_scoreable_batch(rows, self._spec(), set(), set())
        self.assertEqual(len(selected), 150)
        languages = Counter(row.language for row in selected)
        self.assertEqual(languages["PYTHON"], 75)
        self.assertEqual(languages["CPP"], 75)
        cells = Counter((row.language, row.difficulty) for row in selected)
        for language in ("CPP", "PYTHON"):
            for difficulty in ("EASY", "MEDIUM", "HARD"):
                self.assertGreaterEqual(cells[(language, difficulty)], 15)
        flagged = sum(1 for row in selected if row.descriptive_raise)
        self.assertGreaterEqual(flagged, 20)
        self.assertLessEqual(flagged, 30)

    def test_same_seed_is_stable(self):
        rows = self._scoreable_pool()
        first = [row.record_id for row in select_scoreable_batch(rows, self._spec(), set(), set())]
        second = [row.record_id for row in select_scoreable_batch(rows, self._spec(), set(), set())]
        self.assertEqual(first, second)

    def _spec(self):
        return ScoreableBatchSpec(150, 15042, 75, 25, 18, 15, 12)

    def _scoreable_pool(self):
        rows = []
        index = 0
        for language in ("CPP", "PYTHON"):
            for difficulty in ("EASY", "MEDIUM", "HARD"):
                for inner in range(40):
                    flagged = inner < 8
                    rows.append(
                        ScoreableReviewRecord(
                            f"candidate_human|q{index}|{language}|0|h{index}",
                            f"q{index}",
                            language,
                            difficulty,
                            0,
                            120,
                            f"h{index}",
                            0.80 + (inner / 100),
                            False,
                            flagged,
                            0.5 if flagged else 0.1,
                        )
                    )
                    index += 1
        return rows


if __name__ == "__main__":
    unittest.main()

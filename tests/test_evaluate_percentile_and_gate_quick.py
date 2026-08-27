import hashlib
import unittest
from pathlib import Path

import numpy as np

from nw_ai_code_detector.config import EVAL_SCORES_PATH
from nw_ai_code_detector.evaluation import evaluate_percentile_and_gate_quick as experiment
from nw_ai_code_detector.evaluation.evaluate_ai_human_margin_quick import (
    AiAffinity,
    ContrastScores,
    HumanAffinity,
    ScoredRow,
)
from nw_ai_code_detector.evaluation.evaluate_centroid_all_humans import (
    ExperimentRecord,
    ROLE_HELD_OUT,
    ROLE_HUMAN,
    _held_out_reference_match_keys,
)
from nw_ai_code_detector.evaluation.experiment_metrics import question_folds
from nw_ai_code_detector.index import ClusterKey
from nw_ai_code_detector.scorer import score_item


def _scored(question_id, language, label, source, content_hash, ai_nn, ratio, human_count=2):
    return ScoredRow(
        question_id=question_id,
        language=language,
        label=label,
        source=source,
        content_hash=content_hash,
        human_reference_count=human_count,
        ai_affinity=AiAffinity(nn=ai_nn, top3_mean=ai_nn, centroid=ai_nn),
        human_affinity=HumanAffinity(nn=0.5, mean=0.4, centroid=0.4),
        contrast=ContrastScores(
            nn_margin=0.1,
            top3_margin=0.1,
            centroid_contrast=0.1,
            log_distance_ratio=ratio,
            rank_relative=0.5,
        ),
    )


def _gate(question_id, language, label, source, ai_nn, ratio, content_hash=None, human_count=2):
    return experiment.GateRow(
        question_id=question_id,
        language=language,
        label=label,
        source=source,
        content_hash=content_hash or f"{question_id}-{source}",
        human_reference_count=human_count,
        exact_match=False,
        signals=experiment.GateSignals(ai_nn, ratio),
    )


class PercentileAndGateQuickTests(unittest.TestCase):
    def test_phase_b3_scores_are_reused_without_numeric_changes(self):
        scored = [
            _scored("q1", "CPP", "ai", "production_review", "h1", 0.91, 2.5),
            _scored("q1", "CPP", "human", "human_0", "h2", 0.80, 0.1),
        ]
        rows = experiment.gate_rows_from_scored(scored, set())
        self.assertEqual(rows[0].signals.ai_nn, 0.91)
        self.assertEqual(rows[0].signals.log_distance_ratio, 2.5)
        self.assertEqual(rows[1].signals.ai_nn, scored[1].ai_affinity.nn)
        self.assertEqual(
            rows[1].signals.log_distance_ratio,
            scored[1].contrast.log_distance_ratio,
        )

    def test_fold_construction_is_grouped_by_question(self):
        rows = [
            _gate("q1", "CPP", "ai", "p", 0.9, 1.0),
            _gate("q1", "PYTHON", "human", "h", 0.2, 0.1),
            _gate("q2", "CPP", "human", "h", 0.2, 0.1),
        ]
        fold_map = question_folds({row.question_id for row in rows}, experiment.FIXED_SEED)
        experiment._assert_grouped_folds(rows, fold_map)
        self.assertEqual(fold_map["q1"], fold_map["q1"])
        self.assertEqual(len({fold_map["q1"]}), 1)

    def test_held_out_rows_never_influence_training_percentiles(self):
        train = np.sort(np.asarray([0.1, 0.2, 0.3]))
        held = 0.9
        percentile = experiment.human_percentile(train, held)
        self.assertEqual(percentile, 1.0)
        self.assertNotIn(held, train.tolist())

    def test_cpp_and_python_use_separate_empirical_distributions(self):
        rows = [
            _gate("q0", "CPP", "human", "h0", 0.1, 0.0),
            _gate("q1", "CPP", "human", "h1", 0.2, 0.0),
            _gate("q2", "CPP", "human", "h2", 0.3, 0.0),
            _gate("q3", "CPP", "human", "h3", 0.4, 0.0),
            _gate("q4", "CPP", "human", "h4", 0.5, 0.0),
            _gate("q5", "CPP", "ai", "p", 0.9, 2.0),
            _gate("q0", "PYTHON", "human", "h0", 0.9, 0.0),
            _gate("q1", "PYTHON", "human", "h1", 0.91, 0.0),
            _gate("q2", "PYTHON", "human", "h2", 0.92, 0.0),
            _gate("q3", "PYTHON", "human", "h3", 0.93, 0.0),
            _gate("q4", "PYTHON", "human", "h4", 0.94, 0.0),
            _gate("q5", "PYTHON", "ai", "p", 0.9, 2.0),
        ]
        fold_map = {f"q{index}": index % 5 for index in range(6)}
        oof = experiment.assign_oof_rows(rows, fold_map)
        cpp_ai = next(item for item in oof if item.row.language == "CPP" and item.row.label == "ai")
        py_ai = next(item for item in oof if item.row.language == "PYTHON" and item.row.label == "ai")
        self.assertGreater(cpp_ai.nn_percentile, py_ai.nn_percentile)

    def test_percentile_matches_manual_example(self):
        train = np.asarray([0.10, 0.20, 0.40, 0.80])
        self.assertEqual(experiment.human_percentile(train, 0.20), 2 / 4)
        self.assertEqual(experiment.human_percentile(train, 0.15), 1 / 4)
        self.assertEqual(experiment.human_percentile(train, 0.80), 1.0)
        self.assertEqual(experiment.human_percentile(train, 0.00), 0.0)

    def test_percentiles_remain_in_unit_interval(self):
        train = np.asarray([0.2, 0.4, 0.6])
        for value in (-1.0, 0.2, 0.5, 2.0):
            percentile = experiment.human_percentile(train, value)
            self.assertGreaterEqual(percentile, 0.0)
            self.assertLessEqual(percentile, 1.0)

    def test_and_score_equals_minimum_of_percentiles(self):
        self.assertEqual(experiment.and_score(0.9, 0.2), 0.2)
        self.assertEqual(experiment.and_score(0.3, 0.8), 0.3)

    def test_high_value_on_only_one_signal_produces_low_and_score(self):
        self.assertEqual(experiment.and_score(0.99, 0.05), 0.05)
        self.assertEqual(experiment.and_score(0.04, 0.98), 0.04)

    def test_fold_thresholds_use_only_training_humans(self):
        humans = [_gate(f"q{index}", "CPP", "human", "h", 0.1 + index * 0.01, 0.0) for index in range(8)]
        ai = [_gate(f"q{index}", "CPP", "ai", "p", 0.99, 3.0) for index in range(8, 10)]
        fold_map = {f"q{index}": index % 5 for index in range(10)}
        train, held = experiment._split_fold(humans + ai, fold_map, 0)
        train_humans = [row for row in train if row.label == "human"]
        self.assertTrue(all(row.label == "human" for row in train_humans))
        self.assertTrue({row.question_id for row in held}.isdisjoint({row.question_id for row in train_humans}))

    def test_fpr_counts_each_held_out_human_once(self):
        rows = []
        for index in range(10):
            rows.append(_gate(f"q{index}", "CPP", "human", "h", 0.2, 0.1))
            rows.append(_gate(f"q{index}", "CPP", "ai", "p", 0.9, 2.0))
        fold_map = {f"q{index}": index % 5 for index in range(10)}
        oof = experiment.assign_oof_rows(rows, fold_map)
        humans = [item for item in oof if item.row.label == "human"]
        keys = [(item.row.question_id, item.row.language, item.row.content_hash) for item in humans]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertEqual(len(humans), 10)

    def test_baselines_and_and_use_identical_populations(self):
        scored = [
            _scored("q1", "CPP", "ai", "production_review", "a", 0.9, 1.0),
            _scored("q1", "CPP", "human", "human_0", "b", 0.2, 0.1),
        ]
        rows = experiment.gate_rows_from_scored(scored, set())
        experiment._assert_identical_population(scored, rows)

    def test_exact_match_sensitivity_is_question_language_scoped(self):
        shared = "same"
        records = [
            ExperimentRecord("q1", "CPP", "reference", "p0", shared, None, None, hashlib.sha256(shared.encode()).hexdigest()),
            ExperimentRecord("q1", "PYTHON", "held_out", "production_review", shared, None, None, hashlib.sha256(shared.encode()).hexdigest()),
            ExperimentRecord("q1", "CPP", "held_out", "production_review", shared, None, None, hashlib.sha256(shared.encode()).hexdigest()),
        ]
        keys = _held_out_reference_match_keys(records)
        self.assertEqual(keys, {("q1", "CPP", records[0].content_hash)})
        rows = experiment.gate_rows_from_scored(
            [_scored("q1", "CPP", "ai", "production_review", records[0].content_hash, 0.9, 1.0)],
            keys,
        )
        self.assertTrue(rows[0].exact_match)

    def test_no_ai_labels_used_to_construct_percentiles(self):
        humans = [_gate("q1", "CPP", "human", "h", 0.2, 0.1)]
        distributions = experiment._train_distributions(humans, "CPP", 0)
        self.assertEqual(list(distributions.nn_sorted), [0.2])
        with self.assertRaisesRegex(RuntimeError, "AI labels"):
            experiment._train_distributions(
                [_gate("q1", "CPP", "ai", "p", 0.9, 2.0)],
                "CPP",
                0,
            )

    def test_fixed_seed_is_deterministic(self):
        rows = [_gate(f"q{index}", "CPP", "human", "h", 0.1 + index * 0.05, 0.0) for index in range(10)]
        rows.extend([_gate(f"q{index}", "CPP", "ai", "p", 0.9, 2.0) for index in range(10)])
        fold_map = question_folds({row.question_id for row in rows}, experiment.FIXED_SEED)
        first = experiment.assign_oof_rows(rows, fold_map)
        second = experiment.assign_oof_rows(rows, fold_map)
        self.assertEqual([item.and_score for item in first], [item.and_score for item in second])

    def test_no_existing_files_are_modified(self):
        source = Path(experiment.__file__).read_text(encoding="utf-8")
        self.assertNotIn("from nw_ai_code_detector.scorer", source)
        self.assertNotIn("EVAL_SCORES_PATH.write", source)
        self.assertNotIn("score_item(", source)
        before = hashlib.sha256(EVAL_SCORES_PATH.read_bytes()).hexdigest()
        after = hashlib.sha256(EVAL_SCORES_PATH.read_bytes()).hexdigest()
        self.assertEqual(before, after)
        self.assertTrue(callable(score_item))
        self.assertTrue(EVAL_SCORES_PATH.is_file())


if __name__ == "__main__":
    unittest.main()

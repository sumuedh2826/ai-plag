import hashlib
import math
import unittest
from pathlib import Path

import numpy as np

from nw_ai_code_detector.config import EVAL_SCORES_PATH
from nw_ai_code_detector.embedder import l2_normalize
from nw_ai_code_detector.evaluation import evaluate_ai_human_margin_quick as experiment
from nw_ai_code_detector.evaluation.evaluate_centroid_all_humans import (
    ROLE_HELD_OUT,
    ROLE_HUMAN,
    ROLE_REFERENCE,
    ExperimentRecord,
)
from nw_ai_code_detector.index import ClusterKey
from nw_ai_code_detector.scorer import score_item


def _unit(values):
    return np.asarray(l2_normalize(values), dtype=np.float64)


def _record(question_id, language, role, source, text, user_id=None):
    return ExperimentRecord(
        question_id=question_id,
        language=language,
        role=role,
        source=source,
        text=text,
        user_id=user_id,
        raw_code=None,
        content_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
    )


def _item(question_id, language, role, source, text, vector, user_id=None):
    return experiment.VectorItem(
        _record(question_id, language, role, source, text, user_id),
        _unit(vector),
    )


def _ai_bank(question_id, language):
    vectors = np.vstack(
        [
            _unit([1, 0, 0, 0]),
            _unit([0.95, 0.05, 0, 0]),
            _unit([0.9, 0.1, 0, 0]),
            _unit([0, 1, 0, 0]),
            _unit([0, 0.9, 0.1, 0]),
            _unit([0, 0, 1, 0]),
        ]
    )
    return experiment.AiBank(
        key=ClusterKey(question_id, language),
        vectors=vectors,
        hashes=tuple(f"h{index}" for index in range(6)),
        centroid=experiment.centroid_vector(vectors),
    )


class AiHumanMarginQuickTests(unittest.TestCase):
    def test_ai_affinity_uses_only_same_question_language_refs(self):
        bank = _ai_bank("q1", "CPP")
        record = _record("q1", "CPP", ROLE_HELD_OUT, "production_review", "ai")
        resolved = experiment.resolve_ai_bank(
            experiment.PairLookup(record, ClusterKey("q1", "CPP"), {bank.key: bank})
        )
        self.assertEqual(resolved.key.question_id, "q1")
        self.assertEqual(resolved.key.language, "CPP")
        self.assertEqual(resolved.vectors.shape[0], 6)

    def test_human_affinity_uses_only_same_question_language_humans(self):
        humans = [
            _item("q1", "CPP", ROLE_HUMAN, "human_0", "h0", [0, 1, 0, 0]),
            _item("q1", "CPP", ROLE_HUMAN, "human_1", "h1", [0, 0, 1, 0]),
        ]
        folds = experiment.split_human_folds(humans)
        self.assertEqual(folds.key.question_id, "q1")
        self.assertEqual(folds.key.language, "CPP")
        for item in folds.fold_0 + folds.fold_1:
            self.assertEqual(item.record.question_id, "q1")
            self.assertEqual(item.record.language, "CPP")

    def test_cross_question_lookup_fails(self):
        bank = _ai_bank("q1", "CPP")
        record = _record("q2", "CPP", ROLE_HELD_OUT, "production_review", "ai")
        with self.assertRaisesRegex(RuntimeError, "Cross-question"):
            experiment.resolve_ai_bank(
                experiment.PairLookup(record, ClusterKey("q1", "CPP"), {bank.key: bank})
            )

    def test_cross_language_lookup_fails(self):
        humans = [
            _item("q1", "CPP", ROLE_HUMAN, "human_0", "h0", [1, 0, 0, 0]),
            _item("q1", "CPP", ROLE_HUMAN, "human_1", "h1", [0, 1, 0, 0]),
        ]
        folds = experiment.split_human_folds(humans)
        record = _record("q1", "PYTHON", ROLE_HUMAN, "human_0", "other")
        with self.assertRaisesRegex(RuntimeError, "Cross-language"):
            experiment.resolve_human_folds(
                experiment.PairLookup(record, ClusterKey("q1", "CPP"), {folds.key: folds})
            )

    def test_human_evaluation_never_references_self(self):
        humans = [
            _item("q1", "CPP", ROLE_HUMAN, "human_0", "h0", [1, 0, 0, 0]),
            _item("q1", "CPP", ROLE_HUMAN, "human_1", "h1", [0, 1, 0, 0]),
        ]
        folds = experiment.split_human_folds(humans)
        eval_item = folds.fold_0[0]
        refs = experiment.opposite_human_refs(folds, eval_item, 0)
        self.assertTrue(all(id(item.record) != id(eval_item.record) for item in refs))
        self.assertTrue(all(item.record.content_hash != eval_item.record.content_hash for item in refs))

    def test_same_stripped_hash_excluded_from_human_refs(self):
        shared = _item("q1", "CPP", ROLE_HUMAN, "human_0", "same", [1, 0, 0, 0])
        other = _item("q1", "CPP", ROLE_HUMAN, "human_1", "other", [0, 1, 0, 0])
        extra = experiment.VectorItem(
            _record("q1", "CPP", ROLE_HUMAN, "human_2", "same"),
            _unit([0, 0, 1, 0]),
        )
        folds = experiment.split_human_folds([shared, other])
        refs = experiment.opposite_human_refs(folds, extra, 1)
        self.assertTrue(all(item.record.content_hash != extra.record.content_hash for item in refs))

    def test_pairs_with_fewer_than_two_distinct_humans_are_excluded(self):
        duplicate_a = _item("q1", "CPP", ROLE_HUMAN, "human_0", "same", [1, 0, 0, 0])
        duplicate_b = experiment.VectorItem(
            _record("q1", "CPP", ROLE_HUMAN, "human_1", "same"),
            _unit([0, 1, 0, 0]),
        )
        self.assertIsNone(experiment.split_human_folds([duplicate_a, duplicate_b]))
        self.assertIsNone(experiment.split_human_folds([duplicate_a]))

    def test_both_human_folds_are_non_empty(self):
        humans = [
            _item("q1", "CPP", ROLE_HUMAN, "human_0", "h0", [1, 0, 0, 0]),
            _item("q1", "CPP", ROLE_HUMAN, "human_1", "h1", [0, 1, 0, 0]),
            _item("q1", "CPP", ROLE_HUMAN, "human_2", "h2", [0, 0, 1, 0]),
        ]
        folds = experiment.split_human_folds(humans)
        self.assertTrue(len(folds.fold_0) >= 1)
        self.assertTrue(len(folds.fold_1) >= 1)

    def test_every_eligible_human_is_evaluated_once(self):
        population = _tiny_population()
        rows, skipped = experiment.score_population(population)
        humans = [row for row in rows if row.label == "human"]
        keys = [(row.question_id, row.language, row.content_hash) for row in humans]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertEqual(skipped, 0)
        self.assertEqual(len(humans), 2)

    def test_held_out_ai_emitted_once_after_averaging(self):
        population = _tiny_population()
        rows, _skipped = experiment.score_population(population)
        ais = [row for row in rows if row.label == "ai"]
        keys = [(row.question_id, row.language, row.source, row.content_hash) for row in ais]
        self.assertEqual(len(ais), 1)
        self.assertEqual(len(keys), len(set(keys)))

    def test_relative_margin_formulas_match_manual_calculations(self):
        ai = experiment.AiAffinity(nn=0.9, top3_mean=0.8, centroid=0.7)
        human = experiment.HumanAffinity(nn=0.4, mean=0.3, centroid=0.2)
        scores = experiment.contrast_scores(ai, human)
        self.assertAlmostEqual(scores.nn_margin, 0.5)
        self.assertAlmostEqual(scores.top3_margin, 0.4)
        self.assertAlmostEqual(scores.centroid_contrast, 0.5)
        ai_d = max(1 - 0.9, 1e-6)
        human_d = max(1 - 0.4, 1e-6)
        self.assertAlmostEqual(scores.log_distance_ratio, math.log(human_d / ai_d))
        self.assertAlmostEqual(scores.rank_relative, 1 - ai_d / (ai_d + human_d))

    def test_higher_distance_ratio_means_more_ai_like(self):
        closer_to_ai = experiment.contrast_scores(
            experiment.AiAffinity(0.99, 0.9, 0.9),
            experiment.HumanAffinity(0.2, 0.1, 0.1),
        )
        closer_to_human = experiment.contrast_scores(
            experiment.AiAffinity(0.2, 0.2, 0.2),
            experiment.HumanAffinity(0.99, 0.9, 0.9),
        )
        self.assertGreater(closer_to_ai.log_distance_ratio, closer_to_human.log_distance_ratio)
        self.assertGreater(closer_to_ai.rank_relative, closer_to_human.rank_relative)

    def test_fpr_counts_each_human_once(self):
        population = _tiny_population()
        rows, _skipped = experiment.score_population(population)
        humans = [row for row in rows if row.label == "human"]
        self.assertEqual(len(humans), 2)
        self.assertEqual(
            len({(row.question_id, row.language, row.content_hash) for row in humans}),
            2,
        )

    def test_matched_population_nn_uses_same_rows_as_relative_methods(self):
        population = _tiny_population()
        rows, _skipped = experiment.score_population(population)
        nn_ids = [
            (row.question_id, row.language, row.label, row.source, row.content_hash)
            for row in rows
        ]
        self.assertEqual(len(nn_ids), len(set(nn_ids)))
        for row in rows:
            self.assertTrue(math.isfinite(row.ai_affinity.nn))
            self.assertTrue(math.isfinite(row.contrast.nn_margin))

    def test_exact_match_sensitivity_is_pair_scoped(self):
        shared = "same-code"
        records = [
            _record("q1", "CPP", ROLE_REFERENCE, "p0", shared),
            _record("q1", "PYTHON", ROLE_HELD_OUT, "production_review", shared),
            _record("q2", "CPP", ROLE_HELD_OUT, "pair_programming", shared),
            _record("q1", "CPP", ROLE_HELD_OUT, "production_review", shared),
        ]
        keys = experiment._held_out_reference_match_keys(records)
        self.assertEqual(keys, {("q1", "CPP", records[0].content_hash)})

    def test_fixed_seed_is_deterministic(self):
        first = experiment.contrast_scores(
            experiment.AiAffinity(0.8, 0.7, 0.6),
            experiment.HumanAffinity(0.5, 0.4, 0.3),
        )
        second = experiment.contrast_scores(
            experiment.AiAffinity(0.8, 0.7, 0.6),
            experiment.HumanAffinity(0.5, 0.4, 0.3),
        )
        self.assertEqual(first, second)
        humans = [
            _item("q1", "CPP", ROLE_HUMAN, "human_0", "h0", [1, 0, 0, 0]),
            _item("q1", "CPP", ROLE_HUMAN, "human_1", "h1", [0, 1, 0, 0]),
        ]
        self.assertEqual(
            experiment.split_human_folds(humans).fold_0[0].record.content_hash,
            experiment.split_human_folds(humans).fold_0[0].record.content_hash,
        )

    def test_no_existing_output_or_production_file_is_modified(self):
        source = Path(experiment.__file__).read_text(encoding="utf-8")
        self.assertNotIn("from nw_ai_code_detector.scorer", source)
        self.assertNotIn("EVAL_SCORES_PATH.write", source)
        self.assertNotIn("score_item(", source)
        before = (
            hashlib.sha256(EVAL_SCORES_PATH.read_bytes()).hexdigest()
            if EVAL_SCORES_PATH.is_file()
            else None
        )
        after = (
            hashlib.sha256(EVAL_SCORES_PATH.read_bytes()).hexdigest()
            if EVAL_SCORES_PATH.is_file()
            else None
        )
        self.assertEqual(before, after)
        self.assertTrue(callable(score_item))


def _tiny_population():
    ai_items = [
        _item("q1", "CPP", ROLE_REFERENCE, f"p{index}", f"ref{index}", vec)
        for index, vec in enumerate(
            [
                [1, 0, 0, 0],
                [0.95, 0.05, 0, 0],
                [0.9, 0.1, 0, 0],
                [0, 1, 0, 0],
                [0, 0.9, 0.1, 0],
                [0, 0, 1, 0],
            ]
        )
    ]
    humans = [
        _item("q1", "CPP", ROLE_HUMAN, "human_0", "human-a", [0, 0, 0, 1]),
        _item("q1", "CPP", ROLE_HUMAN, "human_1", "human-b", [0.1, 0, 0, 0.9]),
    ]
    held = [_item("q1", "CPP", ROLE_HELD_OUT, "production_review", "ai-code", [1, 0, 0, 0])]
    group = ai_items + humans + held
    bank, folds = experiment._build_pair_banks(group)
    coverage = experiment.MarginCoverage(
        eligible_pairs=(("q1", "CPP"),),
        excluded_few_humans=(),
        duplicates_removed=0,
        human_count_distribution={"2": 1},
        skipped_no_human_refs=0,
    )
    return experiment.MarginPopulation(tuple(group), {bank.key: bank}, {folds.key: folds}, coverage)


if __name__ == "__main__":
    unittest.main()

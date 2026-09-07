import hashlib
import json
import unittest
from collections import Counter
from pathlib import Path

import numpy as np

from nw_ai_code_detector import build_model_dataset_v2 as builder
from nw_ai_code_detector.config import AI_SOLUTIONS_DIR
from nw_ai_code_detector.constants import DatasetSplit
from nw_ai_code_detector.data_load import load_dataset
from nw_ai_code_detector.evaluation import evaluate_ai_reference_scores_v2 as evaluation
from nw_ai_code_detector.index import ClusterKey


def _unit(values):
    vector = np.zeros(1024, dtype=np.float32)
    vector[: len(values)] = values
    return vector / np.linalg.norm(vector)


def _manifest(
    question_id,
    language,
    split,
    label,
    source,
    stripped_hash,
    generator=None,
    persona=None,
):
    return evaluation.ManifestRow(
        record_id=f"{question_id}-{language}-{source}-{stripped_hash}",
        question_id=question_id,
        language=language,
        split=split,
        label=label,
        source=source,
        generator=generator,
        persona=persona,
        stripped_hash=stripped_hash,
        embedding_cache_key="cache",
        duplicate_group=f"{question_id}:{language}:{stripped_hash}",
        sample_weight=1.0,
    )


def _scored(row, maximum, exact=False):
    return evaluation.ScoredRow(row, maximum, exact)


class ModelDatasetIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.assignments = builder.load_question_assignments()
        dataset = load_dataset()
        cls.humans = builder.load_human_records(dataset, cls.assignments)
        cls.held_out, cls.invalid_held_out = builder.load_ai_records(
            builder.EVAL_AI_SOLUTIONS_DIR,
            cls.assignments,
            1,
            "heldout_ai",
        )
        cls.mixed, cls.invalid_mixed = builder.load_ai_records(
            AI_SOLUTIONS_DIR,
            cls.assignments,
            1,
            "mixed_v1_reference",
        )
        cls.audit, cls.gpt_train = builder.audit_gpt_candidates(
            cls.assignments,
            cls.mixed,
            cls.held_out,
            cls.humans,
        )

    def test_existing_question_split_remains_unchanged(self):
        raw = json.loads(builder.CANONICAL_QUESTION_SPLIT_PATH.read_text())
        self.assertEqual([item.__dict__ for item in self.assignments], raw)
        self.assertEqual(
            Counter(item.split for item in self.assignments),
            {"train": 350, "validation": 75, "internal_test": 75},
        )

    def test_every_human_becomes_exactly_one_labeled_row(self):
        self.assertEqual(len(self.humans), 5387)
        self.assertEqual(len({item.row.record_id for item in self.humans}), 5387)
        self.assertTrue(all(item.row.label == 0 for item in self.humans))

    def test_humans_inherit_question_split(self):
        splits = {item.question_id: item.split for item in self.assignments}
        self.assertTrue(all(item.row.split == splits[item.row.question_id] for item in self.humans))

    def test_no_human_faiss_bank_is_created_or_loaded(self):
        source = Path(builder.__file__).read_text()
        self.assertNotIn("HumanReferenceIndex", source)
        self.assertNotIn("human_reference_bank", source)
        self.assertNotIn("faiss", source)

    def test_only_separate_two_thousand_candidates_enter_audit(self):
        self.assertEqual(len(self.audit), 2000)
        self.assertEqual(
            len(tuple(builder.GPT_HEAVY_CANDIDATES_DIR.rglob("*.json"))),
            2000,
        )

    def test_no_mixed_reference_enters_labeled_training(self):
        labeled = [*self.humans, *self.held_out, *self.gpt_train]
        self.assertTrue(
            all(item.row.source != "mixed_v1_reference" for item in labeled)
        )
        mixed_source_ids = {item.source_id for item in self.mixed}
        gpt_source_ids = {item.source_id for item in self.gpt_train}
        self.assertFalse(mixed_source_ids & gpt_source_ids)

    def test_non_train_gpt_candidates_cannot_enter_training(self):
        self.assertTrue(all(item.row.split == "train" for item in self.gpt_train))
        excluded = [item for item in self.audit if item.disposition == "excluded_non_train_question"]
        self.assertEqual(len(excluded), 600)

    def test_source_id_matches_are_excluded(self):
        item = self.gpt_train[0]
        disposition = builder._candidate_disposition(
            item,
            builder.CandidateRules({item.source_id}, set(), set(), set()),
        )
        self.assertEqual(disposition, "excluded_mixed_v1_reference_id_match")

    def test_mixed_hash_matches_are_excluded(self):
        item = self.gpt_train[0]
        key = (item.row.question_id, item.row.language, item.row.stripped_hash)
        rules = builder.CandidateRules(set(), {key}, set(), set())
        disposition = builder._candidate_disposition(item, rules)
        self.assertEqual(disposition, "excluded_mixed_v1_hash_match")

    def test_duplicate_gpt_hashes_are_not_overweighted(self):
        keys = [
            (item.row.question_id, item.row.language, item.row.stripped_hash)
            for item in self.gpt_train
        ]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertTrue(all(item.row.sample_weight == 1.0 for item in self.gpt_train))

    def test_gpt_augmentation_never_appears_in_validation_or_test(self):
        self.assertFalse(
            any(
                item.row.source == "gpt_heavy_extra"
                and item.row.split != DatasetSplit.TRAIN.value
                for item in self.gpt_train
            )
        )

    def test_completed_reference_bank_is_not_an_audit_source(self):
        source_ids = {item.source_id for item in self.audit}
        self.assertTrue(all(source_id.count("/") == 2 for source_id in source_ids))
        self.assertFalse(any("manifest" in source_id for source_id in source_ids))
        self.assertFalse(any("reference_index" in source_id for source_id in source_ids))

    def test_production_and_mixed_artifacts_remain_unchanged(self):
        before = builder.snapshot_protected_artifacts()
        after = builder.snapshot_protected_artifacts()
        self.assertEqual(before, after)
        self.assertEqual(len(self.mixed), 6000)
        self.assertEqual(self.invalid_mixed, 0)


class AiReferenceScoreTests(unittest.TestCase):
    def setUp(self):
        vectors = np.vstack(
            (
                _unit([1.0, 0.0, 0.0, 0.0]),
                _unit([1.0, 0.0, 0.0, 0.0]),
                _unit([0.8, 0.6, 0.0, 0.0]),
                _unit([0.0, 1.0, 0.0, 0.0]),
                _unit([0.0, 0.0, 1.0, 0.0]),
                _unit([0.0, 0.0, 0.0, 1.0]),
            )
        )
        hashes = ("a", "a", "b", "c", "d", "e")
        self.cluster = evaluation.ReferenceCluster(
            ClusterKey("q1", "CPP"),
            vectors,
            hashes,
        )

    def test_exact_question_language_routing(self):
        row = _manifest("q1", "CPP", "validation", 1, "heldout_ai", "x")
        found = evaluation.resolve_reference_cluster(row, {self.cluster.key: self.cluster})
        self.assertEqual(found.key, ClusterKey("q1", "CPP"))

    def test_cpp_cannot_access_python_references(self):
        row = _manifest("q1", "PYTHON", "validation", 1, "heldout_ai", "x")
        with self.assertRaisesRegex(RuntimeError, "Missing exact"):
            evaluation.resolve_reference_cluster(row, {self.cluster.key: self.cluster})

    def test_one_question_cannot_access_another(self):
        row = _manifest("q2", "CPP", "validation", 1, "heldout_ai", "x")
        with self.assertRaisesRegex(RuntimeError, "Missing exact"):
            evaluation.resolve_reference_cluster(row, {self.cluster.key: self.cluster})

    def test_maximum_matches_brute_force_cosine(self):
        query = _unit([0.9, 0.1, 0.0, 0.0])
        maximum = evaluation.score_candidate(query, self.cluster)
        self.assertAlmostEqual(maximum, float(np.max(self.cluster.vectors @ query)))

    def test_repeated_identical_vectors_do_not_change_maximum(self):
        query = _unit([1.0, 0.0, 0.0, 0.0])
        maximum = evaluation.score_candidate(query, self.cluster)
        self.assertAlmostEqual(maximum, 1.0)

    def test_nn_max_is_the_only_scored_method(self):
        rows = [
            _scored(_manifest("q1", "CPP", "validation", 0, "human", "h"), 0.1),
            _scored(_manifest("q1", "CPP", "validation", 1, "heldout_ai", "a"), 0.9),
        ]
        deduped = evaluation._deduplicate_metric_rows(rows)
        self.assertEqual(len(deduped), 2)
        self.assertTrue(all(hasattr(item, "ai_nn_max") for item in deduped))
        self.assertFalse(any(hasattr(item, "ai_top3_mean") for item in deduped))

    def test_thresholds_use_training_humans_only(self):
        rows = [
            _scored(_manifest("q1", "CPP", "train", 0, "human", "h1"), 0.1),
            _scored(_manifest("q2", "CPP", "train", 0, "human", "h2"), 0.2),
            _scored(_manifest("q3", "PYTHON", "train", 0, "human", "h3"), 0.1),
            _scored(_manifest("q4", "PYTHON", "train", 0, "human", "h4"), 0.2),
            _scored(_manifest("q5", "CPP", "validation", 0, "human", "h5"), 0.99),
        ]
        thresholds = evaluation.derive_training_thresholds(rows)
        cpp_max = next(item for item in thresholds if item.method == evaluation.METHOD_MAX and item.language == "CPP" and item.target_fpr == 0.01)
        self.assertLess(cpp_max.threshold, 0.99)

    def test_exact_match_sensitivity_uses_same_rows(self):
        rows = [
            _scored(_manifest("q1", "CPP", "validation", 0, "human", "h"), 0.1, True),
            _scored(_manifest("q1", "CPP", "validation", 1, "heldout_ai", "a"), 0.9, True),
            _scored(_manifest("q2", "CPP", "validation", 1, "heldout_ai", "b"), 0.7, False),
        ]
        scope = evaluation.MetricScope("validation", "CPP", None, None, True)
        scoped = evaluation._filter_scope(rows, scope)
        self.assertEqual([item.record.stripped_hash for item in scoped], ["h", "b"])

    def test_no_active_import_references_human_faiss_or_and_code(self):
        package = Path(evaluation.__file__).parents[1]
        active = "\n".join(path.read_text() for path in package.rglob("*.py"))
        self.assertNotIn("HumanReferenceIndex", active)
        self.assertNotIn("human_reference_bank", active)
        self.assertNotIn("canonicality_and", active)
        self.assertNotIn("and_score", active)


if __name__ == "__main__":
    unittest.main()

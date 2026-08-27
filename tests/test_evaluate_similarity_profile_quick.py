import hashlib
import unittest
from pathlib import Path

import numpy as np

from nw_ai_code_detector.config import EVAL_SCORES_PATH
from nw_ai_code_detector.embedder import l2_normalize
from nw_ai_code_detector.evaluation import evaluate_similarity_profile_quick as experiment
from nw_ai_code_detector.evaluation.evaluate_centroid_all_humans import (
    ROLE_HELD_OUT,
    ROLE_HUMAN,
    ROLE_REFERENCE,
    ExperimentRecord,
)
from nw_ai_code_detector.evaluation.experiment_metrics import question_folds
from nw_ai_code_detector.index import ClusterKey
from nw_ai_code_detector.scorer import score_item


def _unit(values):
    return np.asarray(l2_normalize(values), dtype=np.float64)


def _record(question_id, language, role, source, text):
    return ExperimentRecord(
        question_id=question_id,
        language=language,
        role=role,
        source=source,
        text=text,
        user_id=None,
        raw_code=None,
        content_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
    )


def _cluster(question_id, language, vectors, models):
    matrix = np.vstack(vectors)
    return experiment.ReferenceCluster(
        key=ClusterKey(question_id, language),
        vectors=matrix,
        models=tuple(models),
        sources=tuple(f"p{index}" for index in range(len(vectors))),
        hashes=tuple(f"h{index}" for index in range(len(vectors))),
        centroid=experiment.centroid_vector(matrix),
        pairwise=tuple(
            float(value) for value in experiment.pairwise_reference_similarities(matrix)
        ),
    )


def _profiled(question_id, language, label, source, content_hash, features):
    return experiment.ProfiledSubmission(
        question_id=question_id,
        language=language,
        label=label,
        source=source,
        content_hash=content_hash,
        features=features,
    )


def _six_refs():
    return [
        _unit([1, 0, 0, 0]),
        _unit([0.9, 0.1, 0, 0]),
        _unit([0, 1, 0, 0]),
        _unit([0, 0.9, 0.1, 0]),
        _unit([0, 0, 1, 0]),
        _unit([0, 0, 0.9, 0.1]),
    ]


class SimilarityProfileQuickTests(unittest.TestCase):
    def test_only_same_question_language_references_are_used(self):
        cluster = _cluster("q1", "CPP", _six_refs(), ["m1", "m1", "m2", "m2", "m3", "m3"])
        record = _record("q1", "CPP", ROLE_HELD_OUT, "production_review", "code")
        resolved = experiment.resolve_cluster(
            experiment.ClusterLookup(record, ClusterKey("q1", "CPP"), {cluster.key: cluster})
        )
        self.assertEqual(resolved.key.question_id, "q1")
        self.assertEqual(resolved.key.language, "CPP")
        self.assertEqual(resolved.vectors.shape[0], 6)

    def test_cross_question_lookup_fails(self):
        cluster = _cluster("q1", "CPP", _six_refs(), ["m1", "m1", "m2", "m2", "m3", "m3"])
        record = _record("q2", "CPP", ROLE_HUMAN, "human_0", "code")
        with self.assertRaisesRegex(RuntimeError, "Cross-question"):
            experiment.resolve_cluster(
                experiment.ClusterLookup(
                    record,
                    ClusterKey("q1", "CPP"),
                    {cluster.key: cluster},
                )
            )

    def test_cross_language_lookup_fails(self):
        cluster = _cluster("q1", "CPP", _six_refs(), ["m1", "m1", "m2", "m2", "m3", "m3"])
        record = _record("q1", "PYTHON", ROLE_HUMAN, "human_0", "code")
        with self.assertRaisesRegex(RuntimeError, "Cross-language"):
            experiment.resolve_cluster(
                experiment.ClusterLookup(
                    record,
                    ClusterKey("q1", "CPP"),
                    {cluster.key: cluster},
                )
            )

    def test_six_similarities_produce_sorted_profile(self):
        raw = [0.1, 0.9, 0.4, 0.8, 0.2, 0.7]
        profile = experiment.sorted_profile_features(raw)
        self.assertEqual(profile["max_similarity"], 0.9)
        self.assertEqual(profile["second_similarity"], 0.8)
        self.assertEqual(profile["third_similarity"], 0.7)
        self.assertAlmostEqual(profile["top2_mean"], 0.85)
        self.assertAlmostEqual(profile["top3_mean"], 0.8)
        self.assertAlmostEqual(profile["minimum_similarity"], 0.1)
        self.assertAlmostEqual(profile["similarity_range"], 0.8)
        self.assertAlmostEqual(profile["max_second_gap"], 0.1)

    def test_cross_model_maxima_and_consensus(self):
        similarities = [0.9, 0.2, 0.8, 0.1, 0.4, 0.3]
        models = ["a", "a", "b", "b", "c", "c"]
        maxima = experiment.model_max_similarities(similarities, models)
        self.assertEqual(maxima, {"a": 0.9, "b": 0.8, "c": 0.4})
        consensus = experiment.model_consensus_features(similarities, models)
        self.assertAlmostEqual(consensus["model_consensus_mean"], (0.9 + 0.8 + 0.4) / 3)
        self.assertAlmostEqual(consensus["model_consensus_median"], 0.8)
        self.assertAlmostEqual(consensus["model_consensus_min"], 0.4)
        self.assertAlmostEqual(consensus["model_consensus_range"], 0.5)

    def test_cluster_pairwise_uses_fifteen_unique_pairs(self):
        pairs = experiment.pairwise_reference_similarities(np.vstack(_six_refs()))
        self.assertEqual(len(pairs), experiment.EXPECTED_PAIRWISE_COUNT)

    def test_held_out_reference_exact_matches_are_pair_scoped(self):
        shared = "same-code"
        records = [
            _record("q1", "CPP", ROLE_REFERENCE, "p0", shared),
            _record("q1", "PYTHON", ROLE_HELD_OUT, "production_review", shared),
            _record("q2", "CPP", ROLE_HELD_OUT, "pair_programming", shared),
            _record("q1", "CPP", ROLE_HELD_OUT, "production_review", shared),
        ]
        keys = experiment.same_cluster_exact_match_keys(records)
        self.assertEqual(keys, {("q1", "CPP", records[0].content_hash)})
        self.assertNotIn(("q1", "PYTHON", records[0].content_hash), keys)
        self.assertNotIn(("q2", "CPP", records[0].content_hash), keys)

    def test_grouped_folds_do_not_split_a_question(self):
        question_ids = {f"q{index}" for index in range(10)}
        fold_map = question_folds(question_ids, experiment.FIXED_SEED)
        self.assertEqual(len(fold_map), 10)
        self.assertEqual(set(fold_map), question_ids)
        for question_id in question_ids:
            self.assertIn(fold_map[question_id], range(5))

    def test_scaler_and_model_fit_only_on_training_fold(self):
        rows, fold_map = _classifier_rows()
        result = experiment.grouped_oof_classifier(
            experiment.ClassifierRunInput(rows, ("max_similarity",), fold_map)
        )
        for trace in result.traces:
            overlap = set(trace.train_question_ids) & set(trace.test_question_ids)
            self.assertEqual(overlap, set())
            self.assertTrue(trace.train_row_count > 0)
            self.assertTrue(trace.test_row_count > 0)

    def test_every_output_prediction_is_out_of_fold(self):
        rows, fold_map = _classifier_rows()
        result = experiment.grouped_oof_classifier(
            experiment.ClassifierRunInput(rows, ("max_similarity",), fold_map)
        )
        self.assertEqual(len(result.scores), len(rows))
        predicted_questions = []
        for trace in result.traces:
            predicted_questions.extend(trace.test_question_ids)
        self.assertEqual(set(predicted_questions), {row.question_id for row in rows})

    def test_human_rows_appear_once_in_fpr_calculations(self):
        rows, fold_map = _classifier_rows()
        humans = [row for row in rows if row.label == "human"]
        keys = [(row.question_id, row.language, row.content_hash) for row in humans]
        self.assertEqual(len(keys), len(set(keys)))
        result = experiment.grouped_oof_classifier(
            experiment.ClassifierRunInput(rows, ("max_similarity",), fold_map)
        )
        self.assertEqual(len(result.scores), len(rows))

    def test_persona_and_identifiers_are_not_classifier_features(self):
        overlap = experiment.FORBIDDEN_FEATURE_NAMES.intersection(
            experiment.CLASSIFIER_FEATURE_NAMES
        )
        self.assertEqual(overlap, set())
        with self.assertRaisesRegex(RuntimeError, "identifiers"):
            experiment.grouped_oof_classifier(
                experiment.ClassifierRunInput(
                    _classifier_rows()[0],
                    ("question_id", "max_similarity"),
                    {},
                )
            )

    def test_exact_match_excluded_model_is_retrained_without_excluded_positives(self):
        rows, _fold_map = _classifier_rows()
        match_keys = {(rows[0].question_id, rows[0].language, rows[0].content_hash)}
        self.assertEqual(rows[0].label, "ai")
        remaining = experiment.exclude_exact_match_positives(rows, match_keys)
        remaining_keys = {
            (row.question_id, row.language, row.content_hash)
            for row in remaining
            if row.label == "ai"
        }
        self.assertNotIn(
            (rows[0].question_id, rows[0].language, rows[0].content_hash),
            remaining_keys,
        )
        fold_map = question_folds({row.question_id for row in remaining}, experiment.FIXED_SEED)
        result = experiment.grouped_oof_classifier(
            experiment.ClassifierRunInput(tuple(remaining), ("max_similarity",), fold_map)
        )
        self.assertEqual(len(result.scores), len(remaining))

    def test_fixed_seed_is_deterministic(self):
        rows, fold_map = _classifier_rows()
        first = experiment.grouped_oof_classifier(
            experiment.ClassifierRunInput(rows, ("max_similarity",), fold_map)
        )
        second = experiment.grouped_oof_classifier(
            experiment.ClassifierRunInput(rows, ("max_similarity",), fold_map)
        )
        self.assertEqual(first.scores, second.scores)
        self.assertEqual(first.mean_coefficients, second.mean_coefficients)

    def test_production_scorer_and_eval_scores_remain_unchanged(self):
        source = Path(experiment.__file__).read_text(encoding="utf-8")
        self.assertNotIn("from nw_ai_code_detector.scorer", source)
        self.assertNotIn("EVAL_SCORES_PATH.write", source)
        self.assertNotIn("score_item(", source)
        before = hashlib.sha256(EVAL_SCORES_PATH.read_bytes()).hexdigest() if EVAL_SCORES_PATH.is_file() else None
        after = hashlib.sha256(EVAL_SCORES_PATH.read_bytes()).hexdigest() if EVAL_SCORES_PATH.is_file() else None
        self.assertEqual(before, after)
        self.assertTrue(callable(score_item))


def _classifier_rows():
    rows = []
    for question_index in range(10):
        question_id = f"q{question_index}"
        for language in ("CPP",):
            rows.append(
                _profiled(
                    question_id,
                    language,
                    "ai",
                    "production_review",
                    f"{question_id}-ai",
                    {"max_similarity": 0.9},
                )
            )
            rows.append(
                _profiled(
                    question_id,
                    language,
                    "human",
                    "human_0",
                    f"{question_id}-human",
                    {"max_similarity": 0.1},
                )
            )
    fold_map = question_folds({row.question_id for row in rows}, experiment.FIXED_SEED)
    return tuple(rows), fold_map


class NoNetworkExperimentModuleTests(unittest.TestCase):
    def test_module_does_not_import_voyage_or_openrouter(self):
        source = Path(experiment.__file__).read_text(encoding="utf-8")
        self.assertNotIn("voyageai", source)
        self.assertNotIn("openrouter", source)
        self.assertNotIn("openai", source.lower())


if __name__ == "__main__":
    unittest.main()

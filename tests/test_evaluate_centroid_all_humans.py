import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from nw_ai_code_detector.data_load import Dataset, QuestionRecord
from nw_ai_code_detector.embedder import l2_normalize
from nw_ai_code_detector.evaluation import evaluate_centroid_all_humans as evaluation
from nw_ai_code_detector.evaluation import experiment_metrics
from nw_ai_code_detector.evaluation.experiment_metrics import (
    ClusterScoreGroup,
    LabeledScore,
    conservative_threshold,
    language_calibrated_operating_points,
    macro_auroc_by_question,
    question_folds,
)
from nw_ai_code_detector.constants import EvaluationMode
from nw_ai_code_detector.index import ClusterKey
from nw_ai_code_detector.select_500 import EligibleQuestion
from nw_ai_code_detector.stripper import Language


def _record(
    question_id,
    language,
    role,
    source,
    text,
    raw_code=None,
):
    return evaluation.ExperimentRecord(
        question_id=question_id,
        language=language,
        role=role,
        source=source,
        text=text,
        user_id=None,
        raw_code=raw_code,
        content_hash=evaluation._content_hash(text),
    )


def _scored(
    question_id,
    language,
    label,
    source,
    content_hash,
    nn_score,
    centroid_score,
):
    return evaluation.ScoredEvalRow(
        question_id=question_id,
        language=language,
        label=label,
        source=source,
        content_hash=content_hash,
        user_id=None,
        nn_score=nn_score,
        topk_mean_score=nn_score - 0.01,
        centroid_score=centroid_score,
    )


def _dataset(question_id="q1"):
    question = QuestionRecord(
        question_id=question_id,
        difficulty="EASY",
        tags=(),
        primary_tag="UNTAGGED",
        is_function_completion=True,
        statement_content="",
        boilerplates={"CPP": "class solution { public: int f() { } };"},
    )
    return Dataset(
        questions={question_id: question},
        human_submission_counts={question_id: 4},
        coverage={},
        manifest={},
    )


class CentroidExperimentTests(unittest.TestCase):
    def test_all_human_loading_does_not_stop_after_two(self):
        mapping = {
            "q1:CPP": [
                {"raw_code": f"raw-{index}", "user_id": str(index)}
                for index in range(4)
            ]
        }
        with patch.object(
            evaluation,
            "_strip_human",
            side_effect=lambda row, _boilerplate, _language: row["raw_code"],
        ):
            records = evaluation._humans_for_pair_all(
                _dataset(),
                mapping,
                "q1",
                Language.CPP,
            )
        self.assertEqual([row.source for row in records], [
            "human_0",
            "human_1",
            "human_2",
            "human_3",
        ])

    def test_references_build_index_but_are_not_evaluation_rows(self):
        records, vectors = self._synthetic_cluster()
        index = evaluation._build_reference_index(
            evaluation._as_text_records(records),
            vectors,
        )
        rows = evaluation._score_rows(records, vectors, index)
        self.assertEqual(len(index.get_cluster(ClusterKey("q1", "CPP")).vector_ids), 6)
        self.assertFalse(any(row.source.startswith("ref") for row in rows))

    def test_held_out_vector_does_not_contribute_to_centroid(self):
        records, vectors = self._synthetic_cluster()
        index = evaluation._build_reference_index(
            evaluation._as_text_records(records),
            vectors,
        )
        centroid = evaluation._centroids_from_index(index)["q1:CPP"]
        expected = np.asarray(l2_normalize([3.0, 3.0]), dtype=np.float64)
        np.testing.assert_allclose(centroid, expected, atol=1e-6)
        self.assertFalse(np.allclose(centroid, vectors[6]))

    def test_normalized_centroid_matches_manual_calculation(self):
        records, vectors = self._synthetic_cluster()
        index = evaluation._build_reference_index(
            evaluation._as_text_records(records),
            vectors,
        )
        actual = evaluation._centroids_from_index(index)["q1:CPP"]
        references = np.asarray(vectors[:6], dtype=np.float64)
        expected = np.asarray(
            l2_normalize(np.mean(references, axis=0)),
            dtype=np.float64,
        )
        np.testing.assert_allclose(actual, expected, atol=1e-6)
        self.assertAlmostEqual(float(np.linalg.norm(actual)), 1.0, places=6)

    def test_global_cross_question_overlap_is_not_same_cluster(self):
        records = [
            _record("q1", "CPP", "human", "human_0", "same"),
            _record("q2", "CPP", "reference", "ref", "same"),
        ]
        report = evaluation._content_overlap_report(records)
        self.assertEqual(
            report["global_diagnostics"]["human_and_reference"],
            1,
        )
        self.assertEqual(
            report["same_cluster"]["human_and_reference"][
                "matching_cluster_hashes"
            ],
            0,
        )

    def test_same_question_language_overlap_is_detected_with_sources(self):
        records = [
            _record("q1", "CPP", "held_out", "production_review", "same"),
            _record("q1", "CPP", "reference", "terse", "same"),
        ]
        overlap = evaluation._content_overlap_report(records)["same_cluster"][
            "held_out_and_reference"
        ]
        self.assertEqual(overlap["matching_cluster_hashes"], 1)
        self.assertEqual(overlap["matches"][0]["left_source"], "production_review")
        self.assertEqual(overlap["matches"][0]["right_source"], "terse")

    def test_old_subset_matcher_rejects_unmatched_and_duplicates(self):
        old = [{
            "question_id": "q1",
            "language": "CPP",
            "label": "ai",
            "source": "p",
            "nn_score": 0.9,
            "topk_mean_score": 0.8,
        }]
        duplicate = _scored("q1", "CPP", "ai", "p", "h", 0.9, 0.7)
        alignment = evaluation._align_old_evaluation_rows(
            old,
            [duplicate, duplicate],
        )
        self.assertFalse(alignment["valid"])
        missing = evaluation._align_old_evaluation_rows(old, [])
        self.assertFalse(missing["valid"])
        self.assertEqual(len(missing["missing_original_identities"]), 1)

    def test_old_subset_matcher_reports_exact_row_score_differences(self):
        old = [{
            "question_id": "q1",
            "language": "CPP",
            "label": "ai",
            "source": "p",
            "nn_score": 0.9,
            "topk_mean_score": 0.8,
        }]
        recomputed = [_scored("q1", "CPP", "ai", "p", "h", 0.9, 0.7)]
        alignment = evaluation._align_old_evaluation_rows(old, recomputed)
        self.assertTrue(alignment["valid"])
        self.assertEqual(alignment["matched_recomputed_row_count"], 1)
        self.assertAlmostEqual(
            alignment["score_differences"]["topk_mean"][
                "max_absolute_difference"
            ],
            0.09,
        )

    def test_human_deduplication_is_pair_scoped(self):
        rows = [
            _scored("q1", "CPP", "human", "h0", "same", 0.1, 0.1),
            _scored("q1", "CPP", "human", "h1", "same", 0.1, 0.1),
            _scored("q2", "CPP", "human", "h0", "same", 0.1, 0.1),
        ]
        deduplicated = evaluation._dedupe_if_needed(
            rows,
            evaluation.VARIANT_DEDUPLICATED,
        )
        self.assertEqual(len(deduplicated), 2)

    def test_human_deduplication_never_removes_ai(self):
        rows = [
            _scored("q1", "CPP", "human", "h0", "same", 0.1, 0.1),
            _scored("q1", "CPP", "ai", "p1", "same", 0.9, 0.8),
            _scored("q1", "CPP", "ai", "p2", "same", 0.9, 0.8),
        ]
        deduplicated = evaluation._dedupe_if_needed(
            rows,
            evaluation.VARIANT_DEDUPLICATED,
        )
        self.assertEqual(sum(row.label == "ai" for row in deduplicated), 2)

    def test_exact_reference_match_positive_variant_is_sensitivity_only(self):
        rows = [
            _scored("q1", "CPP", "ai", "p1", "match", 0.9, 0.8),
            _scored("q1", "CPP", "ai", "p2", "other", 0.8, 0.7),
            _scored("q1", "CPP", "human", "h0", "match", 0.2, 0.3),
        ]
        filtered = evaluation._positive_variant_rows(
            rows,
            evaluation.POSITIVE_VARIANT_EXCLUDE_MATCH,
            {("q1", "CPP", "match")},
        )
        self.assertEqual([row.source for row in filtered], ["p2", "h0"])

    def test_macro_auroc_weights_pairs_equally(self):
        items = [
            LabeledScore("q1", "q1:CPP", "CPP", "ai", 0.9),
            LabeledScore("q1", "q1:CPP", "CPP", "human", 0.1),
            LabeledScore("q2", "q2:CPP", "CPP", "ai", 0.1),
        ]
        items.extend(
            LabeledScore("q2", "q2:CPP", "CPP", "human", 0.9)
            for _ in range(20)
        )
        value, count = macro_auroc_by_question(items)
        self.assertEqual(count, 2)
        self.assertEqual(value, 0.5)

    def test_metrics_use_each_human_once_not_pair_deltas(self):
        rows = [
            _scored("q1", "CPP", "ai", "a0", "a0", 0.9, 0.8),
            _scored("q1", "CPP", "ai", "a1", "a1", 0.8, 0.7),
            _scored("q1", "CPP", "human", "h0", "h0", 0.2, 0.3),
            _scored("q1", "CPP", "human", "h1", "h1", 0.1, 0.2),
            _scored("q1", "CPP", "human", "h2", "h2", 0.3, 0.4),
        ]
        metrics = experiment_metrics.method_metrics_bundle(
            "nn",
            evaluation._labeled(rows, "nn"),
            500,
        )
        pair_rows = evaluation._pair_delta_tables(rows)["rows"]
        self.assertEqual(metrics.n_negatives, 3)
        self.assertEqual(len(pair_rows), 6)

    def test_pair_deltas_include_full_within_pair_cartesian_product(self):
        rows = [
            _scored("q1", "CPP", "ai", "a0", "a0", 0.9, 0.8),
            _scored("q1", "CPP", "ai", "a1", "a1", 0.8, 0.7),
            _scored("q1", "CPP", "human", "h0", "h0", 0.2, 0.3),
            _scored("q1", "CPP", "human", "h1", "h1", 0.1, 0.2),
            _scored("q1", "CPP", "human", "h2", "h2", 0.3, 0.4),
        ]
        self.assertEqual(len(evaluation._pair_delta_tables(rows)["rows"]), 6)

    def test_grouped_folds_keep_question_rows_together(self):
        fold_map = question_folds({"q1", "q2", "q3", "q4", "q5"}, 500)
        items = [
            LabeledScore("q1", "q1:CPP", "CPP", "ai", 0.9),
            LabeledScore("q1", "q1:PYTHON", "PYTHON", "human", 0.1),
        ]
        self.assertEqual(
            {fold_map[item.question_id] for item in items},
            {fold_map["q1"]},
        )

    def test_conservative_threshold_handles_ties(self):
        negatives = [0.99, 0.99, 0.90, 0.80, 0.70]
        threshold = conservative_threshold(negatives, 0.01)
        achieved = float(np.mean(np.asarray(negatives) >= threshold))
        self.assertLessEqual(achieved, 0.01)

    def test_language_calibrated_combined_uses_language_thresholds(self):
        items = [
            LabeledScore("q1", "q1:CPP", "CPP", "human", 0.9),
            LabeledScore("q2", "q2:PYTHON", "PYTHON", "human", 0.2),
            LabeledScore("q3", "q3:CPP", "CPP", "ai", 0.95),
            LabeledScore("q4", "q4:PYTHON", "PYTHON", "ai", 0.3),
        ]
        result = language_calibrated_operating_points(items, 500)
        self.assertEqual(result["original_style"]["1%"].recall, 1.0)

    def test_bootstrap_includes_original_and_grouped_oof_intervals(self):
        groups = [
            ClusterScoreGroup(
                token=f"q{index}:CPP",
                language="CPP",
                nn_ai=(0.8,),
                nn_human=(0.2,),
                centroid_ai=(0.9,),
                centroid_human=(0.1,),
            )
            for index in range(6)
        ]
        first = experiment_metrics.cluster_bootstrap_deltas(groups, 500, 20)
        second = experiment_metrics.cluster_bootstrap_deltas(groups, 500, 20)
        self.assertEqual(first, second)
        self.assertIn("original_style_recall_at_1pct_fpr", first)
        self.assertIn(
            "grouped_oof_pooled_single_threshold_recall_at_1pct_fpr",
            first,
        )
        self.assertIn(
            "grouped_oof_language_calibrated_recall_at_1pct_fpr",
            first,
        )

    def test_results_payload_exercises_all_metric_variants(self):
        questions = []
        records = []
        rows = []
        for index in range(6):
            question_id = f"q{index}"
            language = "CPP" if index % 2 == 0 else "PYTHON"
            questions.append(
                EligibleQuestion(
                    question_id=question_id,
                    difficulty="EASY",
                    tags=(),
                    primary_tag="UNTAGGED",
                    coverage_count=1,
                    coverage_bucket="1-2",
                )
            )
            records.extend([
                _record(
                    question_id,
                    language,
                    "reference",
                    "ref",
                    f"ai-{index}",
                ),
                _record(
                    question_id,
                    language,
                    "held_out",
                    "production_review",
                    f"ai-{index}",
                ),
                _record(
                    question_id,
                    language,
                    "held_out",
                    "pair_programming",
                    f"held-{index}",
                ),
                _record(
                    question_id,
                    language,
                    "human",
                    "human_0",
                    f"human-{index}",
                    raw_code=f"raw-{index}",
                ),
            ])
            rows.extend([
                _scored(
                    question_id,
                    language,
                    "ai",
                    "production_review",
                    evaluation._content_hash(f"ai-{index}"),
                    0.9,
                    0.85,
                ),
                _scored(
                    question_id,
                    language,
                    "ai",
                    "pair_programming",
                    evaluation._content_hash(f"held-{index}"),
                    0.8,
                    0.82,
                ),
                _scored(
                    question_id,
                    language,
                    "human",
                    "human_0",
                    evaluation._content_hash(f"human-{index}"),
                    0.2,
                    0.25,
                ),
            ])
        sanity = {
            "groups_with_duplicate_user_ids": {
                "selection_rule": "synthetic",
            }
        }
        payload = evaluation._build_results_payload(
            questions,
            records,
            rows,
            sanity,
            {"available": False},
        )
        variants = payload["metrics"][evaluation.VARIANT_ALL_RECORDS]
        self.assertIn(evaluation.POSITIVE_VARIANT_ALL, variants)
        self.assertIn(evaluation.POSITIVE_VARIANT_EXCLUDE_MATCH, variants)
        combined = variants[evaluation.POSITIVE_VARIANT_ALL][
            "combined_operating_points"
        ]
        self.assertIn("pooled_single_threshold", combined)
        self.assertIn("language_calibrated_combined", combined)
        self.assertEqual(
            payload["meta"]["entropy_bins"]["coverage"][
                "missing_raw_code_records"
            ],
            0,
        )

    def test_incomplete_run_writes_coverage_only_and_does_not_score(self):
        records, _vectors = self._synthetic_cluster()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with self._main_patches(root, records, cached=False), patch.object(
                evaluation,
                "_score_rows",
            ) as score_rows:
                result = evaluation.main()
            self.assertEqual(result, 1)
            score_rows.assert_not_called()
            self.assertTrue((root / "coverage.json").is_file())
            self.assertTrue((root / "report.md").is_file())
            self.assertFalse((root / "scores.json").exists())

    def test_complete_synthetic_run_scores_and_writes_all_outputs(self):
        records, vectors, questions = self._complete_synthetic_records()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with self._main_patches(root, records, cached=True), patch.object(
                evaluation,
                "_load_cached_vectors",
                return_value=vectors,
            ), patch.object(
                evaluation,
                "_compare_old_eval_subset",
                return_value={"available": False},
            ), patch.object(
                evaluation,
                "_load_selected_questions",
                return_value=questions,
            ):
                result = evaluation.main()
            self.assertEqual(result, 0)
            expected = {
                "coverage.json",
                "scores.json",
                "report.md",
                "summary.csv",
                "pairs.csv",
            }
            self.assertEqual({path.name for path in root.iterdir()}, expected)

    def test_default_mode_still_stops_on_incomplete_human_coverage(self):
        records, _vectors = self._synthetic_cluster()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with self._main_patches(root, records, cached=False), patch.object(
                evaluation,
                "_score_rows",
            ) as score_rows:
                result = evaluation.main()
            self.assertEqual(result, 1)
            score_rows.assert_not_called()

    def test_cached_only_skips_uncached_humans(self):
        records, _vectors = self._synthetic_cluster()
        extra = _record("q1", "CPP", "human", "human_1", "uncached-human", raw_code="raw2")
        records = list(records) + [extra]
        cached_texts = {record.text for record in records if record.text != extra.text}
        with patch.object(
            evaluation,
            "cached_vector_for_text",
            side_effect=lambda text: (1.0, 0.0) if text in cached_texts else None,
        ):
            subset = evaluation._select_cached_human_subset(records)
        sources = [record.source for record in subset.records if record.role == "human"]
        self.assertEqual(sources, ["human_0"])
        self.assertEqual(subset.missing_humans_skipped, 1)

    def test_cached_only_fails_for_missing_reference_embedding(self):
        records, _vectors = self._synthetic_cluster()
        cached_texts = {
            record.text for record in records if record.source != "ref0"
        }
        with patch.object(
            evaluation,
            "cached_vector_for_text",
            side_effect=lambda text: (1.0, 0.0) if text in cached_texts else None,
        ):
            with self.assertRaises(RuntimeError):
                evaluation._require_cached_reference_embeddings(records)

    def test_cached_only_fails_for_missing_required_held_out_embedding(self):
        records, _vectors = self._synthetic_cluster()
        cached_texts = {
            record.text for record in records if record.role != "held_out"
        }
        with patch.object(
            evaluation,
            "cached_vector_for_text",
            side_effect=lambda text: (1.0, 0.0) if text in cached_texts else None,
        ):
            with self.assertRaises(RuntimeError):
                evaluation._select_cached_human_subset(records)

    def test_cross_question_scoring_is_impossible(self):
        record = _record("q1", "CPP", "human", "human_0", "human")
        with self.assertRaises(RuntimeError):
            evaluation._assert_same_cluster(record, ClusterKey("q2", "CPP"))

    def test_cross_language_scoring_is_impossible(self):
        record = _record("q1", "CPP", "human", "human_0", "human")
        with self.assertRaises(RuntimeError):
            evaluation._assert_same_cluster(record, ClusterKey("q1", "PYTHON"))

    def test_positive_from_pair_without_cached_humans_is_excluded(self):
        records, _vectors = self._synthetic_cluster()
        records = list(records) + [
            _record("q2", "CPP", "reference", f"ref{index}", f"q2-ref{index}")
            for index in range(6)
        ]
        records.append(_record("q2", "CPP", "held_out", "production_review", "q2-held"))
        records.append(_record("q2", "CPP", "human", "human_0", "q2-human"))
        cached_texts = {
            record.text
            for record in records
            if not (record.question_id == "q2" and record.role == "human")
        }
        with patch.object(
            evaluation,
            "cached_vector_for_text",
            side_effect=lambda text: (1.0, 0.0) if text in cached_texts else None,
        ):
            subset = evaluation._select_cached_human_subset(records)
        held_out_ids = {
            record.question_id
            for record in subset.records
            if record.role == "held_out"
        }
        self.assertEqual(held_out_ids, {"q1"})
        self.assertEqual(len(subset.pairs_excluded_no_cached_human), 1)

    def test_pair_deltas_are_same_question_language_only(self):
        rows = [
            _scored("q1", "CPP", "ai", "a0", "a0", 0.9, 0.8),
            _scored("q1", "PYTHON", "human", "h0", "h0", 0.2, 0.3),
            _scored("q1", "CPP", "human", "h1", "h1", 0.1, 0.2),
        ]
        deltas = evaluation._pair_delta_tables(rows)["rows"]
        self.assertEqual(len(deltas), 1)
        self.assertEqual(deltas[0]["language"], "CPP")

    def test_cached_only_outputs_do_not_overwrite_all_human_paths(self):
        paths = evaluation._output_paths_for_mode(EvaluationMode.CACHED_HUMANS_ONLY)
        all_human = evaluation._output_paths_for_mode(EvaluationMode.ALL_HUMANS)
        self.assertNotEqual(paths.scores, all_human.scores)
        self.assertNotEqual(paths.report, all_human.report)
        self.assertNotEqual(paths.coverage, all_human.coverage)
        self.assertTrue(str(paths.scores).endswith("centroid_cached_humans_scores.json"))

    def test_cached_only_metadata_and_warning_appear_in_outputs(self):
        records, _vectors, questions = self._complete_synthetic_records()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with self._main_patches(root, records, cached=True), patch.object(
                evaluation,
                "_parse_options",
                return_value=argparse.Namespace(
                    limit=None,
                    cached_humans_only=True,
                    bootstrap_iterations=0,
                ),
            ), patch.object(
                evaluation,
                "CENTROID_CACHED_HUMANS_COVERAGE_PATH",
                root / "cached_coverage.json",
            ), patch.object(
                evaluation,
                "CENTROID_CACHED_HUMANS_REPORT_PATH",
                root / "cached_report.md",
            ), patch.object(
                evaluation,
                "CENTROID_CACHED_HUMANS_SCORES_PATH",
                root / "cached_scores.json",
            ), patch.object(
                evaluation,
                "CENTROID_CACHED_HUMANS_SUMMARY_PATH",
                root / "cached_summary.csv",
            ), patch.object(
                evaluation,
                "CENTROID_CACHED_HUMANS_PAIR_DELTAS_PATH",
                root / "cached_pairs.csv",
            ), patch.object(
                evaluation,
                "_compare_old_eval_subset",
                return_value={"available": False},
            ), patch.object(
                evaluation,
                "_load_selected_questions",
                return_value=questions,
            ):
                result = evaluation.main()
            self.assertEqual(result, 0)
            self.assertFalse((root / "scores.json").exists())
            scores = json.loads((root / "cached_scores.json").read_text())
            self.assertEqual(scores["evaluation_mode"], "cached_humans_only")
            self.assertFalse(scores["is_final_all_human_evaluation"])
            report = (root / "cached_report.md").read_text()
            self.assertIn("Preliminary cached-subset evaluation", report)
            self.assertIn(
                "Bootstrap confidence intervals were skipped",
                report,
            )
            self.assertEqual(scores["meta"]["bootstrap_iterations"], 0)
            self.assertEqual(
                scores["meta"]["bootstrap_status"],
                "skipped_for_fast_preliminary_run",
            )
            ci = scores["bootstrap_centroid_minus_nn"]["CPP"][
                "grouped_oof_language_calibrated_recall_at_1pct_fpr"
            ]
            self.assertEqual(ci["status"], "not_calculated")
            self.assertIsNone(ci["low"])

    def test_bootstrap_iterations_zero_is_accepted(self):
        options = argparse.Namespace(bootstrap_iterations=0)
        self.assertEqual(evaluation._bootstrap_iterations(options), 0)

    def test_zero_iterations_bypasses_bootstrap_function(self):
        rows = [
            _scored("q1", "CPP", "ai", "a0", "a0", 0.9, 0.8),
            _scored("q1", "CPP", "human", "h0", "h0", 0.1, 0.2),
        ]
        with patch.object(
            evaluation,
            "cluster_bootstrap_deltas",
        ) as bootstrap:
            payload, status = evaluation._calculate_bootstrap(rows, 0)
        bootstrap.assert_not_called()
        self.assertEqual(status, "skipped_for_fast_preliminary_run")
        self.assertEqual(
            payload["CPP"]["original_style_pooled_auroc"]["status"],
            "not_calculated",
        )

    def test_nonzero_iterations_pass_exact_value_to_bootstrap(self):
        rows = [
            _scored("q1", "CPP", "ai", "a0", "a0", 0.9, 0.8),
            _scored("q1", "CPP", "human", "h0", "h0", 0.1, 0.2),
        ]
        with patch.object(
            evaluation,
            "cluster_bootstrap_deltas",
            return_value={},
        ) as bootstrap:
            _payload, status = evaluation._calculate_bootstrap(rows, 100)
        self.assertEqual(status, "completed")
        self.assertEqual(bootstrap.call_count, 3)
        self.assertTrue(
            all(call.args[2] == 100 for call in bootstrap.call_args_list)
        )

    def _synthetic_cluster(self):
        references = [
            _record("q1", "CPP", "reference", f"ref{index}", f"ref{index}")
            for index in range(6)
        ]
        records = references + [
            _record("q1", "CPP", "held_out", "production_review", "held"),
            _record("q1", "CPP", "human", "human_0", "human", raw_code="raw"),
        ]
        vectors = [
            tuple(l2_normalize([1.0, 0.0])),
            tuple(l2_normalize([1.0, 0.0])),
            tuple(l2_normalize([1.0, 0.0])),
            tuple(l2_normalize([0.0, 1.0])),
            tuple(l2_normalize([0.0, 1.0])),
            tuple(l2_normalize([0.0, 1.0])),
            tuple(l2_normalize([-1.0, 0.0])),
            tuple(l2_normalize([0.5, 0.5])),
        ]
        return records, vectors

    def _complete_synthetic_records(self):
        records = []
        vectors = []
        questions = []
        reference_vectors = [
            tuple(l2_normalize([1.0, 0.0])),
            tuple(l2_normalize([1.0, 0.0])),
            tuple(l2_normalize([1.0, 0.0])),
            tuple(l2_normalize([0.8, 0.2])),
            tuple(l2_normalize([0.8, 0.2])),
            tuple(l2_normalize([0.8, 0.2])),
        ]
        for index in range(6):
            question_id = f"q{index}"
            language = "CPP" if index % 2 == 0 else "PYTHON"
            questions.append(
                EligibleQuestion(
                    question_id=question_id,
                    difficulty="EASY",
                    tags=(),
                    primary_tag="UNTAGGED",
                    coverage_count=1,
                    coverage_bucket="1-2",
                )
            )
            for reference_index in range(6):
                records.append(
                    _record(
                        question_id,
                        language,
                        "reference",
                        f"ref{reference_index}",
                        f"ref-{index}-{reference_index}",
                    )
                )
            vectors.extend(reference_vectors)
            records.extend([
                _record(
                    question_id,
                    language,
                    "held_out",
                    "production_review",
                    f"production-{index}",
                ),
                _record(
                    question_id,
                    language,
                    "held_out",
                    "pair_programming",
                    f"pair-{index}",
                ),
                _record(
                    question_id,
                    language,
                    "human",
                    "human_0",
                    f"human-{index}",
                    raw_code=f"raw-{index}",
                ),
            ])
            vectors.extend([
                tuple(l2_normalize([0.9, 0.1])),
                tuple(l2_normalize([0.7, 0.3])),
                tuple(l2_normalize([0.1, 0.9])),
            ])
        return records, vectors, questions

    def _main_patches(self, root, records, cached):
        question = EligibleQuestion(
            question_id="q1",
            difficulty="EASY",
            tags=(),
            primary_tag="UNTAGGED",
            coverage_count=1,
            coverage_bucket="1-2",
        )
        patchers = [
            patch.object(evaluation, "CENTROID_ALL_HUMANS_COVERAGE_PATH", root / "coverage.json"),
            patch.object(evaluation, "CENTROID_ALL_HUMANS_REPORT_PATH", root / "report.md"),
            patch.object(evaluation, "CENTROID_ALL_HUMANS_SCORES_PATH", root / "scores.json"),
            patch.object(evaluation, "CENTROID_ALL_HUMANS_SUMMARY_PATH", root / "summary.csv"),
            patch.object(evaluation, "CENTROID_ALL_HUMANS_PAIR_DELTAS_PATH", root / "pairs.csv"),
            patch.object(
                evaluation,
                "_parse_options",
                return_value=argparse.Namespace(
                    limit=None,
                    cached_humans_only=False,
                    bootstrap_iterations=0,
                ),
            ),
            patch.object(evaluation, "load_dataset", return_value=_dataset()),
            patch.object(evaluation, "_load_selected_questions", return_value=[question]),
            patch.object(evaluation, "_collect_records", return_value=records),
            patch.object(
                evaluation,
                "cached_vector_for_text",
                side_effect=(lambda _text: (1.0, 0.0) if cached else None),
            ),
        ]

        class PatchContext:
            def __enter__(self):
                for patcher in patchers:
                    patcher.start()
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                for patcher in reversed(patchers):
                    patcher.stop()

        return PatchContext()

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import faiss

from nw_ai_code_detector.build_canonicality_split import (
    assign_human_roles,
    assign_question_splits,
    bank_group_count,
    build_heldout_rows,
    snapshot_protected_artifacts,
)
from nw_ai_code_detector.config import EVAL_SCORES_PATH
from nw_ai_code_detector.constants import (
    CANONICALITY_EMBEDDING_DIMENSION,
    DatasetSplit,
    HumanBankStatus,
    HumanRole,
)
from nw_ai_code_detector.embedder import VoyageEmbedder
from nw_ai_code_detector.evaluation.evaluate_centroid_all_humans import (
    ROLE_HELD_OUT,
    ROLE_HUMAN,
    ROLE_REFERENCE,
    ExperimentRecord,
)
from nw_ai_code_detector.human_reference_index import HumanReferenceIndex, cluster_offset_key
from nw_ai_code_detector.select_500 import EligibleQuestion


def _question(qid, difficulty):
    return EligibleQuestion(
        question_id=qid,
        difficulty=difficulty,
        tags=(),
        primary_tag="UNTAGGED",
        coverage_count=6,
        coverage_bucket="6+",
    )


def _human(qid, language, digest, source, user_id=None, text=None):
    return ExperimentRecord(
        question_id=qid,
        language=language,
        role=ROLE_HUMAN,
        source=source,
        text=text or digest,
        user_id=user_id,
        raw_code=None,
        content_hash=digest,
    )


def _held(qid, language, persona, digest):
    return ExperimentRecord(
        question_id=qid,
        language=language,
        role=ROLE_HELD_OUT,
        source=persona,
        text=digest,
        user_id=None,
        raw_code=None,
        content_hash=digest,
    )


def _unit(seed):
    rng = np.random.default_rng(seed)
    vector = rng.normal(size=CANONICALITY_EMBEDDING_DIMENSION).astype(np.float32)
    vector /= np.linalg.norm(vector)
    return vector


class CanonicalitySplitTests(unittest.TestCase):
    def test_question_split_is_deterministic(self):
        questions = self._fifty_questions()
        first = assign_question_splits(questions, 500)
        second = assign_question_splits(questions, 500)
        self.assertEqual(first, second)
        counts = {item.split: 0 for item in first}
        for item in first:
            counts[item.split] += 1
        self.assertEqual(counts[DatasetSplit.TRAIN.value], 350)
        self.assertEqual(counts[DatasetSplit.VALIDATION.value], 75)
        self.assertEqual(counts[DatasetSplit.INTERNAL_TEST.value], 75)

    def test_both_languages_share_the_question_split(self):
        questions = self._fifty_questions()
        splits = {item.question_id: item.split for item in assign_question_splits(questions, 500)}
        humans = [
            _human("q000", "CPP", "h1", "human_0"),
            _human("q000", "PYTHON", "h2", "human_0"),
        ]
        assigned = assign_human_roles(humans, splits)
        roles_splits = {item.record.language: item.split for item in assigned}
        self.assertEqual(roles_splits["CPP"], roles_splits["PYTHON"])
        self.assertEqual(roles_splits["CPP"], splits["q000"])

    def test_difficulty_stratification_keeps_all_levels(self):
        questions = self._fifty_questions()
        splits = assign_question_splits(questions, 500)
        by_split = {}
        for item in splits:
            by_split.setdefault(item.split, set()).add(item.difficulty)
        for split_name, difficulties in by_split.items():
            self.assertEqual(difficulties, {"EASY", "MEDIUM", "HARD"}, split_name)

    def test_human_role_assignment_is_deterministic(self):
        records = [_human("q1", "CPP", f"h{index}", f"human_{index}") for index in range(6)]
        first = assign_human_roles(records, {"q1": DatasetSplit.TRAIN.value})
        second = assign_human_roles(records, {"q1": DatasetSplit.TRAIN.value})
        self.assertEqual(
            [(item.record_id, item.role) for item in first],
            [(item.record_id, item.role) for item in second],
        )

    def test_six_groups_become_three_bank_and_three_labeled(self):
        records = [_human("q1", "CPP", f"h{index}", f"human_{index}") for index in range(6)]
        assigned = assign_human_roles(records, {"q1": DatasetSplit.TRAIN.value})
        bank = [item for item in assigned if item.role == HumanRole.REFERENCE_BANK.value]
        labeled = [item for item in assigned if item.role == HumanRole.LABELED.value]
        self.assertEqual(bank_group_count(6), 3)
        self.assertEqual(len(bank), 3)
        self.assertEqual(len(labeled), 3)

    def test_identical_hashes_never_cross_roles(self):
        records = [
            _human("q1", "CPP", "same", "human_0", user_id="u1"),
            _human("q1", "CPP", "same", "human_1", user_id="u2"),
            _human("q1", "CPP", "other", "human_2", user_id="u3"),
        ]
        assigned = assign_human_roles(records, {"q1": DatasetSplit.TRAIN.value})
        roles = {item.role for item in assigned if item.record.content_hash == "same"}
        same_roles = roles
        self.assertEqual(len(same_roles), 1)
        self.assertEqual(len({item.record.content_hash for item in assigned}), 2)

    def test_duplicate_bank_hashes_create_one_vector(self):
        from nw_ai_code_detector.build_canonicality_split import BankVector, collect_bank_vectors

        records = [
            _human("q1", "CPP", "same", "human_0", text="code-a"),
            _human("q1", "CPP", "same", "human_1", text="code-a"),
        ]
        assigned = assign_human_roles(records, {"q1": DatasetSplit.TRAIN.value})
        vector = _unit(1)
        with patch(
            "nw_ai_code_detector.build_canonicality_split.cached_vector_for_text",
            return_value=tuple(float(value) for value in vector),
        ), patch(
            "nw_ai_code_detector.build_canonicality_split.embedding_cache_key",
            return_value="cache-key",
        ):
            bank = collect_bank_vectors(assigned)
        self.assertEqual(len(bank), 1)
        self.assertEqual(bank[0].source_record_count, 2)
        self.assertIsInstance(bank[0], BankVector)

    def test_no_human_record_in_both_roles(self):
        records = [_human("q1", "CPP", f"h{index}", f"human_{index}") for index in range(5)]
        assigned = assign_human_roles(records, {"q1": DatasetSplit.VALIDATION.value})
        bank_ids = {item.record_id for item in assigned if item.role == HumanRole.REFERENCE_BANK.value}
        labeled_ids = {item.record_id for item in assigned if item.role == HumanRole.LABELED.value}
        self.assertFalse(bank_ids & labeled_ids)

    def test_labeled_humans_inherit_question_split(self):
        records = [_human("q1", "CPP", f"h{index}", f"human_{index}") for index in range(4)]
        assigned = assign_human_roles(records, {"q1": DatasetSplit.INTERNAL_TEST.value})
        labeled = [item for item in assigned if item.role == HumanRole.LABELED.value]
        self.assertTrue(labeled)
        self.assertTrue(all(item.split == DatasetSplit.INTERNAL_TEST.value for item in labeled))

    def test_held_out_ai_inherits_question_split(self):
        records = [
            _held("q1", "CPP", "production_review", "ai1"),
            ExperimentRecord(
                "q1",
                "CPP",
                ROLE_REFERENCE,
                "terse",
                "ref",
                None,
                None,
                "refhash",
            ),
        ]
        rows = build_heldout_rows(records, {"q1": DatasetSplit.VALIDATION.value})
        self.assertEqual(rows[0]["split"], DatasetSplit.VALIDATION.value)
        self.assertEqual(rows[0]["persona"], "production_review")

    def test_index_row_order_matches_manifest(self):
        directory = self._write_tiny_bank()
        loaded = HumanReferenceIndex.load(directory)
        manifest = [
            json.loads(line)
            for line in (directory / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
            if line
        ]
        self.assertEqual([row["faiss_row_id"] for row in manifest], list(range(len(manifest))))
        cluster = loaded.get_cluster("q1", "CPP")
        self.assertEqual(len(cluster.reference_ids), 2)

    def test_exact_question_language_routing(self):
        loaded = HumanReferenceIndex.load(self._write_tiny_bank())
        self.assertTrue(loaded.has_cluster("q1", "CPP"))
        self.assertTrue(loaded.has_cluster("q1", "PYTHON"))
        self.assertFalse(loaded.has_cluster("q2", "CPP"))

    def test_cpp_query_cannot_access_python_vectors(self):
        loaded = HumanReferenceIndex.load(self._write_tiny_bank())
        cpp = loaded.get_cluster("q1", "CPP").vectors
        python = loaded.get_cluster("q1", "PYTHON").vectors
        self.assertFalse(np.allclose(cpp, python))
        query = python[0]
        result = loaded.score_nn(query, "q1", "CPP")
        brute = float(np.max(cpp @ query))
        self.assertAlmostEqual(result.nn_score, brute)
        self.assertLess(result.nn_score, 0.999)

    def test_one_question_cannot_access_another(self):
        loaded = HumanReferenceIndex.load(self._write_tiny_bank())
        result = loaded.score_nn(_unit(9), "q2", "CPP")
        self.assertEqual(result.status, HumanBankStatus.UNAVAILABLE.value)
        self.assertIsNone(result.nn_score)
        with self.assertRaises(KeyError):
            loaded.get_cluster("q2", "CPP")

    def test_nn_matches_bruteforce_cosine(self):
        loaded = HumanReferenceIndex.load(self._write_tiny_bank())
        cluster = loaded.get_cluster("q1", "CPP")
        query = _unit(3)
        result = loaded.score_nn(query, "q1", "CPP")
        brute = float(np.max(cluster.vectors @ query))
        self.assertAlmostEqual(result.nn_score, brute, places=6)

    def test_missing_cluster_is_explicitly_unavailable(self):
        loaded = HumanReferenceIndex.load(self._write_tiny_bank())
        result = loaded.score_nn(_unit(4), "missing", "CPP")
        self.assertEqual(result.status, HumanBankStatus.UNAVAILABLE.value)
        self.assertEqual(result.reference_count, 0)

    def test_bank_vectors_are_finite_and_unit_normalized(self):
        loaded = HumanReferenceIndex.load(self._write_tiny_bank())
        cluster = loaded.get_cluster("q1", "CPP")
        self.assertTrue(np.isfinite(cluster.vectors).all())
        norms = np.linalg.norm(cluster.vectors, axis=1)
        self.assertTrue(np.allclose(norms, 1.0, atol=1e-3))

    def test_no_embedding_api_path_is_used(self):
        with patch.object(VoyageEmbedder, "embed_texts", side_effect=AssertionError("API called")):
            assign_question_splits(self._fifty_questions(), 500)
            assign_human_roles(
                [_human("q1", "CPP", "h1", "human_0"), _human("q1", "CPP", "h2", "human_1")],
                {"q1": DatasetSplit.TRAIN.value},
            )

    def test_s3_ready_artifacts_omit_raw_code_and_user_ids(self):
        directory = self._write_tiny_bank()
        for path in directory.iterdir():
            if path.suffix not in {".json", ".jsonl"}:
                continue
            text = path.read_text(encoding="utf-8")
            self.assertNotIn('"raw_code"', text)
            self.assertNotIn('"user_id"', text)
            self.assertNotIn('"email"', text)

    def test_existing_eval_scores_remain_unchanged(self):
        first = snapshot_protected_artifacts()
        second = snapshot_protected_artifacts()
        self.assertEqual(first["eval_scores"], second["eval_scores"])
        self.assertEqual(
            first["eval_scores"],
            "ea9eb8a7c14d932895d80424c364c53e05fa5327ddf55e26aff0138f3e278145",
        )
        self.assertTrue(EVAL_SCORES_PATH.is_file())

    def _fifty_questions(self):
        questions = []
        difficulties = ["EASY", "MEDIUM", "HARD"]
        for index in range(500):
            questions.append(_question(f"q{index:03d}", difficulties[index % 3]))
        return questions

    def _write_tiny_bank(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        directory = Path(tmp.name)
        cpp_a = _unit(1)
        cpp_b = _unit(2)
        py_a = _unit(5)
        from nw_ai_code_detector.build_canonicality_split import BankVector
        from nw_ai_code_detector.constants import (
            CANONICALITY_SPLIT_VERSION,
            HUMAN_REFERENCE_BANK_VERSION,
        )
        from nw_ai_code_detector.human_reference_index import CHECKSUM_FILES, DISTANCE_METRIC
        from nw_ai_code_detector.constants import VOYAGE_CODE_3_MODEL, VOYAGE_EMBED_INPUT_TYPE
        from hashlib import sha256

        vectors = [
            BankVector("q1", "CPP", "ref-cpp-a", "h1", "k1", 1, cpp_a, "train", False),
            BankVector("q1", "CPP", "ref-cpp-b", "h2", "k2", 1, cpp_b, "train", False),
            BankVector("q1", "PYTHON", "ref-py-a", "h3", "k3", 1, py_a, "train", True),
        ]
        matrix = np.asarray([item.vector for item in vectors], dtype=np.float32)
        index = faiss.IndexFlatIP(CANONICALITY_EMBEDDING_DIMENSION)
        index.add(matrix)
        faiss.write_index(index, str(directory / "index.faiss"))
        lines = []
        for row_id, item in enumerate(vectors):
            lines.append(
                json.dumps(
                    {
                        "faiss_row_id": row_id,
                        "reference_id": item.reference_id,
                        "question_id": item.question_id,
                        "language": item.language,
                        "stripped_hash": item.stripped_hash,
                        "embedding_cache_key": item.embedding_cache_key,
                        "source_record_count": item.source_record_count,
                        "split_version": CANONICALITY_SPLIT_VERSION,
                        "bank_version": HUMAN_REFERENCE_BANK_VERSION,
                    }
                )
            )
        (directory / "manifest.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
        offsets = {
            cluster_offset_key("q1", "CPP"): {"start": 0, "count": 2, "sparse": False},
            cluster_offset_key("q1", "PYTHON"): {"start": 2, "count": 1, "sparse": True},
        }
        (directory / "cluster_offsets.json").write_text(json.dumps(offsets), encoding="utf-8")
        metadata = {
            "artifact_type": "human_reference_bank",
            "bank_version": HUMAN_REFERENCE_BANK_VERSION,
            "dataset_split_version": CANONICALITY_SPLIT_VERSION,
            "embedding_model": VOYAGE_CODE_3_MODEL,
            "embedding_input_type": VOYAGE_EMBED_INPUT_TYPE,
            "embedding_dimension": CANONICALITY_EMBEDDING_DIMENSION,
            "distance_metric": DISTANCE_METRIC,
            "l2_normalized": True,
            "created_at": "2026-08-27T00:00:00+00:00",
            "total_vector_count": 3,
            "total_cluster_count": 2,
            "network_calls": False,
            "embeddings_generated": False,
        }
        (directory / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        checksums = {
            name: sha256((directory / name).read_bytes()).hexdigest()
            for name in CHECKSUM_FILES
        }
        (directory / "checksums.json").write_text(json.dumps(checksums), encoding="utf-8")
        return directory


if __name__ == "__main__":
    unittest.main()

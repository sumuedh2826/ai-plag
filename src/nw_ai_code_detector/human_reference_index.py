from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from collections.abc import Mapping, Sequence

import faiss
import numpy as np

from nw_ai_code_detector.constants import (
    CANONICALITY_EMBEDDING_DIMENSION,
    CANONICALITY_SPLIT_VERSION,
    HUMAN_REFERENCE_BANK_VERSION,
    HumanBankStatus,
    VOYAGE_CODE_3_MODEL,
    VOYAGE_EMBED_INPUT_TYPE,
)
from nw_ai_code_detector.evaluation.experiment_metrics import UNIT_NORM_TOLERANCE
from nw_ai_code_detector.index import ClusterKey

CHECKSUM_FILES = ("index.faiss", "manifest.jsonl", "cluster_offsets.json", "metadata.json")
DISTANCE_METRIC = "inner_product"


@dataclass(frozen=True)
class HumanCluster:
    key: ClusterKey
    vectors: np.ndarray
    reference_ids: tuple[str, ...]
    sparse: bool


@dataclass(frozen=True)
class HumanNnResult:
    status: str
    nn_score: float | None
    reference_count: int
    sparse: bool


class HumanReferenceIndex:
    def __init__(
        self,
        clusters: Mapping[str, HumanCluster],
        metadata: Mapping[str, object],
    ) -> None:
        self._clusters = dict(clusters)
        self._metadata = dict(metadata)

    @classmethod
    def load(cls, artifact_directory: Path) -> HumanReferenceIndex:
        metadata = _read_json(artifact_directory / "metadata.json")
        _validate_metadata(metadata)
        _validate_checksums(artifact_directory)
        offsets = _read_json(artifact_directory / "cluster_offsets.json")
        vectors = _load_faiss_vectors(artifact_directory / "index.faiss")
        rows = _read_manifest(artifact_directory / "manifest.jsonl")
        _validate_row_order(rows, vectors)
        clusters = _clusters_from_offsets(offsets, vectors, rows)
        return cls(clusters, metadata)

    def has_cluster(self, question_id: str, language: str) -> bool:
        return _cluster_token(question_id, language) in self._clusters

    def get_cluster(self, question_id: str, language: str) -> HumanCluster:
        token = _cluster_token(question_id, language)
        cluster = self._clusters.get(token)
        if cluster is None:
            raise KeyError(f"No human reference cluster for {question_id}/{language}")
        return cluster

    def score_nn(
        self,
        query_vector: Sequence[float],
        question_id: str,
        language: str,
    ) -> HumanNnResult:
        token = _cluster_token(question_id, language)
        cluster = self._clusters.get(token)
        if cluster is None:
            return HumanNnResult(HumanBankStatus.UNAVAILABLE.value, None, 0, False)
        query = np.asarray(query_vector, dtype=np.float32)
        score = float(np.max(cluster.vectors @ query))
        status = HumanBankStatus.SPARSE.value if cluster.sparse else HumanBankStatus.AVAILABLE.value
        return HumanNnResult(status, score, cluster.vectors.shape[0], cluster.sparse)


def cluster_offset_key(question_id: str, language: str) -> str:
    return json.dumps([question_id, language], separators=(",", ":"))


def parse_cluster_offset_key(token: str) -> ClusterKey:
    payload = json.loads(token)
    return ClusterKey(str(payload[0]), str(payload[1]))


def _cluster_token(question_id: str, language: str) -> str:
    return cluster_offset_key(question_id, language)


def _clusters_from_offsets(
    offsets: Mapping[str, Mapping[str, object]],
    vectors: np.ndarray,
    rows: Sequence[Mapping[str, object]],
) -> dict[str, HumanCluster]:
    clusters = {}
    for token, info in offsets.items():
        key = parse_cluster_offset_key(token)
        start = int(info["start"])
        count = int(info["count"])
        slice_vectors = vectors[start : start + count]
        slice_rows = rows[start : start + count]
        _assert_cluster_scope(key, slice_rows)
        clusters[token] = HumanCluster(
            key=key,
            vectors=slice_vectors,
            reference_ids=tuple(str(row["reference_id"]) for row in slice_rows),
            sparse=bool(info["sparse"]),
        )
    return clusters


def _assert_cluster_scope(key: ClusterKey, rows: Sequence[Mapping[str, object]]) -> None:
    for row in rows:
        if row["question_id"] != key.question_id or row["language"] != key.language:
            raise RuntimeError("Human-bank vector stored under the wrong question or language")


def _load_faiss_vectors(path: Path) -> np.ndarray:
    index = faiss.read_index(str(path))
    count = int(index.ntotal)
    dimension = int(index.d)
    vectors = np.zeros((count, dimension), dtype=np.float32)
    index.reconstruct_n(0, count, vectors)
    return vectors


def _read_manifest(path: Path) -> list[dict[str, object]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _validate_row_order(rows: Sequence[Mapping[str, object]], vectors: np.ndarray) -> None:
    if len(rows) != vectors.shape[0]:
        raise RuntimeError("Human-bank manifest length does not match FAISS rows")
    for index, row in enumerate(rows):
        if int(row["faiss_row_id"]) != index:
            raise RuntimeError("FAISS row IDs do not match the manifest order")
    if vectors.shape[1] != CANONICALITY_EMBEDDING_DIMENSION:
        raise RuntimeError("Unexpected human-bank embedding dimension")
    if not np.isfinite(vectors).all():
        raise RuntimeError("Human-bank vectors contain non-finite values")
    norms = np.linalg.norm(vectors, axis=1)
    if np.any(np.abs(norms - 1.0) > UNIT_NORM_TOLERANCE):
        raise RuntimeError("Human-bank vectors are not unit L2 normalized")


def _validate_metadata(metadata: Mapping[str, object]) -> None:
    if metadata.get("bank_version") != HUMAN_REFERENCE_BANK_VERSION:
        raise RuntimeError("Unexpected human-bank version")
    if metadata.get("dataset_split_version") != CANONICALITY_SPLIT_VERSION:
        raise RuntimeError("Unexpected dataset split version")
    if metadata.get("embedding_model") != VOYAGE_CODE_3_MODEL:
        raise RuntimeError("Unexpected embedding model")
    if metadata.get("embedding_input_type") != VOYAGE_EMBED_INPUT_TYPE:
        raise RuntimeError("Unexpected embedding input type")
    if int(metadata["embedding_dimension"]) != CANONICALITY_EMBEDDING_DIMENSION:
        raise RuntimeError("Unexpected embedding dimension")
    if metadata.get("distance_metric") != DISTANCE_METRIC:
        raise RuntimeError("Unexpected distance metric")
    if metadata.get("l2_normalized") is not True:
        raise RuntimeError("Human-bank metadata must declare L2 normalization")
    if metadata.get("network_calls") is not False:
        raise RuntimeError("Human-bank must be cache-only")
    if metadata.get("embeddings_generated") is not False:
        raise RuntimeError("Human-bank must not generate embeddings")


def _validate_checksums(directory: Path) -> None:
    checksums = _read_json(directory / "checksums.json")
    for name in CHECKSUM_FILES:
        digest = sha256((directory / name).read_bytes()).hexdigest()
        if checksums.get(name) != digest:
            raise RuntimeError(f"Checksum mismatch for {name}")


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))

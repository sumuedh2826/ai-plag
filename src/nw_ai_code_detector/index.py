from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Mapping, Sequence

import faiss
import numpy as np

from nw_ai_code_detector.config import REFERENCE_INDEX_DIR


@dataclass(frozen=True)
class ClusterKey:
    question_id: str
    language: str

    @property
    def token(self) -> str:
        return f"{self.question_id}:{self.language}"


@dataclass(frozen=True)
class ClusterVectors:
    key: ClusterKey
    vector_ids: tuple[int, ...]
    vectors: np.ndarray


class ReferenceIndex:
    def __init__(self, clusters: Mapping[str, ClusterVectors], dimension: int) -> None:
        self._clusters = dict(clusters)
        self._dimension = dimension
        self._indexes = {
            token: _build_flat_ip_index(cluster.vectors)
            for token, cluster in self._clusters.items()
        }

    def search(
        self,
        key: ClusterKey,
        query: Sequence[float],
        top_k: int,
    ) -> tuple[float, ...]:
        cluster = self._clusters.get(key.token)
        if cluster is None:
            raise KeyError(f"No reference cluster for {key.token}")
        index = self._indexes[key.token]
        neighbor_count = min(top_k, len(cluster.vector_ids))
        query_array = np.asarray([query], dtype=np.float32)
        scores, _indexes = index.search(query_array, neighbor_count)
        return tuple(float(score) for score in scores[0])

    def get_cluster(self, key: ClusterKey) -> ClusterVectors:
        cluster = self._clusters.get(key.token)
        if cluster is None:
            raise KeyError(f"No reference cluster for {key.token}")
        return cluster

    def cluster_tokens(self) -> tuple[str, ...]:
        return tuple(self._clusters)

    def save(self, directory: Path = REFERENCE_INDEX_DIR) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        manifest = {
            "dimension": self._dimension,
            "clusters": {
                token: {
                    "question_id": cluster.key.question_id,
                    "language": cluster.key.language,
                    "vector_ids": list(cluster.vector_ids),
                    "count": len(cluster.vector_ids),
                }
                for token, cluster in self._clusters.items()
            },
        }
        _write_json(directory / "manifest.json", manifest)
        for token, cluster in self._clusters.items():
            index_path = directory / f"{token.replace(':', '__')}.faiss"
            faiss.write_index(self._indexes[token], str(index_path))
            np.save(directory / f"{token.replace(':', '__')}.npy", cluster.vectors)

    @classmethod
    def from_clusters(
        cls,
        clusters: Sequence[ClusterVectors],
    ) -> ReferenceIndex:
        if not clusters:
            raise ValueError("Reference index requires at least one cluster")
        dimension = int(clusters[0].vectors.shape[1])
        mapping = {cluster.key.token: cluster for cluster in clusters}
        return cls(mapping, dimension)


def _build_flat_ip_index(vectors: np.ndarray) -> faiss.IndexFlatIP:
    index = faiss.IndexFlatIP(int(vectors.shape[1]))
    index.add(np.ascontiguousarray(vectors, dtype=np.float32))
    return index


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    temp_path = path.with_suffix(".json.tmp")
    temp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temp_path.replace(path)

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Mapping

from tools.labeling.constants import (
    FORBIDDEN_IDENTITY_KEYS,
    MANUAL_LABELS_PATH,
    RELABEL_AUDIT_PATH,
    ManualReviewLabel,
)


@dataclass(frozen=True)
class ManualLabelWrite:
    record_id: str
    qid: str
    language: str
    label: ManualReviewLabel
    notes: str


def load_labels(path: Path = MANUAL_LABELS_PATH) -> dict[str, dict[str, object]]:
    if not path.is_file():
        return {}
    labels: dict[str, dict[str, object]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        payload = json.loads(line)
        record_id = str(payload["record_id"])
        labels[record_id] = payload
    return labels


def save_label(write: ManualLabelWrite) -> None:
    payload = {
        "record_id": write.record_id,
        "qid": write.qid,
        "language": write.language,
        "my_label": write.label.value,
        "notes": write.notes,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    _reject_identity_keys(payload)
    labels = load_labels()
    labels[write.record_id] = payload
    _write_labels(labels)


def save_relabel(write: ManualLabelWrite) -> None:
    labels = load_labels()
    previous = labels.get(write.record_id)
    if previous is None:
        raise RuntimeError(f"Cannot re-label missing record: {write.record_id}")
    timestamp = datetime.now(timezone.utc).isoformat()
    updated = {
        "record_id": write.record_id,
        "qid": write.qid,
        "language": write.language,
        "my_label": write.label.value,
        "notes": write.notes,
        "timestamp": timestamp,
    }
    audit = {
        "record_id": write.record_id,
        "qid": write.qid,
        "language": write.language,
        "old_label": previous.get("my_label"),
        "new_label": write.label.value,
        "old_notes": previous.get("notes"),
        "new_notes": write.notes,
        "old_timestamp": previous.get("timestamp"),
        "relabel_timestamp": timestamp,
    }
    _reject_identity_keys(updated)
    _reject_identity_keys(audit)
    _append_relabel_audit(audit)
    labels[write.record_id] = updated
    _write_labels(labels)


def load_relabel_audit(path: Path = RELABEL_AUDIT_PATH) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def first_unlabeled_index(
    record_ids: list[str],
    labels: Mapping[str, object],
) -> int:
    for index, record_id in enumerate(record_ids):
        if record_id not in labels:
            return index
    return max(len(record_ids) - 1, 0)


def _write_labels(labels: Mapping[str, Mapping[str, object]]) -> None:
    MANUAL_LABELS_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(payload, sort_keys=True)
        for payload in labels.values()
    ]
    MANUAL_LABELS_PATH.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _append_relabel_audit(payload: Mapping[str, object]) -> None:
    RELABEL_AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RELABEL_AUDIT_PATH.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, sort_keys=True) + "\n")


def _reject_identity_keys(payload: Mapping[str, object]) -> None:
    overlap = FORBIDDEN_IDENTITY_KEYS.intersection(payload)
    if overlap:
        raise RuntimeError(f"Forbidden identity keys: {sorted(overlap)}")

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

from nw_ai_code_detector.config import (
    AI_SOLUTIONS_DIR,
    EVAL_SCORES_PATH,
    REFERENCE_INDEX_V1_DIR,
)

# data/ and outputs/ are gitignored, so git cannot catch an accidental clobber of the
# v1 bank. Snapshot the cheap identifying artifacts and compare around every v2 stage.
GUARDED_PATHS = (
    REFERENCE_INDEX_V1_DIR / "manifest.json",
    EVAL_SCORES_PATH,
)
SNAPSHOT_PATH = AI_SOLUTIONS_DIR.parent / ".refs_v2_v1_snapshot.json"


class V1MutatedError(RuntimeError):
    pass


def ensure_v1_untouched() -> None:
    """Fail loudly if a v2 stage has altered a v1 artifact."""
    current = _current_digests()
    if not SNAPSHOT_PATH.is_file():
        SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
        SNAPSHOT_PATH.write_text(json.dumps(current, indent=2), encoding="utf-8")
        return
    recorded = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
    changed = sorted(
        key for key, digest in current.items() if recorded.get(key) != digest
    )
    if changed:
        raise V1MutatedError(
            "refs_v2 must not modify the v1 bank; these changed: " + ", ".join(changed)
        )


def _current_digests() -> dict[str, str]:
    return {str(path): _digest(path) for path in GUARDED_PATHS}


def _digest(path: Path) -> str:
    if not path.is_file():
        return "absent"
    return sha256(path.read_bytes()).hexdigest()

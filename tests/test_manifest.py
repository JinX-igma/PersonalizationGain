"""The committed Stage 1 cache must match its recorded hash -- catches silent
corruption or an accidental hand-edit of a results/ CSV that the claim tests
wouldn't otherwise notice until their numbers quietly drifted.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def test_committed_cache_matches_manifest_hashes():
    manifest = json.loads((RESULTS / "manifest.json").read_text())
    for name, entry in manifest["files"].items():
        path = RESULTS / name
        assert path.exists(), f"manifest lists {name} but it is missing from results/"
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        assert actual == entry["sha256"], f"{name} does not match its manifest hash"

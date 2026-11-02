"""Answer provenance: which cached vision answers each artifact was made from.

Every artifact made from vision answers has a sidecar, `<artifact>.answers`: the answer keys it
was made from (cache paths like "llm/<key>.json", one per line, sorted), written before the
artifact itself. The artifacts (ANSWER_ARTIFACTS) are DVC outputs, so their sidecars are versioned
with them, and the answers a commit needs are the union of its live artifacts' sidecars
(sfzmaps.ship). Recording uses llm.recording().

Artifacts made before sidecars existed are covered by pipeline/cache-baseline.json, a committed
list of {artifact path: md5}: provenance unknown, accepted as they are (sfzmaps answers --replay
gives most of them true sidecars at $0; only the rest are listed). An artifact with neither a
sidecar nor a baseline entry is not current: it is read again.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from sfzmaps import config, llm

# Artifact file names made from vision answers (per sheet unless noted). Closed list: ship
# collects sidecars of these only.
ANSWER_ARTIFACTS = ("page.json", "labels.json", "district_labels.json", "district_legend.json",
                    "districts_raw.parquet", "height.parquet")
SUFFIX = ".answers"


def sidecar(artifact: Path) -> Path:
    return Path(artifact).with_name(Path(artifact).name + SUFFIX)


def validate(keys) -> list[str]:
    """Sorted keys; ValueError on anything that isn't a cache answer path (never ship a path
    a sidecar could smuggle in, like ../../.env)."""
    out = sorted(set(keys))
    bad = [k for k in out if not llm.ANSWER_KEY.match(k)]
    if bad:
        raise ValueError(f"not answer keys: {bad[:3]}")
    return out


def write(artifact: Path, keys) -> None:
    """Write artifact's sidecar atomically (before the artifact itself)."""
    p = sidecar(artifact)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text("".join(k + "\n" for k in validate(keys)))
    os.replace(tmp, p)


def read(artifact: Path) -> list[str] | None:
    try:
        return validate(line for line in sidecar(artifact).read_text().splitlines() if line)
    except FileNotFoundError:
        return None


def remove(artifact: Path) -> None:
    """Remove an artifact and its sidecar together."""
    Path(artifact).unlink(missing_ok=True)
    sidecar(artifact).unlink(missing_ok=True)


def baseline_path() -> Path:
    return config.PIPELINE_DIR / "cache-baseline.json"


def _md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _baseline() -> dict[str, str]:
    try:
        return json.loads(baseline_path().read_text())
    except (OSError, ValueError):
        return {}


def baseline_key(artifact: Path) -> str:
    return Path(artifact).resolve().relative_to(config.DATA_DIR.resolve()).as_posix()


def covered(artifact: Path) -> bool:
    """artifact's provenance is known: a sidecar, or a baseline entry for exactly these bytes."""
    if sidecar(artifact).exists():
        return True
    try:
        want = _baseline().get(baseline_key(artifact))
    except ValueError:                 # outside DATA_DIR (tests, demos): only sidecars count
        return False
    return want is not None and Path(artifact).exists() and _md5(Path(artifact)) == want


def consume(artifact: Path) -> None:
    """Fold an artifact's recorded answers into the current recording: the artifact that uses
    it (the series' district read) then names everything it was made from, whether the inner
    artifact was read now or reused, so its sidecar is complete and the same on every run."""
    for k in read(artifact) or []:
        llm._record(k)


def record(keys) -> None:
    """Add keys to the current recording (answers asked inside an inner recording that the
    outer result also depends on)."""
    for k in keys:
        llm._record(k)

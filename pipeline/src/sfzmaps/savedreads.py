"""The saved reads' images, shipped like any output.

pipeline/saved-reads (in git) records every answer the vision models gave the R&D steps; the images those answers were
given for (data/saved-reads/images, too big for git) are named in its images.manifest.tsv with their sha256.  The
frozen build stage `saved-reads` checks the folder against that manifest, and recording the stage puts the folder in
dvc.lock, so `sfzmaps ship` uploads it and `sfzmaps pull` brings it to a fresh checkout.  The images are made by
pipeline/scripts/saved_reads (a replay of the reading scripts), never by the build; nothing else in the build reads them
(build.CODE_EXCLUDE), so this module can change without restaling any other stage."""
import hashlib
import os
from pathlib import Path

from sfzmaps import config


def manifest_path() -> Path:
    return config.PIPELINE_DIR / "saved-reads" / "images.manifest.tsv"


def images_dir() -> Path:
    return config.DATA_DIR / "saved-reads" / "images"


def check(manifest: Path | None = None, images: Path | None = None) -> list[str]:
    """What is wrong with the images folder against the manifest: a missing image, one whose bytes differ, or a file
    the manifest doesn't name.  Empty when the folder is exactly the manifest's images."""
    manifest, images = manifest or manifest_path(), images or images_dir()
    want = {}
    for line in manifest.read_text().splitlines()[1:]:
        if line.strip():
            name, sha, _ = line.split("\t")
            want[name] = sha
    problems = []
    for name, sha in want.items():
        p = images / name
        if not p.is_file():
            problems.append(f"missing: {name}")
        elif hashlib.sha256(p.read_bytes()).hexdigest() != sha:
            problems.append(f"differs from the manifest: {name}")
    for root, _, files in os.walk(images):
        for f in files:
            name = (Path(root) / f).relative_to(images).as_posix()
            if name not in want:
                problems.append(f"not in the manifest: {name}")
    return problems

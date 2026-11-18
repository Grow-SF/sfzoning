"""Where things are in the saved-reads package (pipeline/saved-reads):

  answers/<read>.jsonl.gz                   one line per answer; <read> names the question, e.g. zoning-codes-2000
  answers/experiments-2000/<read>.jsonl.gz  the Nov 10-11 experiments, kept as recorded (package.FROZEN)
  answers/experiments-2000/text-positions/  where those experiments' text sits on each sheet (page px; not answers)
  how-each-was-asked/                       the exact questions, their templates, the answer formats, the recorded runs
  images.manifest.tsv                       every picture's readable name, sha256 and size (the pictures ship by DVC)
"""
from pathlib import Path

ANSWERS = "answers"
POSITIONS = "text-positions"
ASKED = "how-each-was-asked"
QUESTIONS = f"{ASKED}/questions.jsonl"
TEMPLATES = f"{ASKED}/question-templates.jsonl"
FORMATS = f"{ASKED}/answer-formats.jsonl"
RUNS = f"{ASKED}/recorded-runs.jsonl"
MANIFEST = "images.manifest.tsv"


def answer_files(pkg: Path) -> list[Path]:
    """every answer file of the package (not the text positions)"""
    return sorted(p for p in (pkg / ANSWERS).rglob("*.jsonl.gz") if POSITIONS not in p.parts)


def answer_file(pkg: Path, read: str) -> Path:
    return pkg / ANSWERS / f"{read}.jsonl.gz"

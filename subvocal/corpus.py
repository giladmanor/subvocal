"""Read and write the sentence and probe files."""

from __future__ import annotations

import csv
from pathlib import Path

from .phonetics import REPO_ROOT

PROMPTS = REPO_ROOT / "prompts"
SENTENCES_TXT = PROMPTS / "sentences.he.txt"
SENTENCES_TSV = PROMPTS / "sentences.tsv"
PROBE_TSV = PROMPTS / "probe.tsv"


def read_sentence_lines(path: Path = SENTENCES_TXT) -> list[tuple[str, str, str]]:
    """Return (id, section, text) for each sentence in the source text file."""
    rows = []
    section = ""
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            section = line.lstrip("# ").strip()
            continue
        rows.append((f"S{len(rows) + 1:03d}", section, line))
    return rows


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)

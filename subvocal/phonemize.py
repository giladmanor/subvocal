"""Phonemize prompts/sentences.he.txt and write prompts/sentences.tsv.

Usage: python -m subvocal.phonemize
"""

from __future__ import annotations

from .corpus import SENTENCES_TSV, read_sentence_lines, write_tsv
from .phonetics import Phonemizer


def main() -> None:
    ph = Phonemizer()
    rows = []
    for sid, section, text in read_sentence_lines():
        rows.append({"id": sid, "section": section, "text": text, "phonemes": ph(text)})
    write_tsv(SENTENCES_TSV, rows, ["id", "section", "text", "phonemes"])
    print(f"Wrote {len(rows)} sentences to {SENTENCES_TSV}")


if __name__ == "__main__":
    main()

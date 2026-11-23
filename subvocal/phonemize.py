"""Phonemize prompts/sentences.he.txt and write prompts/sentences.tsv.

The TSV gets the phonemes and the text with niqqud. The niqqud comes from phonikud.
To correct a word, add it to prompts/niqqud-fixes.tsv (columns: niqqud_wrong, niqqud, note).
The script then makes the phonemes of that sentence from the corrected niqqud.

Usage: python -m subvocal.phonemize
"""

from __future__ import annotations

from .corpus import PROMPTS, SENTENCES_TSV, read_sentence_lines, read_tsv, write_tsv
from phonikud import phonemize

from .phonetics import Phonemizer, clean_niqqud


def main() -> None:
    ph = Phonemizer()
    fixes_path = PROMPTS / "niqqud-fixes.tsv"
    fixes = {clean_niqqud(r["niqqud_wrong"]): clean_niqqud(r["niqqud"])
             for r in read_tsv(fixes_path)} if fixes_path.exists() else {}
    used = set()
    rows = []
    for sid, section, text in read_sentence_lines():
        niqqud = ph.niqqud(text)
        phonemes = ph(text)
        fixed = niqqud
        for wrong, right in fixes.items():
            if wrong in fixed:
                fixed = fixed.replace(wrong, right)
                used.add(wrong)
        if fixed != niqqud:
            niqqud, phonemes = fixed, phonemize(fixed)
        rows.append({"id": sid, "section": section, "text": text, "niqqud": niqqud, "phonemes": phonemes})
    for wrong in sorted(set(fixes) - used):
        print(f"Warning: niqqud fix not used: {wrong}")
    write_tsv(SENTENCES_TSV, rows, ["id", "section", "text", "niqqud", "phonemes"])
    print(f"Wrote {len(rows)} sentences to {SENTENCES_TSV}")


if __name__ == "__main__":
    main()

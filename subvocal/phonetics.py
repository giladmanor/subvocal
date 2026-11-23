"""Hebrew phoneme tools: phonemize text, split phonemes, and align two phoneme sequences.

The phonemizer is phonikud (https://github.com/thewh1teagle/phonikud).
It needs the ONNX model file. Download it with:

    curl -L -o models/phonikud-1.0.int8.onnx \
        https://huggingface.co/thewh1teagle/phonikud-onnx/resolve/main/phonikud-1.0.int8.onnx
"""

from __future__ import annotations

import os
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = REPO_ROOT / "models" / "phonikud-1.0.int8.onnx"

# The phoneme inventory of Modern Hebrew, as phonikud writes it.
CONSONANTS = [
    "p", "b", "t", "d", "k", "ɡ", "ʔ",
    "f", "v", "s", "z", "ʃ", "ʒ", "χ", "ʁ", "h",
    "ts", "tʃ", "dʒ",
    "m", "n", "l", "j", "w",
]
VOWELS = ["a", "e", "i", "o", "u"]
PHONEMES = CONSONANTS + VOWELS

# "w" occurs only in some loan words. Coverage checks do not require it.
REQUIRED = [p for p in PHONEMES if p != "w"]

_MULTI = ("tʃ", "dʒ", "ts")
_SINGLE = set(PHONEMES)
_ALIASES = {"g": "ɡ", "x": "χ", "r": "ʁ"}


def split_word(word: str) -> list[str]:
    """Split one phonemized word into phoneme tokens. Drop stress marks and other symbols."""
    out: list[str] = []
    i = 0
    while i < len(word):
        pair = word[i : i + 2]
        if pair in _MULTI:
            out.append(pair)
            i += 2
            continue
        ch = _ALIASES.get(word[i], word[i])
        if ch in _SINGLE:
            out.append(ch)
        i += 1
    return out


def split_text(phonemes: str) -> list[list[str]]:
    """Split a phonemized sentence into words, and each word into phoneme tokens."""
    words = [split_word(w) for w in phonemes.split()]
    return [w for w in words if w]


def flatten(words: list[list[str]]) -> list[str]:
    return [p for w in words for p in w]


class Phonemizer:
    """Hebrew text to IPA phonemes. Loads the ONNX model on the first call."""

    def __init__(self, model_path: str | os.PathLike = DEFAULT_MODEL):
        self.model_path = Path(model_path)
        self._model = None

    def _load(self):
        if self._model is None:
            if not self.model_path.exists():
                raise FileNotFoundError(
                    f"phonikud model not found: {self.model_path}. See subvocal/phonetics.py."
                )
            from phonikud_onnx import Phonikud

            self._model = Phonikud(str(self.model_path))
        return self._model

    @lru_cache(maxsize=4096)
    def diacritize(self, text: str) -> str:
        """Text with phonikud diacritics. This includes phonetic marks that are not standard niqqud."""
        return self._load().add_diacritics(text)

    def __call__(self, text: str) -> str:
        from phonikud import phonemize

        return phonemize(self.diacritize(text))

    def niqqud(self, text: str) -> str:
        """Text with standard niqqud, for a person to read."""
        return clean_niqqud(self.diacritize(text))


# phonikud marks that are not standard niqqud: "|" (prefix boundary), U+05AB (stress), U+05BD (meteg).
_PHONIKUD_MARKS = str.maketrans("", "", "|\u05ab\u05bd")


_MARKS = "[\u0591-\u05c7]"
# phonikud writes the holam or the qubuts on the letter before a bare vav ("חֹורֶף", "מְיֻוחָד").
# Standard niqqud puts them on the vav: holam male ("חוֹרֶף") and shuruk ("מְיוּחָד").
_HOLAM_BEFORE_VAV = re.compile(f"({_MARKS}*)[\u05b9\u05c7]({_MARKS}*)ו(?!{_MARKS})")
_QUBUTS_BEFORE_VAV = re.compile(f"({_MARKS}*)\u05bb({_MARKS}*)ו(?!{_MARKS})")


def clean_niqqud(text: str) -> str:
    """Remove the phonikud marks, use the standard forms, and normalize to Unicode NFC."""
    text = unicodedata.normalize("NFC", text.translate(_PHONIKUD_MARKS))
    text = _HOLAM_BEFORE_VAV.sub(lambda m: m[1] + m[2] + "ו\u05b9", text)
    text = _QUBUTS_BEFORE_VAV.sub(lambda m: m[1] + m[2] + "ו\u05bc", text)
    return unicodedata.normalize("NFC", text)


def align(ref: list[str], hyp: list[str]) -> list[tuple[str | None, str | None]]:
    """Align two phoneme sequences with minimum edit distance.

    Returns a list of (ref, hyp) pairs:
    - (p, p): correct;
    - (p, q): substitution of p with q;
    - (p, None): deletion of p;
    - (None, q): insertion of q.
    """
    n, m = len(ref), len(hyp)
    cost = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        cost[i][0] = i
    for j in range(1, m + 1):
        cost[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sub = 0 if ref[i - 1] == hyp[j - 1] else 1
            cost[i][j] = min(cost[i - 1][j - 1] + sub, cost[i - 1][j] + 1, cost[i][j - 1] + 1)

    pairs: list[tuple[str | None, str | None]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and cost[i][j] == cost[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            pairs.append((ref[i - 1], hyp[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and cost[i][j] == cost[i - 1][j] + 1:
            pairs.append((ref[i - 1], None))
            i -= 1
        else:
            pairs.append((None, hyp[j - 1]))
            j -= 1
    pairs.reverse()
    return pairs

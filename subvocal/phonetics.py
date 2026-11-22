"""Hebrew phoneme tools: phonemize text, split phonemes, and align two phoneme sequences.

The phonemizer is phonikud (https://github.com/thewh1teagle/phonikud).
It needs the ONNX model file. Download it with:

    curl -L -o models/phonikud-1.0.int8.onnx \
        https://huggingface.co/thewh1teagle/phonikud-onnx/resolve/main/phonikud-1.0.int8.onnx
"""

from __future__ import annotations

import os
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
    def __call__(self, text: str) -> str:
        from phonikud import phonemize

        return phonemize(self._load().add_diacritics(text))


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

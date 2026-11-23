"""Select the fixed test set: 30 sentences that the app never uses for training.

The test set measures the transcription model before and after the training. Each sound
gets about 10% of its occurrences in the test set (at least 2). A rare sound gives a
maximum of 20% of its occurrences, so that the training set keeps most of them.
The result is prompts/test-set.tsv. Make it one time. Do not change it after you record it.

Usage: python -m subvocal.testset
"""

from __future__ import annotations

import math
from collections import Counter

from .corpus import PROMPTS, SENTENCES_TSV, read_tsv, write_tsv
from .phonetics import REQUIRED, flatten, split_text

TEST_TSV = PROMPTS / "test-set.tsv"
SIZE = 30


def test_ids() -> set[str]:
    return {r["id"] for r in read_tsv(TEST_TSV)} if TEST_TSV.exists() else set()


def choose(rows: list[dict], size: int = SIZE) -> list[dict]:
    phon = {r["id"]: Counter(flatten(split_text(r["phonemes"]))) for r in rows}
    total = sum(phon.values(), Counter())
    want = {p: max(2, round(total[p] * 0.10)) for p in REQUIRED}
    cap = {p: max(2, math.floor(total[p] * 0.20)) for p in REQUIRED}
    have: Counter = Counter()
    chosen: list[dict] = []
    pool = list(rows)
    while pool and len(chosen) < size:
        def score(r):
            c = phon[r["id"]]
            if any(have[p] + c[p] > cap[p] for p in c if p in cap):
                return -1.0
            gain = sum(min(c[p], max(0, want[p] - have[p])) / want[p] for p in c if p in want)
            return gain / math.sqrt(sum(c.values()))
        best = max(pool, key=score)
        if score(best) < 0:
            break
        chosen.append(best)
        have.update(phon[best["id"]])
        pool.remove(best)
    return sorted(chosen, key=lambda r: r["id"])


def main() -> None:
    rows = read_tsv(SENTENCES_TSV)
    chosen = choose(rows)
    write_tsv(TEST_TSV, chosen, ["id", "section", "text", "niqqud", "phonemes"])
    c = Counter(p for r in chosen for p in flatten(split_text(r["phonemes"])))
    print(f"Wrote {len(chosen)} sentences to {TEST_TSV}")
    print("Missing sounds:", [p for p in REQUIRED if c[p] == 0] or "none")
    print("Rare sounds in the test set:", {p: c[p] for p in sorted(REQUIRED, key=lambda p: c[p])[:6]})


if __name__ == "__main__":
    main()

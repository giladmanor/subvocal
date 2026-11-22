"""Report the phoneme coverage of a sentence set.

Usage:
    python -m subvocal.coverage                     # all sentences
    python -m subvocal.coverage --ids plan.txt      # only the sentence IDs in a file
    python -m subvocal.coverage --min 10            # fail if a phoneme occurs fewer than 10 times
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from .corpus import SENTENCES_TSV, read_tsv
from .phonetics import CONSONANTS, REQUIRED, split_text


def count(rows: list[dict[str, str]]) -> dict[str, Counter]:
    """Count each phoneme in total, by position in the word, and each phoneme pair in a word."""
    total, initial, medial, final, pairs = Counter(), Counter(), Counter(), Counter(), Counter()
    for row in rows:
        for word in split_text(row["phonemes"]):
            for i, p in enumerate(word):
                total[p] += 1
                if len(word) == 1 or i == 0:
                    initial[p] += 1
                elif i == len(word) - 1:
                    final[p] += 1
                else:
                    medial[p] += 1
            for a, b in zip(word, word[1:]):
                pairs[(a, b)] += 1
    return {"total": total, "initial": initial, "medial": medial, "final": final, "pairs": pairs}


def clusters(pairs: Counter) -> Counter:
    """Consonant clusters: two consonants with no vowel between them."""
    cons = set(CONSONANTS)
    return Counter({k: v for k, v in pairs.items() if k[0] in cons and k[1] in cons})


def report(rows: list[dict[str, str]], minimum: int) -> bool:
    c = count(rows)
    print(f"Sentences: {len(rows)}")
    print(f"{'phoneme':>8} {'total':>6} {'initial':>8} {'medial':>7} {'final':>6}")
    ok = True
    for p in REQUIRED + ["w"]:
        mark = ""
        if p in REQUIRED and c["total"][p] < minimum:
            mark = "  < minimum"
            ok = False
        print(
            f"{p:>8} {c['total'][p]:>6} {c['initial'][p]:>8} "
            f"{c['medial'][p]:>7} {c['final'][p]:>6}{mark}"
        )
    cl = clusters(c["pairs"])
    print(f"\nPhoneme pairs in words: {len(c['pairs'])} different")
    print(f"Consonant clusters: {len(cl)} different, {sum(cl.values())} in total")
    print("Most frequent clusters: " + ", ".join(f"{a}{b}:{n}" for (a, b), n in cl.most_common(15)))
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sentences", type=Path, default=SENTENCES_TSV)
    ap.add_argument("--ids", type=Path, help="file with one sentence ID on each line")
    ap.add_argument("--min", type=int, default=10, help="minimum count for each phoneme")
    args = ap.parse_args()

    rows = read_tsv(args.sentences)
    if args.ids:
        wanted = [l.split()[0] for l in args.ids.read_text().splitlines() if l.strip()]
        by_id = {r["id"]: r for r in rows}
        rows = [by_id[i] for i in wanted]
    sys.exit(0 if report(rows, args.min) else 1)


if __name__ == "__main__":
    main()

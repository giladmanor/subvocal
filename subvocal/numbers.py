"""Compare Hebrew text with numbers in words and in digits.

The model often writes numbers as digits: "97" for "תשעים ושבע", "052" for "אפס חמישים ושתיים".
These transcripts are correct. canonical() changes the number words to digits, and joins
the digits of one number. The app uses it on the reference and on the transcript, so the
two forms are equal.

Examples:
    "קיבלתי ציון תשעים ושבע"          -> "קיבלתי ציון 97"
    "מתחיל באפס חמישים ושתיים"        -> "מתחיל ב 052"
    "ניצחה שלוש אפס" and "ניצחה 3 0"   -> "ניצחה 30"
"""

from __future__ import annotations

import re

UNITS = {
    "אפס": 0, "אחת": 1, "אחד": 1, "שתיים": 2, "שתים": 2, "שניים": 2, "שני": 2, "שתי": 2,
    "שלוש": 3, "שלושה": 3, "ארבע": 4, "ארבעה": 4, "חמש": 5, "חמישה": 5, "שש": 6, "שישה": 6,
    "שבע": 7, "שבעה": 7, "שמונה": 8, "תשע": 9, "תשעה": 9,
}
TEEN = {"עשר", "עשרה"}
TENS = {"עשרים": 20, "שלושים": 30, "ארבעים": 40, "חמישים": 50, "שישים": 60, "שבעים": 70,
        "שמונים": 80, "תשעים": 90}
HUNDREDS = {"מאה": 100, "מאתיים": 200}
WORDS = set(UNITS) | TEEN | set(TENS) | set(HUNDREDS) | {"מאות", "אלף", "אלפיים", "אלפים"}
PREFIXES = "ובלמכש"


def _split_prefix(tok: str) -> tuple[str, str]:
    """"באפס" -> ("ב", "אפס"). "ושבע" -> ("ו", "שבע"). Return ("", tok) if it is not a number word."""
    if tok in WORDS:
        return "", tok
    for i in range(1, min(3, len(tok))):
        if all(c in PREFIXES for c in tok[:i]) and tok[i:] in WORDS:
            return tok[:i], tok[i:]
    return "", tok


def _parse(words: list[tuple[bool, str]]) -> str:
    """Number words (with a flag for a "ו" before the word) to a digit string.

    A word that cannot continue the current number starts a new number. The digit strings
    of the numbers are joined: "אפס חמישים ושתיים" -> "0" + "52" -> "052".
    """
    out, cur, last = [], None, ""
    for vav, w in words:
        if w in TEEN and last == "unit" and cur is not None and cur % 10 != 0:
            cur += 10
            last = "teen"
        elif w in TEEN:  # "עשר" alone
            out.append(cur) if cur is not None else None
            cur, last = 10, "teen"
        elif w in TENS and cur is not None and cur >= 100 and cur % 100 == 0:
            cur += TENS[w]
            last = "tens"
        elif w in TENS:
            out.append(cur) if cur is not None else None
            cur, last = TENS[w], "tens"
        elif w in UNITS and vav and cur is not None and cur >= 20 and cur % 10 == 0:
            cur += UNITS[w]
            last = "unit"
        elif w in UNITS and cur is not None and last in ("hundreds", "thousand") and cur % 100 == 0:
            cur += UNITS[w]
            last = "unit"
        elif w in UNITS:
            out.append(cur) if cur is not None else None
            cur, last = UNITS[w], "unit"
        elif w == "מאות" and cur is not None and last == "unit":
            cur = (cur // 1000) * 1000 + (cur % 1000) * 100
            last = "hundreds"
        elif w in HUNDREDS and cur is not None and cur >= 1000 and cur % 1000 == 0:
            cur += HUNDREDS[w]
            last = "hundreds"
        elif w in HUNDREDS:
            out.append(cur) if cur is not None else None
            cur, last = HUNDREDS[w], "hundreds"
        elif w in ("אלף", "אלפיים"):
            out.append(cur) if cur is not None else None
            cur, last = (1000 if w == "אלף" else 2000), "thousand"
        elif w == "אלפים" and cur is not None and last == "unit":
            cur *= 1000
            last = "thousand"
    if cur is not None:
        out.append(cur)
    return "".join(str(n) for n in out)


def canonical(text: str) -> str:
    """Text with numbers as digits. The input must be normalized (no niqqud, no punctuation)."""
    text = re.sub(r"([א-ת])(\d)", r"\1 \2", text)
    text = re.sub(r"(\d) ו (?=\d)", r"\1 ", text)  # "5 ו 20" (from "5 ו-20") is one number, as "חמש ועשרים"
    out: list[str] = []
    run: list[tuple[bool, str]] = []
    digits: list[str] = []

    def flush():
        if run:
            digits.append(_parse(run))
            run.clear()
        if digits:
            out.append("".join(digits))
            digits.clear()

    for tok in text.split():
        if tok.isdigit():
            if run:
                digits.append(_parse(run))
                run.clear()
            digits.append(tok)
            continue
        prefix, word = _split_prefix(tok)
        if word in WORDS:
            vav = prefix.endswith("ו")
            start = prefix[:-1] if vav and (run or digits) else prefix
            if start:  # a prefix such as "ב" ends the number before it
                flush()
                out.append(start)
            run.append((vav and bool(run or digits), word))
            continue
        flush()
        out.append(tok)
    flush()
    return " ".join(out)

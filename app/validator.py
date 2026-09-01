"""Groundedness validation: does every figure in the answer come from the context?

In financial Q&A the dangerous failure is not an unhelpful answer, it is a
*confident* one carrying a number nobody filed. A fluent sentence with a wrong
revenue figure is worse than "I don't know", and no amount of prompt instruction
reliably prevents it.

So the answer is checked mechanically. Every currency amount, percentage and
per-share figure in the response is parsed to a magnitude and matched against
figures parsed the same way out of the retrieved documents and tool outputs. A
figure with no match within tolerance is reported as ungrounded.

Tolerance exists because restating is legitimate: $391.04B may be written as
"$391 billion" or "roughly $391.0B". Rounding is fine; invention is not.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

#: $391.04B / $1.2 trillion / 391,035 million / 31.5% / $6.08
_NUMBER = re.compile(
    r"""(?P<currency>[$€£])?\s?
        # The comma-grouped alternative must actually contain a comma group.
        # With `(?:,\d{3})*` it also matches a bare "201" out of "2019", leaving
        # a stray "9" behind and mis-parsing every un-grouped 4+ digit number.
        (?P<value>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)
        \s*
        (?P<suffix>%|pp|percentage\s+points?|percent|bn|billion|b\b|million|mm|m\b|trillion|t\b|k\b|x\b)?
    """,
    re.VERBOSE | re.IGNORECASE,
)

_MULTIPLIER = {
    "b": 1e9, "bn": 1e9, "billion": 1e9,
    "m": 1e6, "mm": 1e6, "million": 1e6,
    "t": 1e12, "trillion": 1e12,
    "k": 1e3,
}

#: relative tolerance for matching a restated figure back to its source
REL_TOLERANCE = 0.02
#: below this magnitude, compare absolutely instead (percentages, EPS, ratios)
ABS_TOLERANCE = 0.06

#: numbers that are almost never claims about data
_IGNORED_LITERALS = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 100.0}


@dataclass(frozen=True)
class Figure:
    raw: str
    magnitude: float
    kind: str  # "currency" | "percent" | "bare"


@dataclass
class GroundingReport:
    grounded: bool
    checked: int
    ungrounded: list[str] = field(default_factory=list)
    matched: list[str] = field(default_factory=list)

    @property
    def rate(self) -> float:
        return 1.0 if self.checked == 0 else len(self.matched) / self.checked


def extract_figures(text: str) -> list[Figure]:
    figures: list[Figure] = []
    for match in _NUMBER.finditer(text or ""):
        raw_value = match.group("value")
        if not raw_value:
            continue
        try:
            value = float(raw_value.replace(",", ""))
        except ValueError:
            continue

        suffix = (match.group("suffix") or "").strip().lower()
        currency = match.group("currency")

        if suffix in ("%", "percent", "pp") or suffix.startswith("percentage"):
            kind, magnitude = "percent", value
        else:
            key = suffix.rstrip(".")
            magnitude = value * _MULTIPLIER.get(key, 1.0)
            kind = "currency" if currency else "bare"

        # a four-digit integer with no unit is a fiscal year, not a claim
        if kind == "bare" and value.is_integer() and 1900 <= value <= 2100:
            continue
        if kind == "bare" and magnitude in _IGNORED_LITERALS:
            continue
        figures.append(Figure(match.group(0).strip(), magnitude, kind))
    return figures


def _matches(candidate: Figure, sources: list[Figure]) -> bool:
    for source in sources:
        if candidate.kind == "percent" and source.kind != "percent":
            continue
        if candidate.kind != "percent" and source.kind == "percent":
            continue
        a, b = candidate.magnitude, source.magnitude
        if abs(a) < 1000 or abs(b) < 1000:
            if abs(a - b) <= ABS_TOLERANCE:
                return True
        if b != 0 and abs(a - b) / abs(b) <= REL_TOLERANCE:
            return True
        # "$391.04B" restated as "391" (billions implied by the sentence)
        for scale in (1e9, 1e6, 1e3):
            if b != 0 and abs(a * scale - b) / abs(b) <= REL_TOLERANCE:
                return True
    return False


def validate_answer(answer: str, context: str) -> GroundingReport:
    """Check every figure in ``answer`` against figures present in ``context``."""
    claimed = extract_figures(answer)
    available = extract_figures(context)

    matched: list[str] = []
    ungrounded: list[str] = []
    for figure in claimed:
        (matched if _matches(figure, available) else ungrounded).append(figure.raw)

    return GroundingReport(
        grounded=not ungrounded,
        checked=len(claimed),
        ungrounded=ungrounded,
        matched=matched,
    )

"""Metadata-aware entity resolution: free text -> strict retrieval filters.

This is the layer that decides *what the index is allowed to return*. The planner
LLM proposes surface forms ("goldman", "op margin", "last fiscal year"); nothing
it says is trusted directly. Every proposal is matched with RapidFuzz against the
vocabularies actually present in the index -- company aliases, canonical metric
ids, sector names -- and only a confident, unambiguous match becomes a hard
Qdrant filter.

Two design choices matter:

*   **Confidence gating.** A match must clear ``fuzzy_threshold`` *and* beat the
    runner-up by ``fuzzy_margin``. "ma" is a real ticker (Mastercard) but also a
    fragment of a dozen words, and "co" is closer to both "COST" and "KO" than it
    is to either alone. When the margin is not met the candidate is demoted to a
    soft hint: it still shapes the query text, but it never removes a document.
    Silently filtering on a wrong company is the worst failure mode in this
    domain -- it produces a fluent, confident answer about the wrong business.

*   **One vocabulary.** Metric aliases come from ``app.metrics``, the same
    registry the corpus builder used to tag payloads, so a resolved filter is
    guaranteed to correspond to something that was actually indexed.
"""

from __future__ import annotations

import functools
import json
import re
from dataclasses import dataclass
from typing import Any, Iterable

from rapidfuzz import fuzz, process, utils

from .config import DATA_DIR, get_settings
from .metrics import alias_index

DOC_TYPES = (
    "income_statement", "balance_sheet", "cash_flow", "ratios",
    "yoy_analysis", "company_profile", "portfolio_position", "portfolio_summary",
)

#: Explicit statement names -> doc_type filter.
#:
#: Only unambiguous, full statement names belong here. A bare "cash flow" must
#: NOT be a hint: it is a substring of "free cash flow" and "operating cash
#: flow", which are metrics living on different cards, so treating it as a
#: statement produces a filter that contradicts the metric and returns nothing.
_STATEMENT_HINTS = {
    "income statement": "income_statement",
    "statement of operations": "income_statement",
    "p&l": "income_statement",
    "profit and loss": "income_statement",
    "balance sheet": "balance_sheet",
    "statement of financial position": "balance_sheet",
    "cash flow statement": "cash_flow",
    "statement of cash flows": "cash_flow",
    "year over year": "yoy_analysis",
    "year-over-year": "yoy_analysis",
}

_PORTFOLIO_WORDS = re.compile(
    r"\b(my|our|client'?s?|portfolio|holding|holdings|position|positions|account|accounts"
    r"|ira|401k|401\(k\)|brokerage|own|owned|we hold|i hold|exposure)\b",
    re.IGNORECASE,
)

_POSSESSIVE = re.compile(r"['’]s$|['’]$", re.IGNORECASE)
_CORPORATE_SUFFIX = re.compile(
    r"\b(inc|corp|corporation|company|co|plc|group|holdings|incorporated|ltd|limited)\b\.?",
    re.IGNORECASE,
)
_SINCE_PATTERN = re.compile(
    r"\b(?:since|after|from)\s+(?:fy\s*)?(20\d{2})\b(?!\s*(?:-|–|—|to|through|thru|until))",
    re.IGNORECASE,
)
_FY_PATTERNS = (
    re.compile(r"\bfy\s*'?(\d{2,4})\b", re.IGNORECASE),
    re.compile(r"\bfiscal\s+(?:year\s+)?(\d{4})\b", re.IGNORECASE),
    re.compile(r"\b(19\d{2}|20\d{2})\b"),
)
_RANGE_PATTERNS = (
    # "between FY2019 and FY2021" -- 'and' only reads as a range separator after
    # "between", otherwise "2023 and 2024" in a two-year comparison would be
    # expanded into a span instead of two discrete years.
    re.compile(r"\bbetween\s+(?:fy\s*)?(20\d{2})\s+and\s+(?:fy\s*)?(20\d{2})\b", re.IGNORECASE),
    re.compile(
        r"\b(?:from\s+)?(?:fy\s*)?(20\d{2})\s*(?:-|–|—|to|through|thru|until)\s*(?:fy\s*)?(20\d{2})\b",
        re.IGNORECASE,
    ),
)


@dataclass(frozen=True)
class Resolution:
    """One resolution attempt, kept for the trace so decisions are auditable."""

    field: str
    surface: str
    resolved: str | None
    score: float
    runner_up: str | None
    margin: float
    strict: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "field": self.field,
            "surface": self.surface,
            "resolved": self.resolved,
            "score": round(self.score, 1),
            "runner_up": self.runner_up,
            "margin": round(self.margin, 1),
            "applied_as": "strict_filter" if self.strict else ("soft_hint" if self.resolved else "dropped"),
        }


class EntityResolver:
    """Fuzzy-maps free text onto the index's actual metadata vocabularies."""

    def __init__(self, companies: list[dict], years: Iterable[int], settings=None) -> None:
        self.settings = settings or get_settings()
        self.companies = companies
        self.by_ticker = {c["ticker"].upper(): c for c in companies}
        self.known_years = sorted(set(years))

        # ---- company vocabulary: every alias, name and ticker -> ticker ----
        self._company_index: dict[str, str] = {}
        for c in companies:
            ticker = c["ticker"].upper()
            self._company_index[ticker.lower()] = ticker
            self._company_index[c["name"].lower()] = ticker
            # "Apple Inc." also answers to "apple"
            stripped = re.sub(
                r"\b(inc|corp|corporation|company|co|plc|group|holdings|incorporated|ltd|the)\b\.?",
                "", c["name"].lower()
            ).strip(" ,.&")
            if stripped:
                self._company_index[stripped] = ticker
            for alias in c.get("aliases", []):
                self._company_index[alias.lower()] = ticker
        self._company_choices = list(self._company_index)

        # ---- metric vocabulary, straight from the registry ----
        self._metric_index = dict(alias_index())
        self._metric_choices = list(self._metric_index)

        # ---- sector vocabulary ----
        self._sector_index = {c["sector"].lower(): c["sector"] for c in companies}
        self._sector_choices = list(self._sector_index)

    # -- generic fuzzy match with confidence gating -------------------------

    def _match(self, field: str, surface: str, index: dict[str, str],
               choices: list[str]) -> Resolution:
        probe = (surface or "").strip().lower()
        if field == "ticker":
            # Strip corporate suffixes from the probe as well as from the index.
            # Otherwise "Hooli Incorporated" scores 85 against "UnitedHealth Group
            # Incorporated" purely on the shared suffix and becomes a strict
            # filter for a company nobody mentioned.
            stripped = _CORPORATE_SUFFIX.sub("", probe).strip(" ,.&")
            if stripped and stripped != probe and stripped not in index:
                probe = stripped
        if not probe:
            return Resolution(field, surface, None, 0.0, None, 0.0, False)

        # exact hit short-circuits: no need to risk fuzz on an unambiguous string
        if probe in index:
            return Resolution(field, surface, index[probe], 100.0, None, 100.0, True)

        results = process.extract(
            probe, choices, scorer=fuzz.WRatio, processor=utils.default_process, limit=5
        )
        if not results:
            return Resolution(field, surface, None, 0.0, None, 0.0, False)

        best_key, best_score, _ = results[0]
        best_value = index[best_key]

        # the runner-up only counts if it resolves to a *different* entity --
        # "apple" and "apple inc" both mean AAPL and must not cancel each other out
        runner_key, runner_score = None, 0.0
        for key, score, _ in results[1:]:
            if index[key] != best_value:
                runner_key, runner_score = key, score
                break

        margin = best_score - runner_score
        strict = (
            best_score >= self.settings.fuzzy_threshold
            and margin >= self.settings.fuzzy_margin
        )
        resolved = best_value if best_score >= self.settings.fuzzy_threshold - 10 else None
        return Resolution(field, surface, resolved, best_score, runner_key, margin, strict)

    # -- public per-field resolvers ----------------------------------------

    def resolve_company(self, surface: str) -> Resolution:
        return self._match("ticker", surface, self._company_index, self._company_choices)

    def resolve_metric(self, surface: str) -> Resolution:
        return self._match("metric", surface, self._metric_index, self._metric_choices)

    def resolve_sector(self, surface: str) -> Resolution:
        return self._match("sector", surface, self._sector_index, self._sector_choices)

    # -- year parsing ------------------------------------------------------

    def resolve_years(self, text: str) -> tuple[list[int], list[Resolution]]:
        """Pull fiscal years out of raw text, including ranges and relative words."""
        found: list[int] = []
        notes: list[Resolution] = []
        if not text:
            return found, notes

        for pattern in _RANGE_PATTERNS:
            for match in pattern.finditer(text):
                lo, hi = int(match.group(1)), int(match.group(2))
                if lo > hi:
                    lo, hi = hi, lo
                span = [y for y in range(lo, hi + 1) if y in self.known_years]
                if span:
                    found.extend(span)
                    notes.append(
                        Resolution("fiscal_year", match.group(0), str(span), 100.0, None, 100.0, True)
                    )
            if found:
                break

        # "since 2019" / "after FY2020" is an open-ended range ending at the
        # newest year we hold, not a single year.
        if not found:
            for match in _SINCE_PATTERN.finditer(text):
                start = int(match.group(1))
                span = [y for y in self.known_years if y >= start]
                if span:
                    found.extend(span)
                    notes.append(
                        Resolution("fiscal_year", match.group(0), str(span), 100.0, None, 100.0, True)
                    )

        if not found:
            for pattern in _FY_PATTERNS:
                for match in pattern.finditer(text):
                    raw = match.group(1)
                    year = int(raw)
                    if year < 100:  # FY'24
                        year += 2000
                    if year in self.known_years:
                        found.append(year)
                        notes.append(
                            Resolution("fiscal_year", match.group(0), str(year), 100.0, None, 100.0, True)
                        )
                if found:
                    break

        lowered = text.lower()
        if not found and self.known_years:
            latest = self.known_years[-1]
            if re.search(r"\b(latest|most recent|current|last year|this year|newest)\b", lowered):
                found.append(latest)
                notes.append(
                    Resolution("fiscal_year", "latest", str(latest), 100.0, None, 100.0, True)
                )

        # de-duplicate, preserve order
        seen: set[int] = set()
        ordered = [y for y in found if not (y in seen or seen.add(y))]
        return ordered, notes

    # -- whole-query resolution -------------------------------------------

    def resolve_query(self, text: str) -> tuple[list[str], list[Resolution]]:
        """Scan raw text for company mentions without the planner's help.

        Runs n-grams (longest first) through the company index. This is the
        safety net for when the planner omits a company the user clearly named,
        and it is what the deterministic no-LLM path relies on entirely.
        """
        # Strip the possessive before matching. "MA's", "chevron's" and
        # "Johnson & Johnson's" are how people actually write these questions,
        # and an unstripped "ma's" matches nothing in the alias index.
        tokens = [_POSSESSIVE.sub("", t) for t in re.findall(r"[A-Za-z&.\-']+", text)]
        tokens = [t for t in tokens if t]
        hits: dict[str, Resolution] = {}
        for size in (4, 3, 2, 1):
            for i in range(len(tokens) - size + 1):
                window = tokens[i:i + size]
                gram = " ".join(window).lower().strip(" .")
                if gram in _STOPWORDS:
                    continue
                # Visa trades as "V". A one-character gram is only ever a ticker
                # when it was actually written as a capital, which keeps "a" and
                # "I" out while letting "V's net margin" resolve.
                if len(gram) < 2 and not (size == 1 and window[0].isupper()):
                    continue
                if gram in self._company_index:
                    ticker = self._company_index[gram]
                    prior = hits.get(ticker)
                    if prior is None or len(gram) > len(prior.surface):
                        hits[ticker] = Resolution("ticker", gram, ticker, 100.0, None, 100.0, True)
        return list(hits), list(hits.values())

    def mentions_portfolio(self, text: str) -> bool:
        return bool(_PORTFOLIO_WORDS.search(text or ""))

    def statement_hints(self, text: str) -> list[str]:
        lowered = (text or "").lower()
        return sorted({dt for phrase, dt in _STATEMENT_HINTS.items() if phrase in lowered})

    def mask_statement_phrases(self, text: str) -> str:
        """Blank out explicit statement names before metric scanning.

        "Show me the cash flow statement" should resolve a document type, not the
        metric `cash_and_equivalents` off the stray word "cash". Consuming the
        phrase here stops one span of text being read as two different filters.
        """
        masked = text or ""
        for phrase in sorted(_STATEMENT_HINTS, key=len, reverse=True):
            pattern = re.compile(re.escape(phrase), re.IGNORECASE)
            masked = pattern.sub(" " * len(phrase), masked)
        return masked


#: single-token words that collide with tickers but are almost never the company
_STOPWORDS = {
    "a", "all", "an", "and", "any", "are", "as", "at", "be", "by", "can", "co",
    "did", "do", "does", "for", "from", "had", "has", "have", "how", "in", "is",
    "it", "its", "me", "my", "of", "on", "or", "our", "over", "show", "so",
    "than", "that", "the", "their", "them", "then", "there", "these", "they",
    "this", "to", "up", "us", "was", "we", "were", "what", "when", "which",
    "who", "why", "will", "with", "you", "your", "vs", "versus", "between",
    "best", "worst", "more", "most", "less", "least", "much", "many", "was",
}


@functools.lru_cache(maxsize=1)
def get_resolver() -> EntityResolver:
    companies = json.loads((DATA_DIR / "companies.json").read_text(encoding="utf-8"))["companies"]
    financials_path = DATA_DIR / "financials.json"
    years: set[int] = set()
    if financials_path.exists():
        financials = json.loads(financials_path.read_text(encoding="utf-8"))
        for record in financials.values():
            years.update(int(y) for y in record["years"])
        # only keep companies we actually have filings for
        available = set(financials)
        companies = [c for c in companies if c["ticker"] in available]
    return EntityResolver(companies, years)

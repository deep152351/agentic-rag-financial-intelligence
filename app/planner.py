"""Planner: question -> decomposed, constrained retrieval plan. (LLM-1)

The planner never touches the index and never sees a document. It reads the
question and proposes *surface forms* -- "goldman", "op margin", "last two years".
Those proposals are then run through ``app.entities``, which is the only thing
allowed to turn text into a filter. So a planner hallucination ("Tesla's FY2031
figures") cannot produce a bogus filter; it produces an unresolvable one that is
dropped or demoted before it reaches Qdrant.

There are two planners behind one interface:

``LLMPlanner``            asks the fast model for a decomposition.
``DeterministicPlanner``  derives one from regex + fuzzy matching alone.

The deterministic planner is not a toy fallback -- it runs whenever no API key is
configured, when the LLM call fails or times out, and whenever the LLM returns an
empty plan. It is also what makes the retrieval half of the system measurable
independently of any model provider.
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from typing import Any, Sequence

from .config import Settings, get_settings
from .entities import DOC_TYPES, EntityResolver, Resolution, get_resolver
from .llm import LLMClient, LLMError, parse_json_object
from .metrics import METRICS
from .schemas import Constraints, Message, Plan, SubQuery

logger = logging.getLogger(__name__)

MAX_SUBQUERY_CHARS = 300

PLANNER_SYSTEM = """\
You decompose questions about US public-company financial statements and a client's \
investment portfolio into retrieval sub-queries.

The corpus contains, per company per fiscal year, one card each for: income statement, \
balance sheet, cash flow, computed ratios, and a year-over-year narrative. It also \
contains company profiles and the client's portfolio positions and account summaries.

Rules:
- Emit ONE sub-query per (company, fiscal year) pair the question needs. A comparison \
of three companies over two years is six sub-queries, not one.
- `text` must be a self-contained search phrase naming the company, the year and the \
metric. It is embedded verbatim, so write it for retrieval, not for a human.
- Put company names, years and metrics in the structured fields too. Use whatever \
wording the user used; they are fuzzy-matched downstream against the real index \
vocabulary, so do not guess ticker symbols or invent canonical names.
- Never invent a fiscal year the user did not ask for. If no year is stated and the \
question implies the most recent, use "latest".
- Set intent to out_of_scope for anything not answerable from company filings or the \
client's portfolio (stock price predictions, tax advice, general market commentary).
- Emit at most {max_subqueries} sub-queries.

Respond with a single JSON object matching the schema."""

PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "intent": {
            "type": "string",
            "enum": ["company_lookup", "comparison", "trend", "portfolio", "screen", "out_of_scope"],
        },
        "reasoning": {"type": "string"},
        "subqueries": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "companies": {"type": "array", "items": {"type": "string"}},
                    "years": {"type": "array", "items": {"type": "string"}},
                    "metrics": {"type": "array", "items": {"type": "string"}},
                    "doc_types": {"type": "array", "items": {"type": "string", "enum": list(DOC_TYPES)}},
                },
                "required": ["text"],
                "additionalProperties": False,
            },
        },
        "portfolio_tools": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["intent", "subqueries"],
    "additionalProperties": False,
}

PORTFOLIO_TOOLS = ("holdings", "exposure", "look_through", "concentration", "screen_holdings")

_TREND_WORDS = re.compile(
    r"\b(trend\w*|over time|history|historical|growth|grew|trajectory|since|each year"
    r"|by year|change\w*|moved?|shift\w*|evolv\w*|progress\w*)\b",
    re.I,
)
_COMPARE_WORDS = re.compile(r"\b(compare|comparison|versus|vs\.?|against|better|worse|outperform|relative to)\b", re.I)
_SCREEN_WORDS = re.compile(r"\b(which|who|any|find|list|screen|rank|top|bottom|highest|lowest|most|least)\b", re.I)
_OUT_OF_SCOPE_WORDS = re.compile(
    r"\b(should i (buy|sell|hold)|will .{0,30}(go up|go down|rise|fall|beat)|price target"
    r"|forecast|predict|prediction|next (week|month|quarter|year)\b.{0,20}\b(price|stock)"
    r"|stock price|share price|tax advice"
    r"|\ba good (buy|investment|stock|pick)\b"      # "is Microsoft a good investment"
    r"|worth (buying|investing)|recommend .{0,20}(buy|sell)"
    r"|what should i invest|invest in this year)\b",
    re.I,
)


@dataclass
class PlanOutcome:
    plan: Plan
    resolutions: list[Resolution]
    source: str  # "llm" | "deterministic" | "llm+deterministic"


# ---------------------------------------------------------------------------
# Shared constraint resolution
# ---------------------------------------------------------------------------


class ConstraintBuilder:
    """Turns raw surface forms into Constraints, recording every decision."""

    def __init__(self, resolver: EntityResolver, settings: Settings) -> None:
        self.resolver = resolver
        self.settings = settings

    def build(self, text: str, companies: Sequence[str] = (), years: Sequence[str] = (),
              metrics: Sequence[str] = (), doc_types: Sequence[str] = (),
              ) -> tuple[Constraints, list[Resolution]]:
        constraints = Constraints()
        notes: list[Resolution] = []

        for surface in companies:
            res = self.resolver.resolve_company(surface)
            notes.append(res)
            if res.strict and res.resolved:
                constraints.tickers.append(res.resolved)
            elif res.resolved:
                constraints.soft_tickers.append(res.resolved)

        # anything the planner missed but the user plainly wrote
        scanned, scan_notes = self.resolver.resolve_query(text)
        for ticker in scanned:
            if ticker not in constraints.tickers:
                constraints.tickers.append(ticker)
        notes.extend(n for n in scan_notes if n.resolved not in {r.resolved for r in notes})

        for surface in metrics:
            res = self.resolver.resolve_metric(surface)
            notes.append(res)
            if res.strict and res.resolved:
                constraints.metrics.append(res.resolved)
            elif res.resolved:
                constraints.soft_metrics.append(res.resolved)

        year_text = " ".join([*years, text])
        parsed_years, year_notes = self.resolver.resolve_years(year_text)
        constraints.fiscal_years.extend(parsed_years)
        notes.extend(year_notes)

        for dt in doc_types:
            if dt in DOC_TYPES:
                constraints.doc_types.append(dt)
        for dt in self.resolver.statement_hints(text):
            if dt not in constraints.doc_types:
                constraints.doc_types.append(dt)

        if self.resolver.mentions_portfolio(text) and not constraints.doc_types:
            constraints.doc_types.extend(["portfolio_position", "portfolio_summary"])

        _reconcile_doc_types(constraints)
        _dedupe(constraints)
        return constraints, notes


def _reconcile_doc_types(c: Constraints) -> None:
    """Drop a statement hint that contradicts the resolved metric.

    "free cash flow" contains the words "cash flow", so the keyword hint proposes
    doc_type=cash_flow -- but `free_cash_flow` is a derived ratio and lives on the
    ratios card. The intersection of the two filters is empty and the query returns
    nothing at all. The metric registry knows which statement each metric belongs
    to, so it wins: a hint that no resolved metric can satisfy is discarded.
    """
    if not (c.metrics and c.doc_types):
        return
    allowed = {METRICS[m].statement for m in c.metrics if m in METRICS}
    # these two card types carry metrics from several statements
    allowed |= {"yoy_analysis", "company_profile"}
    compatible = [dt for dt in c.doc_types if dt in allowed]
    c.doc_types = compatible


def _dedupe(c: Constraints) -> None:
    for field_name in ("tickers", "fiscal_years", "metrics", "sectors",
                       "statements", "doc_types", "account_ids",
                       "soft_tickers", "soft_metrics"):
        values = getattr(c, field_name)
        seen: set = set()
        setattr(c, field_name, [v for v in values if not (v in seen or seen.add(v))])


# ---------------------------------------------------------------------------
# Deterministic planner
# ---------------------------------------------------------------------------


class DeterministicPlanner:
    """Rule-based decomposition. No model, no network, fully reproducible."""

    def __init__(self, resolver: EntityResolver, settings: Settings) -> None:
        self.resolver = resolver
        self.settings = settings
        self.builder = ConstraintBuilder(resolver, settings)

    def plan(self, question: str) -> PlanOutcome:
        tickers, ticker_notes = self.resolver.resolve_query(question)
        years, year_notes = self.resolver.resolve_years(question)
        doc_types = self.resolver.statement_hints(question)
        # scan metrics over text with explicit statement names removed, so
        # "cash flow statement" yields a document type rather than also matching
        # the word "cash" as a balance-sheet metric
        metrics, metric_notes = self._scan_metrics(
            self.resolver.mask_statement_phrases(question)
        )
        is_portfolio = self.resolver.mentions_portfolio(question)
        notes = [*ticker_notes, *year_notes, *metric_notes]

        intent = self._intent(question, tickers, years, is_portfolio)

        subqueries: list[SubQuery] = []
        budget = self.settings.max_subqueries

        if is_portfolio:
            subqueries.append(SubQuery(
                text=question[:MAX_SUBQUERY_CHARS],
                constraints=Constraints(doc_types=["portfolio_position", "portfolio_summary"]),
                rationale="portfolio book lookup",
            ))

        # one sub-query per (company, year) -- the fan-out the parallel wave exists for
        pairs = [(t, y) for t in (tickers or [None]) for y in (years or [None])]
        for ticker, year in pairs:
            if len(subqueries) >= budget:
                break
            if ticker is None and year is None and subqueries:
                continue
            constraints = Constraints(
                tickers=[ticker] if ticker else [],
                fiscal_years=[year] if year else [],
                metrics=list(metrics),
                doc_types=list(doc_types),
            )
            _reconcile_doc_types(constraints)
            subqueries.append(SubQuery(
                text=self._phrase(question, ticker, year, metrics),
                constraints=constraints,
                rationale=f"{ticker or 'any company'} / {f'FY{year}' if year else 'any year'}",
            ))

        if not subqueries:
            subqueries.append(SubQuery(text=question[:MAX_SUBQUERY_CHARS],
                                       constraints=Constraints(),
                                       rationale="unconstrained fallback"))

        return PlanOutcome(
            plan=Plan(
                intent=intent,
                subqueries=subqueries,
                portfolio_tools=self._tools(question, intent),
                reasoning="deterministic decomposition (regex + fuzzy entity matching)",
            ),
            resolutions=notes,
            source="deterministic",
        )

    def _phrase(self, question: str, ticker: str | None, year: int | None,
                metrics: Sequence[str]) -> str:
        """A retrieval-shaped restatement, not a copy of the user's sentence."""
        parts: list[str] = []
        if ticker:
            company = self.resolver.by_ticker.get(ticker, {})
            parts.append(f"{company.get('name', ticker)} ({ticker})")
        if year:
            parts.append(f"FY{year}")
        if metrics:
            parts.extend(METRICS[m].label for m in metrics[:4])
        if not parts:
            return question[:MAX_SUBQUERY_CHARS]
        # keep some of the original wording so the dense branch has context
        return (" ".join(parts) + " — " + question)[:MAX_SUBQUERY_CHARS]

    def _scan_metrics(self, text: str) -> tuple[list[str], list[Resolution]]:
        """Longest-alias-first scan so 'operating margin' beats 'operating income'."""
        lowered = f" {text.lower()} "
        found: list[str] = []
        notes: list[Resolution] = []
        aliases = sorted(
            ((alias, mid) for mid, m in METRICS.items()
             for alias in (m.id.replace("_", " "), m.label.lower(), *m.aliases)),
            key=lambda pair: -len(pair[0]),
        )
        consumed: list[tuple[int, int]] = []
        for alias, metric_id in aliases:
            if metric_id in found or len(alias) < 3:
                continue
            idx = lowered.find(f" {alias} ")
            if idx < 0:
                idx = lowered.find(f" {alias}")
                if idx < 0 or not _is_boundary(lowered, idx + 1 + len(alias)):
                    continue
            span = (idx, idx + len(alias) + 2)
            if any(s <= span[0] < e or s < span[1] <= e for s, e in consumed):
                continue
            consumed.append(span)
            found.append(metric_id)
            notes.append(Resolution("metric", alias, metric_id, 100.0, None, 100.0, True))
        return found, notes

    def _intent(self, question: str, tickers: Sequence[str], years: Sequence[int],
                is_portfolio: bool) -> str:
        # Filings are a record of the past. Anything asking what a price will do,
        # or for a recommendation, cannot be grounded in this corpus no matter how
        # good retrieval is -- refuse at planning time rather than answer fluently
        # from irrelevant documents.
        if _OUT_OF_SCOPE_WORDS.search(question):
            return "out_of_scope"
        if is_portfolio:
            return "portfolio"
        if _COMPARE_WORDS.search(question) or len(tickers) > 1:
            return "comparison"
        # two or more years of a single company is a movement question, not a lookup
        if _TREND_WORDS.search(question) or len(years) >= 2:
            return "trend"
        if not tickers and _SCREEN_WORDS.search(question):
            return "screen"
        return "company_lookup"

    def _tools(self, question: str, intent: str) -> list[str]:
        if intent != "portfolio":
            return []
        lowered = question.lower()
        tools = ["holdings"]
        # Stems carry \w*: a trailing \b straight after "concentrat" can never
        # match "concentrated", which is how the word is actually written.
        if re.search(r"\b(?:exposure|allocation|weight\w*|concentrat\w*|diversif\w*|sector\w*)\b", lowered):
            tools.append("exposure")
        if re.search(r"\b(?:look.?through|underlying|share of|economic|attributable|earnings behind)\b", lowered):
            tools.append("look_through")
        if re.search(r"\b(?:concentrat\w*|risk\w*|largest|biggest|top holdings|hhi)\b", lowered):
            tools.append("concentration")
        if _SCREEN_WORDS.search(lowered):
            tools.append("screen_holdings")
        return list(dict.fromkeys(tools))


def _is_boundary(text: str, index: int) -> bool:
    return index >= len(text) or not (text[index].isalnum() or text[index] == "_")


# ---------------------------------------------------------------------------
# LLM planner
# ---------------------------------------------------------------------------


class LLMPlanner:
    def __init__(self, client: LLMClient, resolver: EntityResolver, settings: Settings) -> None:
        self.client = client
        self.resolver = resolver
        self.settings = settings
        self.builder = ConstraintBuilder(resolver, settings)
        self.deterministic = DeterministicPlanner(resolver, settings)

    async def plan(self, question: str, history: Sequence[Message] = ()) -> PlanOutcome:
        prompt = self._prompt(question, history)
        system = PLANNER_SYSTEM.format(max_subqueries=self.settings.max_subqueries)
        try:
            result = await asyncio.wait_for(
                self.client.complete(system, prompt,
                                     max_tokens=self.settings.planner_max_tokens,
                                     json_schema=PLAN_SCHEMA),
                timeout=self.settings.llm_timeout_s,
            )
            raw = parse_json_object(result.text)
        except (asyncio.TimeoutError, LLMError, Exception) as exc:
            logger.warning("planner LLM unusable (%s); falling back to deterministic", exc)
            outcome = self.deterministic.plan(question)
            outcome.source = "deterministic(llm_failed)"
            return outcome

        outcome = self._materialise(question, raw)
        if not outcome.plan.subqueries:
            fallback = self.deterministic.plan(question)
            fallback.source = "deterministic(empty_llm_plan)"
            return fallback
        return outcome

    def _prompt(self, question: str, history: Sequence[Message]) -> str:
        blocks = []
        if history:
            recent = history[-4:]
            blocks.append("CONVERSATION SO FAR\n" + "\n".join(
                f"{m.role}: {m.content[:400]}" for m in recent
            ))
        blocks.append(f"QUESTION\n{question}")
        return "\n\n".join(blocks)

    def _materialise(self, question: str, raw: dict[str, Any]) -> PlanOutcome:
        intent = raw.get("intent") or "company_lookup"
        notes: list[Resolution] = []
        subqueries: list[SubQuery] = []

        for item in (raw.get("subqueries") or [])[: self.settings.max_subqueries]:
            if not isinstance(item, dict):
                continue
            text = str(item.get("text") or "").strip()
            if not text:
                continue
            constraints, item_notes = self.builder.build(
                text=f"{text} {question}",
                companies=[str(c) for c in item.get("companies") or []],
                years=[str(y) for y in item.get("years") or []],
                metrics=[str(m) for m in item.get("metrics") or []],
                doc_types=[str(d) for d in item.get("doc_types") or []],
            )
            notes.extend(item_notes)
            subqueries.append(SubQuery(
                text=text[:MAX_SUBQUERY_CHARS],
                constraints=constraints,
                rationale=str(item.get("rationale") or ""),
            ))

        tools = [t for t in (raw.get("portfolio_tools") or []) if t in PORTFOLIO_TOOLS]
        if intent == "portfolio" and not tools:
            tools = ["holdings"]

        return PlanOutcome(
            plan=Plan(
                intent=intent if intent in _VALID_INTENTS else "company_lookup",
                subqueries=subqueries,
                portfolio_tools=tools,
                reasoning=str(raw.get("reasoning") or "")[:500],
            ),
            resolutions=notes,
            source="llm",
        )


_VALID_INTENTS = {"company_lookup", "comparison", "trend", "portfolio", "screen", "out_of_scope"}


def build_planner(client: LLMClient | None, settings: Settings | None = None):
    s = settings or get_settings()
    resolver = get_resolver()
    if client is None or getattr(client, "model", "") == "stub":
        return DeterministicPlanner(resolver, s)
    return LLMPlanner(client, resolver, s)

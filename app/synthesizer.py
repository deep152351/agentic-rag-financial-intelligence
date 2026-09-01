"""Synthesizer: retrieved facts -> grounded answer with citations. (LLM-2)

Runs once per request, on the strong model, over context that has already been
retrieved, filtered and (for portfolio questions) computed. Its only job is
faithful prose. It never retrieves, never calculates a portfolio aggregate, and
is told in the system prompt that arithmetic it cannot read off the context is
out of bounds -- the numbers it needs are computed in ``app.wealth`` and handed
to it, precisely so it does not have to add anything up.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Sequence

from .config import Settings, get_settings
from .llm import LLMClient, LLMError
from .schemas import Citation, Message, RetrievedDoc
from .wealth import ToolResult

logger = logging.getLogger(__name__)

SYNTHESIZER_SYSTEM = """\
You are a financial analyst assistant. You answer questions about US public-company \
financial statements and about the client's own investment portfolio.

Absolute rules:
1. Every figure you state must appear in the CONTEXT DOCUMENTS or PORTFOLIO ANALYTICS \
below. Never estimate, extrapolate, or recall a number from memory. If the context \
lacks something the question needs, say exactly what is missing.
2. Do not compute new aggregates. Sums, weights and look-through totals have already \
been computed for you and appear under PORTFOLIO ANALYTICS. Simple differences between \
two figures shown in the context are fine; anything more is not.
3. Cite the document id in square brackets after each claim, e.g. [AAPL-FY2024-ratios].
4. Name the fiscal year with every figure. Fiscal years differ from calendar years for \
many of these filers.
5. These are historical filings. Do not forecast, value a security, or give investment \
advice. If asked to, say plainly that you can report what was filed but not advise.
6. Portfolio positions are stated at cost basis. There are no market prices here, so \
never describe a gain, loss or current market value.

Be concise and specific. Lead with the direct answer, then the supporting figures. \
Use a short markdown table when comparing more than two company-years."""

OUT_OF_SCOPE_ANSWER = (
    "That question can't be answered from this dataset. I work from SEC Form 10-K "
    "filings (FY2019-FY2025) and the client's portfolio book, so I can report what "
    "companies actually filed and what the portfolio holds — but not price forecasts, "
    "valuations, or buy/sell recommendations.\n\n"
    "Things I can answer: a company's revenue, margins, cash flow or balance sheet for "
    "a given fiscal year; how those moved year over year; comparisons across companies; "
    "and the portfolio's holdings, sector exposure, concentration and look-through "
    "share of reported fundamentals."
)


@dataclass
class Synthesis:
    answer: str
    citations: list[Citation]
    context: str
    model: str
    refused: bool = False


def build_context(docs: Sequence[RetrievedDoc], tools: Sequence[ToolResult] = ()) -> str:
    """The exact text the model is allowed to draw figures from.

    The validator checks the answer against this same string, so whatever is
    assembled here defines the boundary of what counts as grounded.
    """
    blocks: list[str] = []
    if tools:
        rendered = "\n\n".join(f"[{t.name}]\n{t.summary}" for t in tools if t.summary)
        if rendered:
            blocks.append("PORTFOLIO ANALYTICS (computed, authoritative)\n" + rendered)

    if docs:
        rendered = "\n\n".join(
            f"[{d.doc_id}] (source: {d.source})\n{d.text}" for d in docs
        )
        blocks.append("CONTEXT DOCUMENTS\n" + rendered)

    return "\n\n".join(blocks) if blocks else "CONTEXT DOCUMENTS\n(none retrieved)"


class Synthesizer:
    def __init__(self, client: LLMClient, settings: Settings | None = None) -> None:
        self.client = client
        self.settings = settings or get_settings()

    async def run(self, question: str, docs: Sequence[RetrievedDoc],
                  tools: Sequence[ToolResult] = (), history: Sequence[Message] = (),
                  intent: str = "company_lookup",
                  deadline_s: float | None = None) -> Synthesis:
        context = build_context(docs, tools)

        if intent == "out_of_scope":
            return Synthesis(OUT_OF_SCOPE_ANSWER, [], context, "n/a (refused at planning)")

        if not docs and not tools:
            return Synthesis(
                "I couldn't find anything in the indexed filings or the portfolio book "
                "that answers that. If you name the company and fiscal year explicitly "
                "(for example \"Apple FY2024 operating margin\") I can look it up directly.",
                [], context, "n/a (no context)",
            )

        prompt = self._prompt(question, context, history)
        timeout = deadline_s or self.settings.llm_timeout_s
        try:
            result = await asyncio.wait_for(
                self.client.complete(SYNTHESIZER_SYSTEM, prompt,
                                     max_tokens=self.settings.synthesizer_max_tokens),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            logger.warning("synthesizer timed out after %.1fs", timeout)
            return Synthesis(_fallback_answer(docs, tools), _citations(docs, tools),
                             context, "timeout")
        except LLMError as exc:
            logger.warning("synthesizer failed: %s", exc)
            return Synthesis(_fallback_answer(docs, tools), _citations(docs, tools),
                             context, "error")

        if result.refused:
            return Synthesis(
                "The model declined to answer this request. Rephrasing it as a direct "
                "question about a filed figure usually works.",
                [], context, result.model, refused=True,
            )

        return Synthesis(result.text.strip(), _citations(docs, tools), context, result.model)

    def _prompt(self, question: str, context: str, history: Sequence[Message]) -> str:
        blocks = []
        if history:
            blocks.append("CONVERSATION SO FAR\n" + "\n".join(
                f"{m.role}: {m.content[:400]}" for m in history[-4:]
            ))
        blocks.append(context)
        blocks.append(f"QUESTION\n{question}")
        return "\n\n".join(blocks)


def _citations(docs: Sequence[RetrievedDoc], tools: Sequence[ToolResult]) -> list[Citation]:
    citations = [
        Citation(
            doc_id=d.doc_id,
            company=d.company,
            fiscal_year=d.fiscal_year,
            source=d.source,
            snippet=d.text[:220],
        )
        for d in docs
    ]
    seen = {c.doc_id for c in citations}
    for tool in tools:
        for doc_id in tool.citations:
            if doc_id not in seen:
                seen.add(doc_id)
                citations.append(Citation(doc_id=doc_id, source=f"portfolio tool: {tool.name}"))
    return citations


def _fallback_answer(docs: Sequence[RetrievedDoc], tools: Sequence[ToolResult]) -> str:
    """When LLM-2 is unavailable, report the retrieved facts rather than nothing."""
    parts = ["The language model was unavailable, so here are the retrieved facts unsummarised.\n"]
    for tool in tools:
        if tool.summary:
            parts.append(tool.summary)
    for d in docs[:5]:
        parts.append(f"[{d.doc_id}]\n{d.text}")
    return "\n\n".join(parts)

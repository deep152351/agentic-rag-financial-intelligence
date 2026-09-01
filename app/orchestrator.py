"""The agent loop.

    plan (LLM-1)
      -> [ parallel hybrid retrieval wave -> RRF fusion -> coverage check ] xN
      -> portfolio tools, if the plan asked for them
      -> synthesise (LLM-2)
      -> validate groundedness

N is governed by ``app.budget``, not by the model. The loop re-retrieves only
when coverage is short *and* the budget allows it, and each retry relaxes the
constraints that failed rather than repeating the same search:

    turn 1   plan constraints as resolved
    turn 2   drop the metric filter (the most common over-constraint -- a metric
             may live on a different statement card than the planner guessed)
    turn 3   drop the year filter too, keeping only company and document type

That ordering is deliberate: company identity is the one filter that must never
be relaxed, because returning the right metric for the wrong company is the
failure mode that produces a confident, completely wrong answer.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Sequence

from .budget import TurnBudget
from .config import Settings, get_settings
from .entities import Resolution
from .llm import DualLLM, get_llms
from .planner import LLMPlanner, build_planner
from .retrieval import HybridRetriever, doc_satisfies
from .schemas import (
    AskRequest, AskResponse, Constraints, Message, Plan, RetrievedDoc,
    SubQuery, Trace, TurnRecord,
)
from .synthesizer import Synthesizer
from .validator import validate_answer
from .wealth import PortfolioAnalytics, ToolResult, get_analytics

logger = logging.getLogger(__name__)


@dataclass
class _State:
    docs: dict[str, RetrievedDoc] = field(default_factory=dict)
    order: list[str] = field(default_factory=list)

    def add(self, docs: Sequence[RetrievedDoc]) -> int:
        new = 0
        for d in docs:
            if d.doc_id not in self.docs:
                self.docs[d.doc_id] = d
                self.order.append(d.doc_id)
                new += 1
            elif d.score > self.docs[d.doc_id].score:
                self.docs[d.doc_id] = d
        return new

    def ranked(self, limit: int) -> list[RetrievedDoc]:
        return sorted(self.docs.values(), key=lambda d: -d.score)[:limit]


class Orchestrator:
    def __init__(self, retriever: HybridRetriever | None = None,
                 llms: DualLLM | None = None,
                 analytics: PortfolioAnalytics | None = None,
                 settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.llms = llms or get_llms(self.settings)
        self.retriever = retriever or HybridRetriever(settings=self.settings)
        self.analytics = analytics or get_analytics()
        self.planner = build_planner(self.llms.planner, self.settings)
        self.synthesizer = Synthesizer(self.llms.synthesizer, self.settings)

    # ------------------------------------------------------------------

    async def answer(self, request: AskRequest) -> AskResponse:
        budget = TurnBudget.from_settings(self.settings, request.max_turns)
        top_k = request.top_k or self.settings.top_k

        # ---- plan -----------------------------------------------------
        outcome = await self._plan(request.question, request.history, budget)
        plan, resolutions = outcome.plan, outcome.resolutions

        if plan.intent == "out_of_scope":
            budget.finish("out_of_scope")
            synthesis = await self.synthesizer.run(
                request.question, [], [], request.history, intent="out_of_scope"
            )
            return self._respond(request, synthesis, [], [], plan, resolutions, budget, [])

        # ---- retrieval waves ------------------------------------------
        state = _State()
        subqueries = plan.subqueries
        turn = 0
        while True:
            allowed, reason = budget.can_start_turn()
            if not allowed:
                budget.finish(reason)
                break

            turn += 1
            started = time.perf_counter()
            fusion = await self.retriever.search(
                subqueries, top_k=max(top_k, self.settings.top_k),
                deadline_s=budget.deadline_for_turn(),
            )
            budget.record_searches(fusion.searches_issued)
            new_docs = state.add(fusion.docs)
            coverage = self.retriever.coverage(subqueries, state.ranked(top_k * 2))

            proceed, decision = budget.should_continue(
                coverage, new_docs, self.settings.coverage_target,
                total_docs=len(state.docs),
            )
            budget.record_turn(TurnRecord(
                turn=turn,
                subqueries=[sq.text for sq in subqueries],
                constraints_applied=[sq.constraints for sq in subqueries],
                docs_retrieved=len(fusion.docs),
                new_docs=new_docs,
                coverage=round(coverage, 3),
                elapsed_ms=round((time.perf_counter() - started) * 1000, 1),
                action="retry_relaxed" if proceed else f"stop:{decision}",
            ))

            if not proceed:
                budget.finish(decision)
                break
            subqueries = self._relax(subqueries, state, turn)

        # ---- portfolio tools ------------------------------------------
        tools = self._run_tools(plan, request.question)

        # ---- synthesise + validate ------------------------------------
        docs = state.ranked(top_k)
        budget.record_synthesizer_call()
        synthesis = await self.synthesizer.run(
            request.question, docs, tools, request.history,
            intent=plan.intent,
            deadline_s=max(2.0, budget.remaining_s),
        )
        return self._respond(request, synthesis, docs, tools, plan, resolutions, budget,
                             [t.name for t in tools])

    # ------------------------------------------------------------------

    async def _plan(self, question: str, history: Sequence[Message], budget: TurnBudget):
        if isinstance(self.planner, LLMPlanner):
            budget.record_planner_call()
            return await self.planner.plan(question, history)
        return self.planner.plan(question)

    def _relax(self, subqueries: Sequence[SubQuery], state: _State, turn: int) -> list[SubQuery]:
        """Widen the filters that came back empty, hardest constraint last."""
        relaxed: list[SubQuery] = []
        for sq in subqueries:
            covered = any(doc_satisfies(d, sq.constraints) for d in state.docs.values())
            if covered:
                relaxed.append(sq)
                continue
            c = sq.constraints
            if turn == 1:
                # metric is the likeliest bad guess: the planner may ask for
                # `operating_margin` on a card tagged `operating_income`
                widened = Constraints(
                    tickers=c.tickers, fiscal_years=c.fiscal_years, sectors=c.sectors,
                    doc_types=c.doc_types, account_ids=c.account_ids,
                    soft_metrics=c.soft_metrics + c.metrics,
                )
            else:
                widened = c.relaxed()
            relaxed.append(SubQuery(text=sq.text, constraints=widened,
                                    rationale=f"relaxed@turn{turn}: {sq.rationale}"))
        return relaxed

    def _run_tools(self, plan: Plan, question: str) -> list[ToolResult]:
        if plan.intent != "portfolio" and not plan.portfolio_tools:
            return []

        wanted = plan.portfolio_tools or ["holdings"]
        kwargs = self._tool_kwargs(plan, question)
        results: list[ToolResult] = []
        for tool in wanted:
            try:
                results.append(self.analytics.run(tool, **kwargs))
            except Exception as exc:  # a broken tool must not sink the request
                logger.warning("portfolio tool %s failed: %s", tool, exc)
        return results

    def _tool_kwargs(self, plan: Plan, question: str) -> dict:
        tickers: list[str] = []
        years: list[int] = []
        metrics: list[str] = []
        for sq in plan.subqueries:
            tickers.extend(sq.constraints.tickers)
            years.extend(sq.constraints.fiscal_years)
            metrics.extend(sq.constraints.metrics)

        kwargs: dict = {}
        if tickers:
            kwargs["tickers"] = list(dict.fromkeys(tickers))
        if years:
            kwargs["fiscal_year"] = max(years)
        if metrics:
            kwargs["metric_id"] = metrics[0]
        lowered = question.lower()
        if "sector" in lowered:
            kwargs["group_by"] = "sector"
        elif "account" in lowered:
            kwargs["group_by"] = "account"
        # substring tests, not word matches -- "deteriorated" and "expanding" must hit
        if any(w in lowered for w in ("fell", "declin", "dropped", "worse", "deteriorat",
                                      "contract", "shrank", "shrunk")):
            kwargs["direction"] = "declined"
        elif any(w in lowered for w in ("improv", "rose", "grew", "increas", "better",
                                        "expand", "gain")):
            kwargs["direction"] = "improved"
        return kwargs

    def _respond(self, request: AskRequest, synthesis, docs, tools, plan: Plan,
                 resolutions: Sequence[Resolution], budget: TurnBudget,
                 tool_names: Sequence[str]) -> AskResponse:
        report = validate_answer(synthesis.answer, synthesis.context)
        if report.ungrounded:
            logger.warning("ungrounded figures in answer: %s", report.ungrounded)

        trace = None
        if request.include_trace:
            merged = Constraints()
            for sq in plan.subqueries:
                c = sq.constraints
                merged.tickers.extend(c.tickers)
                merged.fiscal_years.extend(c.fiscal_years)
                merged.metrics.extend(c.metrics)
                merged.doc_types.extend(c.doc_types)
                merged.sectors.extend(c.sectors)
            for f in ("tickers", "fiscal_years", "metrics", "doc_types", "sectors"):
                setattr(merged, f, list(dict.fromkeys(getattr(merged, f))))

            trace = Trace(
                intent=plan.intent,
                resolved_constraints=merged,
                entity_resolution=[r.as_dict() for r in resolutions],
                turns=budget.turns,
                budget=budget.report(),
                retrieved_doc_ids=[d.doc_id for d in docs],
                portfolio_tools_used=list(tool_names),
                models={**self.llms.describe(), "synthesizer_used": synthesis.model},
            )

        return AskResponse(
            answer=synthesis.answer,
            citations=synthesis.citations,
            grounded=report.grounded,
            ungrounded_figures=report.ungrounded,
            trace=trace,
        )

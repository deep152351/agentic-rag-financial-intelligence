"""Wire and internal contracts."""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

Intent = Literal[
    "company_lookup",     # one company, one or more metrics, one or more years
    "comparison",         # several companies or several years side by side
    "trend",              # movement of a metric over time
    "portfolio",          # anything about the client's own book
    "screen",             # find companies/holdings meeting a condition
    "out_of_scope",       # not answerable from filings or the portfolio
]


# ---------------------------------------------------------------------------
# Retrieval constraints -- the object the whole system is organised around
# ---------------------------------------------------------------------------


class Constraints(BaseModel):
    """Strict metadata filters resolved from a query.

    Produced by the planner LLM as free text, then *disciplined* by
    ``app.entities`` into ids that provably exist in the index. Anything the
    resolver could not bind confidently lands in ``soft_*`` instead, where it
    influences ranking but never removes documents.
    """

    tickers: list[str] = Field(default_factory=list)
    fiscal_years: list[int] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list)
    sectors: list[str] = Field(default_factory=list)
    statements: list[str] = Field(default_factory=list)
    doc_types: list[str] = Field(default_factory=list)
    account_ids: list[str] = Field(default_factory=list)

    soft_tickers: list[str] = Field(default_factory=list)
    soft_metrics: list[str] = Field(default_factory=list)

    def is_empty(self) -> bool:
        return not any(
            (self.tickers, self.fiscal_years, self.metrics, self.sectors,
             self.statements, self.doc_types, self.account_ids)
        )

    def relaxed(self) -> "Constraints":
        """Drop the most selective dimensions -- the retry path when recall is poor."""
        return Constraints(
            tickers=self.tickers,
            sectors=self.sectors,
            doc_types=self.doc_types,
            account_ids=self.account_ids,
            soft_tickers=self.soft_tickers,
            soft_metrics=self.soft_metrics + self.metrics,
        )


class SubQuery(BaseModel):
    """One information need. Each fans out to its own parallel hybrid search."""

    text: str
    constraints: Constraints = Field(default_factory=Constraints)
    rationale: str = ""


class Plan(BaseModel):
    intent: Intent = "company_lookup"
    subqueries: list[SubQuery] = Field(default_factory=list)
    portfolio_tools: list[str] = Field(default_factory=list)
    needs_more_context: bool = False
    reasoning: str = ""


# ---------------------------------------------------------------------------
# Retrieval results
# ---------------------------------------------------------------------------


class RetrievedDoc(BaseModel):
    doc_id: str
    text: str
    score: float
    doc_type: str
    ticker: Optional[str] = None
    company: Optional[str] = None
    sector: Optional[str] = None
    fiscal_year: Optional[int] = None
    #: every fiscal year the chunk speaks to. A year-over-year card carries two,
    #: which is why the year filter and the compliance check both read this list
    #: rather than the display-only `fiscal_year` scalar.
    fiscal_years: list[int] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list)
    source: str = ""
    #: which branches found it, for the trace: {"dense": rank, "sparse": rank}
    branch_ranks: dict[str, int] = Field(default_factory=dict)
    matched_subqueries: list[str] = Field(default_factory=list)


class TurnRecord(BaseModel):
    turn: int
    subqueries: list[str]
    constraints_applied: list[Constraints]
    docs_retrieved: int
    new_docs: int
    coverage: float
    elapsed_ms: float
    action: str


class BudgetReport(BaseModel):
    turns_used: int
    turns_allowed: int
    stop_reason: str
    elapsed_ms: float
    deadline_ms: float
    llm_calls: int
    planner_calls: int
    synthesizer_calls: int
    searches_issued: int


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    history: list[Message] = Field(default_factory=list)
    top_k: Optional[int] = Field(default=None, ge=1, le=50)
    max_turns: Optional[int] = Field(default=None, ge=1, le=6)
    include_trace: bool = True


class Citation(BaseModel):
    doc_id: str
    company: Optional[str] = None
    fiscal_year: Optional[int] = None
    source: str = ""
    snippet: str = ""


class Trace(BaseModel):
    intent: str
    resolved_constraints: Constraints
    entity_resolution: list[dict[str, Any]] = Field(default_factory=list)
    turns: list[TurnRecord] = Field(default_factory=list)
    budget: BudgetReport
    retrieved_doc_ids: list[str] = Field(default_factory=list)
    portfolio_tools_used: list[str] = Field(default_factory=list)
    models: dict[str, str] = Field(default_factory=dict)


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    grounded: bool = True
    ungrounded_figures: list[str] = Field(default_factory=list)
    trace: Optional[Trace] = None

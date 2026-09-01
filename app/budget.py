"""Turn-budget guardrails.

An agentic loop without a budget has two failure modes, and they are the two
things that actually break RAG services in practice: it keeps retrieving until
something times out, or it answers instantly from a bad first wave. This module
is the thing that decides, before each wave, whether another wave is affordable
*and* worth it.

Three ceilings, checked in this order:

1.  **Turn count** -- at most ``max_turns`` retrieval waves. Bounds cost.
2.  **Global deadline** -- wall-clock for the whole request, with a reserve
    carved out so the synthesiser always gets to run. A wave that would eat the
    reserve is refused; answering from what we already have beats a 504.
3.  **Marginal value** -- a wave that adds no new documents, or that already hit
    the coverage target, ends the loop even with budget to spare. Spending a turn
    to re-retrieve the same ten documents is pure latency.

The reserve is why the deadline holds. Retrieval is interruptible and synthesis
is not, so time is spent on retrieval only while enough remains to finish.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from .config import Settings, get_settings
from .schemas import BudgetReport, TurnRecord

#: fraction of the global deadline held back for the synthesis call
SYNTHESIS_RESERVE_RATIO = 0.45
#: never reserve less than this, or synthesis gets cut off on short deadlines
MIN_SYNTHESIS_RESERVE_S = 4.0


@dataclass
class TurnBudget:
    max_turns: int
    global_deadline_s: float
    turn_deadline_s: float

    started_at: float = field(default_factory=time.perf_counter)
    turns_used: int = 0
    planner_calls: int = 0
    synthesizer_calls: int = 0
    searches_issued: int = 0
    stop_reason: str = "not_started"
    turns: list[TurnRecord] = field(default_factory=list)

    @classmethod
    def from_settings(cls, settings: Settings | None = None,
                      max_turns: int | None = None) -> "TurnBudget":
        s = settings or get_settings()
        return cls(
            max_turns=max_turns or s.max_turns,
            global_deadline_s=s.global_deadline_s,
            turn_deadline_s=s.turn_deadline_s,
        )

    # -- clock -------------------------------------------------------------

    @property
    def elapsed_s(self) -> float:
        return time.perf_counter() - self.started_at

    @property
    def remaining_s(self) -> float:
        return max(0.0, self.global_deadline_s - self.elapsed_s)

    @property
    def synthesis_reserve_s(self) -> float:
        return max(MIN_SYNTHESIS_RESERVE_S, self.global_deadline_s * SYNTHESIS_RESERVE_RATIO)

    @property
    def retrieval_time_left_s(self) -> float:
        """Time still spendable on retrieval, keeping the synthesis reserve intact."""
        return max(0.0, self.remaining_s - self.synthesis_reserve_s)

    # -- gates -------------------------------------------------------------

    def can_start_turn(self) -> tuple[bool, str]:
        if self.turns_used >= self.max_turns:
            return False, "turn_limit_reached"
        if self.retrieval_time_left_s <= 0.5:
            return False, "deadline_reserved_for_synthesis"
        return True, "ok"

    def deadline_for_turn(self) -> float:
        """How long this wave may run: the smaller of the per-turn and remaining budget."""
        return max(0.5, min(self.turn_deadline_s, self.retrieval_time_left_s))

    def should_continue(self, coverage: float, new_docs: int,
                        coverage_target: float, total_docs: int = 1) -> tuple[bool, str]:
        """Marginal-value gate, evaluated after a wave completes."""
        if coverage >= coverage_target:
            return False, "coverage_target_met"
        # "no new documents" means the wave was redundant -- but a wave that
        # returned *nothing at all* is an over-constrained filter, which is
        # exactly the case relaxation exists to rescue. Stopping there would
        # answer "not found" without ever widening the search.
        if new_docs == 0 and total_docs > 0:
            return False, "no_new_documents"
        allowed, reason = self.can_start_turn()
        if not allowed:
            return False, reason
        return True, "coverage_below_target"

    # -- recording ---------------------------------------------------------

    def record_turn(self, record: TurnRecord) -> None:
        self.turns_used += 1
        self.turns.append(record)

    def record_planner_call(self) -> None:
        self.planner_calls += 1

    def record_synthesizer_call(self) -> None:
        self.synthesizer_calls += 1

    def record_searches(self, n: int) -> None:
        self.searches_issued += n

    def finish(self, reason: str) -> None:
        if self.stop_reason in ("not_started", ""):
            self.stop_reason = reason

    def report(self) -> BudgetReport:
        return BudgetReport(
            turns_used=self.turns_used,
            turns_allowed=self.max_turns,
            stop_reason=self.stop_reason,
            elapsed_ms=round(self.elapsed_s * 1000, 1),
            deadline_ms=round(self.global_deadline_s * 1000, 1),
            llm_calls=self.planner_calls + self.synthesizer_calls,
            planner_calls=self.planner_calls,
            synthesizer_calls=self.synthesizer_calls,
            searches_issued=self.searches_issued,
        )

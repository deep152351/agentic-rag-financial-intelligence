"""Turn-budget guardrails: the ceilings that keep the agent loop bounded."""

from __future__ import annotations

import time

import pytest

from app.budget import MIN_SYNTHESIS_RESERVE_S, TurnBudget
from app.schemas import TurnRecord


def budget(**kwargs) -> TurnBudget:
    defaults = dict(max_turns=3, global_deadline_s=25.0, turn_deadline_s=8.0)
    defaults.update(kwargs)
    return TurnBudget(**defaults)


def a_turn(n: int = 1) -> TurnRecord:
    return TurnRecord(turn=n, subqueries=[], constraints_applied=[], docs_retrieved=1,
                      new_docs=1, coverage=0.0, elapsed_ms=1.0, action="test")


class TestTurnCeiling:
    def test_allows_turns_up_to_the_limit(self):
        b = budget(max_turns=2)
        assert b.can_start_turn()[0]
        b.record_turn(a_turn(1))
        assert b.can_start_turn()[0]
        b.record_turn(a_turn(2))
        allowed, reason = b.can_start_turn()
        assert not allowed and reason == "turn_limit_reached"

    def test_records_turns(self):
        b = budget()
        b.record_turn(a_turn())
        assert b.turns_used == 1 and len(b.turns) == 1


class TestDeadline:
    def test_reserve_is_held_back_for_synthesis(self):
        b = budget(global_deadline_s=20.0)
        assert b.synthesis_reserve_s >= MIN_SYNTHESIS_RESERVE_S
        assert b.retrieval_time_left_s < b.remaining_s

    def test_short_deadline_still_reserves_a_floor(self):
        b = budget(global_deadline_s=2.0)
        assert b.synthesis_reserve_s == MIN_SYNTHESIS_RESERVE_S
        allowed, reason = b.can_start_turn()
        assert not allowed and reason == "deadline_reserved_for_synthesis"

    def test_turn_deadline_never_exceeds_remaining_retrieval_time(self):
        b = budget(global_deadline_s=10.0, turn_deadline_s=30.0)
        # read the budget first: the clock advances between the two calls, so
        # sampling it afterwards would compare against a smaller number
        available = b.retrieval_time_left_s
        assert b.deadline_for_turn() <= available

    def test_turn_deadline_respects_the_per_turn_cap(self):
        b = budget(global_deadline_s=600.0, turn_deadline_s=5.0)
        assert b.deadline_for_turn() == pytest.approx(5.0, abs=0.1)

    def test_elapsed_time_advances(self):
        b = budget()
        before = b.remaining_s
        time.sleep(0.05)
        assert b.remaining_s < before


class TestMarginalValueGate:
    def test_stops_once_coverage_target_is_met(self):
        proceed, reason = budget().should_continue(coverage=0.9, new_docs=5, coverage_target=0.85)
        assert not proceed and reason == "coverage_target_met"

    def test_stops_when_a_wave_adds_nothing(self):
        """Re-retrieving the same documents is pure latency."""
        proceed, reason = budget().should_continue(coverage=0.2, new_docs=0, coverage_target=0.85)
        assert not proceed and reason == "no_new_documents"

    def test_continues_when_coverage_is_short_and_budget_remains(self):
        proceed, reason = budget().should_continue(coverage=0.4, new_docs=3, coverage_target=0.85)
        assert proceed and reason == "coverage_below_target"

    def test_turn_limit_overrides_low_coverage(self):
        b = budget(max_turns=1)
        b.record_turn(a_turn())
        proceed, reason = b.should_continue(coverage=0.0, new_docs=9, coverage_target=0.85)
        assert not proceed and reason == "turn_limit_reached"


class TestReport:
    def test_report_totals_llm_calls(self):
        b = budget()
        b.record_planner_call()
        b.record_synthesizer_call()
        b.record_searches(12)
        b.finish("coverage_target_met")
        report = b.report()
        assert report.llm_calls == 2
        assert report.planner_calls == 1 and report.synthesizer_calls == 1
        assert report.searches_issued == 12
        assert report.stop_reason == "coverage_target_met"

    def test_first_finish_reason_wins(self):
        b = budget()
        b.finish("turn_limit_reached")
        b.finish("something_else")
        assert b.report().stop_reason == "turn_limit_reached"

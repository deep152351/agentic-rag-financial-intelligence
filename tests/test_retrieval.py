"""RRF fusion and constraint satisfaction -- the pure, testable core of retrieval."""

from __future__ import annotations

import pytest

from app.retrieval import BRANCH_WEIGHTS, doc_satisfies, reciprocal_rank_fusion
from app.schemas import Constraints, RetrievedDoc
from app.vectorstore import build_filter


def doc(doc_id: str, **kwargs) -> RetrievedDoc:
    base = dict(text="", score=0.0, doc_type="income_statement")
    base.update(kwargs)
    return RetrievedDoc(doc_id=doc_id, **base)


class TestReciprocalRankFusion:
    def test_rank_one_scores_one_over_k_plus_one(self):
        fused = reciprocal_rank_fusion([("q", "dense", ["a", "b"])], k=60)
        assert fused["a"] == pytest.approx(1 / 61)
        assert fused["b"] == pytest.approx(1 / 62)

    def test_disagreeing_branches_are_symmetric(self):
        fused = reciprocal_rank_fusion([
            ("q", "dense", ["a", "b", "c"]),
            ("q", "sparse", ["c", "b", "a"]),
        ], k=60)
        # a and c mirror each other (1st on one branch, 3rd on the other), so
        # they must tie. 1/(k+1) + 1/(k+3) slightly exceeds 2/(k+2) because the
        # rank term is convex -- one strong opinion outweighs two lukewarm ones.
        assert fused["a"] == pytest.approx(fused["c"])
        assert fused["a"] > fused["b"]

    def test_agreement_beats_a_single_strong_hit(self):
        fused = reciprocal_rank_fusion([
            ("q", "dense", ["agreed", "solo"]),
            ("q", "sparse", ["agreed", "other"]),
        ], k=60)
        # found first by both branches vs first by one only
        assert fused["agreed"] > fused["solo"]
        assert fused["agreed"] == pytest.approx(2 / 61)

    def test_fusion_is_scale_free(self):
        """Only ranks matter -- that is the entire reason for choosing RRF."""
        a = reciprocal_rank_fusion([("q", "dense", ["x", "y"])], k=60)
        b = reciprocal_rank_fusion([("q", "sparse", ["x", "y"])], k=60)
        assert a == b

    def test_multiple_subqueries_accumulate(self):
        fused = reciprocal_rank_fusion([
            ("sq1", "dense", ["shared", "only1"]),
            ("sq2", "dense", ["shared", "only2"]),
        ], k=60)
        assert fused["shared"] > fused["only1"]
        assert fused["shared"] == pytest.approx(2 / 61)

    def test_weights_are_applied(self):
        fused = reciprocal_rank_fusion(
            [("q", "dense", ["a"]), ("q", "sparse", ["b"])],
            k=60, weights={"dense": 2.0, "sparse": 1.0},
        )
        assert fused["a"] == pytest.approx(2 / 61)
        assert fused["b"] == pytest.approx(1 / 61)

    def test_empty_input(self):
        assert reciprocal_rank_fusion([]) == {}

    def test_default_weights_are_balanced(self):
        assert BRANCH_WEIGHTS["dense"] == BRANCH_WEIGHTS["sparse"]

    def test_larger_k_flattens_rank_advantage(self):
        tight = reciprocal_rank_fusion([("q", "d", ["a", "b"])], k=1)
        loose = reciprocal_rank_fusion([("q", "d", ["a", "b"])], k=1000)
        assert tight["a"] / tight["b"] > loose["a"] / loose["b"]


class TestConstraintSatisfaction:
    def test_ticker_mismatch_fails(self):
        assert not doc_satisfies(doc("d", ticker="MSFT"), Constraints(tickers=["AAPL"]))

    def test_ticker_match_is_case_insensitive(self):
        assert doc_satisfies(doc("d", ticker="aapl"), Constraints(tickers=["AAPL"]))

    def test_year_matches_against_the_full_year_list(self):
        """A FY2025 year-over-year card also speaks to FY2024."""
        yoy = doc("d", ticker="AAPL", fiscal_year=2025, fiscal_years=[2025, 2024])
        assert doc_satisfies(yoy, Constraints(fiscal_years=[2024]))
        assert not doc_satisfies(yoy, Constraints(fiscal_years=[2021]))

    def test_metric_intersection(self):
        d = doc("d", metrics=["revenue", "net_income"])
        assert doc_satisfies(d, Constraints(metrics=["net_income"]))
        assert not doc_satisfies(d, Constraints(metrics=["capex"]))

    def test_empty_constraints_accept_everything(self):
        assert doc_satisfies(doc("d"), Constraints())

    def test_all_conditions_must_hold(self):
        d = doc("d", ticker="AAPL", fiscal_year=2024, fiscal_years=[2024], metrics=["revenue"])
        assert doc_satisfies(d, Constraints(tickers=["AAPL"], fiscal_years=[2024],
                                            metrics=["revenue"]))
        assert not doc_satisfies(d, Constraints(tickers=["AAPL"], fiscal_years=[2023],
                                                metrics=["revenue"]))


class TestFilterConstruction:
    def test_empty_constraints_produce_no_filter(self):
        assert build_filter(Constraints()) is None

    def test_soft_hints_never_become_filters(self):
        """A low-confidence match must not silently remove documents."""
        soft = Constraints(soft_tickers=["AAPL"], soft_metrics=["revenue"])
        assert build_filter(soft) is None

    def test_strict_fields_become_conditions(self):
        flt = build_filter(Constraints(tickers=["AAPL"], fiscal_years=[2024]))
        assert flt is not None and len(flt.must) == 2

    def test_tickers_are_upper_cased(self):
        flt = build_filter(Constraints(tickers=["aapl"]))
        assert flt.must[0].match.any == ["AAPL"]


class TestConstraintRelaxation:
    def test_relaxed_keeps_company_but_drops_year_and_metric(self):
        original = Constraints(tickers=["AAPL"], fiscal_years=[2024],
                               metrics=["operating_margin"], doc_types=["ratios"])
        relaxed = original.relaxed()
        assert relaxed.tickers == ["AAPL"], "company identity must never be relaxed"
        assert relaxed.fiscal_years == []
        assert relaxed.metrics == []
        assert "operating_margin" in relaxed.soft_metrics

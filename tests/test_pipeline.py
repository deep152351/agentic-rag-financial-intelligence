"""Integration: planner decisions, data integrity, and the live API surface.

These need the built corpus and the Qdrant index, so they are skipped rather than
failed when the data pipeline has not been run yet.
"""

from __future__ import annotations

import json

import pytest

from app.config import DATA_DIR, get_settings
from app.entities import get_resolver
from app.metrics import ALL_METRICS, METRICS, format_value
from app.planner import DeterministicPlanner

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

CORPUS = DATA_DIR / "corpus.jsonl"
FINANCIALS = DATA_DIR / "financials.json"
requires_data = pytest.mark.skipif(
    not (CORPUS.exists() and FINANCIALS.exists()),
    reason="run scripts.fetch_sec, scripts.build_corpus, scripts.build_portfolio first",
)


@pytest.fixture(scope="module")
def planner():
    return DeterministicPlanner(get_resolver(), get_settings())


# ---------------------------------------------------------------------------
# metric registry
# ---------------------------------------------------------------------------


class TestMetricRegistry:
    def test_ids_are_unique(self):
        ids = [m.id for m in ALL_METRICS]
        assert len(ids) == len(set(ids))

    def test_reported_metrics_have_tags(self):
        for m in ALL_METRICS:
            if not m.is_derived:
                assert m.tags, f"{m.id} has no us-gaap tags"

    def test_derived_metrics_reference_real_metrics(self):
        for m in ALL_METRICS:
            for dependency in m.derived_from:
                assert dependency in METRICS, f"{m.id} depends on unknown {dependency}"

    def test_formatting_by_unit(self):
        assert format_value("revenue", 391_035_000_000) == "$391.04B"
        assert format_value("operating_margin", 0.3151) == "31.5%"
        assert format_value("eps_diluted", 6.08) == "$6.08"
        assert format_value("current_ratio", 0.867) == "0.87x"

    def test_negative_currency_keeps_its_sign(self):
        assert format_value("net_income", -2_700_000_000).startswith("-$")


# ---------------------------------------------------------------------------
# planner
# ---------------------------------------------------------------------------


class TestPlanner:
    @pytest.mark.parametrize("question,intent", [
        ("What was Apple's operating margin in FY2024?", "company_lookup"),
        ("Compare Apple and Microsoft revenue in FY2024", "comparison"),
        ("How has Tesla free cash flow trended since 2019?", "trend"),
        ("What is my portfolio exposure by sector?", "portfolio"),
        ("Should I buy Nvidia stock?", "out_of_scope"),
        ("What will Apple's share price be next quarter?", "out_of_scope"),
        ("Is Microsoft a good investment?", "out_of_scope"),
    ])
    def test_intent_classification(self, planner, question, intent):
        assert planner.plan(question).plan.intent == intent

    def test_comparison_fans_out_per_company_and_year(self, planner):
        plan = planner.plan("Compare Microsoft and Nvidia revenue from 2022 to 2024").plan
        pairs = {(tuple(sq.constraints.tickers), tuple(sq.constraints.fiscal_years))
                 for sq in plan.subqueries}
        assert len(pairs) == 6, "2 companies x 3 years should be 6 sub-queries"

    def test_subquery_count_is_capped(self, planner):
        plan = planner.plan(
            "Compare Apple, Microsoft, Nvidia and Amazon from 2019 to 2025 on revenue"
        ).plan
        assert len(plan.subqueries) <= get_settings().max_subqueries

    def test_constraints_are_resolved_to_real_ids(self, planner):
        plan = planner.plan("What was coke's op margin in FY'23?").plan
        constraints = plan.subqueries[0].constraints
        assert constraints.tickers == ["KO"]
        assert constraints.fiscal_years == [2023]
        assert constraints.metrics == ["operating_margin"]

    def test_portfolio_tools_are_selected(self, planner):
        assert "concentration" in planner.plan("How concentrated is my portfolio?").plan.portfolio_tools
        assert "exposure" in planner.plan("What is my sector exposure?").plan.portfolio_tools


# ---------------------------------------------------------------------------
# corpus integrity
# ---------------------------------------------------------------------------


@requires_data
class TestCorpusIntegrity:
    @pytest.fixture(scope="class")
    def chunks(self):
        with CORPUS.open(encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]

    def test_doc_ids_are_unique(self, chunks):
        ids = [c["doc_id"] for c in chunks]
        assert len(ids) == len(set(ids))

    def test_every_chunk_has_filterable_metadata(self, chunks):
        for c in chunks:
            assert c["ticker"] and c["doc_type"] and c["text"]
            assert isinstance(c["fiscal_years"], list)

    def test_declared_metrics_exist_in_the_registry(self, chunks):
        for c in chunks:
            for metric_id in c["metrics"]:
                assert metric_id in METRICS

    def test_metrics_match_their_statement(self, chunks):
        """A chunk tagged `balance_sheet` must not claim an income-statement metric."""
        for c in chunks:
            if c["doc_type"] not in ("income_statement", "balance_sheet", "cash_flow"):
                continue
            for metric_id in c["metrics"]:
                assert METRICS[metric_id].statement == c["doc_type"]

    def test_fiscal_years_are_plausible(self, chunks):
        for c in chunks:
            for year in c["fiscal_years"]:
                assert 2015 <= year <= 2030


@requires_data
class TestFinancialsIntegrity:
    @pytest.fixture(scope="class")
    def financials(self):
        return json.loads(FINANCIALS.read_text(encoding="utf-8"))

    def test_derived_ratios_are_consistent(self, financials):
        """operating_margin must equal operating_income / revenue, everywhere."""
        checked = 0
        for record in financials.values():
            for metrics in record["years"].values():
                if not {"operating_margin", "operating_income", "revenue"} <= metrics.keys():
                    continue
                expected = metrics["operating_income"]["value"] / metrics["revenue"]["value"]
                assert metrics["operating_margin"]["value"] == pytest.approx(expected, rel=1e-9)
                checked += 1
        assert checked > 100

    def test_free_cash_flow_is_ocf_minus_capex(self, financials):
        for record in financials.values():
            for metrics in record["years"].values():
                if not {"free_cash_flow", "operating_cash_flow", "capex"} <= metrics.keys():
                    continue
                expected = metrics["operating_cash_flow"]["value"] - metrics["capex"]["value"]
                assert metrics["free_cash_flow"]["value"] == pytest.approx(expected, rel=1e-9)

    def test_per_share_series_have_no_split_discontinuity(self, financials):
        """A >3x jump in share count between adjacent years means a missed split."""
        for ticker, record in financials.items():
            years = sorted(int(y) for y in record["years"])
            counts = [(y, record["years"][str(y)]["shares_diluted"]["value"])
                      for y in years if "shares_diluted" in record["years"][str(y)]]
            for (_, a), (yb, b) in zip(counts, counts[1:]):
                if a and b:
                    ratio = max(a, b) / min(a, b)
                    assert ratio < 3.0, f"{ticker} share count jumps {ratio:.1f}x at FY{yb}"


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------


@requires_data
class TestAPI:
    @pytest.fixture(scope="class")
    def client(self):
        from fastapi.testclient import TestClient

        from app.main import app

        with TestClient(app) as c:
            yield c

    def test_health(self, client):
        body = client.get("/health").json()
        assert body["status"] in ("ok", "degraded")
        assert "planner_model" in body["llm"]
        assert body["retrieval"]["rrf_k"] == get_settings().rrf_k

    def test_universe_lists_indexed_companies(self, client):
        body = client.get("/universe").json()
        assert body["count"] > 30
        assert any(c["ticker"] == "AAPL" for c in body["companies"])

    def test_portfolio_is_labelled_synthetic(self, client):
        body = client.get("/portfolio").json()
        assert "SYNTHETIC" in body["disclaimer"]
        assert body["valuation_basis"] == "cost"

    def test_ask_returns_a_grounded_answer_with_a_trace(self, client):
        response = client.post("/ask", json={
            "question": "What was Apple's operating margin in FY2024?",
        })
        assert response.status_code == 200
        body = response.json()
        assert body["answer"]
        assert body["trace"]["intent"] == "company_lookup"
        assert body["trace"]["resolved_constraints"]["tickers"] == ["AAPL"]
        assert body["trace"]["resolved_constraints"]["fiscal_years"] == [2024]
        assert any(c["doc_id"].startswith("AAPL-FY2024") for c in body["citations"])
        assert body["trace"]["budget"]["turns_used"] >= 1

    def test_out_of_scope_is_refused_without_spending_budget(self, client):
        body = client.post("/ask", json={"question": "Should I buy Tesla stock?"}).json()
        assert body["trace"]["intent"] == "out_of_scope"
        assert body["trace"]["budget"]["turns_used"] == 0
        assert body["trace"]["budget"]["searches_issued"] == 0

    def test_portfolio_question_runs_tools(self, client):
        body = client.post("/ask", json={"question": "What is my exposure by sector?"}).json()
        assert "exposure" in body["trace"]["portfolio_tools_used"]

    def test_empty_question_is_rejected(self, client):
        assert client.post("/ask", json={"question": ""}).status_code == 422

    def test_max_turns_is_bounded_by_the_schema(self, client):
        assert client.post("/ask", json={"question": "hi", "max_turns": 99}).status_code == 422

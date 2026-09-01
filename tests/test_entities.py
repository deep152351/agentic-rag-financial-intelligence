"""Entity resolution: the layer that decides what the index may return."""

from __future__ import annotations

import pytest

from app.entities import get_resolver


@pytest.fixture(scope="module")
def resolver():
    return get_resolver()


class TestCompanyResolution:
    @pytest.mark.parametrize("surface,expected", [
        ("apple", "AAPL"),
        ("Apple Inc.", "AAPL"),
        ("AAPL", "AAPL"),
        ("coca cola", "KO"),
        ("coke", "KO"),
        ("p&g", "PG"),
        ("procter and gamble", "PG"),
        ("jp morgan", "JPM"),
        ("goldman", "GS"),
        ("exxonmobil", "XOM"),
        ("berkshire", "BRK-B"),
    ])
    def test_known_aliases(self, resolver, surface, expected):
        assert resolver.resolve_company(surface).resolved == expected

    @pytest.mark.parametrize("typo,expected", [
        ("Micorsoft", "MSFT"),
        ("APPL", "AAPL"),
        ("nvidia corp", "NVDA"),
        ("tesla motors", "TSLA"),
    ])
    def test_typos_and_variants(self, resolver, typo, expected):
        result = resolver.resolve_company(typo)
        assert result.resolved == expected
        assert result.strict, "a confident match should become a strict filter"

    def test_unknown_company_is_dropped(self, resolver):
        result = resolver.resolve_company("Hooli Incorporated")
        assert not result.strict
        assert result.as_dict()["applied_as"] in ("dropped", "soft_hint")

    def test_exact_match_short_circuits(self, resolver):
        result = resolver.resolve_company("MSFT")
        assert result.score == 100.0 and result.strict


class TestFreeTextScan:
    def test_finds_multiple_companies(self, resolver):
        tickers, _ = resolver.resolve_query(
            "How did Apple and Microsoft compare on operating margin in FY2024?"
        )
        assert set(tickers) == {"AAPL", "MSFT"}

    @pytest.mark.parametrize("question,expected", [
        ("What was MA's cash in FY2023?", "MA"),
        ("What was chevron's diluted EPS?", "CVX"),
        ("What was Johnson & Johnson's total assets?", "JNJ"),
        ("What was wal-mart's equity?", "WMT"),
    ])
    def test_possessive_forms(self, resolver, question, expected):
        """`MA's` must still resolve -- this was a real miss worth 56 eval cases."""
        tickers, _ = resolver.resolve_query(question)
        assert expected in tickers

    def test_single_letter_ticker_needs_capital(self, resolver):
        assert "V" in resolver.resolve_query("What was V's net margin in FY2019?")[0]
        # a lone lowercase letter must never resolve to Visa
        assert "V" not in resolver.resolve_query("give me a summary of revenue")[0]

    def test_stopwords_do_not_resolve(self, resolver):
        tickers, _ = resolver.resolve_query("show me all the most recent results")
        assert tickers == []


class TestMetricResolution:
    @pytest.mark.parametrize("surface,expected", [
        ("op margin", "operating_margin"),
        ("bottom line", "net_income"),
        ("top line", "revenue"),
        ("cash from ops", "operating_cash_flow"),
        ("free cashflow", "free_cash_flow"),
        ("r and d", "rnd_expense"),
        ("debt to equity", "debt_to_equity"),
        ("EBIT", "operating_income"),
    ])
    def test_colloquial_metrics(self, resolver, surface, expected):
        assert resolver.resolve_metric(surface).resolved == expected

    def test_nonsense_metric_dropped(self, resolver):
        assert not resolver.resolve_metric("blorptitude").strict


class TestYearResolution:
    @pytest.mark.parametrize("text,expected", [
        ("revenue in FY24", [2024]),
        ("fiscal year 2022 results", [2022]),
        ("FY'23 numbers", [2023]),
        ("from 2020 to 2023", [2020, 2021, 2022, 2023]),
        ("between fy2019 and fy2021", [2019, 2020, 2021]),
        ("compare 2023 and 2024", [2023, 2024]),
        ("no year mentioned", []),
    ])
    def test_year_forms(self, resolver, text, expected):
        assert resolver.resolve_years(text)[0] == expected

    def test_since_is_open_ended(self, resolver):
        years, _ = resolver.resolve_years("free cash flow since 2022")
        assert years == [y for y in resolver.known_years if y >= 2022]

    def test_latest_resolves_to_newest_year(self, resolver):
        years, _ = resolver.resolve_years("the latest annual report")
        assert years == [max(resolver.known_years)]

    def test_unknown_year_is_not_invented(self, resolver):
        assert resolver.resolve_years("revenue in FY1804")[0] == []


class TestPortfolioDetection:
    @pytest.mark.parametrize("text,expected", [
        ("what is in my IRA", True),
        ("my portfolio exposure", True),
        ("which positions do we hold", True),
        ("what was Apple revenue", False),
        ("compare Microsoft and Oracle", False),
    ])
    def test_portfolio_intent(self, resolver, text, expected):
        assert resolver.mentions_portfolio(text) is expected

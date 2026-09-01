"""Groundedness validation -- the guard against fluent, confident, wrong numbers."""

from __future__ import annotations

import pytest

from app.validator import extract_figures, validate_answer


class TestFigureExtraction:
    def test_currency_suffixes_scale(self):
        figures = {f.raw: f.magnitude for f in extract_figures("$391.04B, $12.5M, $1.2 trillion")}
        assert figures["$391.04B"] == pytest.approx(391.04e9)
        assert figures["$12.5M"] == pytest.approx(12.5e6)
        assert figures["$1.2 trillion"] == pytest.approx(1.2e12)

    def test_four_digit_numbers_are_not_split(self):
        """`\\d{1,3}(?:,\\d{3})*` matched "201" out of "2019" and left a stray "9"."""
        raws = [f.raw for f in extract_figures("Revenue of 6789 million")]
        assert any("6789" in r for r in raws)
        assert "201" not in raws

    def test_bare_years_are_ignored(self):
        assert extract_figures("covering FY2019 through FY2025") == []

    def test_percentages_are_typed_separately(self):
        kinds = {f.raw: f.kind for f in extract_figures("margin of 31.5% on $391.04B")}
        assert kinds["31.5%"] == "percent"
        assert kinds["$391.04B"] == "currency"

    def test_comma_grouped_numbers(self):
        figures = {f.raw: f.magnitude for f in extract_figures("total of 1,234,567 units")}
        assert any(v == pytest.approx(1_234_567) for v in figures.values())


class TestGrounding:
    def test_exact_restatement_is_grounded(self):
        report = validate_answer(
            "Apple's FY2024 revenue was $391.04B.",
            "Total revenue: $391.04B",
        )
        assert report.grounded and report.checked == 1

    def test_rounding_is_tolerated(self):
        report = validate_answer(
            "Revenue was roughly $391 billion.",
            "Total revenue: $391.04B",
        )
        assert report.grounded

    def test_invented_figure_is_caught(self):
        report = validate_answer(
            "Apple's FY2024 revenue was $412.00B.",
            "Total revenue: $391.04B",
        )
        assert not report.grounded
        assert "$412.00B" in report.ungrounded

    def test_percentage_is_not_matched_against_currency(self):
        report = validate_answer("The margin was 391.04%.", "Total revenue: $391.04B")
        assert not report.grounded

    def test_answer_without_figures_is_trivially_grounded(self):
        report = validate_answer("I could not find that filing.", "some context")
        assert report.grounded and report.checked == 0

    def test_partial_grounding_is_reported(self):
        report = validate_answer(
            "Revenue was $391.04B and net income was $999.00B.",
            "Total revenue: $391.04B | Net income: $93.74B",
        )
        assert not report.grounded
        assert report.ungrounded == ["$999.00B"]
        assert report.rate == pytest.approx(0.5)

    def test_scale_restatement_matches(self):
        """"391" in a sentence about billions still refers to $391.04B."""
        report = validate_answer("Revenue reached 391 (in billions).", "revenue $391.04B")
        assert report.grounded

"""Portfolio tools: the wealth-intelligence half.

These are deterministic computations over two files -- the ownership book
(``portfolio.json``, synthetic) and the fundamentals (``financials.json``, real
SEC data). They are not retrieval, and they are not the LLM's arithmetic: the
numbers are computed here and handed to the synthesiser as facts, because an LLM
summing 30 positions is a source of quiet errors that reads perfectly fluently.

The interesting one is ``look_through``. Owning N shares of a business is owning
a claim on N/(shares outstanding) of everything it earns, so a portfolio's
look-through revenue or free cash flow is a real, checkable quantity computed
entirely from reported figures::

    look-through(metric) = sum over holdings of  shares_held x metric / diluted_shares

That gives "your portfolio's share of FY2024 net income" without needing a single
market price -- which this project deliberately does not have.
"""

from __future__ import annotations

import functools
import json
from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

from .config import DATA_DIR
from .metrics import METRICS, format_value

GroupBy = Literal["sector", "ticker", "account", "industry"]


@dataclass
class ToolResult:
    name: str
    summary: str
    data: dict[str, Any] = field(default_factory=dict)
    citations: list[str] = field(default_factory=list)


class PortfolioAnalytics:
    def __init__(self, book: dict, financials: dict) -> None:
        self.book = book
        self.financials = financials
        self.accounts = book["accounts"]
        self.positions = [
            {**p, "account_id": a["account_id"], "account_name": a["name"], "account_type": a["type"]}
            for a in self.accounts for p in a["positions"]
        ]
        self.total_cost = book["totals"]["cost_basis"]
        self.total_at_cost = book["totals"]["value_at_cost"]

    # -- helpers -----------------------------------------------------------

    def _select(self, tickers: Sequence[str] = (), account_ids: Sequence[str] = (),
                sectors: Sequence[str] = ()) -> list[dict]:
        rows = self.positions
        if tickers:
            wanted = {t.upper() for t in tickers}
            rows = [p for p in rows if p["ticker"].upper() in wanted]
        if account_ids:
            rows = [p for p in rows if p["account_id"] in set(account_ids)]
        if sectors:
            rows = [p for p in rows if p["sector"] in set(sectors)]
        return rows

    def _metric(self, ticker: str, metric_id: str, fiscal_year: int) -> float | None:
        record = self.financials.get(ticker)
        if not record:
            return None
        entry = record["years"].get(str(fiscal_year), {}).get(metric_id)
        return entry["value"] if entry else None

    def latest_year(self, ticker: str) -> int | None:
        record = self.financials.get(ticker)
        if not record or not record["years"]:
            return None
        return max(int(y) for y in record["years"])

    @property
    def common_latest_year(self) -> int:
        years = [y for t in {p["ticker"] for p in self.positions}
                 if (y := self.latest_year(t)) is not None]
        return max(set(years), key=years.count) if years else 2024

    # -- tools -------------------------------------------------------------

    def holdings(self, tickers: Sequence[str] = (), account_ids: Sequence[str] = (),
                 sectors: Sequence[str] = ()) -> ToolResult:
        rows = self._select(tickers, account_ids, sectors)
        if not rows:
            return ToolResult("holdings", "No matching positions in the client's book.")
        rows = sorted(rows, key=lambda p: -p["cost_basis"])
        lines = [
            f"- {p['company']} ({p['ticker']}): {p['shares']:,} shares, cost basis "
            f"${p['cost_basis']:,.0f} ({p['cost_basis'] / self.total_at_cost * 100:.2f}% of book), "
            f"held in {p['account_name']}"
            for p in rows
        ]
        total = sum(p["cost_basis"] for p in rows)
        summary = (
            f"Portfolio holdings ({len(rows)} positions, ${total:,.0f} at cost, "
            f"{total / self.total_at_cost * 100:.1f}% of the ${self.total_at_cost:,.0f} book):\n"
            + "\n".join(lines)
        )
        return ToolResult("holdings", summary, {"positions": rows, "total_cost_basis": total})

    def exposure(self, group_by: GroupBy = "sector", tickers: Sequence[str] = (),
                 sectors: Sequence[str] = ()) -> ToolResult:
        rows = self._select(tickers=tickers, sectors=sectors)
        if not rows:
            return ToolResult("exposure", "No matching positions to compute exposure over.")
        key = {"sector": "sector", "ticker": "ticker",
               "account": "account_name", "industry": "industry"}[group_by]
        buckets: dict[str, float] = {}
        for p in rows:
            buckets[p[key]] = buckets.get(p[key], 0.0) + p["cost_basis"]
        ordered = sorted(buckets.items(), key=lambda kv: -kv[1])
        equity_total = sum(buckets.values())
        lines = [
            f"- {name}: ${value:,.0f} ({value / equity_total * 100:.1f}% of equities, "
            f"{value / self.total_at_cost * 100:.1f}% of total book)"
            for name, value in ordered
        ]
        summary = (
            f"Exposure by {group_by} (at cost; cash ${self.book['totals']['cash']:,.0f} "
            f"excluded from the equity percentages):\n" + "\n".join(lines)
        )
        return ToolResult("exposure", summary,
                          {"group_by": group_by, "buckets": dict(ordered),
                           "equity_total": equity_total})

    def look_through(self, metric_id: str = "net_income", fiscal_year: int | None = None,
                     tickers: Sequence[str] = (), sectors: Sequence[str] = ()) -> ToolResult:
        """The portfolio's proportional claim on a reported fundamental."""
        if metric_id not in METRICS:
            return ToolResult("look_through", f"Unknown metric '{metric_id}'.")
        metric = METRICS[metric_id]
        if metric.unit not in ("USD",):
            return ToolResult(
                "look_through",
                f"{metric.label} is a ratio, not an amount; look-through aggregation "
                f"only applies to currency figures.",
            )

        fiscal_year = fiscal_year or self.common_latest_year
        rows = self._select(tickers=tickers, sectors=sectors)
        contributions, skipped = [], []
        for p in rows:
            value = self._metric(p["ticker"], metric_id, fiscal_year)
            shares_out = self._metric(p["ticker"], "shares_diluted", fiscal_year)
            if value is None or not shares_out:
                skipped.append(p["ticker"])
                continue
            share = p["shares"] / shares_out
            contributions.append({
                "ticker": p["ticker"], "company": p["company"], "sector": p["sector"],
                "shares_held": p["shares"], "shares_outstanding": shares_out,
                "ownership_fraction": share,
                "company_value": value,
                "attributable": value * share,
            })

        if not contributions:
            return ToolResult(
                "look_through",
                f"No holdings have both {metric.label} and a diluted share count "
                f"reported for FY{fiscal_year}.",
            )

        contributions.sort(key=lambda c: -c["attributable"])
        total = sum(c["attributable"] for c in contributions)
        lines = [
            f"- {c['company']} ({c['ticker']}): {c['shares_held']:,} of "
            f"{c['shares_outstanding'] / 1e9:.2f}B shares = {c['ownership_fraction'] * 100:.6f}% "
            f"ownership, attributable {metric.label.lower()} "
            f"{format_value(metric_id, c['attributable'])} "
            f"(company total {format_value(metric_id, c['company_value'])})"
            for c in contributions[:12]
        ]
        summary = (
            f"Look-through {metric.label.lower()} for FY{fiscal_year} across "
            f"{len(contributions)} holdings: {format_value(metric_id, total)} attributable to "
            f"this portfolio. Computed as shares held / diluted shares outstanding x the "
            f"company's reported figure.\n" + "\n".join(lines)
        )
        if skipped:
            summary += f"\n(Not included -- no FY{fiscal_year} figure on file: {', '.join(sorted(set(skipped)))}.)"
        citations = [f"{c['ticker']}-FY{fiscal_year}-income_statement" for c in contributions]
        return ToolResult("look_through", summary, {
            "metric": metric_id, "fiscal_year": fiscal_year,
            "total_attributable": total, "contributions": contributions,
        }, citations)

    def concentration(self) -> ToolResult:
        rows = sorted(self.positions, key=lambda p: -p["cost_basis"])
        weights = [p["cost_basis"] / self.total_cost for p in rows]
        hhi = sum(w * w for w in weights)
        top5 = sum(weights[:5])
        top10 = sum(weights[:10])
        effective_n = 1 / hhi if hhi else 0.0
        lines = [
            f"- {p['company']} ({p['ticker']}): {w * 100:.2f}% of equities"
            for p, w in list(zip(rows, weights))[:10]
        ]
        summary = (
            f"Concentration across {len(rows)} equity positions (weights at cost): "
            f"top 5 = {top5 * 100:.1f}%, top 10 = {top10 * 100:.1f}%, "
            f"Herfindahl-Hirschman index = {hhi:.4f}, effective number of positions = "
            f"{effective_n:.1f}. Cash is ${self.book['totals']['cash']:,.0f} "
            f"({self.book['totals']['cash'] / self.total_at_cost * 100:.1f}% of the book).\n"
            + "\n".join(lines)
        )
        return ToolResult("concentration", summary, {
            "hhi": hhi, "effective_positions": effective_n,
            "top5_weight": top5, "top10_weight": top10,
            "weights": {p["ticker"]: w for p, w in zip(rows, weights)},
        })

    def screen_holdings(self, metric_id: str, fiscal_year: int | None = None,
                        direction: Literal["improved", "declined", "any"] = "any",
                        compare_to: int | None = None) -> ToolResult:
        """Which holdings moved a given metric which way, year over year."""
        if metric_id not in METRICS:
            return ToolResult("screen_holdings", f"Unknown metric '{metric_id}'.")
        metric = METRICS[metric_id]
        fiscal_year = fiscal_year or self.common_latest_year
        prior = compare_to or fiscal_year - 1

        results = []
        for p in self.positions:
            cur = self._metric(p["ticker"], metric_id, fiscal_year)
            prev = self._metric(p["ticker"], metric_id, prior)
            if cur is None or prev is None:
                continue
            delta = cur - prev
            pct = (delta / abs(prev)) if prev else None
            if direction == "improved" and delta <= 0:
                continue
            if direction == "declined" and delta >= 0:
                continue
            results.append({
                "ticker": p["ticker"], "company": p["company"], "sector": p["sector"],
                "current": cur, "prior": prev, "delta": delta, "pct_change": pct,
                "weight": p["cost_basis"] / self.total_cost,
            })

        if not results:
            return ToolResult(
                "screen_holdings",
                f"No holdings have {metric.label} reported for both FY{prior} and "
                f"FY{fiscal_year} in the {direction} direction.",
            )

        results.sort(key=lambda r: -r["delta"])
        weight = sum(r["weight"] for r in results)
        lines = []
        for r in results:
            if metric.unit == "percent":
                change = f"{(r['delta']) * 100:+.2f}pp"
            elif r["pct_change"] is not None:
                change = f"{r['pct_change'] * 100:+.1f}%"
            else:
                change = "n/a"
            lines.append(
                f"- {r['company']} ({r['ticker']}): {format_value(metric_id, r['prior'])} "
                f"-> {format_value(metric_id, r['current'])} ({change}), "
                f"{r['weight'] * 100:.2f}% of equities"
            )
        summary = (
            f"Holdings where {metric.label} {direction if direction != 'any' else 'changed'} "
            f"between FY{prior} and FY{fiscal_year}: {len(results)} of "
            f"{len(self.positions)} positions, {weight * 100:.1f}% of equity value at cost.\n"
            + "\n".join(lines)
        )
        citations = [f"{r['ticker']}-FY{fiscal_year}-ratios" for r in results]
        return ToolResult("screen_holdings", summary, {
            "metric": metric_id, "fiscal_year": fiscal_year, "compare_to": prior,
            "direction": direction, "matches": results, "weight_of_matches": weight,
        }, citations)

    # -- dispatch ----------------------------------------------------------

    def run(self, tool: str, **kwargs) -> ToolResult:
        fn = {
            "holdings": self.holdings,
            "exposure": self.exposure,
            "look_through": self.look_through,
            "concentration": self.concentration,
            "screen_holdings": self.screen_holdings,
        }.get(tool)
        if fn is None:
            return ToolResult(tool, f"Unknown portfolio tool '{tool}'.")
        accepted = fn.__code__.co_varnames[: fn.__code__.co_argcount]
        return fn(**{k: v for k, v in kwargs.items() if k in accepted})


@functools.lru_cache(maxsize=1)
def get_analytics() -> PortfolioAnalytics:
    book = json.loads((DATA_DIR / "portfolio.json").read_text(encoding="utf-8"))
    financials = json.loads((DATA_DIR / "financials.json").read_text(encoding="utf-8"))
    return PortfolioAnalytics(book, financials)

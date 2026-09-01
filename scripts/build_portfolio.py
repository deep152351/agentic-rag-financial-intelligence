"""Generate the synthetic wealth book that the portfolio tools operate on.

    python -m scripts.build_portfolio

Outputs ``data/portfolio.json`` and ``data/portfolio_corpus.jsonl``.

What is real and what is not
----------------------------
Every *fundamental* the portfolio is analysed against -- revenue, earnings, free
cash flow, margins -- comes from ``data/financials.json``, i.e. from SEC 10-K
filings. Only the ownership book itself (which accounts exist, how many shares
each holds, what they paid) is synthetic; no such data is public.

Cost basis is not invented out of thin air either: it is the company's actually
reported diluted EPS in the purchase year multiplied by a seeded, sector-typical
earnings multiple. That keeps the book internally coherent -- a position's cost
bears a sane relationship to what the business earned -- without pretending to be
a real price series. Market prices are deliberately absent, so every valuation in
this project is stated *at cost* and every fundamental exposure is a look-through
claim on reported financials rather than a mark-to-market number.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

SEED = 20240917

#: sector -> plausible trailing P/E band used to synthesise a cost basis
PE_BANDS = {
    "Information Technology": (22, 38),
    "Communication Services": (16, 28),
    "Consumer Discretionary": (20, 34),
    "Consumer Staples": (17, 26),
    "Financials": (9, 16),
    "Health Care": (14, 24),
    "Energy": (8, 14),
    "Industrials": (15, 25),
    "Materials": (14, 22),
    "Utilities": (16, 23),
    "Real Estate": (18, 30),
}
DEFAULT_PE = (14, 24)

ACCOUNTS = [
    {
        "account_id": "ACC-TAXABLE-01",
        "name": "Core Taxable Brokerage",
        "type": "taxable_brokerage",
        "owner": "primary",
        "objective": "Long-term growth with a technology and consumer tilt",
        "cash": 84_000.0,
        "positions": [
            ("AAPL", 1400, 2020), ("MSFT", 900, 2020), ("NVDA", 6000, 2021),
            ("GOOGL", 1100, 2021), ("AMZN", 1500, 2022), ("META", 450, 2022),
            ("AVGO", 900, 2023), ("COST", 180, 2021), ("HD", 260, 2020),
            ("NFLX", 320, 2023),
        ],
    },
    {
        "account_id": "ACC-IRA-02",
        "name": "Rollover IRA",
        "type": "traditional_ira",
        "owner": "primary",
        "objective": "Balanced core with dividend and healthcare exposure",
        "cash": 41_500.0,
        "positions": [
            ("JNJ", 700, 2019), ("UNH", 210, 2020), ("LLY", 300, 2021),
            ("PG", 620, 2019), ("KO", 1900, 2019), ("PEP", 480, 2020),
            ("JPM", 640, 2020), ("GS", 150, 2021), ("PFE", 1500, 2019),
            ("XOM", 900, 2021),
        ],
    },
    {
        "account_id": "ACC-401K-03",
        "name": "Employer 401(k)",
        "type": "employer_401k",
        "owner": "primary",
        "objective": "Diversified defensive sleeve with income focus",
        "cash": 22_750.0,
        "positions": [
            ("WMT", 900, 2021), ("CVX", 420, 2021), ("CAT", 190, 2020),
            ("UPS", 340, 2021), ("LIN", 150, 2020), ("NEE", 700, 2020),
            ("AMT", 210, 2021), ("ABBV", 400, 2020), ("MA", 190, 2021),
            ("BA", 260, 2022),
        ],
    },
]


def _cost_per_share(rng: random.Random, financials: dict, ticker: str, year: int) -> float | None:
    """Reported diluted EPS in the purchase year x a seeded sector multiple."""
    record = financials.get(ticker)
    if not record:
        return None
    years = record["years"]
    eps = None
    for candidate in (str(year), str(year - 1), str(year + 1)):
        metrics = years.get(candidate) or {}
        # Visa-style multi-class filers publish no undimensioned diluted EPS,
        # so fall back to basic before giving up on an earnings-based cost.
        for key in ("eps_diluted", "eps_basic"):
            if key in metrics:
                eps = metrics[key]["value"]
                break
        if eps is not None:
            break
    if eps is None:
        return _book_value_cost(rng, years, year)

    low, high = PE_BANDS.get(record["sector"], DEFAULT_PE)
    multiple = rng.uniform(low, high)
    if eps <= 0:
        # loss-making year: price off book value so the cost stays positive
        return _book_value_cost(rng, years, year)
    return round(eps * multiple, 2)


def _book_value_cost(rng: random.Random, years: dict, year: int) -> float | None:
    """Cost basis as a multiple of book value per share, when earnings are unusable."""
    for candidate in (str(year), str(year - 1), str(year + 1)):
        metrics = years.get(candidate) or {}
        equity, shares = metrics.get("shareholders_equity"), metrics.get("shares_diluted")
        if equity and shares and shares["value"]:
            bvps = abs(equity["value"] / shares["value"])
            return round(bvps * rng.uniform(1.5, 3.0), 2)
    return None


def main() -> int:
    rng = random.Random(SEED)
    financials = json.loads((DATA / "financials.json").read_text(encoding="utf-8"))

    accounts: list[dict] = []
    for spec in ACCOUNTS:
        positions = []
        for ticker, shares, year in spec["positions"]:
            cps = _cost_per_share(rng, financials, ticker, year)
            if cps is None:
                # Multi-class filers (Visa, Berkshire) publish no undimensioned
                # per-share facts in companyfacts, so there is no basis to price a
                # synthetic position from. They stay in the retrieval universe.
                print(f"  !! skipping {ticker}: no per-share basis in companyfacts for FY{year}")
                continue
            record = financials[ticker]
            positions.append({
                "ticker": ticker,
                "company": record["name"],
                "sector": record["sector"],
                "industry": record["industry"],
                "shares": shares,
                "purchase_year": year,
                "cost_per_share": cps,
                "cost_basis": round(shares * cps, 2),
            })

        total_cost = round(sum(p["cost_basis"] for p in positions), 2)
        accounts.append({
            **{k: v for k, v in spec.items() if k != "positions"},
            "positions": positions,
            "total_cost_basis": total_cost,
            "total_value_at_cost": round(total_cost + spec["cash"], 2),
        })

    book = {
        "_disclaimer": (
            "SYNTHETIC ownership data for demonstration. Share counts, accounts and "
            "cost bases are generated, not real holdings. All company fundamentals "
            "referenced against this book come from SEC 10-K filings."
        ),
        "as_of": "2026-09-01",
        "base_currency": "USD",
        "valuation_basis": "cost",
        "generator": {"seed": SEED, "method": "reported diluted EPS x seeded sector P/E band"},
        "accounts": accounts,
        "totals": {
            "cash": round(sum(a["cash"] for a in accounts), 2),
            "cost_basis": round(sum(a["total_cost_basis"] for a in accounts), 2),
            "value_at_cost": round(sum(a["total_value_at_cost"] for a in accounts), 2),
            "positions": sum(len(a["positions"]) for a in accounts),
            "distinct_tickers": len({p["ticker"] for a in accounts for p in a["positions"]}),
        },
    }
    (DATA / "portfolio.json").write_text(json.dumps(book, indent=1), encoding="utf-8")

    # ---- retrieval chunks so the portfolio is searchable by the same pipeline ----
    chunks: list[dict] = []
    grand_total = book["totals"]["value_at_cost"]

    for account in accounts:
        for p in positions_of(account):
            weight = p["cost_basis"] / grand_total
            text = (
                f"Portfolio holding: {p['shares']:,} shares of {p['company']} ({p['ticker']}) "
                f"held in the {account['name']} ({account['type'].replace('_', ' ')}, "
                f"account {account['account_id']}). Purchased around {p['purchase_year']} at a "
                f"cost basis of ${p['cost_per_share']:,.2f} per share, ${p['cost_basis']:,.0f} total, "
                f"which is {weight * 100:.2f}% of the household book at cost. "
                f"Sector exposure: {p['sector']} / {p['industry']}. "
                f"This is a position in the client's own wealth portfolio, not a company filing."
            )
            chunks.append({
                "doc_id": f"PORTFOLIO-{account['account_id']}-{p['ticker']}",
                "doc_type": "portfolio_position",
                "text": text,
                "ticker": p["ticker"],
                "company": p["company"],
                "sector": p["sector"],
                "industry": p["industry"],
                "fiscal_year": None,
                "fiscal_years": [],
                "statement": "portfolio",
                "metrics": [],
                "period_end": None,
                "account_id": account["account_id"],
                "source": "Synthetic portfolio book (data/portfolio.json)",
            })

        by_sector: dict[str, float] = {}
        for p in positions_of(account):
            by_sector[p["sector"]] = by_sector.get(p["sector"], 0.0) + p["cost_basis"]
        sector_line = ", ".join(
            f"{s} {v / account['total_cost_basis'] * 100:.1f}%"
            for s, v in sorted(by_sector.items(), key=lambda kv: -kv[1])
        )
        text = (
            f"Portfolio account summary: {account['name']} ({account['account_id']}), a "
            f"{account['type'].replace('_', ' ')} account. Objective: {account['objective']}. "
            f"It holds {len(account['positions'])} equity positions with a total cost basis of "
            f"${account['total_cost_basis']:,.0f} plus ${account['cash']:,.0f} in cash, "
            f"${account['total_value_at_cost']:,.0f} at cost overall. "
            f"Sector allocation at cost: {sector_line}. "
            f"Tickers held: {', '.join(p['ticker'] for p in account['positions'])}."
        )
        chunks.append({
            "doc_id": f"PORTFOLIO-{account['account_id']}-summary",
            "doc_type": "portfolio_summary",
            "text": text,
            "ticker": None,
            "company": account["name"],
            "sector": None,
            "industry": None,
            "fiscal_year": None,
            "fiscal_years": [],
            "statement": "portfolio",
            "metrics": [],
            "period_end": None,
            "account_id": account["account_id"],
            "source": "Synthetic portfolio book (data/portfolio.json)",
        })

    with (DATA / "portfolio_corpus.jsonl").open("w", encoding="utf-8") as fh:
        for c in chunks:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")

    t = book["totals"]
    print(f"{len(accounts)} accounts · {t['positions']} positions · "
          f"{t['distinct_tickers']} distinct tickers")
    print(f"book value at cost: ${t['value_at_cost']:,.0f} (cash ${t['cash']:,.0f})")
    print(f"  -> {DATA / 'portfolio.json'}")
    print(f"  -> {DATA / 'portfolio_corpus.jsonl'} ({len(chunks)} chunks)")
    return 0


def positions_of(account: dict) -> list[dict]:
    return account["positions"]


if __name__ == "__main__":
    raise SystemExit(main())

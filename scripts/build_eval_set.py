"""Generate the evaluation set with exact gold labels.

    python -m scripts.build_eval_set --per-type 24

Gold labels are *derived*, not annotated. Because the corpus builder tagged every
chunk with the company, fiscal year and metric ids it actually contains, the
document that must be retrieved for "Apple's FY2024 operating margin" is known by
construction: ``AAPL-FY2024-ratios``. That removes the usual weak link in a RAG
eval -- hand-labelled relevance that quietly disagrees with the index.

Each case carries two kinds of ground truth:

``gold_doc_ids``      what Recall@10 is measured against.
``gold_constraints``  what the resolved filters *should* have been, which is what
                      constraint compliance is measured against.

Case families deliberately include paraphrase and alias stress (``coke`` for KO,
``op margin`` for operating_margin, ``FY'24`` for 2024) so the eval exercises the
entity resolver rather than just string equality.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from app.metrics import METRICS

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SEED = 7

#: colloquial phrasings -> the canonical metric they should resolve to
METRIC_PHRASINGS = {
    "revenue": ["revenue", "total revenue", "top line", "net sales"],
    "net_income": ["net income", "bottom line", "profit", "net earnings"],
    "operating_income": ["operating income", "operating profit", "EBIT"],
    "operating_margin": ["operating margin", "op margin", "operating profitability"],
    "net_margin": ["net margin", "profit margin"],
    "gross_margin": ["gross margin"],
    "free_cash_flow": ["free cash flow", "FCF", "cash generation"],
    "operating_cash_flow": ["operating cash flow", "cash from operations", "cash from ops"],
    "total_assets": ["total assets", "assets"],
    "shareholders_equity": ["shareholders equity", "book value"],
    "long_term_debt": ["long-term debt", "borrowings"],
    "rnd_expense": ["R&D", "research and development", "research spend"],
    "eps_diluted": ["diluted EPS", "earnings per share"],
    "capex": ["capex", "capital expenditure"],
    "cash_and_equivalents": ["cash", "cash and equivalents"],
    "inventory": ["inventory"],
    "roe": ["ROE", "return on equity"],
    "current_ratio": ["current ratio"],
    "debt_to_equity": ["debt to equity"],
}

YEAR_PHRASINGS = {
    2024: ["FY2024", "fiscal 2024", "FY'24", "the 2024 fiscal year"],
    2023: ["FY2023", "fiscal 2023", "FY'23"],
    2022: ["FY2022", "fiscal 2022"],
    2021: ["FY2021", "fiscal 2021"],
    2020: ["FY2020", "fiscal 2020"],
    2025: ["FY2025", "fiscal 2025", "FY'25"],
}

YOY_METRICS = {"revenue", "operating_income", "net_income", "eps_diluted",
               "free_cash_flow", "operating_margin", "net_margin"}


def company_phrasings(company: dict) -> list[str]:
    options = [company["name"], company["ticker"]]
    options.extend(a for a in company.get("aliases", []) if len(a) > 2)
    return options


def statement_of(metric_id: str) -> str:
    return METRICS[metric_id].statement


def build(per_type: int) -> list[dict]:
    rng = random.Random(SEED)
    financials = json.loads((DATA / "financials.json").read_text(encoding="utf-8"))
    universe = json.loads((DATA / "companies.json").read_text(encoding="utf-8"))["companies"]
    by_ticker = {c["ticker"]: c for c in universe}

    def sample_fact() -> tuple[str, int, str] | None:
        """A (ticker, fiscal_year, metric) triple that genuinely exists."""
        for _ in range(60):
            ticker = rng.choice(list(financials))
            years = financials[ticker]["years"]
            if not years:
                continue
            year = int(rng.choice(list(years)))
            candidates = [m for m in years[str(year)] if m in METRIC_PHRASINGS]
            if candidates:
                return ticker, year, rng.choice(candidates)
        return None

    cases: list[dict] = []
    seen: set[str] = set()

    def add(case: dict) -> None:
        if case["question"] in seen:
            return
        seen.add(case["question"])
        case["id"] = f"{case['family']}-{len(cases):03d}"
        cases.append(case)

    # ---- 1. single-fact lookup ------------------------------------------
    while sum(c["family"] == "lookup" for c in cases) < per_type:
        fact = sample_fact()
        if not fact:
            break
        ticker, year, metric = fact
        company = by_ticker[ticker]
        add({
            "family": "lookup",
            "question": (
                f"What was {rng.choice(company_phrasings(company))}'s "
                f"{rng.choice(METRIC_PHRASINGS[metric])} in "
                f"{rng.choice(YEAR_PHRASINGS.get(year, [f'FY{year}']))}?"
            ),
            "gold_doc_ids": [f"{ticker}-FY{year}-{statement_of(metric)}"],
            "gold_constraints": {"tickers": [ticker], "fiscal_years": [year], "metrics": [metric]},
            "expected_intent": "company_lookup",
        })

    # ---- 2. year-over-year ----------------------------------------------
    while sum(c["family"] == "yoy" for c in cases) < per_type:
        fact = sample_fact()
        if not fact:
            break
        ticker, year, metric = fact
        if metric not in YOY_METRICS or str(year - 1) not in financials[ticker]["years"]:
            continue
        company = by_ticker[ticker]
        add({
            "family": "yoy",
            "question": (
                f"How did {rng.choice(company_phrasings(company))}'s "
                f"{rng.choice(METRIC_PHRASINGS[metric])} change from FY{year - 1} to FY{year}?"
            ),
            "gold_doc_ids": [
                f"{ticker}-FY{year}-yoy_analysis",
                f"{ticker}-FY{year}-{statement_of(metric)}",
                f"{ticker}-FY{year - 1}-{statement_of(metric)}",
            ],
            "gold_constraints": {
                "tickers": [ticker], "fiscal_years": [year - 1, year], "metrics": [metric]
            },
            "expected_intent": "trend",
        })

    # ---- 3. cross-company comparison ------------------------------------
    while sum(c["family"] == "comparison" for c in cases) < per_type:
        a, b = rng.sample(list(financials), 2)
        years_a, years_b = financials[a]["years"], financials[b]["years"]
        shared_years = sorted(set(years_a) & set(years_b))
        if not shared_years:
            continue
        year = int(rng.choice(shared_years))
        shared_metrics = [
            m for m in years_a[str(year)]
            if m in METRIC_PHRASINGS and m in years_b[str(year)]
        ]
        if not shared_metrics:
            continue
        metric = rng.choice(shared_metrics)
        add({
            "family": "comparison",
            "question": (
                f"Compare {rng.choice(company_phrasings(by_ticker[a]))} and "
                f"{rng.choice(company_phrasings(by_ticker[b]))} on "
                f"{rng.choice(METRIC_PHRASINGS[metric])} in FY{year}."
            ),
            "gold_doc_ids": [
                f"{a}-FY{year}-{statement_of(metric)}",
                f"{b}-FY{year}-{statement_of(metric)}",
            ],
            "gold_constraints": {
                "tickers": sorted([a, b]), "fiscal_years": [year], "metrics": [metric]
            },
            "expected_intent": "comparison",
        })

    # ---- 4. multi-year trend --------------------------------------------
    while sum(c["family"] == "trend" for c in cases) < per_type:
        fact = sample_fact()
        if not fact:
            break
        ticker, _, metric = fact
        years = sorted(int(y) for y in financials[ticker]["years"])
        window = [y for y in years if all(
            metric in financials[ticker]["years"].get(str(y), {}) for y in years[:1]
        )]
        window = [y for y in years if metric in financials[ticker]["years"][str(y)]]
        if len(window) < 3:
            continue
        start, end = window[0], window[min(2, len(window) - 1)]
        add({
            "family": "trend",
            "question": (
                f"How has {rng.choice(company_phrasings(by_ticker[ticker]))}'s "
                f"{rng.choice(METRIC_PHRASINGS[metric])} trended from FY{start} to FY{end}?"
            ),
            "gold_doc_ids": [
                f"{ticker}-FY{y}-{statement_of(metric)}" for y in range(start, end + 1)
            ],
            "gold_constraints": {
                "tickers": [ticker],
                "fiscal_years": list(range(start, end + 1)),
                "metrics": [metric],
            },
            "expected_intent": "trend",
        })

    # ---- 5. portfolio ----------------------------------------------------
    portfolio_cases = [
        ("What is my portfolio's exposure by sector?", ["exposure"], []),
        ("How concentrated is my portfolio?", ["concentration"], []),
        ("What are my largest holdings?", ["concentration"], []),
        ("List the positions in my Rollover IRA.", ["holdings"], []),
        ("What is my look-through share of FY2024 net income?", ["look_through"], []),
        ("Which of my holdings had operating margin decline in FY2024?", ["screen_holdings"], []),
        ("How much of my book is in the Information Technology sector?", ["exposure"], []),
        ("What is my portfolio's look-through free cash flow for FY2024?", ["look_through"], []),
    ]
    for question, tools, docs in portfolio_cases:
        add({
            "family": "portfolio",
            "question": question,
            "gold_doc_ids": docs,
            "gold_constraints": {"doc_types": ["portfolio_position", "portfolio_summary"]},
            "expected_intent": "portfolio",
            "expected_tools": tools,
        })

    # ---- 6. out of scope -------------------------------------------------
    for question in [
        "Should I buy Nvidia stock right now?",
        "What will Apple's share price be next quarter?",
        "Give me a price target for Tesla.",
        "Is Microsoft a good investment for my retirement?",
        "What should I invest in this year?",
        "Predict Amazon's revenue for FY2030.",
    ]:
        add({
            "family": "out_of_scope",
            "question": question,
            "gold_doc_ids": [],
            "gold_constraints": {},
            "expected_intent": "out_of_scope",
        })

    return cases


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-type", type=int, default=24)
    ap.add_argument("--out", type=Path, default=DATA / "eval" / "eval_set.json")
    args = ap.parse_args()

    cases = build(args.per_type)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "_note": "Gold labels derived from data/financials.json and the corpus payloads, "
                 "not hand-annotated. Regenerate with scripts.build_eval_set.",
        "seed": SEED,
        "cases": cases,
    }, indent=1), encoding="utf-8")

    from collections import Counter
    counts = Counter(c["family"] for c in cases)
    print(f"{len(cases)} cases -> {args.out}")
    for family, n in sorted(counts.items()):
        print(f"  {family:<14} {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

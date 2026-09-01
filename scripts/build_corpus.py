"""Turn cached SEC companyfacts into (a) a numeric fact store and (b) a chunked corpus.

    python -m scripts.build_corpus
    python -m scripts.build_corpus --start-year 2019 --end-year 2025

Outputs
-------
``data/financials.json``  canonical numbers: ticker -> fiscal_year -> metric -> value,
                          each with the us-gaap tag, period end and accession it came from.
                          The wealth tools and the eval-set builder read this, so answers
                          and gold labels share one source of truth.
``data/corpus.jsonl``     retrieval chunks. Every chunk is a natural-language "fact card"
                          (good for BGE dense vectors) carrying strict metadata
                          (ticker / fiscal_year / statement / metrics) for Qdrant filters.

Fiscal-year labelling
---------------------
A period's fiscal year is ``min(fy)`` across every 10-K entry reporting that period end.
The first 10-K to report a period *is* that period's own annual report; later filings
repeat it as a comparative under their own (higher) ``fy``. This tracks how the company
itself labels the year -- Nike's May-ending FY2024 lands on 2024, not 2023, which a
month-based heuristic would get wrong.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from app.metrics import (
    ALL_METRICS,
    DERIVED_ORDER,
    METRICS,
    REPORTED_METRIC_IDS,
    format_value,
    metrics_for_statement,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"

ANNUAL_FORMS = ("10-K",)
MIN_PERIOD_DAYS, MAX_PERIOD_DAYS = 330, 400


# ---------------------------------------------------------------------------
# XBRL extraction
# ---------------------------------------------------------------------------


def _annual_entries(units: list[dict], period: str) -> list[dict]:
    """Keep only entries that represent a full fiscal year (or its year-end instant)."""
    out = []
    for e in units:
        if not str(e.get("form", "")).startswith(ANNUAL_FORMS):
            continue
        if e.get("fp") != "FY" or "end" not in e or "fy" not in e:
            continue
        if period == "duration":
            if not e.get("start"):
                continue
            try:
                span = (dt.date.fromisoformat(e["end"]) - dt.date.fromisoformat(e["start"])).days
            except ValueError:
                continue
            if not (MIN_PERIOD_DAYS <= span <= MAX_PERIOD_DAYS):
                continue
        else:  # instant
            if e.get("start"):
                continue
        out.append(e)
    return out


def _series_for_tag(facts: dict, tag: str, period: str, unit_keys: Iterable[str],
                    revision: str = "own_filing") -> dict[int, dict]:
    """{fiscal_year: fact} for one us-gaap tag.

    ``revision`` picks *which* reported value to trust when several filings
    restate the same period:

    ``own_filing``          the figure as it appeared in that fiscal year's own
                            10-K (latest amendment of it). Right for currency
                            amounts -- it is what the annual report actually said.
    ``latest_restatement``  the figure from the most recent filing that mentions
                            the period. Required for per-share and share-count
                            metrics, because a stock split retroactively restates
                            them: NVIDIA's 10-for-1 split means FY2024 diluted EPS
                            reads $11.93 in the FY2024 10-K but $1.19 everywhere
                            after it. Taking each year's own filing would produce a
                            series that appears to collapse 75% at the split.
    """
    node = facts.get("facts", {}).get("us-gaap", {}).get(tag)
    if not node:
        return {}

    units: list[dict] = []
    for uk in unit_keys:
        units.extend(node.get("units", {}).get(uk, []))
    if not units:
        return {}

    entries = _annual_entries(units, period)
    if not entries:
        return {}

    by_end: dict[str, list[dict]] = defaultdict(list)
    for e in entries:
        by_end[e["end"]].append(e)

    series: dict[int, dict] = {}
    for end, group in by_end.items():
        fiscal_year = min(int(e["fy"]) for e in group)
        # among the filings that call this period their own FY, take the latest
        # amendment -- i.e. the figure as finally reported in that year's 10-K.
        if revision == "latest_restatement":
            chosen = max(group, key=lambda e: e.get("filed", ""))
        else:
            own = [e for e in group if int(e["fy"]) == fiscal_year]
            chosen = max(own, key=lambda e: e.get("filed", ""))
        prior = series.get(fiscal_year)
        if prior and prior["_end"] >= end:
            continue  # a 52/53-week quirk put two ends in one FY; keep the later
        series[fiscal_year] = {
            "value": float(chosen["val"]),
            "tag": tag,
            "_end": end,
            "period_end": end,
            "form": chosen.get("form"),
            "filed": chosen.get("filed"),
            "accession": chosen.get("accn"),
            "revision": revision,
        }
    return series


_UNIT_KEYS = {
    "USD": ("USD",),
    "USD/shares": ("USD/shares",),
    "shares": ("shares",),
}


#: plausible split ratios, forward and reverse
_SPLIT_CANDIDATES = (1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 15.0, 20.0)
_SPLIT_TOLERANCE = 0.08


def _detect_split(ratio: float) -> float | None:
    """Return the split factor a year-over-year share-count jump implies, else None."""
    if ratio < 1.4:
        return None
    best = min(_SPLIT_CANDIDATES, key=lambda c: abs(ratio / c - 1.0))
    return best if abs(ratio / best - 1.0) <= _SPLIT_TOLERANCE else None


def _normalize_splits(per_year: dict[int, dict[str, dict]]) -> list[dict]:
    """Restate per-share history onto the latest share basis.

    SEC companyfacts only carries a period for as long as some filing still
    reports it -- roughly three years for an income-statement line. So a split
    is reflected in recent years but not in older ones, and the raw series looks
    like NVIDIA's diluted EPS collapsing 75% in FY2023 when nothing of the sort
    happened. Buybacks and issuance move share counts by single-digit percents,
    so a >=1.4x year-over-year jump that lands near a standard split ratio is a
    split; scale earlier share counts up and earlier per-share figures down by
    the cumulative factor.

    ``restated_from_fy`` is the first fiscal year whose *reported* value already
    reflects the split -- i.e. where the restatement window reached. It is not
    the split's announcement date, which XBRL does not carry here.
    """
    years = sorted(per_year)
    shares_id = "shares_diluted"
    raw = {fy: per_year[fy][shares_id]["value"]
           for fy in years if shares_id in per_year[fy]}
    if len(raw) < 2:
        return []

    known = sorted(raw)
    factor_after: dict[int, float] = {}
    events: list[dict] = []
    cumulative = 1.0
    for earlier, later in reversed(list(zip(known, known[1:]))):
        if raw[earlier] > 0:
            split = _detect_split(raw[later] / raw[earlier])
            if split:
                cumulative *= split
                events.append({"restated_from_fy": later, "factor": split})
        factor_after[earlier] = cumulative

    per_share_ids = [m.id for m in ALL_METRICS if m.unit == "USD/shares"]
    for fy, factor in factor_after.items():
        if factor == 1.0:
            continue
        metrics = per_year.get(fy, {})
        if shares_id in metrics:
            metrics[shares_id]["value"] *= factor
            metrics[shares_id]["split_adjusted_by"] = factor
        for mid in per_share_ids:
            if mid in metrics:
                metrics[mid]["value"] /= factor
                metrics[mid]["split_adjusted_by"] = factor

    # A split that coincides with heavy issuance cannot be separated from organic
    # dilution by the share ratio alone -- Tesla's FY2019->FY2020 jump is 3.66x,
    # a 3:1 split on top of ~22% dilution, which sits between two candidate
    # ratios. Rather than publish a per-share series on a basis we cannot pin
    # down, drop the affected years' per-share figures and say so.
    adjusted = {fy: per_year[fy][shares_id]["value"]
                for fy in sorted(per_year) if shares_id in per_year[fy]}
    known = sorted(adjusted)
    boundary: int | None = None
    for earlier, later in zip(known, known[1:]):
        if adjusted[earlier] and max(adjusted[earlier], adjusted[later]) / min(
                adjusted[earlier], adjusted[later]) >= 1.5:
            boundary = later
    if boundary is not None:
        for fy in known:
            if fy < boundary:
                for mid in (shares_id, *per_share_ids):
                    per_year.get(fy, {}).pop(mid, None)
        events.append({"unresolved_basis_before_fy": boundary,
                       "action": "per-share metrics dropped for earlier years"})

    return list(reversed(events))


def extract_company(facts: dict, start_year: int, end_year: int) -> tuple[dict[int, dict[str, dict]], list[dict]]:
    """{fiscal_year: {metric_id: fact}} for one company."""
    per_year: dict[int, dict[str, dict]] = defaultdict(dict)

    for metric_id in REPORTED_METRIC_IDS:
        m = METRICS[metric_id]
        unit_keys = _UNIT_KEYS.get(m.unit, ("USD",))
        # Merge tags in priority order, filling only years still missing. Filers
        # migrate tags mid-history (NVDA moved to the ASC 606 revenue tag in
        # FY2023; Goldman reports revenue net of interest expense), so picking a
        # single "best" tag silently drops years.
        revision = "latest_restatement" if m.unit in ("USD/shares", "shares") else "own_filing"
        merged: dict[int, dict] = {}
        for tag in m.tags:
            series = _series_for_tag(facts, tag, m.period, unit_keys, revision)
            for fy, fact in series.items():
                if start_year <= fy <= end_year and fy not in merged:
                    merged[fy] = fact
            if len(merged) >= (end_year - start_year + 1):
                break
        for fy, fact in merged.items():
            per_year[fy][metric_id] = fact

    splits = _normalize_splits(per_year)

    # derived metrics, in declaration order so fcf_margin can use free_cash_flow
    for fy, metrics in per_year.items():
        values = {mid: f["value"] for mid, f in metrics.items()}
        for metric_id in DERIVED_ORDER:
            m = METRICS[metric_id]
            assert m.formula is not None
            value = m.formula(values)
            if value is None:
                continue
            metrics[metric_id] = {
                "value": value,
                "tag": None,
                "derived_from": list(m.derived_from),
                "period_end": next(
                    (metrics[d]["period_end"] for d in m.derived_from if d in metrics), None
                ),
            }
            values[metric_id] = value

    return dict(per_year), splits


# ---------------------------------------------------------------------------
# Chunk text
# ---------------------------------------------------------------------------


def _pct_change(cur: float, prev: float) -> float | None:
    if prev == 0:
        return None
    return (cur - prev) / abs(prev)


def _describe(metric_id: str, facts: dict[str, dict]) -> str | None:
    f = facts.get(metric_id)
    if not f:
        return None
    return f"{METRICS[metric_id].label} {format_value(metric_id, f['value'])}"


def _statement_text(company: dict, fy: int, statement: str, facts: dict[str, dict],
                    prior: dict[str, dict] | None, period_end: str | None) -> str | None:
    wanted = [m.id for m in metrics_for_statement(statement) if m.id in facts]
    if not wanted:
        return None

    head = (
        f"{company['name']} ({company['ticker']}) — FY{fy} "
        f"{statement.replace('_', ' ')}"
    )
    if period_end:
        head += f", fiscal year ended {period_end}"
    head += f". Sector: {company['sector']}; industry: {company['industry']}. "
    head += "Figures as reported in the company's Form 10-K filed with the SEC.\n"

    lines = []
    for mid in wanted:
        value = format_value(mid, facts[mid]["value"])
        line = f"- {METRICS[mid].label}: {value}"
        if prior and mid in prior and METRICS[mid].unit in ("USD", "USD/shares", "shares"):
            change = _pct_change(facts[mid]["value"], prior[mid]["value"])
            if change is not None:
                direction = "up" if change >= 0 else "down"
                line += (
                    f" ({direction} {abs(change) * 100:.1f}% from "
                    f"{format_value(mid, prior[mid]['value'])} in FY{fy - 1})"
                )
        elif prior and mid in prior:
            delta = (facts[mid]["value"] - prior[mid]["value"]) * (
                100 if METRICS[mid].unit == "percent" else 1
            )
            unit = "pp" if METRICS[mid].unit == "percent" else "x"
            line += (
                f" (vs {format_value(mid, prior[mid]['value'])} in FY{fy - 1}, "
                f"{'+' if delta >= 0 else ''}{delta:.2f}{unit})"
            )
        lines.append(line)

    return head + "\n".join(lines)


def _yoy_text(company: dict, fy: int, facts: dict[str, dict], prior: dict[str, dict]) -> str | None:
    """MD&A-style narrative: what actually changed, in prose."""
    headline = ["revenue", "operating_income", "net_income", "eps_diluted",
                "free_cash_flow", "operating_margin", "net_margin"]
    available = [m for m in headline if m in facts and m in prior]
    if len(available) < 3:
        return None

    parts = [
        f"{company['name']} ({company['ticker']}) — FY{fy} versus FY{fy - 1} "
        f"year-over-year performance review ({company['sector']}).\n"
    ]
    for mid in available:
        cur, prev = facts[mid]["value"], prior[mid]["value"]
        m = METRICS[mid]
        if m.unit == "percent":
            delta_pp = (cur - prev) * 100
            verb = "expanded" if delta_pp >= 0 else "contracted"
            parts.append(
                f"{m.label} {verb} from {format_value(mid, prev)} in FY{fy - 1} to "
                f"{format_value(mid, cur)} in FY{fy}, a move of {delta_pp:+.2f} percentage points."
            )
        else:
            change = _pct_change(cur, prev)
            if change is None:
                continue
            if abs(change) < 0.01:
                verb = "was roughly flat at"
            elif change > 0:
                verb = "grew to" if change < 0.25 else "rose sharply to"
            else:
                verb = "declined to" if change > -0.25 else "fell sharply to"
            parts.append(
                f"{m.label} {verb} {format_value(mid, cur)} in FY{fy} from "
                f"{format_value(mid, prev)} in FY{fy - 1} ({change * 100:+.1f}%)."
            )

    if "revenue" in facts and "net_income" in facts:
        parts.append(
            f"For context, FY{fy} revenue of {format_value('revenue', facts['revenue']['value'])} "
            f"converted into {format_value('net_income', facts['net_income']['value'])} of net income."
        )
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Corpus assembly
# ---------------------------------------------------------------------------


def build_chunks(company: dict, years: dict[int, dict[str, dict]]) -> list[dict]:
    chunks: list[dict] = []
    ticker = company["ticker"]
    sorted_years = sorted(years)

    for fy in sorted_years:
        facts = years[fy]
        prior = years.get(fy - 1)
        period_end = next(
            (f["period_end"] for f in facts.values() if f.get("period_end")), None
        )

        for statement in ("income_statement", "balance_sheet", "cash_flow", "ratios"):
            text = _statement_text(company, fy, statement, facts, prior, period_end)
            if not text:
                continue
            present = [m.id for m in metrics_for_statement(statement) if m.id in facts]
            chunks.append({
                "doc_id": f"{ticker}-FY{fy}-{statement}",
                "doc_type": statement,
                "text": text,
                "ticker": ticker,
                "company": company["name"],
                "sector": company["sector"],
                "industry": company["industry"],
                "fiscal_year": fy,
                "fiscal_years": [fy],
                "statement": statement,
                "metrics": present,
                "period_end": period_end,
                "source": "SEC EDGAR XBRL companyfacts (Form 10-K)",
            })

        if prior:
            text = _yoy_text(company, fy, facts, prior)
            if text:
                present = [m for m in ("revenue", "operating_income", "net_income",
                                       "eps_diluted", "free_cash_flow", "operating_margin",
                                       "net_margin") if m in facts]
                chunks.append({
                    "doc_id": f"{ticker}-FY{fy}-yoy_analysis",
                    "doc_type": "yoy_analysis",
                    "text": text,
                    "ticker": ticker,
                    "company": company["name"],
                    "sector": company["sector"],
                    "industry": company["industry"],
                    "fiscal_year": fy,
                    "fiscal_years": [fy, fy - 1],
                    "statement": "ratios",
                    "metrics": present,
                    "period_end": period_end,
                    "source": "Derived from SEC EDGAR XBRL companyfacts (Form 10-K)",
                })

    # one profile card per company, covering every year we hold
    if sorted_years:
        latest = sorted_years[-1]
        rev = years[latest].get("revenue")
        ni = years[latest].get("net_income")
        span = f"FY{sorted_years[0]}–FY{sorted_years[-1]}"
        summary = (
            f"{company['name']} ({company['ticker']}) is a {company['industry']} company in the "
            f"{company['sector']} sector. Company profile and filing coverage: this dataset holds "
            f"audited annual figures for {span} sourced from SEC Form 10-K filings. "
        )
        if rev:
            summary += f"Most recent fiscal year on file is FY{latest} with total revenue of {format_value('revenue', rev['value'])}"
            if ni:
                summary += f" and net income of {format_value('net_income', ni['value'])}"
            summary += ". "
        summary += "Also known as: " + ", ".join(company["aliases"]) + "."
        chunks.append({
            "doc_id": f"{company['ticker']}-profile",
            "doc_type": "company_profile",
            "text": summary,
            "ticker": company["ticker"],
            "company": company["name"],
            "sector": company["sector"],
            "industry": company["industry"],
            "fiscal_year": latest,
            "fiscal_years": sorted_years,
            "statement": "company_profile",
            "metrics": [],
            "period_end": None,
            "source": "SEC EDGAR company metadata",
        })

    return chunks


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-year", type=int, default=2019)
    ap.add_argument("--end-year", type=int, default=2025)
    ap.add_argument("--min-years", type=int, default=3,
                    help="drop companies with fewer usable fiscal years")
    args = ap.parse_args()

    universe = json.loads((DATA / "companies.json").read_text(encoding="utf-8"))["companies"]
    cik_map = json.loads((RAW / "cik_map.json").read_text(encoding="utf-8"))

    financials: dict[str, Any] = {}
    chunks: list[dict] = []
    skipped: list[str] = []

    for company in universe:
        ticker = company["ticker"]
        raw_path = RAW / f"{ticker}.json"
        if not raw_path.exists():
            skipped.append(f"{ticker} (no raw facts)")
            continue

        facts = json.loads(raw_path.read_text(encoding="utf-8"))
        years, splits = extract_company(facts, args.start_year, args.end_year)
        years = {fy: m for fy, m in years.items() if "revenue" in m or "total_assets" in m}
        if len(years) < args.min_years:
            skipped.append(f"{ticker} ({len(years)} yrs)")
            continue

        financials[ticker] = {
            "cik": cik_map.get(ticker),
            "splits_detected": splits,
            "name": company["name"],
            "sector": company["sector"],
            "industry": company["industry"],
            "entity_name": facts.get("entityName"),
            "years": {
                str(fy): {
                    mid: {
                        "value": f["value"],
                        "tag": f.get("tag"),
                        "derived_from": f.get("derived_from"),
                        "period_end": f.get("period_end"),
                        "accession": f.get("accession"),
                        "revision": f.get("revision"),
                    }
                    for mid, f in sorted(metrics.items())
                }
                for fy, metrics in sorted(years.items())
            },
        }
        chunks.extend(build_chunks(company, years))
        coverage = sum(len(m) for m in years.values())
        parts = [f"{e['factor']:g}:1 restated from FY{e['restated_from_fy']}"
                 for e in splits if "factor" in e]
        parts += [f"per-share basis unresolved before FY{e['unresolved_basis_before_fy']}"
                  f" (those years dropped)"
                  for e in splits if "unresolved_basis_before_fy" in e]
        note = ("  " + "; ".join(parts)) if parts else ""
        print(f"{ticker:<6} {len(years)} fiscal years, {coverage:>4} facts, "
              f"{len(build_chunks(company, years)):>3} chunks{note}")

    (DATA / "financials.json").write_text(json.dumps(financials, indent=1), encoding="utf-8")
    with (DATA / "corpus.jsonl").open("w", encoding="utf-8") as fh:
        for c in chunks:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")

    n_years = sum(len(v["years"]) for v in financials.values())
    print(f"\n{len(financials)} companies · {n_years} company-years · {len(chunks)} chunks")
    print(f"  -> {DATA / 'financials.json'}")
    print(f"  -> {DATA / 'corpus.jsonl'}")
    if skipped:
        print(f"  skipped: {', '.join(skipped)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

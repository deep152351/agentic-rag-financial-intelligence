"""Download SEC EDGAR XBRL ``companyfacts`` for the configured universe.

The raw JSON is cached under ``data/raw/`` so the corpus build is reproducible
offline and so re-runs cost nothing. Nothing here is invented: every number that
later reaches the index traces back to one of these filings.

    python -m scripts.fetch_sec                  # all companies in data/companies.json
    python -m scripts.fetch_sec --tickers AAPL MSFT
    python -m scripts.fetch_sec --force          # ignore cache

SEC asks for a descriptive User-Agent with contact info and caps traffic at
10 requests/second. Set SEC_USER_AGENT in your .env before running against
their servers in anger.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"

TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

DEFAULT_UA = "finrag-research/1.0 (educational project; set SEC_USER_AGENT to override)"
#: SEC fair-access limit is 10 req/s; stay well under it.
REQUEST_INTERVAL_S = 0.15


def _user_agent() -> str:
    return os.getenv("SEC_USER_AGENT", "").strip() or DEFAULT_UA


def _client() -> httpx.Client:
    return httpx.Client(
        headers={
            "User-Agent": _user_agent(),
            "Accept-Encoding": "gzip, deflate",
            "Host": "data.sec.gov",
        },
        timeout=60.0,
        follow_redirects=True,
    )


def load_universe() -> list[dict[str, Any]]:
    payload = json.loads((DATA / "companies.json").read_text(encoding="utf-8"))
    return payload["companies"]


def fetch_ticker_map(client: httpx.Client, force: bool = False) -> dict[str, str]:
    """Ticker -> zero-padded 10-digit CIK, straight from SEC."""
    cache = RAW / "company_tickers.json"
    if cache.exists() and not force:
        raw = json.loads(cache.read_text(encoding="utf-8"))
    else:
        resp = client.get(TICKER_MAP_URL, headers={"Host": "www.sec.gov"})
        resp.raise_for_status()
        raw = resp.json()
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(raw), encoding="utf-8")

    out: dict[str, str] = {}
    for row in raw.values():
        out[str(row["ticker"]).upper()] = f"{int(row['cik_str']):010d}"
    return out


def resolve_cik(company: dict[str, Any], ticker_map: dict[str, str]) -> str | None:
    """Resolve a ticker to a CIK.

    An explicit ``cik`` in companies.json wins. That override exists because SEC's
    ticker map points at the *current* registrant, which after a reorganisation can
    be a brand-new entity with almost no filing history (XOM is the live example:
    the map resolves it to a successor CIK holding a single 10-Q, while the full
    10-K history sits under the legacy CIK).
    """
    override = str(company.get("cik") or "").strip()
    if override:
        return f"{int(override):010d}"

    t = str(company["ticker"]).upper()
    for candidate in (t, t.replace("-", "."), t.replace(".", "-"), t.replace("-", "").replace(".", "")):
        if candidate in ticker_map:
            return ticker_map[candidate]
    return None


def fetch_companyfacts(client: httpx.Client, cik: str, ticker: str, force: bool = False) -> dict | None:
    dest = RAW / f"{ticker}.json"
    if dest.exists() and not force:
        return json.loads(dest.read_text(encoding="utf-8"))

    resp = client.get(COMPANYFACTS_URL.format(cik=cik))
    if resp.status_code == 404:
        print(f"  !! no companyfacts for {ticker} (CIK {cik})", file=sys.stderr)
        return None
    resp.raise_for_status()
    facts = resp.json()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(facts), encoding="utf-8")
    return facts


def main() -> int:
    ap = argparse.ArgumentParser(description="Fetch SEC XBRL companyfacts")
    ap.add_argument("--tickers", nargs="*", help="subset of tickers (default: all)")
    ap.add_argument("--force", action="store_true", help="re-download even if cached")
    args = ap.parse_args()

    RAW.mkdir(parents=True, exist_ok=True)
    universe = load_universe()
    if args.tickers:
        wanted = {t.upper() for t in args.tickers}
        universe = [c for c in universe if c["ticker"].upper() in wanted]

    print(f"User-Agent: {_user_agent()}")
    resolved: dict[str, str] = {}
    failures: list[str] = []

    with _client() as client:
        ticker_map = fetch_ticker_map(client)
        print(f"SEC ticker map: {len(ticker_map):,} tickers")

        for i, company in enumerate(universe, 1):
            ticker = company["ticker"]
            cik = resolve_cik(company, ticker_map)
            if cik is None:
                print(f"[{i:>2}/{len(universe)}] {ticker:<6} CIK NOT FOUND")
                failures.append(ticker)
                continue

            cached = (RAW / f"{ticker}.json").exists() and not args.force
            facts = fetch_companyfacts(client, cik, ticker, force=args.force)
            if facts is None:
                failures.append(ticker)
                continue

            resolved[ticker] = cik
            n_tags = len(facts.get("facts", {}).get("us-gaap", {}))
            print(
                f"[{i:>2}/{len(universe)}] {ticker:<6} CIK {cik}  "
                f"{n_tags:>4} us-gaap tags  {'(cached)' if cached else '(fetched)'}"
            )
            if not cached:
                time.sleep(REQUEST_INTERVAL_S)

    (RAW / "cik_map.json").write_text(json.dumps(resolved, indent=2), encoding="utf-8")
    print(f"\nResolved {len(resolved)}/{len(universe)} companies -> {RAW / 'cik_map.json'}")
    if failures:
        print(f"Failed: {', '.join(failures)}", file=sys.stderr)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())

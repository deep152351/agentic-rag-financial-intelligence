"""Canonical financial-metric registry.

Single source of truth, used in two very different places:

1.  Offline (``scripts/build_corpus.py``) -- to pull the right us-gaap XBRL tags
    out of SEC companyfacts and to compute derived ratios.
2.  Online (``app/entities.py``) -- to fuzzy-map free text in a user query
    ("op margin", "cash from ops") onto the same canonical metric ids, which
    then become strict Qdrant payload filters.

Keeping one registry is what makes the retrieval filters and the indexed
payloads provably agree; two registries would drift.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal, Optional

Statement = Literal["income_statement", "balance_sheet", "cash_flow", "ratios"]
PeriodKind = Literal["duration", "instant"]


@dataclass(frozen=True)
class Metric:
    """A canonical metric plus everything needed to extract and phrase it."""

    id: str
    label: str
    statement: Statement
    period: PeriodKind
    unit: Literal["USD", "USD/shares", "shares", "ratio", "percent"]
    #: us-gaap tags in priority order; first tag with usable annual data wins.
    tags: tuple[str, ...] = ()
    #: surface forms the RapidFuzz resolver matches queries against.
    aliases: tuple[str, ...] = ()
    #: for derived metrics only: computed from already-extracted metric ids.
    derived_from: tuple[str, ...] = ()
    formula: Optional[Callable[[dict[str, float]], Optional[float]]] = field(
        default=None, repr=False, compare=False
    )

    @property
    def is_derived(self) -> bool:
        return self.formula is not None


def _ratio(num: str, den: str) -> Callable[[dict[str, float]], Optional[float]]:
    def _f(v: dict[str, float]) -> Optional[float]:
        n, d = v.get(num), v.get(den)
        if n is None or d is None or d == 0:
            return None
        return n / d

    return _f


def _diff(a: str, b: str) -> Callable[[dict[str, float]], Optional[float]]:
    def _f(v: dict[str, float]) -> Optional[float]:
        x, y = v.get(a), v.get(b)
        if x is None or y is None:
            return None
        return x - y

    return _f


# --------------------------------------------------------------------------
# Reported metrics (pulled straight from XBRL)
# --------------------------------------------------------------------------

_REPORTED: tuple[Metric, ...] = (
    # ---- income statement ----
    Metric(
        "revenue", "Total revenue", "income_statement", "duration", "USD",
        # Order matters. Where a filer tags both `Revenues` and the ASC 606 tag,
        # `Revenues` is the consolidated top line and the ASC 606 tag is a
        # component of it -- American Tower reports $10.6B vs $0.9B, Walmart
        # $681B total revenues vs $674.5B net sales. Where `Revenues` is absent
        # (Apple, Microsoft, Alphabet, Meta, Amazon) the ASC 606 tag *is* the
        # top line, so it comes next.
        tags=(
            "Revenues",
            "RevenueFromContractWithCustomerExcludingAssessedTax",
            "RevenueFromContractWithCustomerIncludingAssessedTax",
            "SalesRevenueNet",
            # banks and brokers report the top line net of interest expense
            "RevenuesNetOfInterestExpense",
            "InterestAndDividendIncomeOperating",
        ),
        aliases=("revenue", "revenues", "total revenue", "net revenue", "sales",
                 "net sales", "top line", "turnover"),
    ),
    Metric(
        "cost_of_revenue", "Cost of revenue", "income_statement", "duration", "USD",
        tags=("CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfGoodsSold"),
        aliases=("cost of revenue", "cost of sales", "cogs", "cost of goods sold"),
    ),
    Metric(
        "gross_profit", "Gross profit", "income_statement", "duration", "USD",
        tags=("GrossProfit",),
        aliases=("gross profit", "gross income"),
    ),
    Metric(
        "operating_income", "Operating income", "income_statement", "duration", "USD",
        tags=("OperatingIncomeLoss",),
        aliases=("operating income", "operating profit", "ebit", "income from operations"),
    ),
    Metric(
        "rnd_expense", "Research & development expense", "income_statement", "duration", "USD",
        tags=("ResearchAndDevelopmentExpense",),
        aliases=("r&d", "rnd", "research and development", "research spend", "r and d"),
    ),
    Metric(
        "sganda_expense", "SG&A expense", "income_statement", "duration", "USD",
        tags=("SellingGeneralAndAdministrativeExpense",),
        aliases=("sg&a", "sga", "selling general and administrative", "overhead"),
    ),
    Metric(
        "interest_expense", "Interest expense", "income_statement", "duration", "USD",
        tags=("InterestExpense", "InterestExpenseDebt"),
        aliases=("interest expense", "interest cost"),
    ),
    Metric(
        "income_tax_expense", "Income tax expense", "income_statement", "duration", "USD",
        tags=("IncomeTaxExpenseBenefit",),
        aliases=("income tax", "tax expense", "provision for income taxes"),
    ),
    Metric(
        "net_income", "Net income", "income_statement", "duration", "USD",
        tags=("NetIncomeLoss", "ProfitLoss"),
        aliases=("net income", "net profit", "earnings", "bottom line",
                 "profit", "net earnings"),
    ),
    Metric(
        "eps_diluted", "Diluted EPS", "income_statement", "duration", "USD/shares",
        tags=("EarningsPerShareDiluted",),
        aliases=("diluted eps", "eps", "earnings per share", "diluted earnings per share"),
    ),
    Metric(
        "eps_basic", "Basic EPS", "income_statement", "duration", "USD/shares",
        tags=("EarningsPerShareBasic",),
        aliases=("basic eps", "basic earnings per share"),
    ),
    Metric(
        "shares_diluted", "Diluted shares outstanding", "income_statement", "duration", "shares",
        # Berkshire and other multi-class filers tag only the basic count at the
        # undimensioned level that companyfacts exposes.
        tags=("WeightedAverageNumberOfDilutedSharesOutstanding",
              "WeightedAverageNumberOfSharesOutstandingBasic"),
        aliases=("diluted shares", "share count", "shares outstanding",
                 "weighted average shares"),
    ),
    # ---- balance sheet ----
    Metric(
        "total_assets", "Total assets", "balance_sheet", "instant", "USD",
        tags=("Assets",),
        aliases=("total assets", "assets", "balance sheet size"),
    ),
    Metric(
        "total_liabilities", "Total liabilities", "balance_sheet", "instant", "USD",
        tags=("Liabilities",),
        aliases=("total liabilities", "liabilities"),
    ),
    Metric(
        "shareholders_equity", "Shareholders' equity", "balance_sheet", "instant", "USD",
        tags=(
            "StockholdersEquity",
            "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
        ),
        aliases=("shareholders equity", "stockholders equity", "book value",
                 "equity", "net worth"),
    ),
    Metric(
        "cash_and_equivalents", "Cash & equivalents", "balance_sheet", "instant", "USD",
        tags=("CashAndCashEquivalentsAtCarryingValue",),
        aliases=("cash", "cash and equivalents", "cash on hand", "cash position"),
    ),
    Metric(
        "current_assets", "Total current assets", "balance_sheet", "instant", "USD",
        tags=("AssetsCurrent",),
        aliases=("current assets",),
    ),
    Metric(
        "current_liabilities", "Total current liabilities", "balance_sheet", "instant", "USD",
        tags=("LiabilitiesCurrent",),
        aliases=("current liabilities",),
    ),
    Metric(
        "inventory", "Inventory", "balance_sheet", "instant", "USD",
        tags=("InventoryNet",),
        aliases=("inventory", "inventories", "stock on hand"),
    ),
    Metric(
        "goodwill", "Goodwill", "balance_sheet", "instant", "USD",
        tags=("Goodwill",),
        aliases=("goodwill",),
    ),
    Metric(
        "long_term_debt", "Long-term debt", "balance_sheet", "instant", "USD",
        tags=("LongTermDebtNoncurrent", "LongTermDebt"),
        aliases=("long term debt", "long-term debt", "debt", "borrowings", "leverage"),
    ),
    # ---- cash flow ----
    Metric(
        "operating_cash_flow", "Cash from operations", "cash_flow", "duration", "USD",
        tags=(
            "NetCashProvidedByUsedInOperatingActivities",
            "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
        ),
        aliases=("operating cash flow", "cash from operations", "cash from ops",
                 "cfo", "ocf", "cash flow from operations", "cash generated from operations"),
    ),
    Metric(
        "capex", "Capital expenditure", "cash_flow", "duration", "USD",
        tags=("PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets"),
        aliases=("capex", "capital expenditure", "capital spending", "ppe purchases"),
    ),
    Metric(
        "investing_cash_flow", "Cash from investing", "cash_flow", "duration", "USD",
        tags=("NetCashProvidedByUsedInInvestingActivities",),
        aliases=("investing cash flow", "cash from investing"),
    ),
    Metric(
        "financing_cash_flow", "Cash from financing", "cash_flow", "duration", "USD",
        tags=("NetCashProvidedByUsedInFinancingActivities",),
        aliases=("financing cash flow", "cash from financing"),
    ),
    Metric(
        "dividends_paid", "Dividends paid", "cash_flow", "duration", "USD",
        tags=("PaymentsOfDividendsCommonStock", "PaymentsOfDividends"),
        aliases=("dividends", "dividends paid", "dividend payments", "shareholder payout"),
    ),
    Metric(
        "buybacks", "Share repurchases", "cash_flow", "duration", "USD",
        tags=("PaymentsForRepurchaseOfCommonStock",),
        aliases=("buybacks", "share repurchases", "stock buyback", "repurchases"),
    ),
)


# --------------------------------------------------------------------------
# Derived metrics (computed from the reported ones)
# --------------------------------------------------------------------------

_DERIVED: tuple[Metric, ...] = (
    Metric(
        "gross_margin", "Gross margin", "ratios", "duration", "percent",
        derived_from=("gross_profit", "revenue"),
        formula=_ratio("gross_profit", "revenue"),
        aliases=("gross margin",),
    ),
    Metric(
        "operating_margin", "Operating margin", "ratios", "duration", "percent",
        derived_from=("operating_income", "revenue"),
        formula=_ratio("operating_income", "revenue"),
        aliases=("operating margin", "op margin", "ebit margin", "operating profitability"),
    ),
    Metric(
        "net_margin", "Net margin", "ratios", "duration", "percent",
        derived_from=("net_income", "revenue"),
        formula=_ratio("net_income", "revenue"),
        aliases=("net margin", "profit margin", "net profit margin"),
    ),
    Metric(
        "free_cash_flow", "Free cash flow", "ratios", "duration", "USD",
        derived_from=("operating_cash_flow", "capex"),
        formula=_diff("operating_cash_flow", "capex"),
        aliases=("free cash flow", "fcf", "cash generation"),
    ),
    Metric(
        "fcf_margin", "Free cash flow margin", "ratios", "duration", "percent",
        derived_from=("free_cash_flow", "revenue"),
        formula=_ratio("free_cash_flow", "revenue"),
        aliases=("fcf margin", "free cash flow margin"),
    ),
    Metric(
        "roe", "Return on equity", "ratios", "duration", "percent",
        derived_from=("net_income", "shareholders_equity"),
        formula=_ratio("net_income", "shareholders_equity"),
        aliases=("roe", "return on equity"),
    ),
    Metric(
        "roa", "Return on assets", "ratios", "duration", "percent",
        derived_from=("net_income", "total_assets"),
        formula=_ratio("net_income", "total_assets"),
        aliases=("roa", "return on assets"),
    ),
    Metric(
        "current_ratio", "Current ratio", "ratios", "instant", "ratio",
        derived_from=("current_assets", "current_liabilities"),
        formula=_ratio("current_assets", "current_liabilities"),
        aliases=("current ratio", "liquidity", "liquidity ratio"),
    ),
    Metric(
        "debt_to_equity", "Debt-to-equity", "ratios", "instant", "ratio",
        derived_from=("long_term_debt", "shareholders_equity"),
        formula=_ratio("long_term_debt", "shareholders_equity"),
        aliases=("debt to equity", "d/e", "leverage ratio", "gearing"),
    ),
    Metric(
        "rnd_intensity", "R&D intensity", "ratios", "duration", "percent",
        derived_from=("rnd_expense", "revenue"),
        formula=_ratio("rnd_expense", "revenue"),
        aliases=("r&d intensity", "research intensity", "r&d as a share of revenue"),
    ),
)


ALL_METRICS: tuple[Metric, ...] = _REPORTED + _DERIVED
METRICS: dict[str, Metric] = {m.id: m for m in ALL_METRICS}
REPORTED_METRIC_IDS: tuple[str, ...] = tuple(m.id for m in _REPORTED)
DERIVED_METRIC_IDS: tuple[str, ...] = tuple(m.id for m in _DERIVED)

#: derived metrics are evaluated in declaration order, so a formula may depend
#: on an earlier derived metric (fcf_margin depends on free_cash_flow).
DERIVED_ORDER: tuple[str, ...] = DERIVED_METRIC_IDS

STATEMENTS: tuple[str, ...] = ("income_statement", "balance_sheet", "cash_flow", "ratios")


def metrics_for_statement(statement: str) -> list[Metric]:
    return [m for m in ALL_METRICS if m.statement == statement]


def alias_index() -> list[tuple[str, str]]:
    """Flat ``(surface_form, metric_id)`` pairs for fuzzy matching."""
    pairs: list[tuple[str, str]] = []
    for m in ALL_METRICS:
        pairs.append((m.id.replace("_", " "), m.id))
        pairs.append((m.label.lower(), m.id))
        for a in m.aliases:
            pairs.append((a.lower(), m.id))
    # de-duplicate, keeping first binding for an ambiguous surface form
    seen: set[str] = set()
    out: list[tuple[str, str]] = []
    for surface, mid in pairs:
        if surface not in seen:
            seen.add(surface)
            out.append((surface, mid))
    return out


def format_value(metric_id: str, value: float) -> str:
    """Human-readable rendering used in both corpus text and answers."""
    m = METRICS[metric_id]
    if m.unit == "percent":
        return f"{value * 100:.1f}%"
    if m.unit == "ratio":
        return f"{value:.2f}x"
    if m.unit == "USD/shares":
        return f"${value:,.2f}"
    if m.unit == "shares":
        return f"{value / 1e9:.2f}B shares" if abs(value) >= 1e9 else f"{value / 1e6:.1f}M shares"
    # USD
    a = abs(value)
    sign = "-" if value < 0 else ""
    if a >= 1e9:
        return f"{sign}${a / 1e9:,.2f}B"
    if a >= 1e6:
        return f"{sign}${a / 1e6:,.1f}M"
    return f"{sign}${a:,.0f}"

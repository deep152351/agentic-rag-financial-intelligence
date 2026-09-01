# Agentic RAG for Financial Statement & Wealth Intelligence

A dual-LLM FastAPI service that answers questions about **US public-company financial
statements** and about a **client's investment portfolio**, grounded in real SEC filings.

Retrieval is a parallel hybrid fan-out — BGE dense + BM25 sparse, both in Qdrant, fused
with Reciprocal Rank Fusion — constrained by company / fiscal-year / metric filters that
are fuzzy-resolved from the question with RapidFuzz, and run under an explicit turn budget.

Every number the system reports traces back to a Form 10-K. Nothing is generated,
estimated, or recalled from model memory, and a validation pass mechanically checks that.

```
                    ┌──────────────────────── FastAPI ────────────────────────┐
   question ───────▶│  /ask   /health   /universe   /portfolio                │
                    └────┬────────────────────────────────────────────────────┘
                         │
                ┌────────▼─────────┐   surface forms      ┌──────────────────────┐
                │  PLANNER  LLM-1  │─────────────────────▶│  Entity resolution   │
                │  fast model      │  "goldman","op mgn"  │  RapidFuzz + gating  │
                │  decompose only  │◀─────────────────────│  -> strict filters   │
                └────────┬─────────┘   resolved ids       └──────────────────────┘
                         │ N sub-queries
        ┌────────────────▼──────────────────────────────────────────┐
        │            PARALLEL RETRIEVAL WAVE  (asyncio.gather)      │
        │   sq₁ ─┬─ BGE dense  ─┐        sq₂ ─┬─ BGE dense  ─┐      │
        │        └─ BM25 sparse─┤             └─ BM25 sparse─┤ …    │  2N searches
        │                       ▼                            ▼      │  concurrently
        │              ┌──────────────────────────────────────┐     │
        │              │  Reciprocal Rank Fusion (k = 60)     │     │
        │              └──────────────────┬───────────────────┘     │
        └─────────────────────────────────┼─────────────────────────┘
                                          │ coverage check
                      ┌───────────────────▼────────────────────┐
                      │  TURN BUDGET: turns · deadline · value │──▶ retry relaxed
                      └───────────────────┬────────────────────┘    (≤ MAX_TURNS)
                                          │
              ┌───────────────────────────▼───────────────────┐
              │  Portfolio tools (deterministic, not the LLM) │
              │  exposure · look-through · concentration      │
              └───────────────────────────┬───────────────────┘
                                          │
                      ┌───────────────────▼────────────────────┐
                      │  SYNTHESIZER  LLM-2   strong model     │
                      │  grounded prose + citations            │
                      └───────────────────┬────────────────────┘
                                          │
                      ┌───────────────────▼────────────────────┐
                      │  Groundedness validation               │
                      │  every figure ⟵ retrieved context      │
                      └────────────────────────────────────────┘
```

---

## Measured results

110 evaluation cases with gold labels **derived from the index**, not hand-annotated
(see [Evaluation](#evaluation)). Reproduce with `make eval-full`.

| Metric | Value |
|---|---|
| **Recall@10** | **99.7 %** |
| Hit@10 | 100 % |
| MRR | 0.864 |
| **Constraint compliance** | **99.1 %** |
| Intent accuracy | 100 % |
| Exact constraint match | 91.8 % |
| Portfolio tool selection | 100 % |
| Retrieval latency p50 / p95 | 233 ms / 381 ms |
| **Groundedness** (end-to-end) | **100 %** |
| Mean turns used | 0.96 / 3 allowed |

### Ablation grid — what each component actually buys

| configuration | filters | Recall@10 | MRR | constraint compliance |
|---|---|---|---|---|
| dense only (BGE) | on | 99.7 % | 0.871 | 98.6 % |
| sparse only (BM25) | on | 99.7 % | 0.865 | 98.6 % |
| **hybrid + RRF** | **on** | **99.7 %** | 0.864 | **99.1 %** |
| dense only (BGE) | off | 78.6 % | 0.551 | 29.5 % |
| sparse only (BM25) | off | 92.6 % | 0.644 | 33.7 % |
| hybrid + RRF | off | 92.8 % | **0.716** | 33.6 % |

Read honestly, this says two things:

* **Metadata filtering is the single biggest lever**: +6.9 pp Recall@10 and **+65.5 pp
  constraint compliance** over the same retriever with filters off. Once a query is
  pinned to one company-year, only a handful of documents survive and ranking barely
  matters — which is why all three filtered rows tie.
* **Fusion earns its place on ranking quality, not raw recall.** With filters off, where
  ranking has to do the work, RRF beats dense-only by **+14.2 pp Recall@10 / +0.164 MRR**
  and beats the stronger single branch (BM25) by **+0.072 MRR**. It puts the right
  document higher even when recall is comparable.

The filtered rows tying is a real result, and it is the interesting one: the value of
this system is concentrated in the entity-resolution layer, not in the embedding model.

---

## The data is real

| | |
|---|---|
| Source | SEC EDGAR XBRL `companyfacts` API (Form 10-K) |
| Coverage | **38 companies**, 11 GICS sectors, **FY2019–FY2025**, 264 company-years |
| Metrics | 26 reported + 10 derived, per company-year |
| Corpus | **1,352 chunks** — income statement, balance sheet, cash flow, ratios, year-over-year narrative, company profile, portfolio positions |

CIKs are resolved at build time from SEC's own ticker map — none are hardcoded. Spot-checked
against published figures: **20/20 exact** (Apple FY2024 revenue $391.04B, NVIDIA FY2025
$130.50B, Amazon FY2024 $637.96B, Exxon FY2022 $413.68B, …), and 237/238 company-years pass
an internal `net_income / diluted_shares ≈ diluted_EPS` consistency check.

**The only synthetic data is the ownership book** — which accounts exist, how many shares
each holds, what they paid. No such data is public. It is generated deterministically
(seeded) and labelled as synthetic in the file, the API response, and the `/portfolio`
endpoint. Even there the cost basis is derived from *reported* diluted EPS times a
sector-typical multiple, so the book stays internally coherent.

There are **no market prices** anywhere in this project. Every valuation is stated *at
cost*, and every fundamental exposure is a look-through claim on reported financials.
That is a deliberate constraint: it means the system cannot fabricate a gain or loss.

### Three data problems worth knowing about

These are the parts that took the most care, and the ones most worth asking about.

1. **Fiscal-year labelling.** A period's fiscal year is `min(fy)` across every 10-K
   reporting that period end — the first 10-K to report a period *is* that period's annual
   report; later filings repeat it as a comparative under their own higher `fy`. A
   month-based heuristic gets Nike's May-ending FY2024 wrong; this doesn't.

2. **Stock splits.** `companyfacts` only restates a period for as long as some filing still
   reports it (~3 years), so a raw per-share series shows NVIDIA's diluted EPS "collapsing"
   75 % at its 10-for-1 split. The extractor detects splits from discontinuities in the
   share-count series and restates earlier years onto the current basis — correctly
   recovering NVIDIA 4:1 and 10:1, Amazon 20:1, Walmart 3:1, Broadcom 10:1, Netflix 10:1.
   Where a split coincides with heavy issuance and the factor is genuinely ambiguous
   (Tesla's FY2020 jump is a 3:1 split *on top of* ~22 % dilution, landing between two
   candidate ratios), the affected per-share figures are **dropped rather than published on
   an unknown basis**.

3. **Tag drift and multi-class filers.** Filers migrate us-gaap tags mid-history (NVIDIA
   moved to the ASC 606 revenue tag in FY2023) so tags are merged year-by-year in priority
   order rather than picking one "best" tag. Where both `Revenues` and the ASC 606 tag
   exist, `Revenues` is the consolidated top line and the other is a component of it —
   American Tower reports $10.6B vs $0.9B. Visa and Berkshire publish no undimensioned
   per-share facts at all, which is a real limitation of the endpoint, not a bug.

---

## Quickstart

```bash
pip install -r requirements.txt

python -m scripts.fetch_sec         # download SEC companyfacts (~165 MB, cached)
python -m scripts.build_corpus      # -> data/financials.json, data/corpus.jsonl
python -m scripts.build_portfolio   # -> data/portfolio.json  (synthetic book)
python -m scripts.build_eval_set    # -> data/eval/eval_set.json
python -m scripts.index_corpus      # embed + index into Qdrant (several minutes)

python -m uvicorn app.main:app --reload
```

Or `make rebuild && make serve`, or on Windows `powershell -ExecutionPolicy Bypass -File setup.ps1`.

**It runs with no API key.** With no key configured the service falls back to a
deterministic rule-based planner and a stub synthesizer, so retrieval, entity resolution,
fusion, budgeting, portfolio analytics and the entire evaluation stay live and measurable.
Every number in this README was produced that way — they measure the retrieval system, not
a language model.

To enable the LLMs, copy `.env.example` to `.env` and set one key:

```bash
LLM_PROVIDER=anthropic
PLANNER_MODEL=claude-haiku-4-5      # fast, runs every turn
SYNTHESIZER_MODEL=claude-opus-5     # strong, runs once
ANTHROPIC_API_KEY=sk-ant-...
```

Anthropic, OpenAI, OpenRouter and Gemini are all supported behind one interface.

### See it work

```bash
python -m scripts.demo                 # scripted walkthrough, full traces
python -m scripts.demo --interactive
make test                              # 125 tests
make eval-full                         # metrics + ablation grid
```

---

## How each piece works

### Dual-LLM

Two models, two jobs, independently configurable:

| slot | job | why this model |
|---|---|---|
| **Planner** (LLM-1) | Read the question, emit sub-queries and candidate filter surface forms as JSON | Runs on *every* turn, schema-constrained, latency-critical → fast model |
| **Synthesizer** (LLM-2) | Turn already-retrieved, already-validated facts into prose with citations | Runs *once*, quality-critical, must not hallucinate a figure → strong model |

The split is the point: planning is cheap and frequent, synthesis is expensive and rare.
One model sized for both either overpays on every turn or under-serves the answer.

Critically, **the planner never touches the index and never sees a document**. It proposes
surface forms; `app/entities.py` is the only component allowed to turn text into a filter.
So a planner hallucination ("Tesla's FY2031 figures") cannot produce a bogus filter — it
produces an unresolvable one, which is dropped before it reaches Qdrant.

### Parallel hybrid retrieval + RRF

A plan carries *N* sub-queries; each is searched on two independent branches, so a wave
issues **2N searches dispatched concurrently** with `asyncio.gather`. A six-company
comparison costs about as long as its slowest single search rather than the sum of twelve.

Fusion is Reciprocal Rank Fusion:

```
score(d) = Σ_branches  weight_b / (k + rank_b(d)) ,  k = 60
```

Cosine similarity lives in [-1, 1]; BM25 is unbounded and corpus-dependent. Any weighted
*score* blend needs normalisation constants that drift as the corpus changes. RRF discards
magnitudes and keeps only ranks, so it is scale-free by construction. The same operator
fuses across sub-queries, so a document answering several parts of a decomposed question
outranks one answering a single part very well.

### Metadata-aware retrieval (RapidFuzz)

Free text → strict Qdrant filters on company, fiscal year and metric. Two things make it
safe rather than merely clever:

* **Confidence gating.** A match must clear a score floor **and** beat the runner-up by a
  margin. Below that it is demoted to a *soft hint* that shapes the query text but never
  removes a document. Silently filtering on the wrong company is the worst failure mode in
  this domain — it produces a fluent, confident answer about the wrong business. Without
  the corporate-suffix guard, "Hooli Incorporated" fuzzy-matches "UnitedHealth Group
  Incorporated" at 85 on the shared suffix alone.
* **One vocabulary.** Metric aliases come from `app/metrics.py` — the same registry the
  corpus builder used to tag payloads. A resolved filter is therefore guaranteed to
  correspond to something actually indexed. Two registries would drift.

It handles aliases (`coke`→KO, `p&g`→PG, `cupertino`→AAPL), typos (`Micorsoft`→MSFT),
possessives (`MA's`, `chevron's` — a bug that alone was worth **56 of 110** eval cases),
single-letter tickers (`V's`, but never a stray lowercase "v"), colloquial metrics
(`op margin`, `bottom line`, `cash from ops`), and year forms (`FY24`, `FY'23`,
`fiscal 2022`, `2020 to 2023`, `between X and Y`, `since 2021`, `latest`).

### Turn-budget guardrails

An agentic loop without a budget either retrieves until something times out or answers
instantly from a bad first wave. Three ceilings, checked in order:

1. **Turn count** — at most `MAX_TURNS` retrieval waves. Bounds cost.
2. **Global deadline** — wall-clock for the request, with a **reserve carved out so the
   synthesizer always gets to run**. Retrieval is interruptible; synthesis is not. A wave
   that would eat the reserve is refused — answering from what we have beats a 504.
3. **Marginal value** — a wave that adds no new documents, or that already hit the coverage
   target, ends the loop *even with budget to spare*. Re-retrieving the same ten documents
   is pure latency.

Retries don't repeat the same search: each relaxes the constraint most likely to be the bad
guess — metric first, then fiscal year — while **company identity is never relaxed**,
because returning the right metric for the wrong company is exactly the confident-and-wrong
failure the whole design is organised against.

Out-of-scope questions are refused at planning time and spend **zero** turns and zero
searches.

### Wealth intelligence

Portfolio analytics are deterministic Python over the ownership book and the SEC
fundamentals — **not** the LLM's arithmetic. An LLM summing 30 positions is a source of
quiet errors that read perfectly fluently. The computed results are handed to the
synthesizer as authoritative facts.

The interesting tool is **look-through**: owning *n* shares is owning a claim on
`n / diluted_shares` of everything the business earns, so

```
look-through(metric) = Σ_holdings  shares_held × metric / diluted_shares
```

is a real, checkable quantity computed entirely from reported figures — "your portfolio's
share of FY2024 net income" — with no market price needed. Also: exposure by
sector/industry/account, concentration (top-5/top-10 weight, HHI, effective position
count), and fundamental screens over holdings ("which of my positions had operating margin
decline in FY2024?").

### Groundedness validation

Every currency amount, percentage and per-share figure in the answer is parsed to a
magnitude and matched against figures parsed the same way out of the exact context the
model was given. Unmatched figures are returned in `ungrounded_figures` and the response is
flagged `grounded: false`. Rounding is tolerated ("$391 billion" ≡ $391.04B); invention is
not. Bare four-digit years are excluded so "FY2024" is never mistaken for a claim.

---

## API

| endpoint | purpose |
|---|---|
| `POST /ask` | the agent — answer, citations, grounding verdict, full trace |
| `GET /health` | Qdrant status, active models, budget and retrieval configuration |
| `GET /universe` | the companies and fiscal years actually indexed |
| `GET /portfolio` | account summary, with the synthetic-data disclaimer |

```bash
curl -s localhost:8000/ask -H 'content-type: application/json' \
  -d '{"question":"What was Apple'\''s operating margin in FY2024?"}'
```

The response carries a **trace** built for auditing, not decoration: resolved constraints,
every entity-resolution decision with its score / runner-up / margin and whether it became
a strict filter or a soft hint, one record per retrieval wave (sub-queries, documents,
new documents, coverage, elapsed, and why the loop continued or stopped), and the budget
report.

---

## Evaluation

Gold labels are **derived, not annotated**. Because the corpus builder tagged every chunk
with the company, fiscal year and metric ids it contains, the document that must be
retrieved for "Apple's FY2024 operating margin" is known by construction:
`AAPL-FY2024-ratios`. That removes the usual weak link in a RAG eval — hand-labelled
relevance that quietly disagrees with the index.

110 cases across six families — `lookup`, `yoy`, `comparison`, `trend`, `portfolio`,
`out_of_scope` — deliberately phrased with alias and paraphrase stress (`cupertino`,
`master card`, `EBIT`, `FY'25`) so the eval exercises the resolver rather than string
equality.

**Constraint compliance is reported as two separate numbers**, because they fail for
different reasons:

* *Filter compliance* — the share of returned documents satisfying the **gold** constraints
  (not the ones the system resolved; grading a filter against itself always scores 100 %).
  Low values mean the wrong company or year got through.
* *Resolution accuracy* — precision/recall/F1 per field against the constraints the question
  actually implies. Precision below 1.0 is an **over**-constraint that silently removes good
  documents; recall below 1.0 is an **under**-constraint.

Separating them matters: a system that applies no filters at all scores perfect filter
compliance and zero resolution recall.

Per-family results:

| family | n | Recall@10 | compliance | intent |
|---|---|---|---|---|
| lookup | 24 | 100 % | 100 % | 100 % |
| comparison | 24 | 100 % | 100 % | 100 % |
| trend | 24 | 99.0 % | 100 % | 100 % |
| yoy | 24 | 100 % | 100 % | 100 % |
| portfolio | 8 | n/a | 88.8 % | 100 % |
| out_of_scope | 6 | n/a | n/a | 100 % |

---

## Project layout

```
app/
  main.py           FastAPI: /ask, /health, /universe, /portfolio
  orchestrator.py   the agent loop; where the turn budget binds
  planner.py        LLM-1 + the deterministic fallback planner
  synthesizer.py    LLM-2, grounded prose with citations
  entities.py       RapidFuzz resolution -> strict filters, with confidence gating
  metrics.py        canonical metric registry (shared by builder and resolver)
  retrieval.py      parallel hybrid fan-out + RRF fusion + coverage
  vectorstore.py    Qdrant hybrid collection, constraint -> filter translation
  embeddings.py     BGE dense + BM25 sparse encoders
  budget.py         turn / deadline / marginal-value ceilings
  wealth.py         portfolio analytics incl. look-through fundamentals
  validator.py      groundedness checking
  llm.py            Anthropic / OpenAI / OpenRouter / Gemini + deterministic stub
  config.py         env-driven settings
scripts/
  fetch_sec.py      SEC EDGAR companyfacts download
  build_corpus.py   XBRL -> fact store + retrieval chunks
  build_portfolio.py synthetic ownership book + its chunks
  build_eval_set.py derived gold labels
  index_corpus.py   embed and index
  evaluate.py       Recall@10, compliance, groundedness, latency, ablations
  demo.py           scripted / interactive walkthrough
tests/              125 tests
```

---

## Known limitations

* **Annual data only.** `companyfacts` quarterly facts are not ingested, so nothing below
  FY granularity is answerable.
* **Multi-class filers.** Visa and Berkshire publish no undimensioned per-share facts, so
  they have no EPS or share-count series and are excluded from the synthetic portfolio.
  They remain fully searchable for statement-level questions.
* **No market prices.** Deliberate — see above. There is no gain/loss, no valuation, no
  total return.
* **Narrative text is generated from structured facts**, not extracted from filing prose.
  The MD&A-style year-over-year cards are faithful to the numbers but are not the
  companies' own words. Ingesting 10-K narrative sections would make the dense branch work
  considerably harder, and is the most obvious next step.
* **The corpus is small** (1,352 chunks). Filters dominate ranking at this scale; the
  ablation grid above shows exactly where that stops being true.
* **Indexing is slow** (~12 min on CPU) because BGE embeds 1,352 chunks locally. It is a
  one-time cost.

## Attribution

Financial data © SEC EDGAR, public domain. This project is for educational and research
use; it is not investment advice.

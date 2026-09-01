"""FastAPI surface."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from .config import get_settings
from .embeddings import get_encoders
from .entities import get_resolver
from .llm import get_llms
from .orchestrator import Orchestrator
from .schemas import AskRequest, AskResponse
from .vectorstore import get_store
from .wealth import get_analytics

logger = logging.getLogger(__name__)
_orchestrator: Orchestrator | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    logger.info("warming embedding models...")
    get_encoders().warm_up()  # pay the model load at startup, not on the first request
    get_resolver()
    get_analytics()

    global _orchestrator
    _orchestrator = Orchestrator(settings=settings)
    logger.info("ready: %s", get_llms(settings).describe())
    yield
    await get_store().close()


app = FastAPI(
    title="Agentic RAG for Financial Statement & Wealth Intelligence",
    description=(
        "Dual-LLM agentic RAG over SEC 10-K filings and a client portfolio book. "
        "Parallel hybrid retrieval (BGE dense + BM25 sparse) in Qdrant, fused with "
        "Reciprocal Rank Fusion, filtered by fuzzy-resolved company / fiscal-year / "
        "metric constraints, under an explicit turn budget."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


def _require_orchestrator() -> Orchestrator:
    if _orchestrator is None:  # pragma: no cover - only before startup completes
        raise HTTPException(status_code=503, detail="service still starting")
    return _orchestrator


@app.get("/health")
async def health() -> dict[str, Any]:
    settings = get_settings()
    store = await get_store().health()
    return {
        "status": "ok" if store.get("points", 0) > 0 else "degraded",
        "qdrant": store,
        "llm": get_llms(settings).describe(),
        "budget": {
            "max_turns": settings.max_turns,
            "global_deadline_s": settings.global_deadline_s,
            "turn_deadline_s": settings.turn_deadline_s,
        },
        "retrieval": {
            "dense_model": settings.dense_model,
            "sparse_model": settings.sparse_model,
            "top_k": settings.top_k,
            "rrf_k": settings.rrf_k,
            "candidates_per_branch": settings.candidates_per_branch,
        },
    }


@app.get("/universe")
async def universe() -> dict[str, Any]:
    """What the index actually covers -- the vocabulary filters are resolved against."""
    resolver = get_resolver()
    return {
        "companies": [
            {"ticker": c["ticker"], "name": c["name"], "sector": c["sector"]}
            for c in resolver.companies
        ],
        "fiscal_years": resolver.known_years,
        "count": len(resolver.companies),
    }


@app.get("/portfolio")
async def portfolio() -> dict[str, Any]:
    analytics = get_analytics()
    return {
        "disclaimer": analytics.book["_disclaimer"],
        "as_of": analytics.book["as_of"],
        "valuation_basis": analytics.book["valuation_basis"],
        "totals": analytics.book["totals"],
        "accounts": [
            {
                "account_id": a["account_id"],
                "name": a["name"],
                "type": a["type"],
                "positions": len(a["positions"]),
                "total_value_at_cost": a["total_value_at_cost"],
            }
            for a in analytics.accounts
        ],
    }


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest) -> AskResponse:
    try:
        return await _require_orchestrator().answer(request)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("request failed")
        raise HTTPException(status_code=500, detail=f"internal error: {exc}") from exc


@app.exception_handler(ValueError)
async def value_error_handler(_request, exc: ValueError) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": str(exc)})

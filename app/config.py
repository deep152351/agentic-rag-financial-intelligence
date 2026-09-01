"""Runtime configuration.

Everything is env-overridable so the same code runs three ways: fully local
(embedded Qdrant + stub LLM, no keys, no network), local Qdrant + a real LLM, or
against a hosted Qdrant cluster.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    # ---------------- vector store ----------------
    qdrant_url: str = Field(default="", description="Hosted Qdrant URL; empty = embedded")
    qdrant_api_key: str = ""
    qdrant_path: Path = Field(
        default=DATA_DIR / "qdrant",
        description="On-disk path for embedded Qdrant when qdrant_url is unset",
    )
    collection: str = "financial_statements"

    # ---------------- embeddings ----------------
    dense_model: str = "BAAI/bge-small-en-v1.5"
    sparse_model: str = "Qdrant/bm25"

    # ---------------- retrieval ----------------
    #: candidates pulled per branch, per sub-query, before fusion
    candidates_per_branch: int = 40
    #: documents handed to the synthesizer after fusion
    top_k: int = 10
    #: RRF damping constant. 60 is the value from Cormack et al. (2009); it makes
    #: fusion insensitive to raw score scales, which matters because cosine
    #: similarity and BM25 are not remotely comparable numbers.
    rrf_k: int = 60
    #: hard ceiling on how many sub-queries the planner may fan out to
    max_subqueries: int = 6

    # ---------------- entity resolution ----------------
    #: RapidFuzz score (0-100) below which a candidate match is discarded.
    fuzzy_threshold: int = 84
    #: a match must beat the runner-up by this much to be applied as a *strict*
    #: filter; otherwise it is kept as a soft hint and the filter is relaxed.
    fuzzy_margin: int = 6

    # ---------------- LLM ----------------
    llm_provider: Literal["anthropic", "openai", "openrouter", "gemini", "stub"] = "anthropic"
    #: Fast, cheap, structured-output model. Runs the planner on every turn.
    planner_model: str = "claude-haiku-4-5"
    #: Stronger model. Runs once per request, on grounded synthesis only.
    synthesizer_model: str = "claude-opus-5"
    planner_max_tokens: int = 2000
    synthesizer_max_tokens: int = 4000

    anthropic_api_key: str = ""
    openai_api_key: str = ""
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    gemini_api_key: str = ""

    # ---------------- turn budget ----------------
    #: max agent turns (a turn = one plan + one parallel retrieval wave)
    max_turns: int = 3
    #: wall-clock ceiling for the whole request
    global_deadline_s: float = 25.0
    #: ceiling for a single retrieval wave
    turn_deadline_s: float = 8.0
    #: ceiling for one LLM call
    llm_timeout_s: float = 20.0
    #: stop early once this share of the plan's information needs are covered
    coverage_target: float = 0.85

    # ---------------- misc ----------------
    log_level: str = "INFO"

    @model_validator(mode="after")
    def _fall_back_to_stub(self) -> "Settings":
        """No key for the chosen provider means the stub LLM, not a crash at request time."""
        needed = {
            "anthropic": self.anthropic_api_key,
            "openai": self.openai_api_key,
            "openrouter": self.openrouter_api_key,
            "gemini": self.gemini_api_key,
            "stub": "n/a",
        }[self.llm_provider]
        if not needed:
            object.__setattr__(self, "llm_provider", "stub")
        return self

    @property
    def uses_embedded_qdrant(self) -> bool:
        return not self.qdrant_url


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

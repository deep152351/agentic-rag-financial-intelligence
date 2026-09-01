"""The dual-LLM layer.

Two models with different jobs, wired independently:

``planner``      Runs on every turn. Its job is decomposition and extraction --
                 read a question, emit sub-queries and candidate filter surface
                 forms as JSON. Latency-critical and schema-constrained, so it
                 gets the fast model and structured output.
``synthesizer``  Runs once, at the end, over documents that already passed
                 retrieval and validation. Its job is careful grounded prose with
                 citations. Quality-critical, so it gets the strong model.

The split is the point: planning is cheap and frequent, synthesis is expensive
and rare. Sizing one model for both would either overpay on every turn or
under-serve the answer. Both slots are independently configurable -- provider and
model id -- so the pair can be Haiku+Opus, Flash+Pro, or mini+4o.

A deterministic ``StubClient`` implements the same protocol, so the whole
pipeline (including the evaluation harness) runs with no API key at all.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Protocol

from .config import Settings, get_settings

logger = logging.getLogger(__name__)


class LLMError(RuntimeError):
    pass


@dataclass
class LLMResult:
    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    stop_reason: str = ""
    refused: bool = False


class LLMClient(Protocol):
    model: str

    async def complete(self, system: str, user: str, *, max_tokens: int,
                       json_schema: dict[str, Any] | None = None) -> LLMResult: ...


# ---------------------------------------------------------------------------
# JSON handling shared by every provider
# ---------------------------------------------------------------------------

_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def parse_json_object(text: str) -> dict[str, Any]:
    """Best-effort extraction of a JSON object from a model response.

    With structured outputs this is a plain ``json.loads``. The fence-stripping
    and brace-matching fallbacks exist for providers without schema enforcement,
    where a model may wrap JSON in prose or a code fence.
    """
    candidate = (text or "").strip()
    if not candidate:
        raise LLMError("empty model response")
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    fenced = _FENCE.search(candidate)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            candidate = fenced.group(1)

    start = candidate.find("{")
    if start >= 0:
        depth, in_string, escape = 0, False, False
        for i, ch in enumerate(candidate[start:], start=start):
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
            elif ch == '"':
                in_string = not in_string
            elif not in_string:
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(candidate[start:i + 1])
                        except json.JSONDecodeError:
                            break
    raise LLMError(f"could not parse JSON from model response: {text[:200]!r}")


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------


@dataclass
class AnthropicClient:
    model: str
    api_key: str
    timeout: float = 20.0
    effort: str | None = None
    #: Opus 5 / Fable 5 may decline a request outright (stop_reason "refusal").
    #: Server-side fallbacks reroute those to another model instead of failing.
    use_refusal_fallback: bool = True
    _client: Any = field(default=None, repr=False)

    def _ensure(self):
        if self._client is None:
            from anthropic import AsyncAnthropic

            self._client = AsyncAnthropic(api_key=self.api_key, timeout=self.timeout)
        return self._client

    async def complete(self, system: str, user: str, *, max_tokens: int,
                       json_schema: dict[str, Any] | None = None) -> LLMResult:
        import anthropic

        client = self._ensure()
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        output_config: dict[str, Any] = {}
        if json_schema is not None:
            output_config["format"] = {"type": "json_schema", "schema": json_schema}
        if self.effort:
            output_config["effort"] = self.effort
        if output_config:
            kwargs["output_config"] = output_config

        try:
            response = await self._create(client, kwargs)
        except anthropic.BadRequestError as exc:
            # Older models reject output_config/effort; retry without them once
            # rather than failing the request.
            if "output_config" in str(exc) or "effort" in str(exc):
                logger.warning("%s rejected output_config; retrying plain", self.model)
                kwargs.pop("output_config", None)
                if json_schema is not None:
                    kwargs["system"] = system + "\n\nRespond with a single JSON object and nothing else."
                response = await self._create(client, kwargs)
            else:
                raise LLMError(f"anthropic bad request: {exc}") from exc
        except anthropic.APIStatusError as exc:
            raise LLMError(f"anthropic error {exc.status_code}: {exc}") from exc
        except anthropic.APIConnectionError as exc:
            raise LLMError(f"anthropic connection error: {exc}") from exc

        stop_reason = getattr(response, "stop_reason", "") or ""
        text = "".join(b.text for b in response.content if getattr(b, "type", "") == "text")
        usage = getattr(response, "usage", None)
        return LLMResult(
            text=text,
            model=self.model,
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
            stop_reason=stop_reason,
            refused=stop_reason == "refusal",
        )

    async def _create(self, client, kwargs: dict[str, Any]):
        if self.use_refusal_fallback:
            try:
                return await client.beta.messages.create(
                    **kwargs,
                    betas=["server-side-fallback-2026-07-01"],
                    fallbacks="default",
                )
            except Exception as exc:  # beta unavailable on this account/model
                logger.debug("refusal fallback unavailable (%s); using standard path", exc)
                self.use_refusal_fallback = False
        return await client.messages.create(**kwargs)


@dataclass
class OpenAIClient:
    """Also serves OpenRouter, which is OpenAI-compatible with a base_url swap."""

    model: str
    api_key: str
    base_url: str | None = None
    timeout: float = 20.0
    _client: Any = field(default=None, repr=False)

    def _ensure(self):
        if self._client is None:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(
                api_key=self.api_key, base_url=self.base_url, timeout=self.timeout
            )
        return self._client

    async def complete(self, system: str, user: str, *, max_tokens: int,
                       json_schema: dict[str, Any] | None = None) -> LLMResult:
        client = self._ensure()
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if json_schema is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "plan", "schema": json_schema, "strict": False},
            }
        try:
            response = await client.chat.completions.create(**kwargs)
        except Exception as exc:
            raise LLMError(f"openai-compatible provider error: {exc}") from exc

        choice = response.choices[0]
        usage = getattr(response, "usage", None)
        return LLMResult(
            text=choice.message.content or "",
            model=self.model,
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
            stop_reason=choice.finish_reason or "",
        )


@dataclass
class GeminiClient:
    model: str
    api_key: str
    timeout: float = 20.0

    async def complete(self, system: str, user: str, *, max_tokens: int,
                       json_schema: dict[str, Any] | None = None) -> LLMResult:
        import google.generativeai as genai

        genai.configure(api_key=self.api_key)
        config: dict[str, Any] = {"max_output_tokens": max_tokens, "temperature": 0.2}
        if json_schema is not None:
            config["response_mime_type"] = "application/json"
        model = genai.GenerativeModel(
            model_name=self.model, system_instruction=system, generation_config=config
        )
        try:
            response = await model.generate_content_async(user)
        except Exception as exc:
            raise LLMError(f"gemini error: {exc}") from exc
        return LLMResult(text=response.text or "", model=self.model)


@dataclass
class StubClient:
    """Deterministic offline stand-in.

    It is not a language model and does not pretend to be one. The planner path
    returns an empty plan, which makes the orchestrator fall back to its
    deterministic rule-based planner; the synthesis path renders the retrieved
    facts directly. That keeps every non-LLM component -- entity resolution,
    hybrid retrieval, RRF, budgeting, validation, the whole eval -- runnable and
    measurable without an API key.
    """

    model: str = "stub"

    async def complete(self, system: str, user: str, *, max_tokens: int,
                       json_schema: dict[str, Any] | None = None) -> LLMResult:
        if json_schema is not None:
            return LLMResult(text=json.dumps({"subqueries": [], "intent": "", "reasoning": "stub"}),
                             model=self.model)
        return LLMResult(text=_stub_answer(user), model=self.model)


def _stub_answer(user_prompt: str) -> str:
    """Render the context the synthesizer was given, without inventing prose."""
    body = user_prompt
    for marker in ("PORTFOLIO ANALYTICS", "CONTEXT DOCUMENTS"):
        # start at whichever context block comes first so computed portfolio
        # analytics are not silently dropped ahead of the retrieved documents
        if marker in body:
            body = body[body.index(marker):]
            break
    else:
        return "[stub mode - no LLM configured] No context was assembled."

    body = body.split("QUESTION\n", 1)[0].strip()
    lines = [ln for ln in body.splitlines() if ln.strip()][:32]
    return (
        "[stub mode - no LLM configured] The retrieval pipeline returned the "
        "following grounded facts:\n\n" + "\n".join(lines)
    )


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def _build(provider: str, model: str, settings: Settings, effort: str | None = None) -> LLMClient:
    if provider == "anthropic":
        return AnthropicClient(model, settings.anthropic_api_key,
                               timeout=settings.llm_timeout_s, effort=effort)
    if provider == "openai":
        return OpenAIClient(model, settings.openai_api_key, timeout=settings.llm_timeout_s)
    if provider == "openrouter":
        return OpenAIClient(model, settings.openrouter_api_key,
                            base_url=settings.openrouter_base_url, timeout=settings.llm_timeout_s)
    if provider == "gemini":
        return GeminiClient(model, settings.gemini_api_key, timeout=settings.llm_timeout_s)
    return StubClient()


@dataclass
class DualLLM:
    planner: LLMClient
    synthesizer: LLMClient
    provider: str

    @property
    def is_stub(self) -> bool:
        return self.provider == "stub"

    def describe(self) -> dict[str, str]:
        return {
            "provider": self.provider,
            "planner_model": self.planner.model,
            "synthesizer_model": self.synthesizer.model,
        }


_dual: DualLLM | None = None


def get_llms(settings: Settings | None = None) -> DualLLM:
    global _dual
    if _dual is None:
        s = settings or get_settings()
        _dual = DualLLM(
            planner=_build(s.llm_provider, s.planner_model, s),
            synthesizer=_build(s.llm_provider, s.synthesizer_model, s, effort="medium"),
            provider=s.llm_provider,
        )
        logger.info("LLMs: %s", _dual.describe())
    return _dual


def reset_llms() -> None:
    """Test hook."""
    global _dual
    _dual = None

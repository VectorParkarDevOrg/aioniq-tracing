"""AioniQ Tracing SDK -- first-party OpenTelemetry instrumentation for any
agent integrating with AioniQ.

Deliberately does NOT monkey-patch the `openai`/`langchain`/any other SDK
(unlike e.g. openlit). Every span is created explicitly by the caller via
`step()`, `trace_llm_call()`, or `trace_embedding()` -- no hidden global
instrumentation, no auto-instrumentation bugs to inherit.

Quickstart:

    import aioniq_tracing

    aioniq_tracing.init(
        base_url="https://your-aioniq-host/governance-api",
        agent_key="ak_live_...",
    )
    aioniq_tracing.set_call_origin("user")

    async with aioniq_tracing.step("answer_question", call_role="primary"):
        async with aioniq_tracing.trace_llm_call(model="openai/gpt-4.1") as call:
            response = await client.chat.completions.create(...)
            call.record_openai(response)

See docs/tracing-integration.md for the full guide, including the
`call_origin`/`call_role` attribution model and what AioniQ's UI does with
each attribute.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from . import _context, _provider
from ._provider import init

__all__ = [
    "init",
    "set_turn_context",
    "set_call_origin",
    "step",
    "trace_llm_call",
    "trace_embedding",
    "LLMCall",
]

_CALL_ORIGIN_VALUES = frozenset({"user", "scheduled"})
_CALL_ROLE_VALUES = frozenset({"primary", "auxiliary"})


def set_turn_context(
    turn_id: str | uuid.UUID | None = None,
    session_id: str | uuid.UUID | None = None,
) -> None:
    """Sets the turn/session identifiers every span created from here on
    (in this asyncio task) will carry, until changed again. Pass None for
    either argument to leave that one untouched, not cleared -- call
    `set_turn_context(turn_id=None, session_id=None)` (the default) is a
    no-op, not a reset."""
    if turn_id is not None:
        _context.turn_id.set(str(turn_id))
    if session_id is not None:
        _context.session_id.set(str(session_id))


def set_call_origin(origin: str) -> None:
    """`"user"` (a live request triggered this) or `"scheduled"` (a
    background job ran with nobody asking). Raises `ValueError` on
    anything else immediately -- fail loud here rather than silently
    losing the attribute on AioniQ's ingest side (which fails soft on an
    unrecognized value)."""
    if origin not in _CALL_ORIGIN_VALUES:
        raise ValueError(
            f"call_origin must be one of {sorted(_CALL_ORIGIN_VALUES)}, got {origin!r}"
        )
    _context.call_origin.set(origin)


def _validate_call_role(role: str | None) -> None:
    if role is not None and role not in _CALL_ROLE_VALUES:
        raise ValueError(
            f"call_role must be one of {sorted(_CALL_ROLE_VALUES)} or None, got {role!r}"
        )


@asynccontextmanager
async def step(
    name: str, *, call_role: str | None = None, purpose: str | None = None
) -> AsyncIterator[None]:
    """A named step-level span (e.g. "memory_recall", "orchestrator") --
    every span created inside this block, including nested steps and
    `trace_llm_call`/`trace_embedding` calls, becomes a real child of it.

    `call_role`: `"primary"` (the actual user-facing response) |
    `"auxiliary"` (everything else running alongside it) | `None` (not
    classified). Cascades onto every span created inside this block; a
    nested `trace_llm_call` doesn't need to repeat it.

    `purpose`: a short, human-readable sentence describing WHY this step
    is running (e.g. "Searching earlier turns in this conversation") --
    AioniQ's Span Graph shows this instead of a generic label when set.

    Safe under cancellation: uses only `tracer.start_as_current_span()`
    (core OTel, whose own `use_span()` guards its attach/detach with a
    real `try/finally`, unlike some third-party auto-instrumentation
    libraries that guard only `except Exception` -- `asyncio.
    CancelledError` is a `BaseException`, not an `Exception`, in Python
    3.8+, and skips that). This package never calls
    `opentelemetry.context.attach`/`detach` directly.
    """
    _validate_call_role(call_role)
    tracer = _provider.get_tracer()
    role_token = _context.call_role.set(call_role) if call_role else None
    purpose_token = _context.purpose.set(purpose) if purpose else None
    try:
        with tracer.start_as_current_span(name) as span:
            if call_role:
                span.set_attribute("aioniq.call_role", call_role)
            if purpose:
                span.set_attribute("aioniq.purpose", purpose)
            yield
    finally:
        if purpose_token is not None:
            _context.purpose.reset(purpose_token)
        if role_token is not None:
            _context.call_role.reset(role_token)


class LLMCall:
    """Yielded by `trace_llm_call()`/`trace_embedding()` -- call one of the
    `record_*` methods after the underlying request completes to attach
    its model/token/cost data to the span."""

    def __init__(self, span: Any) -> None:
        self._span = span

    def record_openai(self, response: Any) -> None:
        """Reads a plain OpenAI-SDK-shaped response: `response.model`,
        `response.usage.prompt_tokens`/`completion_tokens` (chat) or just
        `prompt_tokens` (embeddings, no completion side)."""
        model = getattr(response, "model", None)
        if model:
            self._span.set_attribute("gen_ai.response.model", model)
        usage = getattr(response, "usage", None)
        if usage is not None:
            input_tokens = getattr(usage, "prompt_tokens", None)
            output_tokens = getattr(usage, "completion_tokens", None)
            if input_tokens is not None:
                self._span.set_attribute("gen_ai.usage.input_tokens", input_tokens)
            if output_tokens is not None:
                self._span.set_attribute("gen_ai.usage.output_tokens", output_tokens)

    def record_langchain(self, message: Any) -> None:
        """Reads a LangChain `AIMessage`'s `usage_metadata` (`input_tokens`/
        `output_tokens`) and `response_metadata` (`model_name`) -- the two
        fields `langchain_openai` already populates from the underlying
        OpenAI response. Mirrors the `getattr(resp, "usage_metadata", None)
        or {}` pattern already used at several call sites integrating this
        SDK for the first time."""
        response_metadata = getattr(message, "response_metadata", None) or {}
        model = response_metadata.get("model_name") or response_metadata.get("model")
        if model:
            self._span.set_attribute("gen_ai.response.model", model)
        usage = getattr(message, "usage_metadata", None) or {}
        input_tokens = usage.get("input_tokens")
        output_tokens = usage.get("output_tokens")
        if input_tokens is not None:
            self._span.set_attribute("gen_ai.usage.input_tokens", input_tokens)
        if output_tokens is not None:
            self._span.set_attribute("gen_ai.usage.output_tokens", output_tokens)

    def record_cost(self, cost_usd: float) -> None:
        self._span.set_attribute("gen_ai.usage.cost", cost_usd)

    def set_attribute(self, key: str, value: Any) -> None:
        """Escape hatch for anything not covered by the `record_*` helpers."""
        self._span.set_attribute(key, value)


@asynccontextmanager
async def trace_llm_call(model: str, operation: str = "chat") -> AsyncIterator[LLMCall]:
    """Wraps one LLM call. Usage:

    async with aioniq_tracing.trace_llm_call(model="openai/gpt-4.1") as call:
        response = await client.chat.completions.create(...)
        call.record_openai(response)
    """
    tracer = _provider.get_tracer()
    with tracer.start_as_current_span(f"{operation} {model}") as span:
        span.set_attribute("gen_ai.operation.name", operation)
        span.set_attribute("gen_ai.request.model", model)
        yield LLMCall(span)


@asynccontextmanager
async def trace_embedding(model: str) -> AsyncIterator[LLMCall]:
    """Wraps one embedding call -- same shape as `trace_llm_call`, fixed
    to `operation="embeddings"`."""
    tracer = _provider.get_tracer()
    with tracer.start_as_current_span(f"embeddings {model}") as span:
        span.set_attribute("gen_ai.operation.name", "embeddings")
        span.set_attribute("gen_ai.request.model", model)
        yield LLMCall(span)

"""Per-call context propagated onto every span this package creates.

Plain `contextvars.ContextVar`s, not span-attribute-copying (the pattern
Vera's own pre-SDK `ConversationPropagator` used, which only worked because
`chat/streamer.py` happened to tag one specific parent span explicitly).
ContextVars work regardless of whether the integrating app's web framework
has any auto-instrumented root span at all, and regardless of which
asyncio Task the reading code runs in (each Task gets its own copy at
creation time, same guarantee `asyncio.gather()`-based concurrency already
relies on elsewhere in this ecosystem).
"""

from __future__ import annotations

from contextvars import ContextVar

turn_id: ContextVar[str | None] = ContextVar("aioniq_turn_id", default=None)
session_id: ContextVar[str | None] = ContextVar("aioniq_session_id", default=None)
# Conservative default: a call not explicitly marked as user-triggered is
# assumed automated (a scheduled/background job), never the reverse.
call_origin: ContextVar[str] = ContextVar("aioniq_call_origin", default="scheduled")
# Set by step() for the duration of its own block; None outside any step.
call_role: ContextVar[str | None] = ContextVar("aioniq_call_role", default=None)
purpose: ContextVar[str | None] = ContextVar("aioniq_purpose", default=None)

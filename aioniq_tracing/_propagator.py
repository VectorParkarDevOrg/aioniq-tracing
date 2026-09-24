"""SpanProcessor that stamps every new span with the current turn/session/
call_origin/call_role context -- reads straight from the ContextVars in
`_context.py`, not from a parent span's own attributes. This is what makes
a `trace_llm_call()`/`trace_embedding()` span nested inside a `step()`
block automatically inherit that step's `call_role` (and any turn/session/
call_origin set earlier) without the caller repeating them.
"""

from __future__ import annotations

from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor

from . import _context


class ContextAttributePropagator(SpanProcessor):
    def on_start(self, span: Span, parent_context=None) -> None:  # type: ignore[override]
        turn_id = _context.turn_id.get()
        if turn_id:
            span.set_attribute("aioniq.turn_id", turn_id)
        session_id = _context.session_id.get()
        if session_id:
            span.set_attribute("aioniq.session_id", session_id)
        span.set_attribute("aioniq.call_origin", _context.call_origin.get())
        role = _context.call_role.get()
        if role:
            span.set_attribute("aioniq.call_role", role)
        purpose = _context.purpose.get()
        if purpose:
            span.set_attribute("aioniq.purpose", purpose)

    def on_end(self, span: ReadableSpan) -> None:  # type: ignore[override]
        pass

    def shutdown(self) -> None:  # type: ignore[override]
        return None

    def force_flush(self, timeout_millis: int = 30000) -> bool:  # type: ignore[override]
        return True

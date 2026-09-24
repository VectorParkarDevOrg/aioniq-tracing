"""Test fixtures: a real, isolated TracerProvider per test so spans can be
captured and inspected without needing a live AioniQ instance to export to.
"""

from __future__ import annotations

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from aioniq_tracing import _context, _provider
from aioniq_tracing._propagator import ContextAttributePropagator


@pytest.fixture(autouse=True)
def _reset_context_vars():
    """`_context`'s ContextVars are module-level, so a test that calls
    `set_call_origin`/`set_turn_context` would otherwise leak into
    whichever test happens to run next -- pytest-asyncio does not
    guarantee a fresh `contextvars.Context` per test. Force every test to
    start from (and leave) the documented defaults."""
    yield
    _context.turn_id.set(None)
    _context.session_id.set(None)
    _context.call_origin.set("scheduled")
    _context.call_role.set(None)
    _context.purpose.set(None)


@pytest.fixture
def span_exporter(monkeypatch):
    """Installs a fresh TracerProvider (with the real
    ContextAttributePropagator, exporting to memory instead of AioniQ) for
    the duration of one test, and points `_provider.get_tracer` (which
    `step()`/`trace_llm_call()`/`trace_embedding()` call via module
    reference, not a frozen import-time binding) at it -- bypasses
    `aioniq_tracing.init()`'s real OTLP-exporter setup and its one-shot
    `_initialized` guard, which would otherwise make every test after the
    first a no-op."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    provider.add_span_processor(ContextAttributePropagator())
    monkeypatch.setattr(_provider, "get_tracer", lambda: provider.get_tracer("test"))
    yield exporter

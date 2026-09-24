"""Regression tests for the exact bug that motivated this package: openlit's
own `openai` instrumentation (`instrumentation/openai/async_openai.py`)
does `token = context_api.attach(ctx)` around `await wrapped(...)`, guarded
only by `except Exception` -- but `asyncio.CancelledError` is a
`BaseException`, not an `Exception`, in Python 3.8+. A call cancelled
mid-flight (e.g. by `asyncio.wait_for(..., timeout=...)`) skips openlit's
own `context_api.detach(token)` entirely, leaving the OTel "current
context" corrupted for the rest of the task -- confirmed live as repeated
"Failed to detach context: Token was created in a different Context" log
lines, cascading into every later span in that task.

This package never calls `opentelemetry.context.attach`/`detach` directly
-- only `tracer.start_as_current_span()`, whose own `use_span()` helper
(core `opentelemetry-sdk`, not this package) guards its attach/detach with
a real `try/finally`, so `detach()` always runs regardless of exception
type. These tests prove that guarantee holds through `step()`/
`trace_llm_call()`/`trace_embedding()` specifically, using the same
cancellation shape found live.
"""

import asyncio

import pytest
from opentelemetry import context as otel_context

import aioniq_tracing


async def _hangs_forever():
    await asyncio.sleep(3600)


async def test_trace_llm_call_leaves_context_clean_after_a_timeout_cancellation(span_exporter):
    before = otel_context.get_current()
    async with aioniq_tracing.trace_llm_call(model="openai/gpt-4.1"):
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(_hangs_forever(), timeout=0.05)
    after = otel_context.get_current()
    assert after is before


async def test_step_leaves_context_clean_after_a_timeout_cancellation_inside_it(span_exporter):
    # Matches agent/pipeline.py's real shape: `with traced_step(name):
    # await asyncio.wait_for(_active_llm.ainvoke(messages), timeout=90)`.
    before = otel_context.get_current()
    with pytest.raises(asyncio.TimeoutError):
        async with aioniq_tracing.step("agent_reasoning", call_role="auxiliary"):
            await asyncio.wait_for(_hangs_forever(), timeout=0.05)
    after = otel_context.get_current()
    assert after is before
    # And the call_role ContextVar itself must not have leaked either --
    # a step started fresh afterward must not inherit "auxiliary".
    async with aioniq_tracing.step("title_gen"):
        pass
    span = span_exporter.get_finished_spans()[-1]
    assert "aioniq.call_role" not in span.attributes


async def test_retry_loop_after_a_cancelled_attempt_still_traces_correctly(span_exporter):
    # The real production pattern: 3 retries, first one times out, second
    # one succeeds -- both attempts must still produce correctly-formed,
    # uncorrupted spans.
    for attempt in range(2):
        try:
            async with aioniq_tracing.step("agent_reasoning", call_role="auxiliary"):
                if attempt == 0:
                    await asyncio.wait_for(_hangs_forever(), timeout=0.05)
                else:
                    async with aioniq_tracing.trace_llm_call(model="openai/gpt-4.1"):
                        pass
            break
        except asyncio.TimeoutError:
            continue
    names = [s.name for s in span_exporter.get_finished_spans()]
    assert names.count("agent_reasoning") == 2
    assert "chat openai/gpt-4.1" in names


async def test_nested_trace_llm_call_cancellation_does_not_corrupt_the_outer_step(span_exporter):
    before = otel_context.get_current()
    with pytest.raises(asyncio.TimeoutError):
        async with (
            aioniq_tracing.step("orchestrator", call_role="primary"),
            aioniq_tracing.trace_llm_call(model="openai/gpt-4.1"),
        ):
            await asyncio.wait_for(_hangs_forever(), timeout=0.05)
    after = otel_context.get_current()
    assert after is before
    spans = {s.name: s for s in span_exporter.get_finished_spans()}
    assert spans["orchestrator"].attributes["aioniq.call_role"] == "primary"

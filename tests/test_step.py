import pytest

import aioniq_tracing


async def test_step_creates_a_real_span_with_the_given_name(span_exporter):
    async with aioniq_tracing.step("memory_recall"):
        pass
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "memory_recall"


async def test_step_sets_call_role_and_purpose_attributes(span_exporter):
    async with aioniq_tracing.step(
        "orchestrator", call_role="primary", purpose="Generating your response"
    ):
        pass
    span = span_exporter.get_finished_spans()[0]
    assert span.attributes["aioniq.call_role"] == "primary"
    assert span.attributes["aioniq.purpose"] == "Generating your response"


async def test_step_omits_call_role_and_purpose_when_not_given(span_exporter):
    async with aioniq_tracing.step("scope_gate"):
        pass
    span = span_exporter.get_finished_spans()[0]
    assert "aioniq.call_role" not in span.attributes
    assert "aioniq.purpose" not in span.attributes


async def test_step_rejects_an_invalid_call_role():
    with pytest.raises(ValueError, match="call_role"):
        async with aioniq_tracing.step("x", call_role="not-a-real-role"):
            pass


async def test_nested_step_is_a_real_child_span(span_exporter):
    async with aioniq_tracing.step("outer"), aioniq_tracing.step("inner"):
        pass
    spans = {s.name: s for s in span_exporter.get_finished_spans()}
    assert spans["inner"].parent.span_id == spans["outer"].context.span_id


async def test_nested_step_call_role_does_not_leak_to_a_sibling_step(span_exporter):
    # A nested "auxiliary" step inside a "primary" one must not make a
    # LATER sibling step (back at the outer level) inherit "auxiliary".
    async with aioniq_tracing.step("outer", call_role="primary"):
        async with aioniq_tracing.step("inner", call_role="auxiliary"):
            pass
        async with aioniq_tracing.trace_llm_call(model="gpt-4.1"):
            pass
    spans = {s.name: s for s in span_exporter.get_finished_spans()}
    assert spans["chat gpt-4.1"].attributes["aioniq.call_role"] == "primary"


async def test_nested_step_purpose_does_not_leak_to_a_sibling_step(span_exporter):
    async with aioniq_tracing.step("outer", purpose="Generating your response"):
        async with aioniq_tracing.step(
            "inner", purpose="Searching earlier turns in this conversation"
        ):
            pass
        async with aioniq_tracing.trace_llm_call(model="gpt-4.1"):
            pass
    spans = {s.name: s for s in span_exporter.get_finished_spans()}
    assert spans["chat gpt-4.1"].attributes["aioniq.purpose"] == "Generating your response"


async def test_step_still_cleans_up_call_role_when_the_body_raises(span_exporter):
    with pytest.raises(RuntimeError):
        async with aioniq_tracing.step("orchestrator", call_role="primary"):
            raise RuntimeError("boom")
    # A step started afterward, with no call_role of its own, must not see
    # the crashed step's "primary" leak through.
    async with aioniq_tracing.step("title_gen"):
        pass
    span = span_exporter.get_finished_spans()[-1]
    assert "aioniq.call_role" not in span.attributes

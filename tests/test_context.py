import uuid

import pytest

import aioniq_tracing


async def test_call_origin_defaults_to_scheduled(span_exporter):
    async with aioniq_tracing.step("incident_monitor"):
        pass
    span = span_exporter.get_finished_spans()[0]
    assert span.attributes["aioniq.call_origin"] == "scheduled"


async def test_set_call_origin_user_is_reflected_on_every_subsequent_span(span_exporter):
    aioniq_tracing.set_call_origin("user")
    async with (
        aioniq_tracing.step("orchestrator"),
        aioniq_tracing.trace_llm_call(model="openai/gpt-4.1"),
    ):
        pass
    for span in span_exporter.get_finished_spans():
        assert span.attributes["aioniq.call_origin"] == "user"


def test_set_call_origin_rejects_an_invalid_value():
    with pytest.raises(ValueError, match="call_origin"):
        aioniq_tracing.set_call_origin("not-a-real-origin")


async def test_set_turn_context_stamps_turn_and_session_id_on_every_span(span_exporter):
    turn = uuid.uuid4()
    session = uuid.uuid4()
    aioniq_tracing.set_turn_context(turn_id=turn, session_id=session)
    async with aioniq_tracing.step("memory_recall"):
        pass
    span = span_exporter.get_finished_spans()[0]
    assert span.attributes["aioniq.turn_id"] == str(turn)
    assert span.attributes["aioniq.session_id"] == str(session)


async def test_span_has_no_turn_or_session_attribute_when_never_set(span_exporter):
    async with aioniq_tracing.step("memory_recall"):
        pass
    span = span_exporter.get_finished_spans()[0]
    assert "aioniq.turn_id" not in span.attributes
    assert "aioniq.session_id" not in span.attributes

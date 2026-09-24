from types import SimpleNamespace

import aioniq_tracing


async def test_trace_llm_call_names_the_span_operation_and_model(span_exporter):
    async with aioniq_tracing.trace_llm_call(model="openai/gpt-4.1", operation="chat"):
        pass
    span = span_exporter.get_finished_spans()[0]
    assert span.name == "chat openai/gpt-4.1"
    assert span.attributes["gen_ai.operation.name"] == "chat"
    assert span.attributes["gen_ai.request.model"] == "openai/gpt-4.1"


async def test_trace_embedding_names_the_span_and_fixes_operation(span_exporter):
    async with aioniq_tracing.trace_embedding(model="text-embedding-3-small"):
        pass
    span = span_exporter.get_finished_spans()[0]
    assert span.name == "embeddings text-embedding-3-small"
    assert span.attributes["gen_ai.operation.name"] == "embeddings"


async def test_record_openai_reads_model_and_token_usage(span_exporter):
    response = SimpleNamespace(
        model="openai/gpt-4.1",
        usage=SimpleNamespace(prompt_tokens=120, completion_tokens=45),
    )
    async with aioniq_tracing.trace_llm_call(model="openai/gpt-4.1") as call:
        call.record_openai(response)
    span = span_exporter.get_finished_spans()[0]
    assert span.attributes["gen_ai.response.model"] == "openai/gpt-4.1"
    assert span.attributes["gen_ai.usage.input_tokens"] == 120
    assert span.attributes["gen_ai.usage.output_tokens"] == 45


async def test_record_openai_embeddings_response_has_no_completion_side(span_exporter):
    # Embeddings responses carry prompt_tokens but no completion_tokens.
    response = SimpleNamespace(
        model="text-embedding-3-small", usage=SimpleNamespace(prompt_tokens=8)
    )
    async with aioniq_tracing.trace_embedding(model="text-embedding-3-small") as call:
        call.record_openai(response)
    span = span_exporter.get_finished_spans()[0]
    assert span.attributes["gen_ai.usage.input_tokens"] == 8
    assert "gen_ai.usage.output_tokens" not in span.attributes


async def test_record_langchain_reads_usage_metadata_and_response_metadata(span_exporter):
    message = SimpleNamespace(
        response_metadata={"model_name": "openai/gpt-4.1"},
        usage_metadata={"input_tokens": 200, "output_tokens": 60, "total_tokens": 260},
    )
    async with aioniq_tracing.trace_llm_call(model="openai/gpt-4.1") as call:
        call.record_langchain(message)
    span = span_exporter.get_finished_spans()[0]
    assert span.attributes["gen_ai.response.model"] == "openai/gpt-4.1"
    assert span.attributes["gen_ai.usage.input_tokens"] == 200
    assert span.attributes["gen_ai.usage.output_tokens"] == 60


async def test_record_langchain_tolerates_a_message_with_no_usage_metadata(span_exporter):
    # e.g. history_summary/rca_conclude call sites that don't read usage
    # today -- must not crash the wrapper.
    message = SimpleNamespace()
    async with aioniq_tracing.trace_llm_call(model="openai/gpt-4.1") as call:
        call.record_langchain(message)
    span = span_exporter.get_finished_spans()[0]
    assert "gen_ai.usage.input_tokens" not in span.attributes


async def test_record_cost_sets_the_cost_attribute(span_exporter):
    async with aioniq_tracing.trace_llm_call(model="openai/gpt-4.1") as call:
        call.record_cost(0.0042)
    span = span_exporter.get_finished_spans()[0]
    assert span.attributes["gen_ai.usage.cost"] == 0.0042


async def test_trace_llm_call_inherits_the_enclosing_steps_call_role_and_purpose(span_exporter):
    async with (
        aioniq_tracing.step(
            "orchestrator", call_role="primary", purpose="Generating your response"
        ),
        aioniq_tracing.trace_llm_call(model="openai/gpt-4.1"),
    ):
        pass
    spans = {s.name: s for s in span_exporter.get_finished_spans()}
    llm_span = spans["chat openai/gpt-4.1"]
    assert llm_span.attributes["aioniq.call_role"] == "primary"
    assert llm_span.attributes["aioniq.purpose"] == "Generating your response"

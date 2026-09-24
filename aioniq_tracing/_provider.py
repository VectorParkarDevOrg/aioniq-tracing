"""OTel SDK wiring: a TracerProvider with a batched OTLP/HTTP exporter
pointed at AioniQ's trace-ingestion endpoint, plus the context-attribute
propagator. No monkey-patching of any third-party client anywhere in this
package -- callers create spans explicitly via `step()`/`trace_llm_call()`/
`trace_embedding()` in `__init__.py`.
"""

from __future__ import annotations

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from ._propagator import ContextAttributePropagator

_TRACER_NAME = "aioniq_tracing"
_initialized = False


def init(base_url: str, agent_key: str, application_name: str = "aioniq-agent") -> None:
    """Call once at process startup. Idempotent -- a second call is a
    no-op, so it's harmless if an integrator calls this from both a
    module-level import and a framework startup hook by mistake.

    `base_url`: AioniQ's own base URL (e.g. the same host used for the
    LLM/MCP gateway), NOT including `/v1/traces` -- that suffix is added
    here to match `docs/tracing-integration.md`'s documented endpoint.
    `agent_key`: the same agent API key (`ak_live_...`) used for LLM/MCP
    gateway calls -- one credential for both, per AioniQ's own auth
    contract (`app/tracing/ingest_router.py`'s `get_current_agent`).
    """
    global _initialized
    if _initialized:
        return
    endpoint = base_url.rstrip("/") + "/v1/traces"
    resource = Resource.create({SERVICE_NAME: application_name})
    provider = TracerProvider(resource=resource)
    exporter = OTLPSpanExporter(endpoint=endpoint, headers={"Authorization": f"Bearer {agent_key}"})
    provider.add_span_processor(BatchSpanProcessor(exporter))
    provider.add_span_processor(ContextAttributePropagator())
    trace.set_tracer_provider(provider)
    _initialized = True


def get_tracer() -> trace.Tracer:
    return trace.get_tracer(_TRACER_NAME)

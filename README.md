# aioniq-tracing

First-party OpenTelemetry tracing SDK for agents integrating with
AioniQ. Full guide: <https://aioniqdocs.parkarlabs.in/#/1.0/tracing-sdk>.

AioniQ's gateways already trace every governed LLM, MCP and SSH call on their
own. This SDK is for the parts of your agent the gateways can't see — which
subagent ran, which internal step it was on — and correlates them into the
same Turn Detail view.

Deliberately does **not** monkey-patch the `openai`/`langchain`/any other SDK
(unlike e.g. OpenLIT). Every span is created explicitly by the caller via
`step()`, `trace_llm_call()`, or `trace_embedding()` — no hidden global
instrumentation, no auto-instrumentation bugs to inherit.

It was written to replace OpenLIT in one of our own agents specifically
because OpenLIT's own `openai` instrumentation had a real,
confirmed bug: its streaming-call wrapper did `token = context_api.attach(ctx)`
around `await wrapped(...)`, guarded only by `except Exception` — but
`asyncio.CancelledError` is a `BaseException`, not an `Exception`, since
Python 3.8+, so a call cancelled mid-flight (e.g. `asyncio.wait_for(...,
timeout=...)` firing) skipped OpenLIT's own detach entirely, corrupting the
OTel "current context" for every later span in that task. This package only
ever uses `tracer.start_as_current_span()` (core `opentelemetry-sdk`, whose
own `use_span()` guards attach/detach with a real `try/finally`), so that
class of bug can't happen here — see `tests/test_cancellation_safety.py`.

## Install

```bash
pip install aioniq-tracing
```

[![PyPI](https://img.shields.io/pypi/v/aioniq-tracing)](https://pypi.org/project/aioniq-tracing/)

Or from a tagged release on GitHub:

```bash
pip install "aioniq-tracing @ git+https://github.com/VectorParkarDevOrg/aioniq-tracing.git@v0.1.0"
```

Requires Python 3.10+.

## Quickstart

```python
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
```

`agent_key` is the same agent API key your agent already uses for the LLM and
MCP gateways. See the [AioniQ documentation](https://aioniqdocs.parkarlabs.in/#/1.0/tracing-sdk)
for the full guide, including the `call_origin`/`call_role` attribution model
and what AioniQ's UI does with each attribute.

## Development

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"

.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/mypy aioniq_tracing --ignore-missing-imports --no-strict-optional
.venv/bin/pytest tests/ -v
```

## Releasing

Bump `version` in `pyproject.toml`, then push a matching tag:

```bash
git tag v0.1.1 && git push origin v0.1.1
```

The Release workflow builds the wheel and source package, attaches them to a
GitHub Release, and publishes to PyPI via trusted publishing (no token
stored; gated by the `PYPI_PUBLISH` repository variable).

## License

Apache License 2.0 — see [LICENSE](LICENSE).

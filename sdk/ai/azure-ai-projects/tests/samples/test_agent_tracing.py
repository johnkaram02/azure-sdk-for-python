# ------------------------------------
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# ------------------------------------

import ast
import asyncio
import importlib.util
import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from azure.core.exceptions import ClientAuthenticationError, ResourceNotFoundError, ServiceRequestError
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PACKAGE_ROOT))

from samples import util

PRIVACY_SETTINGS = [
    "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT",
    "AZURE_TRACING_GEN_AI_INCLUDE_BINARY_DATA",
    "AZURE_TRACING_GEN_AI_TRACE_CONTEXT_PROPAGATION_INCLUDE_BAGGAGE",
]


@pytest.fixture
def tracing(monkeypatch):
    import azure.ai.projects.telemetry
    import azure.monitor.opentelemetry

    provider = TracerProvider(shutdown_on_exit=False)
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    state = SimpleNamespace(active=False)
    instrumentor = MagicMock()
    instrumentor.is_instrumented.side_effect = lambda: state.active
    instrumentor.instrument.side_effect = lambda: setattr(state, "active", True)
    instrumentor.uninstrument.side_effect = lambda: setattr(state, "active", False)
    configure = MagicMock(side_effect=lambda **_: monkeypatch.setattr(trace, "_TRACER_PROVIDER", provider))
    monkeypatch.setattr(azure.monitor.opentelemetry, "configure_azure_monitor", configure)
    monkeypatch.setattr(azure.ai.projects.telemetry, "AIProjectInstrumentor", lambda: instrumentor)
    monkeypatch.setattr(trace, "_TRACER_PROVIDER", None)
    monkeypatch.setattr(util, "_tracing_state", util._SampleTracingState())
    monkeypatch.delenv(util._TRACING_GATE, raising=False)
    for setting in PRIVACY_SETTINGS:
        monkeypatch.delenv(setting, raising=False)
    client = SimpleNamespace(
        telemetry=SimpleNamespace(
            get_application_insights_connection_string=MagicMock(return_value="test-connection-not-a-secret")
        )
    )
    yield SimpleNamespace(
        provider=provider,
        exporter=exporter,
        instrumentor=instrumentor,
        configure=configure,
        client=client,
        state=state,
        monkeypatch=monkeypatch,
    )
    provider.shutdown()


def test_automatic_setup_explicitly_instruments_and_ends_scenario_before_flush(tracing):
    flush = MagicMock(wraps=tracing.provider.force_flush)
    tracing.monkeypatch.setattr(tracing.provider, "force_flush", flush)
    with util.agent_tracing(tracing.client, scenario="original-conversation"):
        assert tracing.state.active
        assert trace.get_current_span().is_recording()
        flush.assert_not_called()
    assert [span.name for span in tracing.exporter.get_finished_spans()] == ["original-conversation"]
    tracing.configure.assert_called_once_with(
        connection_string="test-connection-not-a-secret", logger_name="azure.ai.projects.samples"
    )
    tracing.instrumentor.instrument.assert_called_once_with()
    tracing.instrumentor.uninstrument.assert_called_once_with()
    flush.assert_called_once_with(timeout_millis=5_000)
    assert not tracing.state.active


def test_setup_does_not_opt_in_to_content_binary_or_baggage_capture(tracing):
    with util.agent_tracing(tracing.client, scenario="privacy-defaults"):
        assert all(setting not in util.os.environ for setting in PRIVACY_SETTINGS)
    assert all(setting not in util.os.environ for setting in PRIVACY_SETTINGS)


@pytest.mark.parametrize("setting", PRIVACY_SETTINGS)
@pytest.mark.parametrize("value", ["false", "true"])
def test_explicit_privacy_settings_are_preserved(tracing, setting, value):
    tracing.monkeypatch.setenv(setting, value)
    with util.agent_tracing(tracing.client, scenario="explicit-privacy"):
        assert util.os.environ[setting] == value
    assert util.os.environ[setting] == value


def test_inactive_instrumentation_warns_and_can_retry_without_duplicate_configuration(tracing, caplog):
    tracing.instrumentor.instrument.side_effect = None
    with util.agent_tracing(tracing.client, scenario="inactive"):
        assert util._tracing_state.users == 0
    assert "GenAI instrumentation did not activate" in caplog.text
    assert not tracing.exporter.get_finished_spans()
    tracing.instrumentor.instrument.side_effect = lambda: setattr(tracing.state, "active", True)
    with util.agent_tracing(tracing.client, scenario="retry"):
        assert tracing.state.active
    tracing.configure.assert_called_once()
    assert [span.name for span in tracing.exporter.get_finished_spans()] == ["retry"]
    assert util._tracing_state.users == 0


@pytest.mark.parametrize("setting", ["false", "FALSE", "", "0"])
def test_explicit_opt_out_avoids_connection_lookup_and_optional_setup(tracing, setting):
    tracing.monkeypatch.setenv(util._TRACING_GATE, setting)
    with util.agent_tracing(tracing.client, scenario="disabled"):
        pass
    tracing.client.telemetry.get_application_insights_connection_string.assert_not_called()
    tracing.configure.assert_not_called()
    tracing.instrumentor.instrument.assert_not_called()


@pytest.mark.parametrize(
    "error",
    [
        ResourceNotFoundError("missing"),
        ClientAuthenticationError("denied"),
        ServiceRequestError("offline"),
        ValueError("invalid connection"),
    ],
)
def test_expected_connection_failure_warns_and_preserves_conversation(tracing, caplog, error):
    tracing.client.telemetry.get_application_insights_connection_string.side_effect = error
    with caplog.at_level(logging.WARNING), util.agent_tracing(tracing.client, scenario="unavailable"):
        result = "original answer"
    assert result == "original answer"
    assert type(error).__name__ in caplog.text
    assert "ingestion has not been verified" in caplog.text
    tracing.configure.assert_not_called()


def test_unexpected_discovery_error_is_not_swallowed(tracing):
    tracing.client.telemetry.get_application_insights_connection_string.side_effect = AssertionError("bug")
    with pytest.raises(AssertionError, match="bug"), util.agent_tracing(tracing.client, scenario="unexpected"):
        pytest.fail("A programming error must not enter the scenario.")


def test_missing_optional_package_warns_without_blocking_conversation(tracing, caplog):
    tracing.configure.side_effect = ModuleNotFoundError("optional package")
    with util.agent_tracing(tracing.client, scenario="no-package"):
        pass
    assert "install azure-monitor-opentelemetry" in caplog.text


@pytest.mark.parametrize("error", [ValueError("test-connection-not-a-secret"), RuntimeError("configuration")])
def test_exporter_configuration_failure_is_explicit_and_redacted(tracing, caplog, error):
    tracing.configure.side_effect = error
    with util.agent_tracing(tracing.client, scenario="configuration-failure"):
        pass
    assert type(error).__name__ in caplog.text
    assert "test-connection-not-a-secret" not in caplog.text


def test_repeated_contexts_share_provider_without_shutting_it_down(tracing):
    shutdown = MagicMock()
    tracing.monkeypatch.setattr(tracing.provider, "shutdown", shutdown)
    for name in ("first", "second"):
        with util.agent_tracing(tracing.client, scenario=name):
            pass
    tracing.configure.assert_called_once()
    shutdown.assert_not_called()
    assert [span.name for span in tracing.exporter.get_finished_spans()] == ["first", "second"]
    assert util._tracing_state.users == 0


def test_nested_contexts_preserve_outer_instrumentation(tracing):
    with util.agent_tracing(tracing.client, scenario="outer"):
        with util.agent_tracing(tracing.client, scenario="inner"):
            assert util._tracing_state.users == 2
        assert tracing.state.active
    tracing.configure.assert_called_once()
    tracing.instrumentor.uninstrument.assert_called_once()
    spans = tracing.exporter.get_finished_spans()
    assert spans[0].parent.span_id == spans[1].context.span_id


def test_preexisting_provider_is_left_unchanged(tracing, caplog):
    tracing.monkeypatch.setattr(trace, "_TRACER_PROVIDER", tracing.provider)
    with util.agent_tracing(tracing.client, scenario="caller-managed"):
        pass
    tracing.configure.assert_not_called()
    tracing.instrumentor.instrument.assert_not_called()
    assert "caller-managed" in caplog.text


def test_preexisting_instrumentation_is_not_reconfigured_or_removed(tracing):
    tracing.state.active = True
    with util.agent_tracing(tracing.client, scenario="caller-instrumented"):
        pass
    tracing.instrumentor.instrument.assert_not_called()
    tracing.instrumentor.uninstrument.assert_not_called()


def test_different_destination_does_not_duplicate_exporters(tracing, caplog):
    with util.agent_tracing(tracing.client, scenario="first-destination"):
        pass
    tracing.client.telemetry.get_application_insights_connection_string.return_value = "different"
    with util.agent_tracing(tracing.client, scenario="different-destination"):
        pass
    tracing.configure.assert_called_once()
    assert "different telemetry provider or destination" in caplog.text


def test_conversation_failure_is_preserved_and_owned_telemetry_flushed(tracing):
    flush = MagicMock(wraps=tracing.provider.force_flush)
    tracing.monkeypatch.setattr(tracing.provider, "force_flush", flush)
    with pytest.raises(RuntimeError, match="agent failed"):
        with util.agent_tracing(tracing.client, scenario="failed-conversation"):
            raise RuntimeError("agent failed")
    flush.assert_called_once()
    tracing.instrumentor.uninstrument.assert_called_once()
    assert tracing.exporter.get_finished_spans()[0].status.status_code == trace.StatusCode.ERROR


def test_scenario_error_does_not_export_exception_content(tracing):
    with pytest.raises(RuntimeError, match="PRIVATE-CONVERSATION-CONTENT"):
        with util.agent_tracing(tracing.client, scenario="private-error"):
            raise RuntimeError("PRIVATE-CONVERSATION-CONTENT")
    span = tracing.exporter.get_finished_spans()[0]
    assert span.status.status_code == trace.StatusCode.ERROR
    assert "PRIVATE-CONVERSATION-CONTENT" not in repr((span.attributes, span.events, span.status.description))


@pytest.mark.parametrize("flush_result", [False, OSError("flush failed")])
def test_flush_failure_warns_without_masking_original_error(tracing, caplog, flush_result):
    flush = MagicMock(return_value=flush_result) if flush_result is False else MagicMock(side_effect=flush_result)
    tracing.monkeypatch.setattr(tracing.provider, "force_flush", flush)
    with pytest.raises(RuntimeError, match="original failure"):
        with util.agent_tracing(tracing.client, scenario="flush-failure"):
            raise RuntimeError("original failure")
    assert "flush" in caplog.text


def test_cleanup_failure_does_not_mask_conversation_error(tracing, caplog):
    tracing.instrumentor.uninstrument.side_effect = RuntimeError("cleanup")
    with pytest.raises(ValueError, match="conversation failure"):
        with util.agent_tracing(tracing.client, scenario="cleanup-failure"):
            raise ValueError("conversation failure")
    assert "instrumentation cleanup" in caplog.text


def test_async_uses_same_configuration_and_lifecycle(tracing):
    getter = AsyncMock(return_value="test-connection-not-a-secret")
    tracing.client.telemetry.get_application_insights_connection_string = getter

    async def conversation():
        async with util.agent_tracing_async(tracing.client, scenario="async-conversation"):
            assert tracing.state.active
            await asyncio.sleep(0)

    asyncio.run(conversation())
    getter.assert_awaited_once()
    tracing.configure.assert_called_once()
    tracing.instrumentor.uninstrument.assert_called_once()
    assert tracing.exporter.get_finished_spans()[0].name == "async-conversation"


def test_async_opt_out_skips_discovery(tracing):
    getter = AsyncMock()
    tracing.client.telemetry.get_application_insights_connection_string = getter
    tracing.monkeypatch.setenv(util._TRACING_GATE, "false")

    async def conversation():
        async with util.agent_tracing_async(tracing.client, scenario="disabled-async"):
            pass

    asyncio.run(conversation())
    getter.assert_not_awaited()
    tracing.configure.assert_not_called()


def test_async_missing_connection_preserves_scenario(tracing, caplog):
    tracing.client.telemetry.get_application_insights_connection_string = AsyncMock(
        side_effect=ResourceNotFoundError("no connection")
    )

    async def conversation():
        async with util.agent_tracing_async(tracing.client, scenario="missing-async"):
            return "original answer"

    assert asyncio.run(conversation()) == "original answer"
    assert "ResourceNotFoundError" in caplog.text


def test_async_cancellation_closes_span_and_flushes(tracing):
    tracing.client.telemetry.get_application_insights_connection_string = AsyncMock(
        return_value="test-connection-not-a-secret"
    )
    flush = MagicMock(wraps=tracing.provider.force_flush)
    tracing.monkeypatch.setattr(tracing.provider, "force_flush", flush)

    async def conversation():
        async with util.agent_tracing_async(tracing.client, scenario="cancelled-async"):
            raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(conversation())
    flush.assert_called_once()
    assert not tracing.state.active
    assert tracing.exporter.get_finished_spans()[0].name == "cancelled-async"


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("connection", [None, ""])
def test_empty_connection_warns_once_and_preserves_scenario(tracing, caplog, asynchronous, connection):
    if asynchronous:
        tracing.client.telemetry.get_application_insights_connection_string = AsyncMock(return_value=connection)

        async def conversation():
            async with util.agent_tracing_async(tracing.client, scenario="empty-connection"):
                return "original answer"

        assert asyncio.run(conversation()) == "original answer"
    else:
        tracing.client.telemetry.get_application_insights_connection_string.return_value = connection
        with util.agent_tracing(tracing.client, scenario="empty-connection"):
            result = "original answer"
        assert result == "original answer"
    assert caplog.text.count("empty Application Insights connection string") == 1
    tracing.configure.assert_not_called()


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    "live,console,agent_sample,configured,expected",
    [
        (False, False, True, "true", "false"),
        (False, True, True, "true", "true"),
        (True, False, True, "true", "true"),
        (True, False, True, "false", "false"),
        (False, False, False, "true", "true"),
    ],
)
def test_sample_executor_playback_disables_only_automatic_cloud_setup(
    tmp_path, monkeypatch, asynchronous, live, console, agent_sample, configured, expected
):
    import sample_executor

    monkeypatch.setattr(sample_executor, "is_live", lambda: live)
    monkeypatch.setenv(util._TRACING_GATE, "inherited")
    filename = "sample_agent_with_console_tracing.py" if console else "sample_agent_basic.py"
    sample_dir = tmp_path / "samples" / ("agents" if agent_sample else "evaluations")
    sample_dir.mkdir(parents=True)
    sample = sample_dir / filename
    sample.write_text(
        "import os\n"
        f"{'async ' if asynchronous else ''}def main():\n"
        f"    global observed\n    observed = os.environ.get({util._TRACING_GATE!r})\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(sample_dir))
    executor_type = sample_executor.AsyncSampleExecutor if asynchronous else sample_executor.SyncSampleExecutor
    executor = executor_type(
        SimpleNamespace(get_credential=MagicMock()), str(sample), env_vars={util._TRACING_GATE: configured}
    )
    if asynchronous:
        asyncio.run(executor.execute_async(patched_open_fn=open))
    else:
        executor.execute(patched_open_fn=open)
    assert executor.module.observed == expected
    assert util.os.environ[util._TRACING_GATE] == "inherited"


@pytest.mark.parametrize("folder", ["", "tools", "telemetry", "voice"])
def test_folder_shims_share_the_same_setup_functions_and_ownership_state(folder):
    path = PACKAGE_ROOT / "samples" / "agents" / folder / "util.py"
    spec = importlib.util.spec_from_file_location("sample_tracing_shim", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.agent_tracing is util.agent_tracing
    assert module.agent_tracing_async is util.agent_tracing_async


def test_every_eligible_entrypoint_reuses_setup_before_client_acquisition():
    agents = PACKAGE_ROOT / "samples" / "agents"
    included = []
    for path in sorted(agents.rglob("sample*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
        names = [ast.unparse(node.func) for node in calls]
        if not any(
            ("responses." in name and name.rsplit(".", 1)[-1] in ("create", "stream"))
            or ("realtime" in name and name.endswith(".connect"))
            for name in names
        ):
            continue
        if path.name == "sample_agent_basic_with_console_tracing.py":
            assert not any(name in ("agent_tracing", "agent_tracing_async") for name in names)
            continue
        included.append(path)
        helper_calls = [node for node in calls if ast.unparse(node.func) in ("agent_tracing", "agent_tracing_async")]
        assert len(helper_calls) == 1, path
        helper_call = helper_calls[0]
        assert "configure_azure_monitor" not in names, path
        assert not any("get_application_insights_connection_string" in name for name in names), path
        client_calls = [node for node in calls if ast.unparse(node.func).endswith(".get_openai_client")]
        assert all(helper_call.lineno < node.lineno for node in client_calls), path
        assert "azure-monitor-opentelemetry" in ast.get_docstring(tree), path
        assert "samples/agents/telemetry/README.md" in ast.get_docstring(tree), path
    assert len(included) == 59

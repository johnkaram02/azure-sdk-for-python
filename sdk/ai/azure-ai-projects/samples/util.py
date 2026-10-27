# ------------------------------------
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# ------------------------------------

import asyncio
import hashlib
import logging
import os
import sys
import tempfile
import zipfile
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING, Any, AsyncIterator, Iterator, Optional, Tuple

from azure.core.exceptions import AzureError

_PKG_ROOT = Path(__file__).resolve().parents[1]
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from azure.ai.projects import AIProjectClient
from azure.ai.projects.aio import AIProjectClient as AsyncAIProjectClient
from azure.ai.projects.models import (
    AgentBlueprintReference,
    AgentDefinition,
    AgentEndpointConfig,
    AgentVersionDetails,
    FixedRatioVersionSelectionRule,
    ProtocolConfiguration,
    ResponsesProtocolConfiguration,
    VersionSelector,
)

if TYPE_CHECKING:
    from azure.ai.projects.telemetry import AIProjectInstrumentor
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.trace import Tracer


@dataclass
class _SampleTracingState:
    provider: Optional["TracerProvider"] = None
    destination: Optional[bytes] = None
    instrumentor: Optional["AIProjectInstrumentor"] = None
    users: int = 0


_TRACING_GATE = "AZURE_EXPERIMENTAL_ENABLE_GENAI_TRACING"
_TRACING_LOGGER = logging.getLogger("azure.ai.projects.samples")
_TRACING_LOCK = RLock()
_tracing_state = _SampleTracingState()


def _tracing_disabled() -> bool:
    setting = os.environ.get(_TRACING_GATE)
    return setting is not None and setting.lower() != "true"


def _tracing_warning(reason: str) -> None:
    _TRACING_LOGGER.warning(
        "Application Insights sample tracing is unavailable (%s). "
        "The agent scenario will continue. Check the project's App Insights connection, "
        "connection-read permissions, and the setup in samples/agents/telemetry/README.md. "
        "Trace ingestion has not been verified.",
        reason,
    )


def _start_sample_tracing(connection_string: Optional[str]) -> Optional["Tracer"]:
    if not connection_string:
        return None

    try:
        from azure.ai.projects.telemetry import AIProjectInstrumentor
        from azure.monitor.opentelemetry import configure_azure_monitor
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.trace import ProxyTracerProvider

        with _TRACING_LOCK:
            provider = trace.get_tracer_provider()
            destination = hashlib.sha256(connection_string.encode("utf-8")).digest()
            if _tracing_state.provider is None:
                if not isinstance(provider, ProxyTracerProvider):
                    _tracing_warning("caller-managed telemetry; its provider and destination were left unchanged")
                    return None
                configure_azure_monitor(
                    connection_string=connection_string,
                    logger_name="azure.ai.projects.samples",
                )
                configured_provider = trace.get_tracer_provider()
                if not isinstance(configured_provider, TracerProvider):
                    _tracing_warning("Azure Monitor did not install a supported tracer provider")
                    return None
                _tracing_state.provider = configured_provider
                _tracing_state.destination = destination
            elif provider is not _tracing_state.provider or destination != _tracing_state.destination:
                _tracing_warning("a different telemetry provider or destination is already configured")
                return None

            os.environ.setdefault(_TRACING_GATE, "true")
            instrumentor = AIProjectInstrumentor()
            if not instrumentor.is_instrumented():
                instrumentor.instrument()
                if instrumentor.is_instrumented():
                    _tracing_state.instrumentor = instrumentor
            if instrumentor.is_instrumented():
                _tracing_state.users += 1
                return trace.get_tracer("azure.ai.projects.samples")
            _tracing_warning("GenAI instrumentation did not activate")
    except ImportError:
        _tracing_warning("install azure-monitor-opentelemetry and azure-core-tracing-opentelemetry")
    except (AzureError, OSError, ValueError, RuntimeError) as error:
        _tracing_warning(type(error).__name__)
    return None


def _finish_sample_tracing() -> None:
    with _TRACING_LOCK:
        _tracing_state.users -= 1
        if _tracing_state.users == 0 and _tracing_state.instrumentor is not None:
            try:
                _tracing_state.instrumentor.uninstrument()
            except (AzureError, OSError, ValueError, RuntimeError) as error:
                _tracing_warning(f"instrumentation cleanup: {type(error).__name__}")
            finally:
                _tracing_state.instrumentor = None
        provider = _tracing_state.provider
    if provider is not None:
        try:
            if not provider.force_flush(timeout_millis=5_000):
                _tracing_warning("telemetry flush timed out")
        except (AzureError, OSError, ValueError, RuntimeError) as error:
            _tracing_warning(f"telemetry flush: {type(error).__name__}")


@contextmanager
def _scenario_span(tracer: Optional["Tracer"], scenario: str) -> Iterator[None]:
    if tracer is None:
        yield
        return
    with tracer.start_as_current_span(scenario, record_exception=False, set_status_on_exception=False) as span:
        try:
            yield
        finally:
            # Exception messages and stack traces can contain content even when message capture is off.
            if sys.exc_info()[0] is not None:
                from opentelemetry.trace import StatusCode

                span.set_status(StatusCode.ERROR)


@contextmanager
def agent_tracing(project_client: AIProjectClient, *, scenario: str) -> Iterator[None]:
    """Trace an existing agent scenario to the project's linked App Insights, when available.

    :param project_client: Existing authenticated project client.
    :type project_client: ~azure.ai.projects.AIProjectClient
    :keyword str scenario: Name of the span enclosing the original agent scenario.
    :return: A tracing context that preserves the original conversation outcome.
    :rtype: ~typing.ContextManager[None]
    """
    connection_string = None
    if not _tracing_disabled():
        try:
            connection_string = project_client.telemetry.get_application_insights_connection_string()
            if not connection_string:
                _tracing_warning("the project returned an empty Application Insights connection string")
        except (AzureError, ValueError) as error:
            _tracing_warning(type(error).__name__)
    tracer = _start_sample_tracing(connection_string)
    try:
        with _scenario_span(tracer, scenario):
            yield
    finally:
        if tracer is not None:
            _finish_sample_tracing()


@asynccontextmanager
async def agent_tracing_async(project_client: AsyncAIProjectClient, *, scenario: str) -> AsyncIterator[None]:
    """Async adapter for the shared project-linked tracing setup and lifecycle.

    :param project_client: Existing authenticated async project client.
    :type project_client: ~azure.ai.projects.aio.AIProjectClient
    :keyword str scenario: Name of the span enclosing the original agent scenario.
    :return: An async tracing context that preserves the original conversation outcome.
    :rtype: ~typing.AsyncContextManager[None]
    """
    connection_string = None
    if not _tracing_disabled():
        try:
            connection_string = await project_client.telemetry.get_application_insights_connection_string()
            if not connection_string:
                _tracing_warning("the project returned an empty Application Insights connection string")
        except (AzureError, ValueError) as error:
            _tracing_warning(type(error).__name__)
    tracer = _start_sample_tracing(connection_string)
    try:
        with _scenario_span(tracer, scenario):
            yield
    finally:
        if tracer is not None:
            await asyncio.to_thread(_finish_sample_tracing)


def _resolve_zip_file(zip_file: str) -> Path:
    zip_path = Path(zip_file)
    if zip_path.is_absolute():
        return zip_path

    for base_path in (Path.cwd(), Path(__file__).resolve().parent, Path(__file__).resolve().parents[1]):
        candidate = base_path / zip_path
        if candidate.exists():
            return candidate.resolve()

    return (Path.cwd() / zip_path).resolve()


def zip_directory(
    source_dir: Path, zip_filename: Optional[str] = None, *, env_var: str = "ZIP_FILE_PATH"
) -> Tuple[bytes, str, Path]:
    """Return zip bytes, SHA-256, and path for a source directory.

    Direct sample runs create a normal zip from ``source_dir`` in the temp folder.
    Callers can set ``env_var`` to load an existing zip file instead of creating
    a new archive.
    """
    configured_zip_file = os.environ.get(env_var)
    if configured_zip_file:
        zip_path = _resolve_zip_file(configured_zip_file)
        zip_bytes = zip_path.read_bytes()
        zip_sha256 = hashlib.sha256(zip_bytes).hexdigest()
        print(f"Loaded zip from {zip_path}: {len(zip_bytes)} bytes, sha256={zip_sha256}")
        return zip_bytes, zip_sha256, zip_path

    zip_path = Path(tempfile.gettempdir()).resolve() / (zip_filename or f"{source_dir.name}.zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in sorted(source_dir.rglob("*")):
            if not file_path.is_file():
                continue
            if "__pycache__" in file_path.parts or file_path.suffix == ".pyc" or file_path.name == "README.md":
                continue
            zf.write(file_path, file_path.relative_to(source_dir).as_posix())

    zip_bytes = zip_path.read_bytes()
    zip_sha256 = hashlib.sha256(zip_bytes).hexdigest()
    print(f"Built zip from {source_dir}: {len(zip_bytes)} bytes, sha256={zip_sha256}, path={zip_path}")
    return zip_bytes, zip_sha256, zip_path


@contextmanager
def create_version_with_endpoint(
    project_client: AIProjectClient,
    agent_name: str,
    *,
    definition: AgentDefinition,
    metadata: Optional[dict[str, str]] = None,
    description: Optional[str] = None,
    blueprint_reference: Optional[AgentBlueprintReference] = None,
    draft: Optional[bool] = None,
    **kwargs: Any,
) -> Iterator[AgentVersionDetails]:
    """Create a version, point the agent endpoint to it, then restore and delete on exit."""
    created_version = None
    original_agent_endpoint = None

    try:
        created_version = project_client.agents.create_version(
            agent_name=agent_name,
            definition=definition,
            metadata=metadata,
            description=description,
            blueprint_reference=blueprint_reference,
            draft=draft,
            **kwargs,
        )

        original_agent_endpoint = project_client.agents.get(agent_name=agent_name).agent_endpoint
        endpoint_config = AgentEndpointConfig(
            version_selector=VersionSelector(
                version_selection_rules=[
                    FixedRatioVersionSelectionRule(agent_version=created_version.version, traffic_percentage=100),
                ]
            ),
            protocol_configuration=ProtocolConfiguration(responses=ResponsesProtocolConfiguration()),
        )
        project_client.agents.update_details(agent_name=agent_name, agent_endpoint=endpoint_config)
        print(f"Agent endpoint configured for version {created_version.version}")

        yield created_version
    finally:
        if original_agent_endpoint is not None:
            project_client.agents.update_details(agent_name=agent_name, agent_endpoint=original_agent_endpoint)

        if created_version is not None:
            project_client.agents.delete_version(
                agent_name=agent_name, agent_version=created_version.version, force=True
            )


@asynccontextmanager
async def create_version_with_endpoint_async(
    project_client: AsyncAIProjectClient,
    agent_name: str,
    *,
    definition: AgentDefinition,
    metadata: Optional[dict[str, str]] = None,
    description: Optional[str] = None,
    blueprint_reference: Optional[AgentBlueprintReference] = None,
    draft: Optional[bool] = None,
    **kwargs: Any,
) -> AsyncIterator[AgentVersionDetails]:
    """Async variant of :func:`create_version_with_endpoint`."""
    created_version = None
    original_agent_endpoint = None

    try:
        created_version = await project_client.agents.create_version(
            agent_name=agent_name,
            definition=definition,
            metadata=metadata,
            description=description,
            blueprint_reference=blueprint_reference,
            draft=draft,
            **kwargs,
        )

        original_agent_endpoint = (await project_client.agents.get(agent_name=agent_name)).agent_endpoint
        endpoint_config = AgentEndpointConfig(
            version_selector=VersionSelector(
                version_selection_rules=[
                    FixedRatioVersionSelectionRule(agent_version=created_version.version, traffic_percentage=100),
                ]
            ),
            protocol_configuration=ProtocolConfiguration(responses=ResponsesProtocolConfiguration()),
        )
        await project_client.agents.update_details(agent_name=agent_name, agent_endpoint=endpoint_config)
        print(f"Agent endpoint configured for version {created_version.version}")

        yield created_version
    finally:
        if original_agent_endpoint is not None:
            await project_client.agents.update_details(agent_name=agent_name, agent_endpoint=original_agent_endpoint)

        if created_version is not None:
            await project_client.agents.delete_version(
                agent_name=agent_name, agent_version=created_version.version, force=True
            )

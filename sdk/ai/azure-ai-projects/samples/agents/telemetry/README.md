# Application Insights for Python agent conversations

The validated agent conversation samples share one setup implementation in
[`samples/util.py`](https://github.com/Azure/azure-sdk-for-python/blob/main/sdk/ai/azure-ai-projects/samples/util.py). They reuse the Application Insights
connection already attached to your Foundry project. They do not deploy Azure
resources, replace connections, or assign roles.

This rollout includes samples with passing live or recorded execution. Samples
without a confirmed pass remain unchanged, including workflows, memory search,
computer use, Azure Functions, agent-to-agent calls and realtime voice.
Included samples explicitly enter `agent_tracing` or `agent_tracing_async`;
the console-tracing example keeps its existing console-only destination.

## Set up once

1. Use an existing Foundry project with a connected Application Insights
   resource. Keep the project's endpoint, model deployment and any tool-specific
   connections required by your chosen sample.
2. Authenticate as usual with `DefaultAzureCredential`, for example with
   `az login`. The caller must be able to read the project's connection and
   retrieve its credentials. Access to query/view traces is separate from
   permission to run the agent or send telemetry; have your administrator grant
   the appropriate access to the actual Application Insights/workspace.
3. Install your sample's dependencies, including the shared tracing dependencies:

   ```console
   pip install azure-monitor-opentelemetry azure-core-tracing-opentelemetry
   ```

4. Set `FOUNDRY_PROJECT_ENDPOINT` and the model/tool settings documented in your
   sample, then run its existing command. There is no need to copy an Application
   Insights connection string into each sample.

## Automatic behavior and privacy

When a usable project connection exists, the shared helper configures Azure
Monitor and explicitly enables the SDK's **experimental GenAI instrumentation**
before the sample acquires its OpenAI client.
It supplies `AZURE_EXPERIMENTAL_ENABLE_GENAI_TRACING=true` only when that variable
is unset. This enables preview tracing whose span names and attributes may change.

To opt out of this automatic client-side setup, set:

```dotenv
AZURE_EXPERIMENTAL_ENABLE_GENAI_TRACING=false
```

Opting out does not disconnect Application Insights or disable service-generated
project traces. An existing caller-managed provider or SDK instrumentation is
left unchanged, rather than reconfigured or given another exporter. In that case
the caller remains responsible for its destination, instrumentation and lifecycle.

Message content and tool arguments/results are not recorded by default. Keep
`OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT` unset or `false`. Set it to
`true` only if you explicitly intend to collect potentially sensitive content.
Raw audio/image payloads are not added by this helper. Baggage propagation
remains off unless explicitly enabled through the SDK's existing controls.
Project/service-side content collection settings are separate.

If connection discovery, permissions, optional packages or tracing configuration
are unavailable, the helper warns and the original agent scenario continues.
The warning is not a successful tracing result. Conversation errors still
propagate normally.

The console-tracing sample intentionally remains console-only. Its explicit
instrumentation and environment setup are documented in that sample.

## Reuse in another agent sample

After constructing the existing authenticated project client, enter the shared
context before constructing the OpenAI client:

```python
from util import agent_tracing

with (
    project_client,
    agent_tracing(project_client, scenario="my-agent-scenario"),
    project_client.get_openai_client(agent_name=agent_name) as openai_client,
):
    response = openai_client.responses.create(input="Hello")
```

Async samples use `agent_tracing_async` in an `async with` statement. Their
connection lookup is asynchronous, and the bounded flush runs off the event
loop. Both adapters share configuration and ownership logic; do not copy the
exporter setup into each sample or hide it in agent creation.

The context stays active while streams and function-tool continuations
complete. It ends the scenario span and attempts
a five-second trace flush even on failure/cancellation, without masking the
original exception. Only instrumentation activated by the helper is removed.
The Azure Monitor provider is configured once per process and retains its
normal process-exit shutdown; it is not shut down between sample contexts.

## View traces and use them for evaluation

Open the project's Tracing view and select the agent and execution time window.
Correlate the actual response/conversation IDs and trace IDs from that run.
Successful configuration, a successful answer, or an old trace does not prove
that a new execution was ingested. Allow for ingestion delay when querying.

The evaluation examples illustrate the same existing-project connection lookup
and Azure Monitor configuration, and how to consume existing traces:

- [`sample_human_evaluations.py`](https://github.com/Azure/azure-sdk-for-python/blob/main/sdk/ai/azure-ai-projects/samples/evaluations/sample_human_evaluations.py)
  configures Azure Monitor and emits feedback events with illustrative IDs.
- [`sample_multiturn_trace_evaluation_by_id.py`](https://github.com/Azure/azure-sdk-for-python/blob/main/sdk/ai/azure-ai-projects/samples/evaluations/sample_multiturn_trace_evaluation_by_id.py)
  selects real conversations/traces for evaluation.
- [`sample_multiturn_trace_evaluation_agent_filter.py`](https://github.com/Azure/azure-sdk-for-python/blob/main/sdk/ai/azure-ai-projects/samples/evaluations/sample_multiturn_trace_evaluation_agent_filter.py)
  selects traces by agent/version and time window.

Those evaluation examples do not share an existing bootstrap function: their
setup is inline or they consume traces already produced elsewhere. The agent
helper reuses the same public SDK connection-discovery and Azure Monitor APIs;
it does not import/run an evaluation scenario. Evaluation identity/query access
is separate from this client-side exporter setup.

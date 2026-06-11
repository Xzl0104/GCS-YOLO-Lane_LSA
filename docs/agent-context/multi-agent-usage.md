# Multi-Agent Usage

This project uses a single project-wide delegation rule for Codex Agents.

## Available Agents

Project role names:

- `gcs_explorer`: read-only exploration
- `gcs_implementer`: workspace-write implementation
- `gcs_reviewer`: read-only review
- `gcs_tester`: workspace-write validation
- `gcs_experiment_analyst`: read-only experiment analysis
- `gcs_docs_researcher`: read-only documentation and source research
- `gcs_security_auditor`: read-only integrity and security review
- `gcs_integrator`: workspace-write synthesis

All project Agents use `gpt-5.5` with `xhigh` reasoning.

## Runtime Role Mapping

The project role names above are stable policy names. Live `multi_agent_v1` `agent_type` values are runtime-discovered and may differ by Codex App version and tool discovery state.

If the `multi_agent_v1.spawn_agent` tool schema lists project roles such as `gcs_reviewer` or `gcs_experiment_analyst`, assistant-originated runtime delegation may pass those project roles directly as `agent_type`.

If the current runtime exposes only built-in roles, use this fallback mapping for live calls:

| Project role | Live `agent_type` |
| --- | --- |
| `gcs_explorer` | `explorer` |
| `gcs_implementer` | `worker` |
| `gcs_reviewer` | `explorer` |
| `gcs_tester` | `worker` |
| `gcs_experiment_analyst` | `explorer` |
| `gcs_docs_researcher` | `explorer` |
| `gcs_security_auditor` | `explorer` |
| `gcs_integrator` | `default` |

When using the fallback mapping, put the project role in `message`, for example: `Act as project role gcs_explorer`. Do not pass `gcs_explorer` as the live `agent_type` unless the current `multi_agent_v1.spawn_agent` schema explicitly lists it as a supported runtime role.

## Runtime Spawn Contract

Use this contract before any multi-Agent work.

There are three distinct delegation paths:

- User-originated App/Skill orchestration: when the user sends a `$gcs-*` Skill command in Codex CLI/App, the App may own spawn, routing, wait, and close orchestration.
- Assistant-originated runtime delegation: when the user explicitly asks for multi-Agent, delegation, subagents, or parallel agent work, and `multi_agent_v1.spawn_agent` is available after tool discovery, the assistant may call that tool directly.
- Repository wrapper calls: project code that owns a host `spawn_agent` callable must use the adapter or payload normalizer described below.

Default to natural-language or Skill-triggered delegation in Codex CLI/App for user-originated Skill commands. In that path, Codex owns spawn, routing, wait, and close orchestration, and project code does not construct low-level `spawn_agent` payloads.

Skill loading is not delegation. A `<skill>...</skill>` block or loaded `SKILL.md` only gives the assistant workflow instructions; it does not mean an Agent was spawned. If the assistant writes `$gcs-review-change ...` in a reply, that is plain text and does not trigger App orchestration.

For assistant-originated runtime delegation, first verify that `multi_agent_v1.spawn_agent` is available. The `multi_agent_v1` tools may be deferred and absent from the initial tool list. If `multi_agent_v1.spawn_agent` is not visible and the user explicitly requested multi-Agent work, the assistant must call `tool_search` with query `multi_agent_v1 spawn_agent` when `tool_search` is available before declaring subagents unavailable. Do not say that runtime subagents are unavailable merely because `multi_agent_v1.spawn_agent` is absent from the initial callable list. If neither `multi_agent_v1.spawn_agent` nor `tool_search` is available, report that the current API/tool surface does not expose runtime multi-Agent delegation, then continue only with local workflow execution. Do not simulate Agent roles or describe local work as delegated Agent output.

Do not hand-write raw low-level `spawn_agent` JSON in chat unless you are testing the wrapper itself. Use the exposed `multi_agent_v1.spawn_agent` tool when assistant-originated runtime delegation is authorized and available.

Recommended prompt shape:

```text
$gcs-review-change Review the current branch.

Spawn one read-only agent for each topic:
1. correctness
2. official-val/test leakage
3. tests and missing validation
4. maintainability

Wait for all agents and summarize findings with severity, evidence, confidence, and suggested fixes.
```

First launch one gate Agent with a narrow task. If the gate Agent fails, stop delegation immediately and continue locally or report the blocker.

## Low-Level Wrapper Rule

Use low-level `spawn_agent` JSON only inside a wrapper or when explicitly debugging the runtime call. Assistant tool calls through `multi_agent_v1.spawn_agent` are runtime delegation, not repository wrapper code.

Every live call must pass `agent_type` and exactly one payload field: either `message` or `items`. Do not pass `message` and `items` together.

Field meaning:

```text
agent_type       = which runtime role to launch
items            = structured input / material package
fork_context     = whether the subagent inherits current conversation context
reasoning_effort = how deeply the subagent should reason
service_tier     = service tier
message          = concrete task brief for this run; alternative to items
```

`message` is the concrete task brief for this subagent run.

`items` is the explicit material package for the subagent. Use it for task notes, `AGENTS.md`, `docs/agent-context/current-contracts.md`, `docs/agent-context/experiment-rules.md`, relevant source files, run summaries, and error logs.

`message` and `items` are alternative payload fields in the current spawn tool. Use `message` for a simple task string. Use `items` for structured text, skill, mention, image, or local image input.

`fork_context` controls whether the subagent inherits the current conversation context. When `fork_context=false`, `items` is especially important. When `fork_context=true`, `items` is still useful, but avoid repeating too much context.

For this project, use `reasoning_effort: "xhigh"` for project Agent calls unless the user explicitly changes the policy.

Put the intended project role, read-only/write scope, no-edit constraints, and task scope in whichever payload field you use. If using `message`, list materials in a `Materials:` sentence. If using `items`, include the task instruction as a text item alongside any material references supported by the runtime.

Wrapper code must drop empty strings and empty arrays at the final adapter boundary before calling `spawn_agent`; an empty `items` array and an empty `message` string are still treated as present by the runtime.

Repository wrappers that own the host `spawn_agent` callable must call `scripts/gcs_spawn_adapter.py::spawn_agent_with_normalized_payload`. If a wrapper cannot use that adapter directly, it must call `scripts/gcs_spawn_payload.py::normalize_spawn_payload` immediately before sending the low-level runtime call, after any UI or schema defaults have been applied.

The low-level chat-exposed low-level spawn surface must not be used directly when it serializes omitted payload fields as empty defaults. Use user-originated Skill/App orchestration, assistant-originated `multi_agent_v1.spawn_agent` tool calls, or route repository wrapper calls through the adapter above in the actual UI/tool layer.

Do not describe local work as delegated Agent output. In short, do not describe local analysis or loaded Skill execution as if a subagent produced it.

## When To Use Agents

Use project role `gcs_explorer` for read-only mapping, `gcs_implementer` for scoped edits, `gcs_reviewer` for code-review findings, `gcs_tester` for contract checks and small reproductions, `gcs_experiment_analyst` for run comparison, `gcs_docs_researcher` for version-sensitive behavior, `gcs_security_auditor` for leakage and path risks, and `gcs_integrator` for consolidating multiple outputs. For live calls, use tool-discovered project `agent_type` values when available; otherwise map those roles to `explorer`, `worker`, or `default` as shown above.

## Skills

Stable workflows:

- `$gcs-explore-codebase`
- `$gcs-plan-change`
- `$gcs-implement-change`
- `$gcs-review-change`
- `$gcs-debug-issue`
- `$gcs-fix-ci`
- `$gcs-experiment-review`
- `$gcs-integrate-results`

Use the skill name explicitly when you want that workflow to trigger reliably.

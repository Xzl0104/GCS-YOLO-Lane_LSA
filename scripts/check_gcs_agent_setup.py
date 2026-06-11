"""Validate project-scoped Codex agents, skills, context files, and AGENTS.md discovery."""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

from gcs_spawn_adapter import spawn_agent_with_normalized_payload
from gcs_spawn_payload import PAYLOAD_XOR_ERROR, normalize_spawn_payload


ROOT = Path(__file__).resolve().parents[1]
AGENT_FILES = {
    "gcs-explorer.toml": ("gcs_explorer", "read-only", "gpt-5.5", "xhigh"),
    "gcs-implementer.toml": ("gcs_implementer", "workspace-write", "gpt-5.5", "xhigh"),
    "gcs-reviewer.toml": ("gcs_reviewer", "read-only", "gpt-5.5", "xhigh"),
    "gcs-tester.toml": ("gcs_tester", "workspace-write", "gpt-5.5", "xhigh"),
    "gcs-experiment-analyst.toml": ("gcs_experiment_analyst", "read-only", "gpt-5.5", "xhigh"),
    "gcs-docs-researcher.toml": ("gcs_docs_researcher", "read-only", "gpt-5.5", "xhigh"),
    "gcs-security-auditor.toml": ("gcs_security_auditor", "read-only", "gpt-5.5", "xhigh"),
    "gcs-integrator.toml": ("gcs_integrator", "workspace-write", "gpt-5.5", "xhigh"),
}
SKILLS = (
    "gcs-explore-codebase",
    "gcs-plan-change",
    "gcs-implement-change",
    "gcs-review-change",
    "gcs-debug-issue",
    "gcs-fix-ci",
    "gcs-experiment-review",
    "gcs-integrate-results",
)
SKILL_RUNTIME_AGENT_TYPES = {
    "gcs-explore-codebase": "explorer",
    "gcs-plan-change": "explorer",
    "gcs-implement-change": "worker",
    "gcs-review-change": "explorer",
    "gcs-debug-issue": "worker",
    "gcs-fix-ci": "worker",
    "gcs-experiment-review": "explorer",
    "gcs-integrate-results": "default",
}
RUNTIME_AGENT_TYPES = {"explorer", "worker", "default"}
CONTEXT_FILES = (
    "project-context.md",
    "current-contracts.md",
    "commands.md",
    "experiment-rules.md",
    "known-bottlenecks.md",
    "decision-log.md",
    "multi-agent-usage.md",
    "implementation-manual.md",
)
XHIGH_POLICY = {
    "gcs_explorer": "gpt-5.5",
    "gcs_implementer": "gpt-5.5",
    "gcs_reviewer": "gpt-5.5",
    "gcs_tester": "gpt-5.5",
    "gcs_experiment_analyst": "gpt-5.5",
    "gcs_docs_researcher": "gpt-5.5",
    "gcs_security_auditor": "gpt-5.5",
    "gcs_integrator": "gpt-5.5",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def parse_frontmatter(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8")
    require(text.startswith("---\n"), f"{path} must start with YAML frontmatter")
    end = text.find("\n---\n", 4)
    require(end >= 0, f"{path} has no closing YAML frontmatter")
    metadata: dict[str, str] = {}
    for line in text[4:end].splitlines():
        if not line.strip():
            continue
        key, separator, value = line.partition(":")
        require(bool(separator), f"{path} has invalid frontmatter line: {line!r}")
        metadata[key.strip()] = value.strip()
    return metadata


def parse_simple_yaml(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        key, separator, value = line.partition(":")
        require(bool(separator), f"{path} has invalid yaml line: {line!r}")
        data[key.strip()] = value.strip()
    return data


def check_spawn_payload_normalization() -> None:
    message_only = normalize_spawn_payload(
        {
            "agent_type": "explorer",
            "fork_context": False,
            "reasoning_effort": "xhigh",
            "service_tier": "auto",
            "message": "review this",
            "items": [],
            "model": "",
        }
    )
    require("message" in message_only, "message-only spawn params must keep message")
    require("items" not in message_only, "message-only spawn params must drop empty items")
    require("model" not in message_only, "spawn params must drop empty model override")

    items_only = normalize_spawn_payload(
        {
            "agent_type": "explorer",
            "fork_context": False,
            "reasoning_effort": "xhigh",
            "service_tier": "auto",
            "message": "",
            "items": [{"type": "text", "text": "review this"}],
        }
    )
    require("items" in items_only, "items-only spawn params must keep items")
    require("message" not in items_only, "items-only spawn params must drop empty message")

    try:
        normalize_spawn_payload({"message": "review this", "items": [{"type": "text", "text": "duplicate"}]})
    except ValueError as error:
        require(str(error) == PAYLOAD_XOR_ERROR, "message/items conflict must raise the expected error")
    else:
        raise AssertionError("message/items conflict must fail")

    try:
        normalize_spawn_payload({"agent_type": "explorer", "message": "   ", "items": []})
    except ValueError as error:
        require(str(error) == PAYLOAD_XOR_ERROR, "empty payload fields must fail the expected XOR error")
    else:
        raise AssertionError("empty message plus empty items must fail")


def check_spawn_adapter_boundary() -> None:
    calls: list[dict] = []

    def fake_spawn_agent(**kwargs):
        calls.append(kwargs)
        return {"ok": True}

    result = spawn_agent_with_normalized_payload(
        fake_spawn_agent,
        {
            "agent_type": "explorer",
            "fork_context": False,
            "reasoning_effort": "xhigh",
            "message": "review this",
            "items": [],
            "model": "",
        },
    )
    require(result == {"ok": True}, "spawn adapter must return the host spawn_agent result")
    require(len(calls) == 1, "spawn adapter must call host spawn_agent exactly once")
    require("message" in calls[0], "spawn adapter must preserve message payload")
    require("items" not in calls[0], "spawn adapter must remove UI-injected empty items before host call")
    require("model" not in calls[0], "spawn adapter must remove empty model override before host call")


def check_config() -> None:
    path = ROOT / ".codex" / "config.toml"
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    agents = data.get("agents", {})
    require(agents.get("max_threads") == 8, "agents.max_threads must be 8")
    require(agents.get("max_depth") == 1, "agents.max_depth must be 1")
    require(agents.get("job_max_runtime_seconds") == 1800, "agent CSV worker timeout must be 1800 seconds")


def check_agents() -> None:
    root = ROOT / ".codex" / "agents"
    actual = {path.name for path in root.glob("*.toml")}
    require(actual == set(AGENT_FILES), f"unexpected agent files: {sorted(actual ^ set(AGENT_FILES))}")
    names: set[str] = set()
    for filename, (expected_name, expected_sandbox, expected_model, expected_effort) in AGENT_FILES.items():
        path = root / filename
        data = tomllib.loads(path.read_text(encoding="utf-8"))
        require(data.get("name") == expected_name, f"{path} name must be {expected_name}")
        require(re.fullmatch(r"[a-z][a-z0-9_]*", expected_name) is not None, f"invalid agent name: {expected_name}")
        require(bool(data.get("description")), f"{path} needs description")
        require(bool(data.get("developer_instructions")), f"{path} needs developer_instructions")
        require(data.get("sandbox_mode") == expected_sandbox, f"{path} sandbox must be {expected_sandbox}")
        if expected_model is None:
            require("model" not in data, f"{path} must inherit the parent model")
        else:
            require(data.get("model") == expected_model, f"{path} model must be {expected_model}")
        require(
            data.get("model_reasoning_effort") == expected_effort,
            f"{path} model_reasoning_effort must be {expected_effort}",
        )
        require(expected_name not in names, f"duplicate agent name: {expected_name}")
        names.add(expected_name)


def check_skills() -> None:
    root = ROOT / ".agents" / "skills"
    actual = {path.name for path in root.iterdir() if path.is_dir()}
    require(actual == set(SKILLS), f"unexpected skill directories: {sorted(actual ^ set(SKILLS))}")
    for name in SKILLS:
        skill_path = root / name / "SKILL.md"
        metadata = parse_frontmatter(skill_path)
        require(set(metadata) == {"name", "description"}, f"{skill_path} frontmatter must contain only name and description")
        require(metadata["name"] == name, f"{skill_path} name mismatch")
        require(bool(metadata["description"]), f"{skill_path} needs description")
        skill_text = skill_path.read_text(encoding="utf-8")
        require("TODO" not in skill_text, f"{skill_path} still contains TODO")
        require("## Runtime Delegation Guard" in skill_text, f"{skill_path} must include the runtime delegation guard")
        require(
            "initial tool list alone" in skill_text
            and "`tool_search`" in skill_text
            and "`multi_agent_v1 spawn_agent`" in skill_text
            and "current API/tool surface does not expose runtime multi-Agent delegation" in skill_text
            and "continue only with local workflow execution" in skill_text
            and "track each spawned agent id" in skill_text
            and "wait_agent" in skill_text
            and "close_agent" in skill_text
            and "A loaded Skill is not a spawned Agent" in skill_text,
            f"{skill_path} must require tool_search and runtime lifecycle cleanup before declaring subagents unavailable",
        )
        ui_path = root / name / "agents" / "openai.yaml"
        require(ui_path.is_file(), f"missing UI metadata: {ui_path}")
        require("TODO" not in ui_path.read_text(encoding="utf-8"), f"{ui_path} still contains TODO")
        ui_metadata = parse_simple_yaml(ui_path)
        expected_agent_type = SKILL_RUNTIME_AGENT_TYPES[name]
        require(
            ui_metadata.get("agent_type") == expected_agent_type,
            f"{ui_path} agent_type must be live runtime type {expected_agent_type}",
        )
        require(
            ui_metadata.get("agent_type") in RUNTIME_AGENT_TYPES,
            f"{ui_path} uses unsupported live runtime agent_type {ui_metadata.get('agent_type')!r}",
        )


def check_context_and_index() -> None:
    context_root = ROOT / "docs" / "agent-context"
    for filename in CONTEXT_FILES:
        path = context_root / filename
        require(path.is_file() and path.stat().st_size > 0, f"missing context file: {path}")

    agents_path = ROOT / "AGENTS.md"
    content = agents_path.read_text(encoding="utf-8")
    require(len(agents_path.read_bytes()) <= 32768, "root AGENTS.md must remain within the default 32 KiB read limit")
    require(content.startswith("# AGENTS.md"), "root AGENTS.md must be the concise instruction file")
    require("超详细小白实现手册" not in content, "historical implementation manual must not be stored in root AGENTS.md")
    require("## Agent Coordination Rules" in content, "AGENTS.md is missing agent coordination rules")
    require("All project Agents use `gpt-5.5` with `xhigh` reasoning." in content, "AGENTS.md is missing all-xhigh model policy")
    require(
        "exactly one payload field" in content
        and "message = simple task brief" in content
        and "items   = structured input / material package" in content
        and "message and items are alternative payload fields" in content
        and "reasoning_effort = how deeply" in content
        and "Default to natural-language or Skill-triggered delegation" in content
        and "drop empty strings and empty arrays at the final adapter boundary" in content
        and "spawn_agent_with_normalized_payload" in content
        and "normalize_spawn_payload" in content
        and "Stop delegation immediately" in content,
        "AGENTS.md is missing the runtime delegation shape rule",
    )
    require(
        "Assistant-originated multi-Agent delegation is allowed" in content
        and "multi_agent_v1.spawn_agent" in content
        and "explicitly asks for multi-Agent" in content
        and "subagents" in content
        and "deferred and absent from the initial tool list" in content
        and "must call `tool_search` with query `multi_agent_v1 spawn_agent`" in content
        and "Do not say that runtime subagents are unavailable merely because" in content
        and "current API/tool surface does not expose runtime multi-Agent delegation" in content
        and "Do not simulate Agent roles" in content
        and "maintain an active agent id list" in content
        and "wait_agent" in content
        and "close_agent" in content
        and "Completed agents remain open and count against `max_threads` until closed" in content
        and "agents.max_threads` is 8" in content
        and "does not mean an Agent was spawned" in content
        and "Live `multi_agent_v1` `agent_type` values are runtime-discovered" in content
        and "built-in roles `explorer`, `worker`, and `default`" in content
        and "$gcs-review-change" in content,
        "AGENTS.md must document assistant-originated delegation and runtime agent_type discovery",
    )
    for role, model_name in XHIGH_POLICY.items():
        require(role in content and model_name in content, f"AGENTS.md is missing model policy for {role}")
    for filename in CONTEXT_FILES:
        require(f"docs/agent-context/{filename}" in content, f"AGENTS.md index is missing {filename}")
    require("python scripts/check_gcs_agent_setup.py" in content, "AGENTS.md must document the setup check")
    multi_agent_usage = (context_root / "multi-agent-usage.md").read_text(encoding="utf-8")
    require("## Runtime Spawn Contract" in multi_agent_usage, "multi-agent-usage.md is missing the runtime spawn contract")
    require("## Runtime Role Mapping" in multi_agent_usage, "multi-agent-usage.md is missing runtime role mapping")
    require(
        "Live `multi_agent_v1` `agent_type` values" in multi_agent_usage
        and "runtime-discovered" in multi_agent_usage
        and "Assistant-originated runtime delegation" in multi_agent_usage
        and "deferred and absent from the initial tool list" in multi_agent_usage
        and "must call `tool_search` with query `multi_agent_v1 spawn_agent`" in multi_agent_usage
        and "Do not say that runtime subagents are unavailable merely because" in multi_agent_usage
        and "current API/tool surface does not expose runtime multi-Agent delegation" in multi_agent_usage
        and "Do not simulate Agent roles" in multi_agent_usage
        and "Runtime Lifecycle Rule" in multi_agent_usage
        and "agents.max_threads` is 8" in multi_agent_usage
        and "active agent list" in multi_agent_usage
        and "wait_agent" in multi_agent_usage
        and "close_agent" in multi_agent_usage
        and "Completed agents remain open and count toward `max_threads` until `close_agent` is called" in multi_agent_usage
        and "resume_agent" in multi_agent_usage
        and "Skill loading is not delegation" in multi_agent_usage
        and "does not mean an Agent was spawned" in multi_agent_usage
        and "If the assistant writes `$gcs-review-change ...`" in multi_agent_usage
        and "multi_agent_v1.spawn_agent" in multi_agent_usage
        and "tool-discovered project `agent_type` values" in multi_agent_usage
        and "explorer" in multi_agent_usage
        and "worker" in multi_agent_usage
        and "default" in multi_agent_usage
        and "alternative payload fields" in multi_agent_usage
        and "exactly one payload field" in multi_agent_usage
        and "concrete task brief" in multi_agent_usage
        and "fork_context` controls whether the subagent inherits" in multi_agent_usage
        and "reasoning_effort = how deeply" in multi_agent_usage
        and "Default to natural-language or Skill-triggered delegation" in multi_agent_usage
        and "drop empty strings and empty arrays at the final adapter boundary" in multi_agent_usage
        and "spawn_agent_with_normalized_payload" in multi_agent_usage
        and "normalize_spawn_payload" in multi_agent_usage
        and '"agent_type": "gcs_explorer"' not in multi_agent_usage,
        "multi-agent-usage.md must document assistant runtime delegation and live agent_type discovery",
    )
    require("single project-wide delegation rule" in multi_agent_usage, "multi-agent-usage.md must define one project-wide delegation rule")
    require("gate Agent" in multi_agent_usage, "multi-agent-usage.md must require a gate Agent before batch launch")
    require("stop delegation immediately" in multi_agent_usage, "multi-agent-usage.md must stop after gate failure")
    require("do not describe local" in multi_agent_usage, "multi-agent-usage.md must not let local work imply delegation")
    decision_log = (context_root / "decision-log.md").read_text(encoding="utf-8")
    require(
        "Permit assistant-originated runtime delegation through `multi_agent_v1.spawn_agent`" in decision_log
        and "must call `tool_search` with query `multi_agent_v1 spawn_agent`" in decision_log
        and "current API/tool surface does not expose runtime multi-Agent delegation" in decision_log
        and "must not simulate Agent roles" in decision_log
        and "Skill loading remains separate from delegation" in decision_log
        and "Close completed runtime agents and raise max_threads to 8" in decision_log
        and "wait_agent" in decision_log
        and "close_agent" in decision_log
        and "wrapper calls still must use" in decision_log,
        "decision-log.md must record the assistant-originated runtime delegation policy",
    )
    for token in ("agent_type", "fork_context", "reasoning_effort", "service_tier", "message", "items"):
        require(token in multi_agent_usage, f"multi-agent-usage.md is missing delegation term: {token}")
    review_skill = (ROOT / ".agents" / "skills" / "gcs-review-change" / "SKILL.md").read_text(encoding="utf-8")
    require("Runtime Spawn Contract" in review_skill, "gcs-review-change skill must reference the runtime spawn contract")
    require("gate Agent" in review_skill, "gcs-review-change skill must require a gate Agent before batch launch")
    require("stop delegation" in review_skill, "gcs-review-change skill must stop after gate failure")
    require("exactly one payload field" in review_skill, "gcs-review-change skill must require message/items payload exclusivity")
    require("Assistant-originated runtime delegation" in review_skill, "gcs-review-change skill must allow authorized assistant runtime delegation")
    require("call `tool_search` with query `multi_agent_v1 spawn_agent`" in review_skill, "gcs-review-change skill must require tool_search before declaring subagents unavailable")
    require("A loaded Skill is not a spawned Agent" in review_skill, "gcs-review-change skill must distinguish loaded skills from spawned agents")
    require("tool-discovered project `agent_type` values" in review_skill, "gcs-review-change skill must document runtime-discovered project agent types")
    require("spawn_agent_with_normalized_payload" in review_skill, "gcs-review-change skill must point wrappers to the spawn adapter")
    require("normalize_spawn_payload" in review_skill, "gcs-review-change skill must point wrappers to the payload normalizer")
    require("chat-exposed low-level" in review_skill, "gcs-review-change skill must warn against broken chat-level spawn surfaces")
    manual = (context_root / "implementation-manual.md").read_text(encoding="utf-8")
    require(manual.startswith("# GCS-YOLO-Lane 超详细小白实现手册"), "implementation manual archive has the wrong content")
    require("## Codex 多 Agent 与 Skills 协作约定" not in manual, "active Agent rules must not be duplicated in the historical manual")


def check_delegation_templates() -> None:
    """Runtime delegation docs are validated in check_context_and_index."""

    return


def check_project_contract_paths() -> None:
    required = (
        "ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12.yaml",
        "data/tusimple_gcs_fixed_y_960x544.yaml",
        "tools/train_gcs.py",
        "tools/infer_gcs.py",
        "tools/eval_gcs.py",
        "tools/eval_tusimple_official.py",
        "tools/sweep_tusimple_official.py",
        "scripts/verify_loss_cleanup.py",
        "tools/check_gcs_count_head_topk_contract.py",
        "tools/check_gcs_decode_meta_contract.py",
        "tools/check_gcs_algorithm_contract.py",
    )
    for relative in required:
        require((ROOT / relative).is_file(), f"documented project path does not exist: {relative}")

    check_model_source = (ROOT / "tools" / "check_model.py").read_text(encoding="utf-8")
    head_dependency_source = (ROOT / "tools" / "check_gcs_head_dependency.py").read_text(encoding="utf-8")
    commands = (ROOT / "docs" / "agent-context" / "commands.md").read_text(encoding="utf-8")
    require('"--cfg"' in check_model_source, "tools/check_model.py must expose --cfg")
    require("check_model.py --cfg " in commands, "commands.md must call check_model.py with --cfg")
    require("check_model.py --model " not in commands, "commands.md must not use the nonexistent check_model --model flag")
    require('"--weights"' in head_dependency_source, "tools/check_gcs_head_dependency.py must expose --weights")
    require(
        "check_gcs_head_dependency.py --weights " in commands,
        "commands.md must call check_gcs_head_dependency.py with --weights",
    )
    require(
        "check_gcs_head_dependency.py --model " not in commands,
        "commands.md must not use the nonexistent check_gcs_head_dependency --model flag",
    )


def main() -> int:
    checks = (
        ("config", check_config),
        ("agents", check_agents),
        ("skills", check_skills),
        ("context/index", check_context_and_index),
        ("delegation templates", check_delegation_templates),
        ("spawn payload normalization", check_spawn_payload_normalization),
        ("spawn adapter boundary", check_spawn_adapter_boundary),
        ("project paths", check_project_contract_paths),
    )
    try:
        for label, check in checks:
            check()
            print(f"[PASS] {label}")
    except (AssertionError, OSError, UnicodeError, tomllib.TOMLDecodeError) as error:
        print(f"[FAIL] {error}", file=sys.stderr)
        return 1
    print("GCS Codex agent setup: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

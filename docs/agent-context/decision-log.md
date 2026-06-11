# Decision Log

This file records non-trivial decisions about architecture, training, evaluation, data, loss, decode, postprocess, experiment policy, and Agent coordination.

Every non-trivial decision should include decision, why, alternatives considered, tradeoffs accepted, validation evidence, and whether the decision affects mainline or only an experiment.

---

## Decision: Use Q=12 as the default model

Status: current mainline

Decision:

Use:

```text
ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12.yaml
```

as the default model config.

Why:

Q=12 provides more lane query candidates than Q=8 and is better aligned with candidate coverage needs in complex TuSimple scenes.

Alternatives considered:

- Q=8 legacy config
- Q=10 experimental configs
- larger Q values

Tradeoff:

Q=12 increases candidate capacity and compute compared with Q=8, but remains manageable and is the current stable default.

Validation evidence:

Current project contracts and checks are centered on Q=12 outputs.

Mainline or experiment:

Mainline.

---

## Decision: Protect test from tuning

Status: hard policy

Decision:

Use official-val for threshold, checkpoint, and postprocess parameter selection. Use test only for final evaluation.

Why:

This prevents test leakage and preserves final metric credibility.

Alternatives considered:

- test-time sweep
- combined val/test search
- full-test guided tuning

Tradeoff:

Test metrics become final evidence only, not a search signal.

Validation evidence:

`tools/sweep_tusimple_official.py` defaults to validation and must reject test sweeps.

Mainline or experiment:

Hard policy.

---

## Decision: Current 7-loss setup is the default, not a permanent restriction

Status: current policy

Decision:

The current default mainline uses:

```text
exist_loss
point_loss
point_valid_loss
line_iou_loss
count_cls_loss
count_sum_loss
quality_loss
```

But this does not ban old or new losses.

Why:

A clean baseline is useful for debugging and reproducibility, but the project is an algorithm research project whose objective is official ACC improvement.

Alternatives considered:

- permanently ban removed losses
- freely mix old losses into mainline
- keep default clean while allowing controlled candidates

Tradeoff:

Candidates need explicit flags, traceability, and official-val comparison before promotion.

Mainline or experiment:

Policy.

---

## Decision: Install project Agent and Skill configuration

Status: current policy

Decision:

Use project-scoped Agents under `.codex/agents/`, workflow Skills under `.agents/skills/`, and context documents under `docs/agent-context/`.

Why:

The project needs repeatable roles for exploration, implementation, review, testing, experiment analysis, documentation research, security review, and integration.

Alternatives considered:

- keep all guidance in root `AGENTS.md`
- rely only on ad hoc prompts
- split active guidance from the historical implementation notes

Tradeoff:

The active root instructions are shorter and easier for Codex to load, while detailed historical notes are archived separately.

Validation evidence:

Run `python scripts/check_gcs_agent_setup.py` after changes.

Mainline or experiment:

Policy.

---

## Decision: Map project roles to live runtime Agent types

Status: current policy, with tool-discovered project roles preferred when available

Decision:

Keep project role names such as `gcs_explorer`, `gcs_implementer`, and `gcs_reviewer` in `.codex/agents/` as policy/configuration roles. When `multi_agent_v1.spawn_agent` exposes those project role names in its current tool schema, assistant-originated runtime delegation may pass them directly as `agent_type`.

For Skill UI metadata and runtimes that expose only built-in role names, use this fallback mapping:

```text
gcs_explorer -> explorer
gcs_implementer -> worker
gcs_reviewer -> explorer
gcs_tester -> worker
gcs_experiment_analyst -> explorer
gcs_docs_researcher -> explorer
gcs_security_auditor -> explorer
gcs_integrator -> default
```

Why:

Different Codex App/runtime surfaces expose different live `agent_type` values. Some surfaces expose project roles from `.codex/agents/`; older or Skill UI surfaces may expose only `explorer`, `worker`, and `default`.

Alternatives considered:

- always pass project role names directly as live `agent_type`
- remove project role names entirely
- keep project roles and map them to runtime roles only as a fallback

Validation evidence:

`python scripts/check_gcs_agent_setup.py` validates Skill UI fallback mappings and documentation.

Mainline or experiment:

Policy.

---

## Decision: Permit assistant-originated runtime delegation through `multi_agent_v1.spawn_agent`

Status: current policy

Decision:

Allow the assistant to create project subagents directly with `multi_agent_v1.spawn_agent` only when both conditions are true:

- the user explicitly asks for multi-Agent, delegation, subagents, or parallel agent work
- the `multi_agent_v1.spawn_agent` tool is available after tool discovery

Because `multi_agent_v1` tools may be deferred and absent from the initial tool list, the assistant must call `tool_search` with query `multi_agent_v1 spawn_agent` when `tool_search` is available before reporting that runtime subagents are unavailable.

If discovery still fails, the correct conclusion is that the current API/tool surface does not expose runtime multi-Agent delegation. The assistant must not simulate Agent roles.

Skill loading remains separate from delegation. A loaded `<skill>...</skill>` block or `SKILL.md` gives the assistant a local workflow; it does not mean an Agent was spawned.

Why:

The documentation must separate user-originated Skill/App orchestration, assistant local Skill execution, assistant runtime tool calls, and repository wrapper calls.

Alternatives considered:

- require the user to send `$gcs-*` commands for every multi-Agent run
- treat loaded Skills as if they were spawned Agents
- allow raw low-level `spawn_agent` JSON in chat

Tradeoff:

Assistant-originated runtime delegation is available only on tool surfaces that expose `multi_agent_v1.spawn_agent`, and only behind explicit user authorization and tool discovery. Repository wrapper calls still must use `scripts/gcs_spawn_adapter.py::spawn_agent_with_normalized_payload` or `scripts/gcs_spawn_payload.py::normalize_spawn_payload` at the final boundary.

Validation evidence:

`python scripts/check_gcs_agent_setup.py` validates updated delegation guidance, required tool discovery before declaring subagents unavailable, Skill-loaded-vs-Agent-spawned distinction, fallback role mapping, and wrapper payload normalizer requirements.

Mainline or experiment:

Policy.

---

## Decision: Close completed runtime agents and raise max_threads to 8

Status: current policy

Decision:

Set project `agents.max_threads` to 8 and require assistant-originated runtime delegation to track every spawned agent id, collect final results with `wait_agent`, and call `close_agent` for completed, stale, cancelled, failed, interrupted, or no-longer-needed agents before ending the turn.

Why:

Completed runtime agents can remain open after producing a final result and continue to count against the open-thread limit until they are explicitly closed. Without explicit cleanup, later delegation can fail with an agent thread limit error even when no useful work is still running.

Alternatives considered:

- keep `max_threads=4` and rely on manual cleanup
- increase the limit without adding lifecycle cleanup
- avoid assistant-originated runtime delegation entirely

Tradeoff:

The project can run wider read-only fan-out when useful, but the main assistant must do explicit lifecycle bookkeeping. Closed agents can be resumed with `resume_agent` when needed, so closing completed agents is preferred over keeping them open only for context preservation.

Validation evidence:

`python scripts/check_gcs_agent_setup.py` validates `agents.max_threads=8` and checks that root and multi-agent guidance require the `spawn_agent` -> `wait_agent` -> `close_agent` lifecycle.

Mainline or experiment:

Policy.

---

## Decision: Normalize low-level spawn payloads at the final adapter boundary

Status: current policy

Decision:

Repository low-level Agent wrappers must call `scripts/gcs_spawn_payload.py::normalize_spawn_payload` immediately before sending a `spawn_agent` runtime call.

Why:

Some UI or schema surfaces can rewrite a message-only delegation request into a payload that includes both `message` and `items: []`. The runtime treats the empty array as a present payload field, so repository-side parameter generation can pass while the final serialized call still fails.

Alternatives considered:

- rely only on documentation
- keep per-wrapper ad hoc cleanup
- avoid low-level wrappers entirely

Tradeoff:

The normalizer adds a small shared utility and a stricter setup check. It does not change natural-language or Skill-triggered delegation behavior.

Validation evidence:

`python scripts/check_gcs_agent_setup.py` validates that message-only payloads drop empty `items`, items-only payloads drop empty `message`, non-empty `message` plus non-empty `items` fails, and all-empty payloads fail.

Mainline or experiment:

Policy.

---

## Decision: Use Skill delegation by default and route low-level spawn through the project adapter

Status: current policy

Decision:

Default project delegation should use natural-language or Skill-triggered Codex App/CLI orchestration. Low-level wrapper code that owns a host `spawn_agent` callable must call `scripts/gcs_spawn_adapter.py::spawn_agent_with_normalized_payload`; wrappers that cannot use the adapter directly must call `scripts/gcs_spawn_payload.py::normalize_spawn_payload` immediately before the runtime call.

This decision governs user-originated Skill/App orchestration and repository wrapper calls. It does not prohibit assistant-originated `multi_agent_v1.spawn_agent` tool calls when the user explicitly requests multi-Agent work and the tool is available.

Why:

The chat-exposed low-level `spawn_agent` surface can serialize omitted payload fields as empty defaults, for example `message` plus `items: []`, before the runtime validates the XOR contract. Natural-language/Skill delegation avoids project-side JSON construction, while the adapter gives real UI/tool code a concrete final-boundary hook.

Alternatives considered:

- keep only documentation guidance
- call the low-level chat surface directly and rely on callers to omit fields
- use only `normalize_spawn_payload` without a callable adapter boundary

Tradeoff:

The adapter is deliberately small and does not own runtime orchestration. Host UI/tool code still has to call it at the actual final boundary.

Validation evidence:

`python scripts/check_gcs_agent_setup.py` validates natural-language/Skill guidance, the low-level payload normalizer, and that `spawn_agent_with_normalized_payload` removes UI-injected empty fields before calling a host `spawn_agent` callable.

Mainline or experiment:

Policy.

---

## Decision: Use conservative count-generalization defaults and document Count Boundary output

Status: current mainline, requires fresh official-val confirmation before any improvement claim

Decision:

Keep Count Head decode as the default policy, document the active Count Boundary output:

```text
pred_count_boundary_logits: B x 2
```

and use these mainline training defaults:

```text
gcs_count_sum = 0.03
gcs_quality = 0.4
gcs_quality_neg_weight = 0.5
gcs_count_cls_w2/w3/w4/w5 = 0.5/1.2/1.4/1.8
gcs_point_valid_gt5_pos_weight = 2.0
gcs_gt5_edge_loss_weight = 1.15
gcs_gt5_oversample_weight = 1.0
gcs_group_sampler_ratios = 2:0.01,3:0.29,4:0.42,5:0.28
```

Keep `gcs_last_lane_rescue`, `gcs_edge_last_lane_rescue`, and `gcs_soft_count_decision` default-off unless an explicit official-val sweep selects them.

Why:

Recent official-val diagnostics separate two issues: Count Head GT5 underprediction and final K=5 output shortfall. The parent count-aware run `gcs_yolo_lane_s_q12_countaware_e160_ft_cw4x5_qneg05_hard_lastlane` had strong GT5 Count Head behavior on official-val (`gt5_output5_rate=0.932432`, `gt5_count_head_under_rate=0.027027`, `count_acc_5=0.932432`). Later/aggressive GT5-only continuations improved neither count generalization nor GT5 under-detection consistently; one constrained official-best row had `gt5_count_head_under_rate=0.135135`, and the ordinary best checkpoint was much worse for GT5 count.

Alternatives considered:

- promote aggressive GT5 oversampling/hard-edge/soft-count/rescue as default
- keep the old weaker count/quality defaults
- disable Count Boundary by default

Tradeoff:

The conservative defaults reduce GT5-specific over-concentration while preserving Count Head supervision. Count Boundary remains part of the active model/decode path, so the output contract must mention it explicitly. These defaults are not proof of a new metric gain until retrained and selected on official-val.

Validation evidence:

Local contract validation passed after the change:

```text
python -m py_compile tools/train_gcs.py ultralytics/models/yolo/gcs_lane/train.py ultralytics/utils/gcs_loss.py tests/test_gcs_count_aware.py
python -m pytest tests/test_gcs_count_aware.py -q
python tools/check_gcs_count_head_topk_contract.py
python tools/check_gcs_decode_meta_contract.py
python tools/check_gcs_algorithm_contract.py
python scripts/verify_loss_cleanup.py
python tools/check_model.py --cfg ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12.yaml --imgsz 544 960
python scripts/check_gcs_agent_setup.py
```

Mainline or experiment:

Mainline defaults. Any claim of improvement requires a fresh official-val sweep; test remains reserved for one-shot final evaluation.

---

## Decision: Sync code changes to GitHub after validation

Status: current workflow policy

Decision:

After each code change, sync the validated published source to:

```text
https://github.com/Xzl0104/GCS-YOLO-Lane_LSA
```

Each sync should be a normal Git commit on `main`. Do not force-push or rewrite published history, so earlier algorithm states remain recoverable by commit SHA or `git revert`.

The GitHub repository is limited to these folders:

```text
data
gcs_tools
scripts
tests
tools
ultralytics
docs
```

Why:

The user wants GitHub to stay current with future code changes while keeping local-only artifacts out of the public repository.
Keeping ordinary Git history also gives each validated algorithm state a rollback point.

Alternatives considered:

- keep GitHub synchronization as an ad hoc manual step
- publish the entire local workspace, including runs, outputs, archives, and checkpoints
- track only the algorithm source folders in the published repository

Tradeoff:

The local workspace is not a standard repository root, so synchronization must preserve the published-folder boundary and avoid pushing generated or large local artifacts.

Validation evidence:

The initial GitHub publish was created from a clean mirror containing only the allowed folders, excluding `__pycache__` and `.pyc` files.

Mainline or experiment:

Workflow policy.

# AGENTS.md

## Language

- Respond in English unless the user explicitly asks for another language.
- Keep code, paths, command names, script names, model names, and parameter names unchanged.

## Project Identity

This branch is the current GCS-YOLO-Lane mainline, imported from the historical `5-25-3.zip` algorithm.

The branch keeps the historical 5-25-3 algorithm body and changes only the TuSimple fixed-y contract needed by the current request.

It is not the current mainline Count Head / Quality Head branch. Current behavior is governed by `docs/agent-context/current-contracts.md`.

## Required Context

Before non-trivial work, read:

- `docs/agent-context/project-context.md`
- `docs/agent-context/current-contracts.md`
- `docs/agent-context/commands.md`
- `docs/agent-context/experiment-rules.md`
- `docs/agent-context/known-bottlenecks.md`
- `docs/agent-context/decision-log.md`
- `docs/agent-context/multi-agent-usage.md`
- `docs/agent-context/implementation-manual.md`
- `docs/agent-context/environment.md`

Historical or mainline notes must not override `docs/agent-context/current-contracts.md`.

## Local And Remote Environments

Use the existing Windows CUDA conda environment for local Codex validation, inference smoke checks, and contract checks:

```text
D:\miniconda3\envs\lsa_yolo
```

Environment name:

```text
lsa_yolo
```

Prefer `conda activate lsa_yolo`. If activation is unavailable, use `D:\miniconda3\envs\lsa_yolo\python.exe`.

Run formal TuSimple training and official-val evaluation on the remote CUDA server, not locally from Codex, unless the user explicitly asks otherwise. From the primary Windows workstation, connect with the local SSH config alias:

```bash
ssh gcs-ebcloud-lane
```

Activate the remote conda environment:

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate ssh_lane
```

Use hardware-aware run strategy:

- Local RTX 4060 8GB: local Codex is for contract checks, label/oracle validation, model-shape checks, and tiny smoke runs only. Do not run formal algorithm training locally unless the user explicitly asks.
- Remote RTX 4090 24GB: formal TuSimple training and official-val evaluation should run on the server. Use `batch=32` as the default formal-training starting point for current Q12/K56 TuSimple runs, reducing only for OOM/instability.

## Hard Contract

For TuSimple, always use:

```bash
--imgsz 544 960
```

This is H,W order. Do not reverse it.

Default model:

```text
ultralytics/cfg/models/gcs/gcs-yolo-lane-s.yaml
```

Default data:

```text
data/tusimple_gcs_fixed_y_960x544.yaml
```

Current fixed-y label contract:

```text
point_mode = fixed_y
fixed_y_start = 710 / 720 = 0.9861111111111112
fixed_y_end = 160 / 720 = 0.2222222222222222
K = 56
Q = 12
```

The 56 fixed y anchors are exactly:

```text
710, 700, 690, ..., 160
```

The K56 labels must be regenerated from original TuSimple JSON and images, not resampled from historical K32 labels.

Historical q12-k56 experiment notes and compatibility paths must stay as old records. They do not override the active 5-25-3 K56 mainline contract.

The active source code is based on rollback commit `50999d6af` (`Document 5-25-3 K56 as mainline`) plus default-disabled experiment knobs for `duplicate_margin_loss`, `spurious_margin_loss`, `lane_balanced_point_loss`, `short_valid_recall_loss`, `far_spurious_survival_loss`, `gt5_rank_consistency_loss`, and `gt3_extra_survival_loss`. Other later experiment commits and their documentation are legacy conclusions only unless a future task explicitly re-enables those mechanisms.
The `gt3_extra_survival_loss` knob is a narrow follow-up to `dupmargin005`; it does not change the model structure, decoder, eval, or final-test selection rules unless explicitly enabled for training.

## Branch Scope

Do not silently import later mainline mechanisms into this branch. In particular, do not add Count Head, Count Boundary, Quality Head, Survival Head, near-miss mining, official-best checkpoint preservation, or mainline K56 candidate scripts unless a future task explicitly asks for that algorithm change.

Agent/Skill tooling may exist in a local Codex workspace, but it is not part of the server-side algorithm payload for this branch. Datasets, generated runs, checkpoints, caches, converted labels, and large runtime artifacts must stay out of Git.

## Output Contract

The default K56 model must produce:

```text
pred_points: B x 12 x 56 x 2
pred_logits: B x 12
pred_valid_logits: B x 12 x 56
aux_mask_logits: B x 2 x H x W
aux_edge_logits: B x 1 x H x W
```

## Loss Contract

Default logged loss items on this branch:

```text
exist_loss
point_loss
lane_balanced_point_loss
point_valid_loss
short_valid_recall_loss
smooth_loss
curve_loss
mask_loss
edge_loss
count_loss
count_under5_loss
duplicate_margin_loss
spurious_margin_loss
far_spurious_survival_loss
gt5_rank_consistency_loss
gt3_extra_survival_loss
```

## Agent Coordination Rules

Project roles:

- `gcs_explorer`
- `gcs_implementer`
- `gcs_reviewer`
- `gcs_tester`
- `gcs_experiment_analyst`
- `gcs_docs_researcher`
- `gcs_security_auditor`
- `gcs_integrator`

All project Agents use `gpt-5.5` with `xhigh` reasoning.

Default to one writable implementation agent in the main worktree. Use read-only agents for exploration, review, experiment analysis, documentation research, and security review. Parallel implementations require separate worktrees or clearly disjoint ownership.

Each subagent must return evidence, uncertainty, confidence, and a recommended next action. The integrator deduplicates findings, ranks risks, and chooses the smallest safe path.

## Runtime Delegation

Assistant-originated multi-Agent delegation is allowed only when the user explicitly asks for multi-Agent, delegation, subagents, or parallel agent work and the `multi_agent_v1.spawn_agent` tool is available after discovery.

`multi_agent_v1` tools may be deferred and absent from the initial tool list. Before declaring runtime subagents unavailable, the assistant must call `tool_search` with query `multi_agent_v1 spawn_agent` when `tool_search` is available. Do not say that runtime subagents are unavailable merely because `multi_agent_v1.spawn_agent` is absent from the initial callable list.

If the current API/tool surface does not expose runtime multi-Agent delegation, continue only with local workflow execution. Do not simulate Agent roles or describe local work as delegated Agent output.

Runtime agent lifecycle is mandatory: maintain an active agent id list for every `spawn_agent` call, use `wait_agent` to collect final results, then call `close_agent` for each completed, stale, cancelled, or no-longer-needed agent before ending the turn. Completed agents remain open and count against `max_threads` until closed. Project `agents.max_threads` is 8 and `max_depth` is 1.

Skill loading is not delegation. A loaded Skill or `SKILL.md` workflow does not mean an Agent was spawned. If the assistant writes `$gcs-review-change ...` in a reply, that is plain text and does not trigger App orchestration.

Live `multi_agent_v1` `agent_type` values are runtime-discovered. Prefer tool-discovered project `agent_type` values when the schema lists them. If only built-in roles `explorer`, `worker`, and `default` are exposed, use the mapping in `docs/agent-context/multi-agent-usage.md` and state the intended project role in the payload.

Start complex delegation with one gate Agent. Stop delegation immediately if the gate fails.

Low-level spawn payloads must contain exactly one payload field:

```text
message = simple task brief
items   = structured input / material package
fork_context = whether the subagent inherits current context
reasoning_effort = how deeply the subagent should reason
service_tier = runtime service tier
```

message and items are alternative payload fields. Default to natural-language or Skill-triggered delegation when Codex App/CLI owns orchestration.

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

## Validation

Recommended local checks after code/config changes:

```bash
python -m py_compile <changed-python-files>
python tools/check_model.py --cfg ultralytics/cfg/models/gcs/gcs-yolo-lane-s.yaml --imgsz 544 960 --batch 1 --device cpu
python tools/check_gcs_label_order_split.py --dataset-root <path-to-tusimple_fixed_y_k56_960x544>
```

For changed Python helper files, also run:

```bash
python -m py_compile <changed-python-files>
```

Formal training should run on the remote CUDA server, not on the local 8GB GPU, unless the user explicitly asks otherwise.

## Research Integrity

Use official-val for threshold, checkpoint, and postprocess selection. Use test only once for final evaluation of a selected candidate.

Do not tune on test, use GT during inference or decode, fabricate lanes, silently change official metrics, or claim improvement without official-val evidence.

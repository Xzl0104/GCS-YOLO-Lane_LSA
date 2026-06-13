# AGENTS.md

## Language

- Respond in English unless the user explicitly asks for another language.
- Keep code, paths, command names, script names, model names, and parameter names unchanged.

## Project Identity

This repository is GCS-YOLO-Lane.

It is not a standard YOLO segmentation project. The goal is to modify YOLO11 into a structured lane detection network that predicts lane instances as ordered 2D point sequences.

The main research objective is to improve TuSimple official Accuracy under a clean, reproducible, and leakage-free protocol.

## Required Context

Before non-trivial work, read the relevant project context:

- `docs/agent-context/project-context.md`
- `docs/agent-context/current-contracts.md`
- `docs/agent-context/commands.md`
- `docs/agent-context/experiment-rules.md`
- `docs/agent-context/known-bottlenecks.md`
- `docs/agent-context/decision-log.md`
- `docs/agent-context/multi-agent-usage.md`
- `docs/agent-context/implementation-manual.md`
- `docs/agent-context/environment.md`

Current behavior is governed by `docs/agent-context/current-contracts.md`. Historical notes are background only.

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

Run experiment training and official-val evaluation on the remote CUDA server, not locally from Codex, unless the user explicitly asks otherwise. From the primary Windows workstation, connect with the local SSH config alias:

```bash
ssh gcs-ebcloud-lane
```

Activate the remote conda environment:

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate ssh_lane
```

Keep the remote training/evaluation loop in a dedicated Git clone of the published source, with datasets, archives, runs, and checkpoints linked or copied as runtime artifacts.

Use hardware-aware run strategy:

- Local RTX 4060 8GB: local Codex is for contract checks, label/oracle validation, model-shape checks, and tiny smoke runs only. Do not run formal algorithm training locally unless the user explicitly asks. Keep local smoke batches small enough to leave CUDA headroom.
- Remote RTX 4090 24GB: formal TuSimple training and official-val evaluation should run on the server. Use `batch=32` as the default formal-training starting point for current Q12/K56 TuSimple runs, reducing only for OOM/instability and increasing only after an explicit throughput check that preserves the same official-val protocol.

## Current Hard Contracts

For TuSimple, always use:

```bash
--imgsz 544 960
```

This is H,W order. Do not reverse it.

Default model:

```text
ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12.yaml
```

Current label contract:

```text
point_mode = fixed_y
fixed_y_start = 710 / 720 = 0.9861111111111112
fixed_y_end = 0.25
K = 32
```

Active experimental K56 contract:

```text
data = data/tusimple_gcs_fixed_y_k56_960x544.yaml
model = ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12-k56.yaml
point_mode = fixed_y
fixed_y_start = 710 / 720 = 0.9861111111111112
fixed_y_end = 160 / 720 = 0.2222222222222222
K = 56
```

The K56 labels must be regenerated from original TuSimple JSON and images, not resampled from existing K32 labels.

The model output must include:

```text
pred_points: B x Q x K x 2
pred_logits: B x Q
pred_valid_logits: B x Q x K
pred_quality_logits: B x Q
pred_count_logits: B x 4
pred_count_boundary_logits: B x 2
```

Current conservative count-generalization defaults:

```text
gcs_count_sum = 0.03
gcs_quality = 0.4
gcs_quality_neg_weight = 0.5
gcs_count_cls_w2/w3/w4/w5 = 0.5/1.2/1.4/1.8
gcs_count_boundary_gt5_pos_weight = 1.15
gcs_point_valid_gt5_pos_weight = 2.0
gcs_gt5_edge_loss_weight = 1.15
gcs_candidate_gt5_edge_weight = 1.10
gcs_point_valid_gt5_edge_continuity = 0.05
gcs_point_valid_gt5_edge_continuity_thr = 0.55
gcs_gt5_oversample_weight = 1.0
gcs_group_sampler_ratios = 2:0.01,3:0.29,4:0.42,5:0.28
```

The GT5 candidate-quality knobs are training-side only. They strengthen real matched GT5 edge-query supervision inside existing loss items and do not change decode, use GT during inference/decode, or fabricate lanes.

Current default-off training-side experimental knobs:

```text
gcs_quality_gt5_edge_floor = 0.0
gcs_quality_hard_negative_from_head = False
gcs_hard_negative_visible_segment = False
gcs_hard_negative_visible_thr = 0.5
gcs_hard_negative_visible_support_points = 12.0
gcs_point_valid_gt5_edge_segment = 0.0
gcs_point_valid_gt5_edge_segment_thr = 0.65
gcs_point_valid_gt5_edge_segment_min_points = 5
```

`gcs_quality_gt5_edge_floor` is training-side only. When enabled above `0.0`, it floors the matched Quality Head target for real left/right edge lanes in GT5 images only; it does not change decode, use GT during inference, or fabricate lanes.

When `gcs_quality_hard_negative_from_head` is enabled, Quality Head hard negatives must be mined from unmatched queries only. Hungarian-matched queries remain matched quality targets even when their current continuous quality target is `0.0`.

The visible-segment hard-negative and GT5 edge Quality floor knobs remain default-off after the 2026-06-13 `gcs_yolo_lane_s_q12_gt5segq_vishn_countvis_ft12_seed1_b8w0` and `gcs_yolo_lane_s_q12_quality_gt5edgefloor_ft12_seed1_b8w0` official-val gates; neither recipe is promoted to mainline defaults.

`gcs_soft_count_decision`, `gcs_last_lane_rescue`, and `gcs_edge_last_lane_rescue` stay default-off unless official-val selection promotes them.

Current default loss items:

```text
exist_loss
point_loss
point_valid_loss
line_iou_loss
count_cls_loss
count_sum_loss
quality_loss
```

The 7-loss setup is the default baseline, not a permanent research restriction.

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

message and items are alternative payload fields. Repository wrappers must drop empty strings and empty arrays at the final adapter boundary before calling the host runtime.

Use `scripts/gcs_spawn_adapter.py::spawn_agent_with_normalized_payload` or `scripts/gcs_spawn_payload.py::normalize_spawn_payload` for wrapper-owned low-level calls. Default to natural-language or Skill-triggered delegation when Codex App/CLI owns orchestration.

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

## Research Integrity

Use official-val for checkpoint, threshold, postprocess, and parameter selection. Use test only for one-shot final evaluation of a selected candidate.

Do not tune on test, use GT during inference or decode, fabricate lanes, silently change official metrics, or claim improvement without official-val evidence.

Old or removed mechanisms may return as controlled experimental candidates when explicit, configurable, traceable, and validated under the same protocol.

## Git Sync Policy

`AGENTS.md` is part of the project source and must be tracked and synchronized with the GitHub branch. Do not treat it as a local-only instruction file when it contains project workflow, environment, contract, or research-policy changes.

After every Git sync, report a concise sync summary to the user. The summary must include changed files, validation performed, commit SHA, push status, and any remaining unsynced or ignored local files that matter to the requested work.

## Validation

After Agent, Skill, context, or delegation-policy changes, run:

```bash
python scripts/check_gcs_agent_setup.py
```

For changed Python helper files, also run:

```bash
python -m py_compile <changed-python-files>
```

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

Default-off K56 fifth-candidate verifier experiment:

```text
model = ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12-k56-fifthness-v1.yaml
```

This opt-in YAML may additionally emit `pred_fifthness_logits: B x Q` and enables Count Head fifth-candidate evidence. The default K32 and K56 YAMLs must not emit `pred_fifthness_logits`.
The first `gcs_yolo_lane_s_q12_k56_fifthness_v1_ft8_seed1_b32w4` short gate is rejected: best official-val was epoch 5 `0.959006`, below the K56 parent `0.959315`, with worse FP/FN and high GT4-to-5 pressure. Do not start full/e180 training from that recipe.
The follow-up `gcs_yolo_lane_s_q12_k56_fifthness_gt5neg_ft8_seed1_b32w4` gate is also not promotable: best official-val was epoch 6 `0.959319`, only `+0.000004` over the parent, but FP worsened to `0.047429` and `rate_4_to_5` rose to `0.121212`. Do not treat this as a full-training candidate.
The fifthness verifier is now wired into inference/evaluation through default-off decode switches. It is ignored unless `gcs_use_fifthness_decode=True`, and when enabled it may only gate or re-rank the selected fifth lane and fifth-lane rescue candidates. Enabling fifthness decode against a model that does not emit `pred_fifthness_logits` must fail fast.
A 2026-06-15 closed-loop official-val sweep of the `fifthness_gt5neg` checkpoint with fifthness decode thresholds `0.30/0.50/0.70/0.85` is not promotable. Low thresholds keep ACC at `0.959319` but preserve worse FP/`rate_4_to_5`; high threshold `0.85` lowers FP and `rate_4_to_5` but drops ACC to `0.959023` and worsens GT5 `5->4`. Do not start full/e180 from these decode settings.

The K56 labels must be regenerated from original TuSimple JSON and images, not resampled from existing K32 labels.

Current K56 official-val state:

```text
label oracle = 0.998256
baseline official_best = 0.959315 at epoch 152
best val-only min-points grid = 0.959750 with point_valid_thr=0.40, candidate_min_points=5, final_min_points=9, fifth_min_points=4
vs current-code K32 0.953756 = +0.005559
vs legacy 0.959224 = +0.000091
```

K56 is still experimental, not mainline-promoted, and has no final/promotable official-test claim. A user-requested diagnostic-only official-test audit was run on 2026-06-14 for the K56 parent, min-points, `cqcalib`, `curveaux`, and `lowfp_joint` rows; those numbers must not be used for checkpoint, threshold, postprocess, loss, or model selection. The rejected K56 gates are `gcs_yolo_lane_s_q12_k56_cqcalib_ft12_seed1_b32w4`, `gcs_yolo_lane_s_q12_k56_cqcalib_lr1e4_ft8_seed1_b32w4`, `gcs_yolo_lane_s_q12_k56_curveaux_ft8_seed1_b32w4`, `gcs_yolo_lane_s_q12_k56_lowfp_joint_ft8_seed1_b32w4`, `gcs_yolo_lane_s_q12_k56_countadj_lowmargin_ft8_seed1_b32w4`, and `gcs_yolo_lane_s_q12_k56_countff_supp_ft8_seed1_b32w4`; do not rerun these exact recipes as the next path. The `countff_supp` source-side switches have been removed; keep only its summaries and diagnostics as audit history.

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
gcs_geometry_curvature = 0.0
gcs_geometry_curvature_beta_px = 5.0
gcs_quality_pairwise = 0.0
gcs_quality_pairwise_margin = 0.2
gcs_fifthness = 0.0
gcs_fifthness_pairwise = 0.0
gcs_fifthness_margin = 0.2
gcs_fifthness_negative_topk = 2
gcs_fifthness_negative_score_thr = 0.1
gcs_fifthness_include_gt5_negatives = False
gcs_count_cumulative = 0.0
gcs_count_cumulative_label_smoothing = 0.0
```

Current default-off fifthness decode knobs:

```text
gcs_use_fifthness_decode = False
gcs_fifthness_decode_thr = 0.0
gcs_fifthness_decode_rank_weight = 1.0
```

`gcs_quality_gt5_edge_floor` is training-side only. When enabled above `0.0`, it floors the matched Quality Head target for real left/right edge lanes in GT5 images only; it does not change decode, use GT during inference, or fabricate lanes.

`gcs_fifthness*`, `gcs_quality_pairwise*`, and `gcs_count_cumulative*` are default-off training-side K56 fifth-candidate calibration candidates. Fifthness positives are GT5 edge matched lanes, negatives are competitive unmatched outside candidates, Quality pairwise ranks GT5 edge matches over competitive false fifth candidates, and cumulative count supervision uses the existing `pred_count_logits: B x 4` without replacing the Count Head output contract. By default, fifthness negatives are mined from GT3/GT4 images; `gcs_fifthness_include_gt5_negatives` is a default-off follow-up switch that also mines GT5 same-image unmatched outside candidates. These training-side knobs do not by themselves change decode, use GT during inference, fabricate lanes, or alter official metrics.

When `gcs_quality_hard_negative_from_head` is enabled, Quality Head hard negatives must be mined from unmatched queries only. Hungarian-matched queries remain matched quality targets even when their current continuous quality target is `0.0`.

The visible-segment hard-negative and GT5 edge Quality floor knobs remain default-off after the 2026-06-13 `gcs_yolo_lane_s_q12_gt5segq_vishn_countvis_ft12_seed1_b8w0` and `gcs_yolo_lane_s_q12_quality_gt5edgefloor_ft12_seed1_b8w0` official-val gates; neither recipe is promoted to mainline defaults.

`gcs_soft_count_decision`, `gcs_last_lane_rescue`, and `gcs_edge_last_lane_rescue` stay default-off unless official-val selection promotes them.

Current default loss items:

```text
exist_loss
point_loss
point_valid_loss
line_iou_loss
curvature_loss
count_cls_loss
count_sum_loss
quality_loss
```

The 8-loss setup is the default logging contract. `curvature_loss` is default-off unless
`gcs_geometry_curvature > 0.0`; it is training-side only and targets fixed-y GT5 edge-lane
geometry without changing decode, official metrics, or inference GT usage.

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

Use official-val for checkpoint, threshold, postprocess, and parameter selection. Use test only for one-shot final evaluation of a selected candidate, and require `tools/eval_tusimple_official.py --split test` to carry an official-val `--selection-summary` unless the run is explicitly marked `--diagnostic-only-test`.

Do not tune on test, use GT during inference or decode, fabricate lanes, silently change official metrics, or claim improvement without official-val evidence.
Selection and diagnostic tools must not accept TuSimple test GT through `--split val --gt-json ...`; keep the explicit GT-source and `test_set` path guards enabled when changing official-val tooling.

Old or removed mechanisms may return as controlled experimental candidates when explicit, configurable, traceable, and validated under the same protocol.

When a controlled experimental direction has completed its planned same-protocol official-val gate and is rejected as not useful, remove its active source path so it cannot affect later work. Delete the failed direction's source-side switches, CLI args, config defaults, model YAMLs, loss/decode branches, runnable command templates, and dedicated tests unless they are still needed for an incomplete isolation gate, a diagnostic tool, or a clearly planned follow-up. Keep concise documentation records, official-val evidence, experiment summaries, and diagnostics as audit history. Large rejected-run checkpoint artifacts should be deleted or archived after summaries and diagnostics are preserved. Previously removed mechanisms may return only as a fresh controlled candidate with an explicit hypothesis and official-val validation.

## Git Sync Policy

`AGENTS.md` and `README.md` are part of the project source and must be tracked and synchronized with the GitHub branch. Do not treat them as local-only files when they contain project workflow, environment, contract, research-policy, or handoff-summary changes.

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

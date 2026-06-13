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
gcs_count_boundary_gt5_pos_weight = 1.15
gcs_point_valid_gt5_pos_weight = 2.0
gcs_gt5_edge_loss_weight = 1.15
gcs_candidate_gt5_edge_weight = 1.10
gcs_point_valid_gt5_edge_continuity = 0.05
gcs_point_valid_gt5_edge_continuity_thr = 0.55
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
- rely on decode-side rescue instead of improving real matched query candidates

Tradeoff:

The conservative defaults reduce GT5-specific over-concentration while preserving Count Head supervision. Count Boundary remains part of the active model/decode path, so the output contract must mention it explicitly. The new training-side candidate-quality defaults concentrate a small amount of extra gradient on real matched GT5 edge queries/lanes and adjacent visible point-valid anchors. `gcs_candidate_gt5_edge_weight` is matched edge-query/lane weighting, not per-anchor positive-target-only weighting. This can improve candidate quality only if retraining confirms it on official-val; it may also over-weight rare edge lanes if the defaults are too high.

Validation evidence:

Latest local validation for the training-side GT5 candidate-quality update:

```text
python -m py_compile ultralytics/utils/gcs_loss.py ultralytics/models/yolo/gcs_lane/train.py tools/train_gcs.py tests/test_gcs_count_aware.py
python -m pytest tests/test_gcs_count_aware.py -q
python scripts/verify_loss_cleanup.py
python scripts/check_gcs_agent_setup.py
```

No official-val result is available yet for the new training-side knobs.

Previous local contract validation for the Count Boundary/count-aware mainline:

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

Each archive commit must include a concise note based on the completed work. After pushing the archive commit, the assistant must give the user a work summary covering changed files, validation performed, commit SHA, and GitHub sync status.

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
Per-archive notes make the rollback history understandable, and the post-archive summary gives the user immediate confirmation of what was saved.

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

---

## Decision: Rank lane candidates by visible-segment quality instead of all-anchor mean valid

Status: current decode default, requires fresh training confirmation before any 0.97-range improvement claim

Decision:

Use the longest contiguous visible segment to compute rank visibility:

```text
rank_score = exist_score * visible_segment_mean_valid * min(1, visible_segment_points / 12)
```

Keep the all-anchor mean as diagnostic metadata:

```text
mean_valid_score_all
```

`mean_valid_score` now means the visible-segment mean used by rank and quality-gated rescue.

Why:

The fixed-y TuSimple label has `K=32`, but the fifth edge lane in GT5 images is often visible for only about 5-7 anchors. The old all-anchor mean divided high-confidence visible anchors by all 32 anchors, so a real short edge lane could receive a rank score near zero even when its visible segment was reliable.

Official-val diagnosis of:

```text
runs/gcs_lane/gcs_yolo_lane_s_q12_cb_gt45_ft6_officialtopk_seed2_b8w0
```

showed this structural failure:

```text
old official_best.pt: official_acc=0.952530
gt5_output5_rate=0.000000
count_acc_5=0.000000
gt5_rank5_score_low_rate=0.648649
rank_score_low=48/74 GT5 images
s5 mean=0.005515
```

With visible-segment rank on the same checkpoint and official-val split:

```text
official_acc=0.954186
gt5_output5_rate=0.594595
count_acc_5=0.594595
gt5_rank5_score_low_rate=0.013514
rank_score_low=1/74 GT5 images
s5 mean=0.214644
rescue_precision=0.891304
```

Alternatives considered:

- continue threshold, NMS, and rescue sweeps
- keep the old all-anchor mean and lower `s5_low_thr`
- add a default-off rank mode
- train only, leaving decode rank unchanged

Tradeoff:

The new rank removes a root bias against short visible lanes, but it also exposes the next bottleneck: Quality Head and Count Head must distinguish true fifth lanes from false fifth candidates. On the older strong FT6 `last.pt`, visible-segment rank reached `gt5_output5_rate=0.743243` but official-val Accuracy was `0.954474`, below the old selected row `0.954782`, because FP/GT4-to-5 cost rose. Raising the quality gate to `0.75` improved rescue precision but reduced GT5 recall and did not improve Accuracy. Therefore the next progress must come from training-side quality/count separation, not from simply tightening decode thresholds.

Validation evidence:

Local checks passed:

```text
python -m py_compile ultralytics/utils/gcs_postprocess.py tools/diagnose_gcs_gt5.py ultralytics/utils/gcs_loss.py tools/train_gcs.py tools/infer_gcs.py tools/check_gcs_count_head_topk_contract.py tools/check_gcs_decode_meta_contract.py tools/check_gcs_algorithm_contract.py tests/test_gcs_count_aware.py
python tools/check_gcs_count_head_topk_contract.py
python tools/check_gcs_decode_meta_contract.py
python tools/check_gcs_algorithm_contract.py
python -m pytest tests/test_gcs_count_aware.py -q
python scripts/verify_loss_cleanup.py
python scripts/check_gcs_agent_setup.py
```

Official-val evidence, not test:

```text
runs/gcs_lane/gcs_yolo_lane_s_q12_cb_gt45_ft6_officialtopk_seed2_b8w0/analysis_official_best_visible_segment_rank_val_sweep/tusimple_official_sweep_summary.json
runs/gcs_lane/gcs_yolo_lane_s_q12_cb_gt45_ft6_officialtopk_seed2_b8w0/analysis_official_best_visible_segment_rank_gt5_diag_val/gt5_rank_diagnostics_summary.json
runs/gcs_lane/gcs_yolo_lane_s_q12_e180_countboundary_rankfix_balgt45_v1/analysis_official_best_visible_segment_rank_val_sweep/tusimple_official_sweep_summary.json
runs/gcs_lane/gcs_yolo_lane_s_q12_cb_gt45_ft6_from_official_best_b8w0_v1/analysis_last_visible_segment_rank_val_sweep/tusimple_official_sweep_summary.json
```

Mainline or experiment:

Decode default. Improvement claims still require retraining and official-val selection; test remains final-only.

---

## Decision: Treat the 2026-06-12 Count Boundary GT4/GT5 fine-tune as a promising experiment, not a mainline promotion

Status: experimental candidate, requires longer official-val confirmation

Decision:

Do not promote `gcs_soft_count_decision` or threshold/NMS/rescue sweeps from the 2026-06-12 analysis.

Keep the short Count Boundary / GT4-GT5 fine-tune result as the next experimental direction. The strongest local candidate is:

```text
runs/gcs_lane/gcs_yolo_lane_s_q12_cb_gt45_ft6_from_official_best_b8w0_v1/weights/last.pt
```

Its selected official-val decode row used:

```text
conf = 0.005 through 0.15 tied
point_valid_thr = 0.15
nms_dist_px = 18
max_det = 5
min_points = 6
rank_min_points = 5:4
```

The next smallest safe action is to rerun this direction for a longer controlled fine-tune with official-val checkpoint/top-k preservation, using official-val only for selection.

Why:

The original run `gcs_yolo_lane_s_q12_e180_countboundary_rankfix_balgt45_v1` showed that threshold, NMS, and rescue were not the main bottlenecks. Candidate-pool shortfall, GT5 valid-points failure, and GT5 NMS suppression were near zero in the selected official-val rows, while Count Head / Count Boundary GT4/GT5 confusion remained visible.

Soft-count decode was tested directly and did not improve official-val:

```text
official_best.pt baseline:      official_acc=0.954137
official_best.pt soft-count:    official_acc=0.953481
best.pt baseline:               official_acc=0.953319
best.pt soft-count:             official_acc=0.953319, with higher FP
```

The short Count Boundary / GT4-GT5 fine-tune improved the official-val selected row:

```text
original official_best.pt: official_acc=0.954137, FP=0.051377, FN=0.036272,
                           count_acc_4=0.818182, count_acc_5=0.864865,
                           gt5_count_head_under_rate=0.135135

ft6 last.pt:              official_acc=0.954782, FP=0.047658, FN=0.034665,
                           count_acc_4=0.818182, count_acc_5=0.891892,
                           gt5_count_head_under_rate=0.094595
```

Alternatives considered:

- Continue threshold, NMS, or rescue sweeps.
- Promote `gcs_soft_count_decision`.
- Promote the 6-epoch fine-tune immediately.
- Revisit training-side Count Boundary / GT4-GT5 weighting with longer official-val checkpoint selection.

Tradeoff:

The 6-epoch result is useful but not stable enough for mainline promotion. The same short fine-tune run's ordinary `best.pt` regressed GT5 behavior:

```text
ft6 best.pt: official_acc=0.953056, count_acc_5=0.797297,
             gt5_count_head_under_rate=0.189189
```

This means ordinary validation fitness is not sufficient for this experiment family. Future runs need official-val checkpoint preservation or top-k official-val candidates, and test must remain reserved for one-shot final confirmation after candidate selection.

Validation evidence:

Official-val sweeps were run with `--imgsz 544 960` on the 363-image official-val subset:

```text
runs/gcs_lane/gcs_yolo_lane_s_q12_e180_countboundary_rankfix_balgt45_v1/analysis_official_best_val_sweep/tusimple_official_sweep_summary.json
runs/gcs_lane/gcs_yolo_lane_s_q12_e180_countboundary_rankfix_balgt45_v1/analysis_official_best_softcount_val_sweep/tusimple_official_sweep_summary.json
runs/gcs_lane/gcs_yolo_lane_s_q12_e180_countboundary_rankfix_balgt45_v1/analysis_best_val_sweep/tusimple_official_sweep_summary.json
runs/gcs_lane/gcs_yolo_lane_s_q12_e180_countboundary_rankfix_balgt45_v1/analysis_best_softcount_val_sweep/tusimple_official_sweep_summary.json
runs/gcs_lane/gcs_yolo_lane_s_q12_cb_gt45_ft6_from_official_best_b8w0_v1/analysis_best_val_sweep/tusimple_official_sweep_summary.json
runs/gcs_lane/gcs_yolo_lane_s_q12_cb_gt45_ft6_from_official_best_b8w0_v1/analysis_last_val_sweep/tusimple_official_sweep_summary.json
```

One failed training attempt is also part of the experiment record: a 20-epoch batch-32 run failed on the local Windows environment with CUDA OOM followed by `WinError 1455`; the successful short run used `batch=8` and `workers=0`.

Mainline or experiment:

Experiment only. Current mainline defaults and contracts are unchanged.

---

## Decision: Do not promote the 2026-06-12 GT5 calibration mainline seed0 run

Status: experimental result, rejected for mainline promotion

Decision:

Do not promote:

```text
runs/gcs_lane/gcs_yolo_lane_s_q12_e180_gt5calib_mainline_seed0
```

as a mainline improvement. Keep its artifacts as diagnostic evidence that the current GT5-calibration mainline recipe can improve some GT5 output behavior but does not improve clean official-val Accuracy.

The official-val selected checkpoint for this run remains:

```text
runs/gcs_lane/gcs_yolo_lane_s_q12_e180_gt5calib_mainline_seed0/weights/official_best.pt
```

but it is not better than the prior countboundary baseline or the short GT4/GT5 fine-tune result.

Why:

The run was evaluated with the same protected protocol: threshold/rank selection on official-val only, and test used only afterward as confirmation/diagnostic evidence.

Official-val results:

```text
best.pt:          official_acc=0.951660, FP=0.061111, FN=0.047521,
                  conf=0.15, point_valid_thr=0.15, rank_min_points=none,
                  count_acc_4=0.787879, count_acc_5=0.932432,
                  gt5_count_head_under_rate=0.027027,
                  gt5_valid_points_fail_rate=0.040541

official_best.pt: official_acc=0.953507, FP=0.048714, FN=0.037649,
                  conf=0.005, point_valid_thr=0.15, rank_min_points=none,
                  count_acc_4=0.818182, count_acc_5=0.905405,
                  gt5_count_head_under_rate=0.054054,
                  gt5_valid_points_fail_rate=0.040541
```

This is below the prior local official-val references:

```text
gcs_yolo_lane_s_q12_e180_countboundary_rankfix_balgt45_v1 official_best.pt: official_acc=0.954137
gcs_yolo_lane_s_q12_cb_gt45_ft6_from_official_best_b8w0_v1 last.pt:          official_acc=0.954782
```

Test results were recorded only as diagnostic evidence:

```text
best.pt:          Accuracy=0.955855, FP=0.037946, FN=0.038761
official_best.pt: Accuracy=0.955466, FP=0.035436, FN=0.038342
```

The ordinary `best.pt` test Accuracy being slightly higher must not be used to choose the checkpoint, because it had worse official-val Accuracy and test must not participate in checkpoint or threshold selection.

GT5 diagnosis on official-val showed:

```text
official_best.pt: GT5 count-under=4/74, valid-points fail=3/74, rescue success=0
best.pt:          GT5 count-under=2/74, valid-points fail=2/74, rescue success=0
```

`best.pt` improved GT5 count/output but paid for it with worse GT4-to-5 and FP behavior. `official_best.pt` was more balanced, but still did not beat the previous official-val candidates. Candidate-pool shortfall and GT5 NMS suppression were not the main failure mode.

Alternatives considered:

- Promote `official_best.pt` because it is the run's official-val best.
- Select ordinary `best.pt` because its test Accuracy was slightly higher.
- Continue broad threshold, NMS, rank, or rescue sweeps.
- Abandon GT5 calibration entirely.
- Treat GT5 calibration as a useful direction only when combined with GT4/GT5 joint calibration and official-val checkpoint preservation.

Tradeoff:

Rejecting promotion avoids locking in a lower official-val result. The test comparison is useful for understanding split instability, but it cannot justify test-driven checkpoint selection. The useful signal is narrower: GT5-specific weighting can help fifth-lane output, but it must be constrained by GT4 false-to-5 behavior and overall FP/FN.

Validation evidence:

Official-val sweeps and GT5 diagnostics were generated under:

```text
runs/gcs_lane/gcs_yolo_lane_s_q12_e180_gt5calib_mainline_seed0/analysis_best_val_sweep/tusimple_official_sweep_summary.json
runs/gcs_lane/gcs_yolo_lane_s_q12_e180_gt5calib_mainline_seed0/analysis_official_best_val_sweep/tusimple_official_sweep_summary.json
runs/gcs_lane/gcs_yolo_lane_s_q12_e180_gt5calib_mainline_seed0/analysis_best_gt5_diag_val/gt5_rank_diagnostics_summary.json
runs/gcs_lane/gcs_yolo_lane_s_q12_e180_gt5calib_mainline_seed0/analysis_official_best_gt5_diag_val/gt5_rank_diagnostics_summary.json
```

Test summaries, diagnostic only:

```text
runs/gcs_lane/gcs_yolo_lane_s_q12_e180_gt5calib_mainline_seed0/analysis_best_test_official_from_val/tusimple_official_summary.json
runs/gcs_lane/gcs_yolo_lane_s_q12_e180_gt5calib_mainline_seed0/analysis_official_best_test_official_from_val/tusimple_official_summary.json
```

Next smallest safe action:

Run a controlled GT4/GT5 joint-calibration experiment from the stronger countboundary or short fine-tune checkpoint, with official-val checkpoint/top-k preservation. Do not spend the next iteration on broad threshold/NMS/rescue sweeps unless a new hypothesis changes those controls.

Mainline or experiment:

Experiment only. Current mainline defaults and contracts are unchanged.

---

## Decision: Preserve official-val Top-K checkpoints during unstable fine-tunes

Status: current workflow support, default-compatible

Decision:

Add `gcs_official_best_top_k` to training configuration. The default is:

```text
gcs_official_best_top_k = 1
```

so existing behavior is unchanged. When set above `1`, training still updates:

```text
weights/official_best.pt
```

strictly by official-val `official_acc`, and additionally preserves the retained Top-K official-val candidate checkpoints under:

```text
weights/official_topk/
```

with metadata recorded in:

```text
official_best_summary.json
```

Why:

The 2026-06-12 experiment evidence shows ordinary `best.pt`, `last.pt`, and `official_best.pt` can disagree in short GT4/GT5 fine-tunes. The promising 6-epoch Count Boundary / GT4-GT5 fine-tune improved official-val Accuracy on `last.pt`, while ordinary `best.pt` regressed GT5 output. The newest `gt5calib_mainline_seed0` run also showed a val/test ranking mismatch: ordinary `best.pt` had slightly higher diagnostic test Accuracy, but worse official-val Accuracy and worse FP/GT4-to-5 behavior.

The project objective is official Accuracy under official-val selection. Preserving official-val Top-K candidates gives the next fine-tune run more recoverable checkpoints without changing model architecture, loss, decode, official metric calculation, or test usage.

Alternatives considered:

- rely on ordinary `best.pt`
- rely on `last.pt`
- enable `save_period` epoch checkpoints and sweep them manually
- change checkpoint selection to `official_score`, GT5 diagnostics, or test Accuracy
- continue broad threshold, NMS, soft-count, or rescue sweeps

Tradeoff:

Top-K preservation uses extra disk space only when `gcs_official_best_top_k > 1`. It does not solve GT4/GT5 calibration by itself; it removes a checkpoint-selection failure mode so the next controlled fine-tune can be judged cleanly on official-val.

Validation evidence:

Local checks passed after the implementation:

```text
python -m py_compile tools/train_gcs.py ultralytics/models/yolo/gcs_lane/train.py ultralytics/cfg/__init__.py tests/test_gcs_boundary_decode_plumbing.py tests/test_gcs_count_aware.py tools/check_gcs_algorithm_contract.py
python -m pytest tests/test_gcs_boundary_decode_plumbing.py tests/test_gcs_count_aware.py -q
python scripts/verify_loss_cleanup.py
python tools/check_gcs_algorithm_contract.py
python tools/check_gcs_count_head_topk_contract.py
python tools/check_gcs_decode_meta_contract.py
python scripts/check_gcs_agent_setup.py
```

Mainline or experiment:

Workflow support. Default behavior is unchanged at Top-K `1`; use `--gcs-official-best-top-k 3` for the next controlled GT4/GT5 fine-tune.

---

## Decision: Add default-off GT5 segment-quality training candidate

Status: experimental candidate, default-off

Decision:

Add explicit training-side knobs:

```text
gcs_quality_hard_negative_from_head = False
gcs_point_valid_gt5_edge_segment = 0.0
gcs_point_valid_gt5_edge_segment_thr = 0.65
gcs_point_valid_gt5_edge_segment_min_points = 5
```

`gcs_quality_hard_negative_from_head` lets `quality_loss` mine unmatched hard negatives directly from high `pred_quality_logits`, instead of relying only on the existing `pred_logits.sigmoid() * mean_valid` hard-negative mask.

`gcs_point_valid_gt5_edge_segment` adds a soft longest-visible-segment support penalty for matched left/right GT5 edge lanes. It is separate from the existing adjacent-anchor continuity penalty and targets the observed `K=5` to final-output-4 failure mode.

Why:

The 2026-06-13 analysis of:

```text
runs/gcs_lane/gcs_yolo_lane_s_q12_cb_gt45_ft8_visrank_qhard_seed1_b8w0
```

showed the run is not promotable and does not solve the current bottleneck:

```text
best.pt official-val sweep:          official_acc=0.949357
official_best.pt official-val sweep: official_acc=0.953179
official_best.pt GT5 valid fail:     0.216216
official_best.pt k5_to_output4:      0.259259
official_best.pt gt5_output5_rate:   0.729730
official_best.pt GT5 candidate shortfall: 0.027027
official_best.pt GT5 NMS suppression:    0.013514
matched/unmatched quality mean:      0.864285 / 0.748277
```

The failure is therefore not mainly threshold, candidate-pool size, or NMS. It is fifth-lane valid support after Count Head K selection plus weak Quality Head false-candidate separation.

Alternatives considered:

- Continue threshold, point-valid, rank-min-points, NMS, or rescue sweeps.
- Promote the FT8 `official_best.pt` checkpoint.
- Increase GT5 count weighting only.
- Make Quality Head ranking or rescue stricter at decode time.
- Add a training-side candidate that improves the signals used by the existing visible-segment rank and quality-gated rescue.

Tradeoff:

The new knobs preserve mainline defaults and do not change decode, official metrics, GT usage at inference, or lane fabrication rules. They add focused training pressure only when explicitly enabled. The risk is over-suppressing rare real edge candidates if matcher misses them early, or overfitting the small GT5 subset if the segment gain is too high.

Validation evidence:

Metric artifacts:

```text
runs/gcs_lane/gcs_yolo_lane_s_q12_cb_gt45_ft8_visrank_qhard_seed1_b8w0/analysis_best_val_sweep/tusimple_official_sweep_summary.json
runs/gcs_lane/gcs_yolo_lane_s_q12_cb_gt45_ft8_visrank_qhard_seed1_b8w0/analysis_official_best_val_sweep/tusimple_official_sweep_summary.json
runs/gcs_lane/gcs_yolo_lane_s_q12_cb_gt45_ft8_visrank_qhard_seed1_b8w0/analysis_best_test_official_from_val/tusimple_official_summary.json
runs/gcs_lane/gcs_yolo_lane_s_q12_cb_gt45_ft8_visrank_qhard_seed1_b8w0/analysis_official_best_test_official_from_val/tusimple_official_summary.json
```

Test results are diagnostic/final-only and did not drive checkpoint or parameter selection:

```text
best.pt test Accuracy:          0.954517
official_best.pt test Accuracy: 0.955443
```

Local code validation passed:

```text
python -m py_compile ultralytics/utils/gcs_loss.py tools/train_gcs.py ultralytics/models/yolo/gcs_lane/train.py ultralytics/cfg/__init__.py tests/test_gcs_count_aware.py
python -m pytest tests/test_gcs_count_aware.py -q
python scripts/verify_loss_cleanup.py
python tools/check_gcs_count_head_topk_contract.py
python tools/check_gcs_decode_meta_contract.py
python tools/check_gcs_algorithm_contract.py
python tools/check_model.py --cfg ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12.yaml --imgsz 544 960
python scripts/check_gcs_agent_setup.py
```

Mainline or experiment:

Experiment only. Improvement claims require a fresh official-val run using the new knobs and official-best Top-K preservation. Test remains final-only after official-val selection.

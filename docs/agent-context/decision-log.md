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

Each sync should be a normal Git commit on the active published branch. Do not force-push or rewrite published history, so earlier algorithm states remain recoverable by commit SHA or `git revert`.

Each archive commit must include a concise note based on the completed work. After every pushed archive commit, the assistant must give the user a sync summary covering changed files, validation performed, commit SHA, GitHub push status, and any remaining unsynced or ignored local files that matter to the requested work.

The GitHub repository includes root project instructions, root project overview, and the allowed source/documentation folders:

```text
AGENTS.md
README.md
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
- track root `AGENTS.md` and `README.md` with the source so project instructions and overview stay synchronized

Tradeoff:

The local workspace is not a standard repository root, so synchronization must preserve the published-source boundary and avoid pushing generated or large local artifacts. Root `AGENTS.md` and `README.md` are now explicit exceptions because they carry active project instructions and handoff context for future agents and humans.

Validation evidence:

The initial GitHub publish was created from a clean mirror containing only the allowed folders, excluding `__pycache__` and `.pyc` files. On 2026-06-13, the workflow policy was updated to track root `AGENTS.md` and `README.md`, and to require a sync summary after every pushed archive commit.

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

---

## Decision: Restrict Quality Head hard-negative mining to unmatched queries

Status: current implementation, experimental-knob behavior fix

Decision:

When `gcs_quality_hard_negative_from_head` is enabled, Quality Head hard-negative mining uses the explicit Hungarian unmatched-query mask. A matched query remains part of the matched quality-target loss even if its current continuous `target_quality` is `0.0`.

Why:

The first GT5 segment-quality fine-tune:

```text
runs/gcs_lane/gcs_yolo_lane_s_q12_gt5segq_ft10_seed1_b8w0_v1
```

was not promotable. Its best independent official-val row was `last.pt`:

```text
official_acc=0.953994
official_fp=0.042424
official_fn=0.035354
count_acc_5=0.662162
gt5_output5_rate=0.662162
gt5_count_head_under_rate=0.148649
gt5_valid_points_fail_rate=0.189189
decode/k5_to_output4_rate=0.208955
```

This was only a small gain over the FT8 `official_best.pt` (`0.953179`) and remained below the prior `0.954137` countboundary baseline and `0.954782` FT6 reference. The candidate improved Quality Head separation (`matched/unmatched=0.870587/0.716851`), but it reduced GT5 output relative to the FT8 `official_best.pt` (`0.729730 -> 0.662162`) and pushed GT5 underprediction close to the diagnostic limit.

The implementation review found that `quality_loss` used `target_quality == 0.0` as the negative mask. Because continuous quality targets can be `0.0` for matched lanes with poor current geometry, this allowed real matched GT5 edge candidates to be reweighted as Quality Head hard negatives. That directly conflicts with the intended "unmatched hard negative" contract and can suppress rare real fifth-lane candidates early in fine-tuning.

Alternatives considered:

- Keep `target_quality == 0.0` as the negative definition and document that zero-quality matched lanes are also hard negatives.
- Reduce the hard-negative weight only.
- Disable `gcs_quality_hard_negative_from_head`.
- Keep the feature but use explicit matched/unmatched masks.

Tradeoff:

Matched lanes with very poor current geometry still receive BCE pressure toward their continuous quality target, including `0.0`; they are just not multiplied by unmatched hard-negative weights. This preserves false-candidate suppression while reducing the risk of training against real matched edge lanes.

Validation evidence:

Local checks after the fix:

```text
python -m py_compile ultralytics/utils/gcs_loss.py tests/test_gcs_count_aware.py
python -m pytest tests/test_gcs_count_aware.py -q
```

The test suite includes a K=32 fixed-y case with `fixed_y_start=710/720` to verify that matched zero-quality lanes are not mined as hard negatives, and that GT5 edge segment support applies only to GT5 left/right edge lanes.

Mainline or experiment:

Experimental-knob behavior fix. Mainline defaults remain unchanged because `gcs_quality_hard_negative_from_head=False` and `gcs_point_valid_gt5_edge_segment=0.0` by default.

---

## Decision: Use a dedicated remote Git clone and minimal TuSimple test archive for server evaluation

Status: current workflow policy

Decision:

For remote CUDA runs, use a dedicated Git clone of the published source as the working directory. If the server also has a non-Git project copy containing datasets, runs, checkpoints, or pretrained weights, keep it as a runtime artifact source and do not run `git pull` inside it or blindly overwrite it.

For final TuSimple official test evaluation on the server, prepare the original TuSimple test archive under the remote Git clone's `archive/` tree. A minimal test-only archive is valid when it contains:

```text
archive/TUSimple/test_label.json
archive/TUSimple/test_set/clips/<date>/<clip>/<frame>.jpg
archive/TUSimple/train_set/
```

where the image set contains exactly the frames referenced by `test_label.json`. `train_set/` may be an empty placeholder for test-only evaluation because `find_tusimple_archive_root()` validates the conventional TuSimple root shape.

Why:

The remote server may contain a useful non-Git training directory with local datasets and historical runs, but it is not safe to use that directory as the Git synchronization target. A separate Git clone keeps source updates reproducible while preserving local runtime artifacts.

`tools/eval_tusimple_official.py --split test` needs the original TuSimple `raw_file` images and official label JSON. The fixed-y converted dataset under `datasets/tusimple_fixed_y_960x544` is sufficient for training and custom GCS validation, but it is not the original TuSimple archive expected by the official metric path.

Alternatives considered:

- Convert the existing non-Git remote directory into a Git worktree.
- Selectively overwrite only folders that already exist in the non-Git directory.
- Upload the full TuSimple `test_set/clips` archive.
- Keep only fixed-y converted test images on the server.

Tradeoff:

The dedicated Git clone requires linking or copying runtime artifacts such as `datasets/`, `archive/`, and pretrained weights. The minimal TuSimple test archive is much smaller than the full raw test dump, but it is sufficient only for final test evaluation over the 2,782 labeled test records; it is not a replacement for the complete original dataset archive.

Validation evidence:

Remote full-dataset 1-epoch smoke training completed successfully from the dedicated Git clone:

```text
run name: codex_full1ep_20260613_170533
train images: 3263
steps: 816/816
exit code: 0
```

The remote minimal TuSimple test archive was verified after extraction:

```text
records: 2782
images: 2782
missing raw_file images: 0
test_set size: about 591M
```

Mainline or experiment:

Workflow and data-preparation policy. It does not change model behavior, training defaults, official metrics, or the rule that test is final-only after official-val candidate selection.

---

## Decision: Align Count Head candidate evidence with visible-segment decode semantics

Status: current implementation, requires server official-val training evidence before any improvement claim

Decision:

`CandidateAwareCountHead` now computes its primary per-query lane quality and top4/top5 cardinality evidence with:

```text
visible_lane_quality = exist_score * visible_segment_mean_valid * visible_support_score
visible_support_score = min(1, visible_segment_points / 12)
```

The visible segment is the longest contiguous run where point-valid probability is at least `0.5`. This same visible-lane quality drives top-query candidate selection inside the Count Head. The all-anchor valid mean is retained as an auxiliary aggregate feature, but it no longer defines the primary Count Head fifth-lane evidence.

Why:

Decode ranking had already moved from all-anchor mean valid to visible-segment rank because short TuSimple edge lanes can have only 5-7 reliable anchors out of `K=32`. A code review found the Count Head still used all-anchor `valid_mean` and `exist * valid_mean` for candidate evidence. That left a structural mismatch: a real short edge lane could survive decode ranking but still look weak to Count Head K selection.

The observed metric failures fit this mismatch: candidate-pool shortfall and NMS were low, while GT5 Count Head underprediction and `K=5 -> output4` failures remained material.

Alternatives considered:

- Keep Count Head on all-anchor valid mean and only tune thresholds.
- Add more decode rescue or soft-count policy.
- Add new Count Head feature dimensions for visible segment evidence.
- Replace same-width Count Head feature semantics while preserving checkpoint shape compatibility.

Tradeoff:

The implementation preserves Count Head module dimensions and output contracts, so existing checkpoints remain loadable. Because the semantic distribution of score features changes, old checkpoints should be fine-tuned or reselected under official-val before any metric claim. This is a root code alignment, not a proven ACC improvement by itself.

Validation evidence:

Local checks after the implementation:

```text
D:\miniconda3\envs\lsa_yolo\python.exe -m py_compile ultralytics/nn/modules/gcs_lane.py tests/test_gcs_count_aware.py tools/check_gcs_count_head_topk_contract.py
D:\miniconda3\envs\lsa_yolo\python.exe -m pytest tests/test_gcs_count_aware.py -q -p no:cacheprovider --basetemp .tmp_pytest\basetemp
D:\miniconda3\envs\lsa_yolo\python.exe scripts/verify_loss_cleanup.py
D:\miniconda3\envs\lsa_yolo\python.exe tools/check_gcs_count_head_topk_contract.py
D:\miniconda3\envs\lsa_yolo\python.exe tools/check_gcs_decode_meta_contract.py
D:\miniconda3\envs\lsa_yolo\python.exe tools/check_gcs_algorithm_contract.py
D:\miniconda3\envs\lsa_yolo\python.exe tools/check_model.py --cfg ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12.yaml --imgsz 544 960
D:\miniconda3\envs\lsa_yolo\python.exe scripts/check_gcs_agent_setup.py
```

The new unit/contract tests build a `K=32` short-edge fixture where six contiguous high-valid anchors are suppressed by the old all-anchor mean but remain count-visible through visible-segment evidence.

Mainline or experiment:

Current implementation. The first clean FT6 control after this change was not promotable; continue with a joint Count calibration gate rather than claiming this implementation as an ACC improvement.

---

## Decision: Reject the Count-visible FT6 control and add a default-off adjacent Count margin candidate

Status: experimental candidate, default-off

Decision:

Do not promote:

```text
gcs_yolo_lane_s_q12_cb_gt45_ft6_countvis_clean_seed1_b8w0
```

Add a default-off training-side adjacent Count Head margin inside the existing `count_cls_loss`:

```text
gcs_count_adjacent_margin = 0.2
gcs_count_adjacent_margin_gain = 0.0
gcs_count_adjacent_margin_gt45_weight = 1.0
```

When enabled, the margin penalizes neighboring count logits that are too close to or above the GT count logit. It does not change model outputs, decode, official metric calculation, GT usage during inference, or the seven logged loss items.

Why:

The remote clean FT6 control from commit `ec9cf5f47` reached best official-val:

```text
official_acc=0.953415
official_fp=0.045592
official_fn=0.037190
count_acc_3=0.950673
count_acc_4=0.848485
count_acc_5=0.824324
gt5_output5_rate=0.824324
gt5_count_head_under_rate=0.040541
gt5_valid_points_fail_rate=0.135135
```

This is below both official-val references:

```text
gcs_yolo_lane_s_q12_e180_countboundary_rankfix_balgt45_v1 official_best.pt: 0.954137
gcs_yolo_lane_s_q12_cb_gt45_ft6_from_official_best_b8w0_v1 last.pt:          0.954782
```

The selected-threshold GT5 diagnosis showed candidate supply and rank are no longer the dominant bottleneck:

```text
GT5 images: 74
kept: 61
count_head_under_predict: 3
valid_points_fail: 3
quality_too_low: 5
candidate_pool_shortfall: 2
gt5_rank5_score_low_rate: 0.0
valid_points_5 mean/median: 5.824324 / 6.0
```

The remaining count confusion still creates an FP/FN tradeoff:

```text
3->4: 11
4->3: 6
4->5: 4
5->4: 12
```

Alternatives considered:

- Promote the clean FT6 control because it recovered some GT5 availability.
- Run the mild segment-quality candidate immediately.
- Sweep thresholds, NMS, rescue, or soft-count decode.
- Add a new output head or change decode policy.
- Add a smaller default-off margin term to make adjacent count separation trainable under official-val selection.

Tradeoff:

The adjacent margin is a direct count-calibration pressure, but it is still an experimental knob. If too strong, it can over-separate neighboring counts and increase GT4/GT5 false transitions. Keeping it default-off preserves baseline reproducibility; using it only in the next gate keeps the result attributable.

Validation evidence:

Local checks after adding the knob:

```text
D:\miniconda3\envs\lsa_yolo\python.exe -m py_compile ultralytics/utils/gcs_loss.py tools/train_gcs.py ultralytics/cfg/__init__.py tests/test_gcs_count_aware.py
D:\miniconda3\envs\lsa_yolo\python.exe -m pytest tests/test_gcs_count_aware.py tests/test_gcs_boundary_decode_plumbing.py -q -p no:cacheprovider --basetemp .tmp_pytest\basetemp
D:\miniconda3\envs\lsa_yolo\python.exe scripts/verify_loss_cleanup.py
D:\miniconda3\envs\lsa_yolo\python.exe tools/check_gcs_count_head_topk_contract.py
D:\miniconda3\envs\lsa_yolo\python.exe tools/check_gcs_decode_meta_contract.py
D:\miniconda3\envs\lsa_yolo\python.exe tools/check_gcs_algorithm_contract.py
D:\miniconda3\envs\lsa_yolo\python.exe tools/check_model.py --cfg ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12.yaml --imgsz 544 960
D:\miniconda3\envs\lsa_yolo\python.exe scripts/check_gcs_agent_setup.py
```

Remote control audit artifacts:

```text
runs/gcs_lane/gcs_yolo_lane_s_q12_cb_gt45_ft6_countvis_clean_seed1_b8w0/args.yaml
runs/gcs_lane/gcs_yolo_lane_s_q12_cb_gt45_ft6_countvis_clean_seed1_b8w0/results.csv
runs/gcs_lane/gcs_yolo_lane_s_q12_cb_gt45_ft6_countvis_clean_seed1_b8w0/weights/official_best_summary.json
official-val sweep summary and GT5 diagnostic output under the same run analysis area
```

Next smallest safe action:

Run one remote gate experiment from the parent countboundary `official_best.pt` with joint Count calibration plus:

```text
gcs_count_adjacent_margin = 0.25
gcs_count_adjacent_margin_gain = 0.25
gcs_count_adjacent_margin_gt45_weight = 1.5
```

Use official-val Top-K preservation. Defer the mild segment-quality candidate until this gate shows whether the remaining official-val failures are still `quality_too_low`, `valid_points_fail`, or `min_points`.

Mainline or experiment:

Experiment only. No improvement claim is valid until the remote gate beats the official-val references under the protected protocol.

---

## Decision: Reject adjacent Count margin gate and add visible-segment hard-negative mining candidate

Status: experimental candidate, default-off

Decision:

Do not promote or continue the adjacent Count margin gate:

```text
gcs_yolo_lane_s_q12_jointcount_adjmargin_countvis_ft12_seed1_b8w0
```

Add a default-off training-side hard-negative mining option:

```text
gcs_hard_negative_visible_segment = False
gcs_hard_negative_visible_thr = 0.5
gcs_hard_negative_visible_support_points = 12.0
```

When enabled, the shared unmatched hard-negative mask uses:

```text
exist_score * visible_segment_mean_valid * min(1, visible_segment_points / support_points)
```

instead of `exist_score * all_anchor_mean_valid`. The mask remains unmatched-only, so Hungarian-matched queries are protected even when their current quality target is low.

Why:

The adjacent margin gate from commit `632634eb6` reached independent official-val:

```text
official_acc=0.953113
official_fp=0.041736
official_fn=0.035354
count_acc_3=0.946188
count_acc_4=0.818182
count_acc_5=0.675676
gt5_output5_rate=0.675676
gt5_count_head_under_rate=0.108108
gt5_valid_points_fail_rate=0.216216
```

This is below the active references:

```text
countboundary baseline official_best.pt: 0.954137
old FT6 reference:                     0.954782
clean count-visible FT6 control:       0.953415
```

The adjacent margin worsened GT5 `5->4` failure (`22/74`) and did not expose candidate-pool or NMS as the primary blocker:

```text
GT5 candidate_pool_shortfall_rate=0.027027
GT5 top5_suppressed_by_nms_rate=0.0
matched/unmatched quality mean=0.880241/0.771234
```

The remaining root issue is fifth-lane survival and false fifth-lane quality separation after Count/decode visible-segment alignment. Hard-negative mining still used all-anchor mean valid, so short visible false fifth-lane candidates could avoid the negative pressure that now matches Count/decode evidence.

Alternatives considered:

- Continue or strengthen adjacent Count margin.
- Return to broad threshold, NMS, rescue, or soft-count sweeps.
- Run the previous strong hard-negative recipe again.
- Add a visible-segment hard-negative option and keep matched real lanes protected.

Tradeoff:

Visible-segment hard-negative mining is more aligned with the current Count/decode evidence, but it can increase pressure on short edge-like false positives. The next gate therefore uses mild quality and GT5 edge-segment weights, explicit `gcs_hard_negative_quality_thr=0.40`, `gcs_hard_negative_topk=2`, official-best Top-K preservation, and official-val-only selection. It does not change decode, official metrics, GT usage during inference, or the default mainline.

Validation evidence:

Local checks after implementation:

```text
D:\miniconda3\envs\lsa_yolo\python.exe -m py_compile ultralytics/utils/gcs_loss.py ultralytics/models/yolo/gcs_lane/train.py tools/train_gcs.py ultralytics/cfg/__init__.py tests/test_gcs_count_aware.py
D:\miniconda3\envs\lsa_yolo\python.exe -m pytest tests/test_gcs_count_aware.py tests/test_gcs_boundary_decode_plumbing.py -q -p no:cacheprovider --basetemp .tmp_pytest\basetemp
D:\miniconda3\envs\lsa_yolo\python.exe scripts/verify_loss_cleanup.py
D:\miniconda3\envs\lsa_yolo\python.exe tools/check_gcs_count_head_topk_contract.py
D:\miniconda3\envs\lsa_yolo\python.exe tools/check_gcs_decode_meta_contract.py
D:\miniconda3\envs\lsa_yolo\python.exe tools/check_gcs_algorithm_contract.py
D:\miniconda3\envs\lsa_yolo\python.exe tools/check_model.py --cfg ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12.yaml --imgsz 544 960
D:\miniconda3\envs\lsa_yolo\python.exe scripts/check_gcs_agent_setup.py
```

The unit tests cover default-off unchanged behavior, short visible-segment unmatched mining, config/CLI/trainer plumbing, and matched-query protection under the intended remote recipe.

Mainline or experiment:

Experiment only. Improvement claims require a fresh remote official-val run. Test remains final-only after official-val selection.

---

## Decision: Reject visible-segment hard-negative remote gate

Status: rejected experiment, default-off infrastructure retained

Decision:

Do not promote:

```text
gcs_yolo_lane_s_q12_gt5segq_vishn_countvis_ft12_seed1_b8w0
```

Keep the code/config support from commit:

```text
4881bcebc16ff09a72d1f90458bc3a99296ea597
```

as default-off experimental infrastructure only:

```text
gcs_hard_negative_visible_segment = False
gcs_hard_negative_visible_thr = 0.5
gcs_hard_negative_visible_support_points = 12.0
```

Do not change mainline defaults, do not rerun the same recipe as the next gate, and do not use test to rescue or tune it.

Why:

The 12-epoch remote gate used `ssh_lane` on the remote CUDA server, trained from:

```text
runs/gcs_lane/gcs_yolo_lane_s_q12_e180_countboundary_rankfix_balgt45_v1/weights/official_best.pt
```

with visible-segment hard-negative mining, `gcs_quality_hard_negative_from_head`, mild GT5 edge segment support, `gcs_hard_negative_quality_thr=0.40`, and `gcs_hard_negative_topk=2`.

Training-time official-val Top-K selected epoch 5:

```text
official_acc=0.953639
official_fp=0.044674
official_fn=0.036272
```

The independent official-val resweep on `weights/official_best.pt` confirmed the same best row:

```text
official_acc=0.953639
count_acc_3=0.923767
count_acc_4=0.863636
count_acc_5=0.635135
gt5_output5_rate=0.635135
gt5_count_head_under_rate=0.121622
gt5_valid_points_fail_rate=0.243243
gt5_candidate_pool_shortfall_rate=0.0
gt5_top5_suppressed_by_nms_rate=0.013514
matched/unmatched quality mean=0.864663/0.726468
```

This does not beat the active official-val references:

```text
countboundary baseline official_best.pt: 0.954137
old FT6 reference:                     0.954782
```

It only slightly exceeds already rejected controls:

```text
clean count-visible FT6 control: 0.953415
adjacent Count margin gate:      0.953113
```

More importantly, it worsened the target GT5 behavior versus the clean count-visible control:

```text
clean count-visible GT5 output5: 0.824324
visible hard-negative GT5 output5: 0.635135
```

GT5 diagnosis on `official_best.pt` showed:

```text
GT5 images: 74
kept: 47
quality_too_low: 17
count_head_under_predict: 9
valid_points_fail: 1
candidate_pool_shortfall: 0
rank5_score_low: 0
GT5 NMS suppression rate: 0.013514
valid_points_5 mean/median: 5.621622 / 6.0
s5 mean/median: 0.218729 / 0.243419
```

The root blocker is therefore not candidate supply, rank, or NMS. It is quality-gated fifth-lane survival after Count Head K selection, with secondary Count underprediction. The hard-negative pressure creates usable matched/unmatched quality separation, but it suppresses or fails to preserve real short fifth-lane candidates enough to lose official-val.

Alternatives considered:

- Promote the recipe because it beats the two most recent rejected gates.
- Continue the same recipe for more epochs.
- Increase hard-negative or duplicate-negative pressure.
- Use decode/test sweeps to recover the official metric.
- Stop and document the result as non-promotable.

Tradeoff:

Keeping the default-off code preserves a useful controlled mechanism and tests, but the specific recipe should not become the next main path. The next idea needs a different hypothesis that preserves GT5 real-candidate quality while controlling false fifth lanes, rather than simply increasing unmatched hard-negative pressure.

Validation evidence:

Local validation before remote training:

```text
D:\miniconda3\envs\lsa_yolo\python.exe -m py_compile ultralytics/utils/gcs_loss.py ultralytics/models/yolo/gcs_lane/train.py tools/train_gcs.py ultralytics/cfg/__init__.py tests/test_gcs_count_aware.py
D:\miniconda3\envs\lsa_yolo\python.exe -m pytest tests/test_gcs_count_aware.py tests/test_gcs_boundary_decode_plumbing.py -q -p no:cacheprovider --basetemp .tmp_pytest\basetemp
D:\miniconda3\envs\lsa_yolo\python.exe scripts/verify_loss_cleanup.py
D:\miniconda3\envs\lsa_yolo\python.exe tools/check_gcs_count_head_topk_contract.py
D:\miniconda3\envs\lsa_yolo\python.exe tools/check_gcs_decode_meta_contract.py
D:\miniconda3\envs\lsa_yolo\python.exe tools/check_gcs_algorithm_contract.py
D:\miniconda3\envs\lsa_yolo\python.exe tools/check_model.py --cfg ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12.yaml --imgsz 544 960
D:\miniconda3\envs\lsa_yolo\python.exe scripts/check_gcs_agent_setup.py
```

Remote artifacts:

```text
runs/gcs_lane/gcs_yolo_lane_s_q12_gt5segq_vishn_countvis_ft12_seed1_b8w0/args.yaml
runs/gcs_lane/gcs_yolo_lane_s_q12_gt5segq_vishn_countvis_ft12_seed1_b8w0/results.csv
runs/gcs_lane/gcs_yolo_lane_s_q12_gt5segq_vishn_countvis_ft12_seed1_b8w0/official_best_summary.json
runs/gcs_lane/gcs_yolo_lane_s_q12_gt5segq_vishn_countvis_ft12_seed1_b8w0/weights/official_best.pt
runs/gcs_lane/gcs_yolo_lane_s_q12_gt5segq_vishn_countvis_ft12_seed1_b8w0/weights/official_topk/
runs/gcs_lane/gcs_yolo_lane_s_q12_gt5segq_vishn_countvis_ft12_seed1_b8w0/analysis_official_best_val_sweep/tusimple_official_sweep_summary.json
runs/gcs_lane/gcs_yolo_lane_s_q12_gt5segq_vishn_countvis_ft12_seed1_b8w0/analysis_official_best_gt5_diag_val/gt5_rank_diagnostics_summary.json
```

Mainline or experiment:

Rejected experiment. Infrastructure remains default-off. No test evidence was used.

## 2026-06-13: Add default-off GT5 edge Quality target floor candidate

Decision:

Add `gcs_quality_gt5_edge_floor` as a default-off training-side experimental knob. When the value is greater than `0.0`, matched Quality Head targets for real left/right edge lanes in GT5 images are clamped to at least that floor. Mainline defaults remain unchanged at `0.0`.

Why:

The current-code reliability audit found a baseline-protocol drift and a persistent GT5 fifth-lane survival bottleneck. The default-off rescue official-val sweep on:

```text
runs/gcs_lane/gcs_yolo_lane_s_q12_e180_countboundary_rankfix_balgt45_v1/weights/official_best.pt
```

selected:

```text
official_acc=0.953756
FP=0.046006
FN=0.036961
artifact: runs/gcs_lane/reliability_audit_20260613_baseline_current_default_val_sweep
```

This is below the archived countboundary summary (`0.954137`) and old FT6 reference (`0.954782`). The matching GT5 diagnosis at:

```text
runs/gcs_lane/reliability_audit_20260613_baseline_current_default_gt5_diag
```

showed:

```text
GT5 images=74
kept=49
quality_too_low=14
count_head_under_predict=7
valid_points_fail=3
candidate_pool_shortfall=1
gt5_rank5_score_low_rate=0.0
gt5_top5_suppressed_by_nms_rate=0.0
```

Candidate supply, visible-segment rank, and NMS are therefore not the active primary bottleneck. A floor on matched GT5 edge Quality targets directly tests whether true short edge lanes receive too-low continuous quality targets, while leaving decode and official evaluation untouched.

Alternatives considered:

- Continue the rejected visible-segment hard-negative recipe.
- Re-open threshold, NMS, rescue, or soft-count decode sweeps.
- Change Quality Head ranking or rescue gates at inference time.
- Add a default-off training-side floor for matched GT5 edge-lane quality targets only.

Tradeoff:

The floor may increase GT5 retention but can also raise false fifth-lane pressure if the trained Quality Head stops separating real and spurious fifth candidates. Keeping the knob default-off and testing it as a single controlled remote gate preserves baseline reproducibility and keeps attribution clean.

Validation evidence:

Local checks after implementation:

```text
D:\miniconda3\envs\lsa_yolo\python.exe -m py_compile ultralytics/utils/gcs_loss.py ultralytics/models/yolo/gcs_lane/train.py tools/train_gcs.py ultralytics/cfg/__init__.py tests/test_gcs_count_aware.py
D:\miniconda3\envs\lsa_yolo\python.exe -m pytest tests/test_gcs_count_aware.py -q --basetemp=.tmp_pytest\basetemp -o cache_dir=.tmp_pytest\cache
D:\miniconda3\envs\lsa_yolo\python.exe -m pytest tests/test_gcs_boundary_decode_plumbing.py -q
D:\miniconda3\envs\lsa_yolo\python.exe scripts/verify_loss_cleanup.py
D:\miniconda3\envs\lsa_yolo\python.exe tools/check_gcs_count_head_topk_contract.py
D:\miniconda3\envs\lsa_yolo\python.exe tools/check_gcs_decode_meta_contract.py
D:\miniconda3\envs\lsa_yolo\python.exe tools/check_gcs_algorithm_contract.py
D:\miniconda3\envs\lsa_yolo\python.exe tools/check_model.py --cfg ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12.yaml --imgsz 544 960
D:\miniconda3\envs\lsa_yolo\python.exe tools/check_dataset.py
D:\miniconda3\envs\lsa_yolo\python.exe tools/check_gcs_label_order_split.py
D:\miniconda3\envs\lsa_yolo\python.exe scripts/check_gcs_agent_setup.py
```

Mainline or experiment:

Experimental candidate. Default-off. No decode change, no official metric change, no test usage.

## 2026-06-13: Reject GT5 edge Quality target floor 0.65 gate

Decision:

Do not promote `gcs_quality_gt5_edge_floor=0.65`. Keep `gcs_quality_gt5_edge_floor` default-off at `0.0`.

Why:

The remote gate:

```text
gcs_yolo_lane_s_q12_quality_gt5edgefloor_ft12_seed1_b8w0
commit: 7adbf03a6
weights init: runs/gcs_lane/gcs_yolo_lane_s_q12_e180_countboundary_rankfix_balgt45_v1/weights/official_best.pt
```

completed 12 epochs on the remote CUDA server with official-best Top-K preservation. Training-time official_best selected epoch 12. Independent official-val resweep on `weights/official_best.pt` confirmed:

```text
official_acc=0.953587
FP=0.048990
FN=0.035583
count_acc_3/4/5=0.910314/0.803030/0.716216
gt5_output5_rate=0.716216
gt5_count_head_under_rate=0.027027
gt5_valid_points_fail_rate=0.256757
matched/unmatched quality mean=0.857180/0.703256
```

This does not beat the current-code audit baseline:

```text
official_acc=0.953756
FP=0.046006
FN=0.036961
gt5_output5_rate=0.662162
```

and remains below the archived countboundary baseline (`0.954137`) and old FT6 reference (`0.954782`).

GT5 diagnosis on 74 GT5 images showed:

```text
kept=53
quality_too_low=16
count_head_under_predict=2
valid_points_fail=2
candidate_pool_shortfall=1
gt5_candidate_pool_shortfall_rate=0.013514
gt5_rank5_score_low_rate=0.0
gt5_top5_suppressed_by_nms_rate=0.0
valid_points_5 mean/median=5.837838/6.0
s5 mean/median=0.203062/0.216318
```

The floor reduced Count Head underprediction and eventually recovered more GT5 output, but it did not solve quality-gated fifth-lane survival and introduced a worse FP tradeoff. The root bottleneck remains short fifth-lane quality/valid-point separation, not raw candidate supply, rank, NMS, or Count Head K alone.

Alternatives considered:

- Promote the gate because it slightly improved FN and GT5 output versus the current-code audit baseline.
- Try the same floor for more epochs.
- Sweep floor values immediately.
- Keep the code path default-off and treat this as a negative gate.

Tradeoff:

The negative result is useful because it isolates a training-side hypothesis without decode or metric changes. It does not justify changing mainline defaults. A lower floor or schedule could be a future controlled candidate, but it must be justified as a new experiment and compared against the same official-val baseline.

Validation evidence:

Local source validation before the remote run:

```text
D:\miniconda3\envs\lsa_yolo\python.exe -m py_compile ultralytics/utils/gcs_loss.py ultralytics/models/yolo/gcs_lane/train.py tools/train_gcs.py ultralytics/cfg/__init__.py tests/test_gcs_count_aware.py
D:\miniconda3\envs\lsa_yolo\python.exe -m pytest tests/test_gcs_count_aware.py -q --basetemp=.tmp_pytest\basetemp -o cache_dir=.tmp_pytest\cache
D:\miniconda3\envs\lsa_yolo\python.exe scripts/verify_loss_cleanup.py
D:\miniconda3\envs\lsa_yolo\python.exe tools/check_gcs_count_head_topk_contract.py
D:\miniconda3\envs\lsa_yolo\python.exe tools/check_gcs_decode_meta_contract.py
D:\miniconda3\envs\lsa_yolo\python.exe tools/check_gcs_algorithm_contract.py
D:\miniconda3\envs\lsa_yolo\python.exe tools/check_model.py --cfg ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12.yaml --imgsz 544 960
D:\miniconda3\envs\lsa_yolo\python.exe tools/check_gcs_label_order_split.py
D:\miniconda3\envs\lsa_yolo\python.exe scripts/check_gcs_agent_setup.py
```

Remote preflight:

```text
python -m py_compile ultralytics/utils/gcs_loss.py ultralytics/models/yolo/gcs_lane/train.py tools/train_gcs.py ultralytics/cfg/__init__.py tests/test_gcs_count_aware.py
python scripts/verify_loss_cleanup.py
python tools/check_gcs_count_head_topk_contract.py
python tools/check_gcs_decode_meta_contract.py
python tools/check_gcs_algorithm_contract.py
```

Remote artifacts:

```text
runs/gcs_lane/gcs_yolo_lane_s_q12_quality_gt5edgefloor_ft12_seed1_b8w0/args.yaml
runs/gcs_lane/gcs_yolo_lane_s_q12_quality_gt5edgefloor_ft12_seed1_b8w0/results.csv
runs/gcs_lane/gcs_yolo_lane_s_q12_quality_gt5edgefloor_ft12_seed1_b8w0/official_best_summary.json
runs/gcs_lane/gcs_yolo_lane_s_q12_quality_gt5edgefloor_ft12_seed1_b8w0/weights/official_best.pt
runs/gcs_lane/gcs_yolo_lane_s_q12_quality_gt5edgefloor_ft12_seed1_b8w0/weights/official_topk/
runs/gcs_lane/gcs_yolo_lane_s_q12_quality_gt5edgefloor_ft12_seed1_b8w0/analysis_official_best_val_sweep/tusimple_official_sweep_summary.json
runs/gcs_lane/gcs_yolo_lane_s_q12_quality_gt5edgefloor_ft12_seed1_b8w0/analysis_official_best_gt5_diag_val/gt5_rank_diagnostics_summary.json
```

Mainline or experiment:

Rejected experiment. Infrastructure remains default-off. No test evidence was used.

## 2026-06-13: Identify K32 fixed-y official-grid representation bottleneck

Decision:

For the `0.97` TuSimple official Accuracy objective, stop treating another `K=32` GT5 quality/count fine-tune as the next main path. Select a separate `Q12-K56` fixed-y candidate aligned exactly to the TuSimple official h-sample grid (`160..710 step 10` in official top-to-bottom order, stored as `710..160 step -10` in the model's bottom-to-top fixed-y order) as the smallest credible next experimental direction.

The current mainline contract remains unchanged:

```text
point_mode = fixed_y
fixed_y_start = 710 / 720
fixed_y_end = 0.25
K = 32
```

The `K=56` path is experimental only until implemented, locally checked, and selected on official-val.

Why:

The integrated `gcs_integrator` conclusion is that the current `K=32` label geometry is now the higher-level bottleneck for moving from the `0.95` range toward `0.97`.

The current `K=32` fixed-y official-val label oracle is:

```text
Accuracy=0.956249
FP=0
FN=0.003444
```

Current references are:

```text
current-code audit baseline: 0.953756
old FT6 reference:          0.954782
```

That leaves only about `0.0015` to `0.0025` practical headroom under the current label geometry, far short of `0.97`. The 2026-06-13 `K=32` GT5 gates confirm that more pressure on quality/count inside the same contract is not enough:

```text
gcs_yolo_lane_s_q12_gt5segq_vishn_countvis_ft12_seed1_b8w0:     0.953639
gcs_yolo_lane_s_q12_quality_gt5edgefloor_ft12_seed1_b8w0:       0.953587
```

Simulated label-oracle alternatives show that official h-sample alignment changes the ceiling:

```text
K=56 aligned to official h-samples 160..710 step 10, stored bottom-to-top as 710..160 step -10: Accuracy=0.998256
K=64 over 710->160:                         Accuracy=0.967817
```

This suggests that exact official-grid alignment matters more than simply increasing K.

Alternatives considered:

- Continue another `K=32` GT5 quality/count/rescue fine-tune.
- Sweep decode thresholds, NMS, soft-count, or rescue again.
- Promote legacy no-count-head or older near-`0.960` rows.
- Jump directly to remote training with a new K value.
- First implement and validate a separate `Q12-K56` official-h-sample-aligned candidate.

Tradeoff:

Changing from `K=32` to `K=56` changes the label contract, output shape, memory profile, and checkpoint compatibility. Old `K=32` checkpoints and old legacy rows are not promotion evidence for the new contract. The benefit is that the next experiment tests the measured representation ceiling instead of continuing low-headroom tuning inside `K=32`.

Validation evidence:

The decision is based on official-val oracle and audit evidence above, plus the rejected official-val gates. The exact oracle script, commit, split artifact, and label conversion path must be recorded with the `K=56` implementation before any promotion claim.

Next smallest safe action:

Implement the `Q12-K56` path as an explicit experiment with a separate data YAML and label root. Run local label-oracle, dataset, model-shape, and contract checks before selecting any remote training command. Use official-val only for checkpoint and parameter selection, preserve `official_best` Top-K, and keep test final-only.

Mainline or experiment:

Experimental candidate. The current mainline remains `Q12-K32`; no improvement is claimed without future official-val evidence.

## 2026-06-13: Implement Q12-K56 official h-sample branch and launch remote baseline

Decision:

Implement the `Q12-K56` official-h-sample-aligned path as an isolated experimental branch, with separate data and model configs:

```text
data/tusimple_gcs_fixed_y_k56_960x544.yaml
ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12-k56.yaml
```

The K56 label contract is:

```text
point_mode = fixed_y
fixed_y_start = 710 / 720 = 0.9861111111111112
fixed_y_end = 160 / 720 = 0.2222222222222222
K = 56
```

Labels are rebuilt from original TuSimple JSON and images into:

```text
datasets/tusimple_fixed_y_k56_960x544
```

They must not be resampled from existing K32 labels.

Why:

The prior K32 representation oracle (`Accuracy=0.956249`) leaves too little geometry headroom for the `0.97` objective. The rebuilt K56 official-grid labels align exactly to TuSimple official `h_samples=710..160 step -10` and raise the official-val label oracle to:

```text
Accuracy=0.998256
FN=0.001377
FP=-0.000689
images=363
```

Alternatives considered:

- Continue K32 Count/Quality fine-tuning.
- Increase K without exact official-grid alignment.
- Resample existing K32 labels into K56.
- Build K56 from original TuSimple JSON and keep it isolated from K32.

Tradeoff:

K56 changes output shape and memory profile:

```text
pred_points: B x 12 x 56 x 2
pred_valid_logits: B x 12 x 56
```

K32 checkpoints are not compatible promotion evidence for K56. Formal K56 training belongs on the remote RTX 4090 24GB server, not on the local RTX 4060 8GB workstation. The default formal remote starting point is `batch=32 workers=4`, with changes only after explicit OOM/stability or throughput evidence.

Validation evidence:

Local and remote K56 validation passed:

```text
python -m py_compile tools/rebuild_tusimple_fixed_y_k56_from_reference_split.py tools/check_tusimple_fixed_y_label_oracle.py tools/train_gcs.py tests/test_gcs_k56_contract.py
python -m pytest tests/test_gcs_k56_contract.py
python tools/check_tusimple_fixed_y_label_oracle.py --data data/tusimple_gcs_fixed_y_k56_960x544.yaml --label-split val --archive-root archive
python tools/check_model.py --cfg ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12-k56.yaml --imgsz 544 960 --batch 1
```

Remote baseline:

```text
run: gcs_yolo_lane_s_q12_k56_offhs_e180_seed1_b32w4
batch: 32
workers: 4
official_best: enabled
official_best_top_k: 5
```

An earlier conservative `batch=4 workers=4` smoke run was stopped after the user pointed out that the 24GB server GPU was underused. The `batch=32` run is the formal baseline path.

Mainline or experiment:

Experimental candidate. The K32 mainline remains current until K56 is selected on official-val. No test evidence has been used.

## 2026-06-14: Continue Q12-K56 b32 baseline after early official-val check

Decision:

Continue the remote K56 formal baseline:

```text
run: gcs_yolo_lane_s_q12_k56_offhs_e180_seed1_b32w4
batch: 32
workers: 4
epochs: 180
official_best_top_k: 5
```

Do not stop it, do not replace it with a new experiment yet, and do not use test to tune anything.

Why:

A read-only remote check at `2026-06-14 07:22 CST` found the run alive at epoch `6/180`, using about `17.7 GiB` on the RTX 4090 24GB server. The log scan found no OOM, NaN, traceback, runtime error, or shape error.

The current official-best row is early and weak:

```text
best_epoch: 5
official_acc: 0.904071
official_fp: 0.180624
official_fn: 0.154040
count_acc_3/4/5: 0.847534 / 0.909091 / 0.351351
gt5_output5_rate: 0.351351
gt5_count_head_under_rate: 0.0
gt5_valid_points_fail_rate: 0.648649
decode/k5_to_output4_rate: 0.733945
```

The ordinary validation row for epoch `6` showed continuing progress:

```text
val/precision: 0.866769
val/recall: 0.870769
val/f1: 0.868764
val/decode/count_head_k: 3.96694
val/decode/final_pred_lanes: 3.5978
val/decode/k5_to_output4_rate: 0.962406
```

This is too early to judge the full K56 baseline. It only says the representation branch trains without NaN/shape/runtime failure and that early GT5 valid-point survival is still weak.

Alternatives considered:

- Stop the run because epoch 5 official-val is far below legacy references.
- Launch a geometry or Count/Quality auxiliary experiment immediately.
- Continue the current healthy formal baseline until enough official-val epochs exist.

Tradeoff:

Continuing spends server time, but preserves the first clean K56 baseline under the new label contract. Starting a new experiment from an epoch-5 failure interpretation would risk chasing early-training noise before the baseline has matured.

Validation evidence:

Read-only checks confirmed the command, process, log health, `results.csv`, and `official_best_summary.json`. No test evidence was used.

Mainline or experiment:

Experimental baseline monitoring decision. K56 is still not promoted over K32, and no official-test claim is available.

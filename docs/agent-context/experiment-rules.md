# Experiment Rules

This branch is for the imported `5-25-3.zip` algorithm under the K56 TuSimple fixed-y contract.

## Integrity Rules

Do not:

- tune on test
- use GT during inference or decode
- fabricate lanes
- silently change the official metric
- hide contract changes
- compare runs under different protocols without saying so
- claim improvement without official-val evidence

## Selection Policy

Use official-val for threshold, checkpoint, postprocess, and run selection. Do not choose the final checkpoint only from `val/total_loss`, internal `val/f1`, or `best.pt` provenance unless that checkpoint is also selected by official-val evidence.

For formal TuSimple training, enable periodic training-time official-val selection. The training hook uses `tools/sweep_tusimple_official_cached.py` and writes:

```text
weights/official_best.pt
weights/official_best_sweep.json
weights/official_best_decode.yaml
official_sweeps/epoch*/prediction_cache/manifest.json
official_sweeps/epoch*/prediction_cache/predictions.pt
```

Checkpoint/decode selection priority:

1. prefer `strict_order_valid=true`
2. if tied, lower `ordered_slot_order_violations`
3. if tied, maximize `official_acc`
4. if tied, maximize `official_score`
5. if tied, lower `official_FP`
6. if tied, lower `official_FN`
7. if tied, higher `count_acc_4`
8. if tied, higher `count_acc`
9. if tied, higher `count_acc_5`

For ordered-slot training-time official-best candidates, use slot order with
`order_check=warn` and `uses_runtime_sort=false` so training records order
violations instead of stopping early. Final ordered-slot official evaluation
still uses `order_check=error`; runtime sorting is never a main-result path.

Use test only once for final evaluation of a candidate already selected on official-val.

## Branch-Specific Rule

Do not add later mainline Count/Quality/Survival/near-miss machinery to this
branch unless a future task explicitly changes the algorithm scope. The
2026-06-27 `official_best` hook is an explicit selection-protocol addition
only; it does not change the 5-25-3 algorithm body. The 2026-06-27
`gcs_hard_sampling` option is a separate user-requested, default-off
train-dataloader sampler only; it must not affect validation/test dataloaders,
labels, decode, or official metrics. As of 2026-09-08, active query `GCSLoss`
is the five-term loss contract; historical/default-off `count_boundary_loss`
and `gcs_spurious_neg` keys remain only for old run interpretation and command
compatibility unless a future task explicitly restores those loss terms.

Any future E3-lite spurious-negative restoration must initialize from the
intended validation-selected predecessor and define its own ablation contract.
Keep `gcs_hard_sampling` disabled for that experiment unless a future task
explicitly starts a separate ablation.

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

For formal TuSimple training, enable periodic training-time official-val selection. The training hook writes:

```text
weights/official_best.pt
weights/official_best_sweep.json
weights/official_best_decode.yaml
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

Do not add later mainline Count/Quality/Survival/near-miss machinery to this branch unless a future task explicitly changes the algorithm scope. The 2026-06-27 `official_best` hook is an explicit selection-protocol addition only; it does not change the 5-25-3 algorithm body. The 2026-06-27 `count_boundary_loss` is a separate user-requested, default-off loss option for GT3/GT4/GT5 adjacent count-score boundaries; it must be selected only by official-val evidence. The 2026-06-27 `gcs_hard_sampling` option is a separate user-requested, default-off train-dataloader sampler only; it must not affect validation/test dataloaders, labels, decode, or official metrics. The 2026-06-27 `gcs_spurious_neg` option is a separate user-requested, default-off E3-lite loss; it must not change data sampling, matcher logic, point/smooth/curve losses, decode, NMS, or official metrics.

The E3-lite spurious-negative experiment must initialize from the E1 count-boundary checkpoint, not from an E2 hard-sampling or count-aware top-k run. Keep `gcs_hard_sampling` disabled for this experiment unless a future task explicitly starts a separate ablation.

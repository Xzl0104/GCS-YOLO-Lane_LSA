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

1. maximize `official_acc`
2. if tied, maximize `official_score`
3. if tied, lower `official_FP`
4. if tied, lower `official_FN`
5. if tied, higher `count_acc_4`

Use test only once for final evaluation of a candidate already selected on official-val.

## Branch-Specific Rule

Do not add later mainline Count/Quality/Survival/near-miss machinery to this branch unless a future task explicitly changes the algorithm scope. The 2026-06-27 `official_best` hook is an explicit selection-protocol addition only; it does not change the 5-25-3 algorithm body. The 2026-06-27 `count_boundary_loss` is a separate user-requested, default-off loss option for GT3/GT4/GT5 adjacent count-score boundaries; it must be selected only by official-val evidence.

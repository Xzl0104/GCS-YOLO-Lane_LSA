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

Use official-val for threshold, checkpoint, postprocess, and run selection. Use test only once for final evaluation of a candidate already selected on official-val.

## Branch-Specific Rule

Do not add later mainline Count/Quality/Survival/near-miss/official-best machinery to this branch unless a future task explicitly changes the algorithm scope. This branch only adapts Q=12/K=56, fixed-y anchors, image size, and data/model defaults.

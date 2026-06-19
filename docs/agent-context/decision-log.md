# Decision Log

This file records decisions for branch `codex/5-25-3-k56`.

## 2026-06-19: Import 5-25-3 as a Separate K56 Branch

Decision:

Import the historical `5-25-3.zip` GCS-YOLO-Lane algorithm as branch `codex/5-25-3-k56` and adapt only the explicit TuSimple fixed-y contract:

```text
legacy archive: Q=8, K=32, fixed_y=[0.98, 0.25]
this branch:    Q=12, K=56, fixed_y=[710/720, 160/720]
```

Use:

```text
model = ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12-k56.yaml
data  = data/tusimple_gcs_fixed_y_k56_960x544.yaml
imgsz = 544 960
```

Why:

The user requested a separate branch for the `5-25-3.zip` algorithm while preserving the current K56/710 TuSimple contract.

Alternatives considered:

- Port later mainline Count/Quality/Survival/near-miss mechanisms into the branch.
- Keep the legacy Q8/K32 contract.
- Resample old K32 labels to K56.

Tradeoff:

The branch is intentionally narrower than current mainline. It preserves 5-25-3 algorithm behavior and does not include later mainline helpers, tests, or official-best machinery. K56 labels must be regenerated from original TuSimple JSON and images.

Validation evidence:

Local checks performed during branch setup:

```text
python -m py_compile <changed GCS Python files>
python tools/check_model.py --cfg ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12-k56.yaml --imgsz 544 960 --batch 1 --device cpu
YAML contract assertions for Q=12, K=56, fixed_y_start=710/720, fixed_y_end=160/720
fixed-y anchor assertion for 56 y pixels: 710,700,...,160
sample label split/order check against an external K56 dataset root
```

Mainline or experiment:

Separate historical algorithm branch, not a mainline promotion.

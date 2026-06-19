# AGENTS.md

## Language

- Respond in English unless the user explicitly asks for another language.
- Keep code, paths, command names, script names, model names, and parameter names unchanged.

## Project Identity

This branch is a separate GCS-YOLO-Lane algorithm branch imported from `5-25-3.zip`.

The branch keeps the historical 5-25-3 algorithm body and changes only the TuSimple fixed-y contract needed by the current request.

## Required Context

Before non-trivial work, read:

- `docs/agent-context/project-context.md`
- `docs/agent-context/current-contracts.md`
- `docs/agent-context/commands.md`
- `docs/agent-context/environment.md`

Historical or mainline notes must not override `docs/agent-context/current-contracts.md`.

## Hard Contract

For TuSimple, always use:

```bash
--imgsz 544 960
```

This is H,W order. Do not reverse it.

Default model:

```text
ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12-k56.yaml
```

Default data:

```text
data/tusimple_gcs_fixed_y_k56_960x544.yaml
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

## Branch Scope

Do not silently import later mainline mechanisms into this branch. In particular, do not add Count Head, Count Boundary, Quality Head, Survival Head, near-miss mining, official-best checkpoint preservation, or mainline K56 candidate scripts unless a future task explicitly asks for that algorithm change.

This branch does not track `datasets/`, `scripts/`, or `tests/` from the zip import. Datasets are runtime artifacts and must stay out of Git.

## Output Contract

The default K56 model must produce:

```text
pred_points: B x 12 x 56 x 2
pred_logits: B x 12
pred_valid_logits: B x 12 x 56
aux_mask_logits: B x 2 x H x W
aux_edge_logits: B x 1 x H x W
```

## Validation

Use the existing local Python environment for quick validation:

```text
D:\miniconda3\envs\lsa_yolo
```

Recommended local checks after code/config changes:

```bash
python -m py_compile <changed-python-files>
python tools/check_model.py --cfg ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12-k56.yaml --imgsz 544 960 --batch 1 --device cpu
python tools/check_gcs_label_order_split.py --dataset-root <path-to-tusimple_fixed_y_k56_960x544>
```

Formal training should run on the remote CUDA server, not on the local 8GB GPU, unless the user explicitly asks otherwise.

## Research Integrity

Use official-val for threshold, checkpoint, and postprocess selection. Use test only once for final evaluation of a selected candidate.

Do not tune on test, use GT during inference or decode, fabricate lanes, silently change official metrics, or claim improvement without official-val evidence.

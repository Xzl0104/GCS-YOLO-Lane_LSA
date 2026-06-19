# Known Bottlenecks

This file applies to branch `codex/5-25-3-k56`.

## Branch Scope

The branch imports the historical `5-25-3.zip` algorithm and changes only the TuSimple fixed-y contract to Q12/K56 with official h-sample anchors.

Do not read mainline Count Head, Count Boundary, Quality Head, Survival Head, near-miss, or official-best bottlenecks as active branch behavior. Those mechanisms are not part of this 5-25-3 branch.

## Data And Geometry

- The active fixed-y anchors must be `710, 700, 690, ..., 160` normalized by original height `720`.
- Do not mix old `fixed_y=[0.98,0.25]` or K32 labels with this branch.
- Do not resample historical K32 labels into K56. Regenerate K56 labels from original TuSimple JSON and images.
- Preserve `--imgsz 544 960` in H,W order.
- The converted fixed-y dataset alone is not the full original TuSimple archive; official TuSimple evaluation still needs original raw-file image resolution/path context.

## Validation

- Local validation can check parser defaults, YAML contracts, fixed-y anchors, model output shape, and sample labels.
- Formal training and official-val evaluation should run on the remote CUDA server.
- This branch does not include later mainline official-val helper scripts; if official-val is required, generate predictions with this branch and evaluate them from a compatible evaluation checkout.

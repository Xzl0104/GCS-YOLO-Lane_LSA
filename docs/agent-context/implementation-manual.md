# GCS-YOLO-Lane 超详细小白实现手册（历史归档）

This file archives the long historical implementation manual that previously lived in root `AGENTS.md`.

The active project instructions now live in root `AGENTS.md`, and active contracts are split across files in `docs/agent-context/`.

Use this archive only for historical background and implementation context. It must not override:

- `AGENTS.md`
- `docs/agent-context/current-contracts.md`
- `docs/agent-context/experiment-rules.md`
- `docs/agent-context/commands.md`

## Historical Scope

The original manual described how to implement GCS-YOLO-Lane from a beginner perspective, including YOLO11 modification, structured lane labels, TuSimple/CULane preparation, fixed-y labels, model checks, dataset checks, overfit checks, training, inference, evaluation, ablations, and paper-style result organization.

The current active project state is more advanced than that original beginner manual. Current work should follow the Q=12 fixed-y mainline, Count Head and Quality Head contracts, official-val/test separation, and the validation commands recorded in the active context files.

## Historical Principle

Historical notes are useful for understanding why the repository looks the way it does. They are not allowed to override current contracts unless the user explicitly asks to create a controlled experimental candidate.

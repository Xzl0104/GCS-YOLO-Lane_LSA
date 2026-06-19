# Multi-Agent Usage

This branch does not include project `.codex/`, `.agents/`, or repository wrapper scripts from current mainline. Use Codex runtime tools directly when the user explicitly asks for subagents or multi-agent work.

## Policy

- Prefer one writable owner in the worktree.
- Use read-only subagents for final review or focused exploration.
- Do not simulate subagent output if runtime delegation is unavailable.
- Track spawned agent ids, wait for results when needed, and close completed or failed agents before ending the turn.

## Scope

Subagents reviewing this branch should check:

- Q=12/K=56 fixed-y contract
- `--imgsz 544 960` H,W order
- K56 labels regenerated from original TuSimple JSON/images
- no accidental import of later mainline Count/Quality/Survival/near-miss/official-best mechanisms
- training commands that match this branch's available CLI

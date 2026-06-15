# Experiment Rules

This project is an algorithm research project. The objective is clean TuSimple official Accuracy under a reproducible protocol.

## Baseline And Candidates

The current Q=12 fixed-y model is the default baseline, not a permanent limitation.

Any module, loss, head, decoder rule, postprocess strategy, sampling strategy, training strategy, rescue mechanism, or previously removed component may be added or reintroduced as a controlled experimental candidate when it may improve official ACC.

Do not reject an idea only because:

- it appeared in an older experiment
- it was previously removed from the mainline
- it is not part of the current explicit loss contract
- it changes the current architecture
- it adds a new auxiliary branch
- it adds a new training signal
- it revisits old count, rank, validity, continuity, duplicate, boundary, or rescue mechanisms

## Candidate Requirements

Every experimental candidate must be:

- explicit
- configurable
- traceable
- documented as baseline or experimental
- validated with the smallest useful contract checks
- compared on official-val before any promotion claim

## Failed Candidate Cleanup

When a controlled candidate has completed its planned official-val gate and the evidence shows the direction is not useful, clean it out of the active source path instead of leaving dead switches that can influence later work.

Apply cleanup when all are true:

- the candidate has a completed, same-protocol official-val result
- the result fails the promotion objective, such as lower ACC or a worse FP/FN/GT4/GT5 tradeoff
- the code path is not part of the current baseline contract
- the code path is not still needed for an incomplete isolation gate, a diagnostic tool, or a clearly planned follow-up

Cleanup means:

- remove source-side switches, CLI args, config defaults, model YAMLs, loss/decode branches, and dedicated tests for that failed direction
- remove runnable command templates that could restart the rejected recipe
- keep concise documentation records in `current-contracts.md`, `known-bottlenecks.md`, `commands.md`, and/or `decision-log.md`
- keep experiment summaries and diagnostic artifacts needed to understand the negative result
- delete or archive large rejected-run checkpoint artifacts when summaries and diagnostics have already been preserved

Do not delete historical documentation, official-val evidence, or diagnostic tools merely because the first recipe failed. Previously removed mechanisms may return only as a new controlled candidate with an explicit hypothesis, fresh source changes, and official-val validation.

## Integrity Rules

Do not:

- tune on test
- use GT during inference or decode
- fabricate lanes
- silently change the official metric
- hide contract changes
- compare runs under different protocols without saying so
- merge into the mainline without clean official-val evidence
- claim improvement without official-val evidence or a clearly labeled diagnostic result

## Selection Policy

Use official-val for threshold, checkpoint, postprocess, rescue, ranking, and count-policy selection.

Use test only once for final evaluation of a candidate already selected on official-val.

## Recommended Experiment Flow

1. Define the hypothesis and touched contracts.
2. Add the candidate as an isolated explicit option.
3. Keep the current baseline reproducible.
4. Run targeted contract checks.
5. Run official-val sweep or focused official-val validation. For unstable short fine-tunes, preserve official-val candidates with `gcs_official_best_top_k > 1`.
6. Compare official ACC against the baseline.
7. Analyze FP, FN, GT4/GT5 confusion, output5 rate, rescue precision, candidate shortfall, and valid-points failure.
8. Promote only with official-val evidence.
9. If the candidate is rejected under the cleanup rule, remove its active source path and preserve the negative-result record.
10. Use test only for final confirmation.

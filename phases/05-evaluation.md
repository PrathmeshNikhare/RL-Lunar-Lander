# Phase 05 — Controlled Evaluation

## Objective
Compare policies on the same original task objective.

## Tasks
- Evaluate with reward wrappers disabled.
- Collect per-episode original return, length, success and crash indicators.
- Define success/crash rules from environment semantics and test them.
- Evaluate at least 100 episodes per configuration where practical.
- Use at least three training seeds where practical.
- Preserve raw episode-level data.
- Report mean and variability; state sample sizes and limitations.
- Compare hovering diagnostics separately from task success.

## Acceptance criteria
- No custom reward bonuses enter evaluation return.
- Every configuration has comparable evaluation records.
- Metrics are derived from raw episode data.
- Report distinguishes training shaped return from original evaluation return.
- Missing runs and deviations are explicit.

## Verification
Metric unit tests, evaluation smoke test, data schema validation.

## Review
Experiment Analyst + Test Engineer.

## Execution instructions

First produce a plan only. Do not edit until the user approves. Implement only
this phase. Run the specified checks, request the listed specialist review,
update progress files and create a handoff. Ask for approval before advancing.

## Required phase report

Record: files changed, commands run, test results, review findings, acceptance
criteria evidence, unresolved issues, Git checkpoint and next action.

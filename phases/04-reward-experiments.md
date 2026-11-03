# Phase 04 — Four Reward Experiments

## Objective
Implement and train the four planned reward configurations.

## Tasks
- A: original reward baseline.
- B: configurable small per-step time penalty.
- C: configurable upright-and-slow per-step bonus intended to probe reward hacking.
- D: potential-based shaping with documented Phi and terminal handling.
- Keep PPO settings and budgets comparable initially.
- Train using explicit seeds and separate output directories.
- Log original and shaped reward components independently.
- Save run metadata and model checkpoints.
- Inspect videos and diagnostics; do not presume the flawed reward creates hovering.

## Acceptance criteria
- All four configurations are selectable by configuration.
- Each run is reproducible from recorded metadata.
- Reward calculations have unit tests.
- At least one run per configuration completes, or failures are documented.
- Claims about reward hacking are backed by measured behaviour.

## Verification
Unit tests, configuration validation, short smoke training for each mode.
Full training is explicit and may take longer.

## Review
Reward Engineer + RL Engineer + Experiment Analyst.

## Execution instructions

First produce a plan only. Do not edit until the user approves. Implement only
this phase. Run the specified checks, request the listed specialist review,
update progress files and create a handoff. Ask for approval before advancing.

## Required phase report

Record: files changed, commands run, test results, review findings, acceptance
criteria evidence, unresolved issues, Git checkpoint and next action.

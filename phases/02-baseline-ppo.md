# Phase 02 — Baseline PPO

## Objective
Train and evaluate PPO using the unmodified environment reward.

## Tasks
- Implement configuration-driven PPO training.
- Use `MlpPolicy`, `[64, 64]` architecture and documented initial hyperparameters.
- Start with one million timesteps, configurable from CLI/config.
- Add TensorBoard logging and unique run directories.
- Save model and effective config.
- Implement independent deterministic evaluation.
- Record videos of selected evaluation episodes.

## Acceptance criteria
- Short train/save/load smoke test passes.
- Full training can be launched explicitly.
- Evaluation reports original environment return.
- Training logs and metadata are retained.
- A landing score >=200 is a target, not a guaranteed gate; if missed, document
  diagnostics and next controlled adjustment.

## Verification
Unit tests, short training smoke test, save/load test, evaluation schema test.
Do not run full training automatically during verification.

## Review
RL Engineer + Test Engineer + Architecture Reviewer.

## Execution instructions

First produce a plan only. Do not edit until the user approves. Implement only
this phase. Run the specified checks, request the listed specialist review,
update progress files and create a handoff. Ask for approval before advancing.

## Required phase report

Record: files changed, commands run, test results, review findings, acceptance
criteria evidence, unresolved issues, Git checkpoint and next action.

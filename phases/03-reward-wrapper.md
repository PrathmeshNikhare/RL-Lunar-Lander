# Phase 03 — Reward Wrapper

## Objective
Implement a reusable Gymnasium wrapper that adds configured reward components.

## Tasks
- Separate pure reward calculations from wrapper mechanics.
- Preserve original reward in info/logging.
- Expose each custom component separately.
- Ensure wrapper reset/step contract and info propagation are correct.
- Validate configuration and thresholds.
- Add deterministic unit tests with hand-calculated transitions.
- Add tests proving evaluation can instantiate the environment without shaping.

## Acceptance criteria
- Baseline mode is exactly equivalent to original reward.
- Components are independently observable.
- Invalid configs fail clearly.
- Wrapper passes short integration tests.
- No hidden Box2D internals are accessed.

## Verification
Run unit and wrapper integration tests.

## Review
Reward Engineer + Test Engineer + Architecture Reviewer.

## Execution instructions

First produce a plan only. Do not edit until the user approves. Implement only
this phase. Run the specified checks, request the listed specialist review,
update progress files and create a handoff. Ask for approval before advancing.

## Required phase report

Record: files changed, commands run, test results, review findings, acceptance
criteria evidence, unresolved issues, Git checkpoint and next action.

# Phase 01 — Environment Exploration

## Objective
Validate LunarLander-v3 and understand its observation, action and episode API.

## Tasks
- Build a minimal environment factory.
- Create the environment with human rendering for manual inspection.
- Run random actions with a bounded episode loop.
- Print observation shape, values, action space and reward.
- Handle `terminated` and `truncated` separately.
- Document observation ordering, units/scales and action meanings from official docs.
- Add a short seeded smoke test.

## Acceptance criteria
- LunarLander-v3 initializes and steps successfully.
- Observation has the expected eight components.
- Action space is discrete with four actions.
- Reset/step API is handled correctly.
- Random rollout can finish and reset safely.

## Verification
Run the random rollout manually and the short environment test.

## Review
RL Engineer + Test Engineer.

## Execution instructions

First produce a plan only. Do not edit until the user approves. Implement only
this phase. Run the specified checks, request the listed specialist review,
update progress files and create a handoff. Ask for approval before advancing.

## Required phase report

Record: files changed, commands run, test results, review findings, acceptance
criteria evidence, unresolved issues, Git checkpoint and next action.

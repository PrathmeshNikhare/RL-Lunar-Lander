# Phase 00 — Foundation and Local Setup

## Objective
Establish a clean, reproducible Windows 11 / Python 3.12 project environment
and validate dependency compatibility without implementing the RL application.

## Tasks
- Inspect repository and Git state.
- Confirm Python 3.12 availability and document PowerShell commands.
- Create `.venv` only after user approval.
- Install compatible Gymnasium Box2D extra, Stable-Baselines3, PyTorch,
  TensorBoard, Matplotlib and development/test tools.
- Verify imports and record exact package versions.
- Add dependency specification and setup instructions.
- Document Box2D/SWIG troubleshooting only if relevant to observed install issue.

## Acceptance criteria
- Python interpreter is 3.12.
- Imports for Gymnasium, Box2D, Stable-Baselines3 and Torch succeed.
- Dependency versions are recorded.
- `.venv` is ignored by Git.
- Setup steps are documented and reproducible.

## Verification
- `python --version`
- `python -m pip --version`
- import/version smoke test
- `git status --short`

## Review
Test Engineer + Documentation Reviewer.

## Execution instructions

First produce a plan only. Do not edit until the user approves. Implement only
this phase. Run the specified checks, request the listed specialist review,
update progress files and create a handoff. Ask for approval before advancing.

## Required phase report

Record: files changed, commands run, test results, review findings, acceptance
criteria evidence, unresolved issues, Git checkpoint and next action.

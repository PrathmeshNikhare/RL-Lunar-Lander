# Current Handoff

## State
- **Phase 00** (foundation) is complete (`70e3445`).
- **Phase 01** (environment exploration) is complete and committed locally as
  "Phase 01: environment exploration" (not pushed).
- **Phase 01 added:**
  - the package `lunar_lander_rl/` (`environment.py`, `rollout.py`);
  - `scripts/random_rollout.py`;
  - 37 passing tests in total;
  - `docs/environment-notes.md`, the semantic reference for Phases 03–05.
- **Not started:** reward wrappers, PPO and training.

Full record: `progress/handoffs/2026-11-03-phase-01.md`.

## Current phase
`02-baseline-ppo`: not started. Phases 00–01 are complete.

## Next safe action
1. Run `/start-phase 02-baseline-ppo` and present the plan only.
2. Wait for explicit user approval before editing.

## Environment notes
- Always use `.\.venv\Scripts\python.exe`. The machine's default `py`/`python`
  is 3.14.
- Do not run pytest with a blanket `-W error::DeprecationWarning`: Box2D's SWIG
  import crashes the interpreter (see D-013).
- The package is not installed. pytest uses `pythonpath = .`, and the scripts
  add the repository root to `sys.path` (D-015). Run commands from the
  repository root.
- Before designing rewards or metrics, read `docs/environment-notes.md`. Key
  points:
  - the terminal ±100 is *assigned*, not added;
  - "at rest" (+100) is not the same as "on the pad", and the pad is
    x ∈ [−0.2, 0.2];
  - leg flags are unreliable at termination;
  - `angle` is not wrapped;
  - there is no termination at the top of the screen or for height;
  - seeding also drives the terrain and per-step engine dispersion.

## Constraints
- Windows 11, PowerShell, Python 3.12.
- Intermediate portfolio scope.
- One phase at a time.
- The original environment reward is the evaluation ground truth.
- Do not invent results or claim reward hacking without evidence.
- Do not push to `origin` without user approval.

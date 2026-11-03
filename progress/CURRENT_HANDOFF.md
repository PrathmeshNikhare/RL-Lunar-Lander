# Current Handoff

## State
Phase 00 (foundation and local setup) is complete and committed locally as
"Phase 00: foundation and local setup" (not pushed). A Python 3.12 `.venv` exists
locally (git-ignored), and dependencies are recorded in `requirements*.txt`.
No RL application code, environment exploration or training has been started.

Full record: `progress/handoffs/2026-11-03-phase-00.md`.

## Current phase
`01-environment`: not started. Phase 00 is complete.

## Next safe action
1. Run `/start-phase 01-environment` and present the plan only.
2. Wait for explicit user approval before editing.

## Environment notes
- Always use `.\.venv\Scripts\python.exe`. The machine's default `py`/`python`
  is 3.14.
- Do not run pytest with a blanket `-W error::DeprecationWarning`: Box2D's SWIG
  import crashes the interpreter (see D-013).

## Constraints
- Windows 11, PowerShell, Python 3.12.
- Intermediate portfolio scope.
- One phase at a time.
- The original environment reward is the evaluation ground truth.
- Do not invent results or claim reward hacking without evidence.
- Do not push to `origin` without user approval.

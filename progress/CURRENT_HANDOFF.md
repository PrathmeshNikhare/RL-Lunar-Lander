# Current Handoff

## State
- **Phases 00–05** are complete. `origin/main` is at `e594275` (Phase 05
  3/3).
- **Phase 06** (visualization and analysis) is complete: `53079be` (1/2),
  `892252a` (2/3) and "Phase 06 (3/3)" (results, record, docs, progress),
  all local, not pushed. The user accepted it ("commit").
  - Experiment Analyst REVISE and Documentation Reviewer ACCEPT (after
    fixes); all required fixes are applied.

Full record: `progress/handoffs/2026-11-07-phase-06.md`.

## Current phase
`07-publication`: not started. Phases 00–06 are complete.

## Next safe action
1. Run `/start-phase 07-publication` and present the plan only.
2. Wait for explicit user approval before editing.

## Environment notes
- Always use `.\.venv\Scripts\python.exe`. The machine's default `py`/`python`
  is 3.14.
- Do not run pytest with a blanket `-W error::DeprecationWarning`: Box2D's SWIG
  import crashes the interpreter (see D-013).
- The package is not installed. pytest uses `pythonpath = .`, and the scripts
  add the repository root to `sys.path` (D-015). Run commands from the
  repository root.
- Never launch a training run from tests. A 1M-step run takes about 4–6
  minutes (CPU, 16 envs).
- Environment semantics are in `docs/environment-notes.md`. In particular,
  +100 means at rest (the Box2D body is asleep), which is not the same as
  "on the pad". Orientation-engine impulses wake the body.
- Metrics and outcome rules: `docs/07-metrics.md`. Figures, traces and clips:
  `docs/visualization.md`.
- Every results script refuses an existing output path. Write re-runs to a
  new path.
- In Git Bash, avoid `sed` with `\|`, which GNU sed reads as alternation.
  For multi-line or table edits, use a script file or the Edit tool.
- **Clips are not watched yet** (owner's choice). Describe them only by their
  data labels until they are.

## Constraints
- Windows 11, PowerShell, Python 3.12.
- Intermediate portfolio scope.
- One phase at a time.
- The original environment reward is the evaluation ground truth.
- Do not invent results or claim reward hacking without evidence.
- Do not push to `origin` without user approval.

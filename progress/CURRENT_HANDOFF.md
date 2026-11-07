# Current Handoff

## State
- **Phases 00–06** are complete. `origin/main` is at `31d02ca` (Phase 06
  3/3, pushed by the user).
- **Phase 07** (GitHub publication) is complete and accepted by the user.
  It is committed locally on `main` and not pushed.
  - Reviews: Documentation REVISE and Architecture REVISE (all required
    fixes applied); Test Engineer ACCEPT.

Full record: `progress/handoffs/2026-11-07-phase-07.md`.

## Current phase
None. All planned phases (00–07) are complete.

- **Post-release README metrics** (D-047, D-048): implemented, reviewed
  (two REVISE verdicts, fixes applied) and committed locally (`82eab58`,
  then "Post-release (2/2)"), not pushed. Record:
  `progress/handoffs/2026-11-07-post-release-metrics.md`.

## Next safe action
1. Push to `origin` only after separate, explicit approval.
2. Optional follow-ups (only if the user asks): view the Mermaid diagrams
   on GitHub, watch the 9 non-bonus clips, Architecture L1/L3/L6.

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

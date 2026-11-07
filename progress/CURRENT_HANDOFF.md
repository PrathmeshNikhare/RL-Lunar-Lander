# Current Handoff

## State
- **Phases 00–07** are complete. `origin/main` is at `e42672a`
  ("Post-release (2/2)"); everything through the post-release metrics work
  is pushed (checked with `git fetch` on 2026-11-07).
- **Phase 07** (GitHub publication) was accepted by the user.
  - Reviews: Documentation REVISE and Architecture REVISE (all required
    fixes applied); Test Engineer ACCEPT.

Full record: `progress/handoffs/2026-11-07-phase-07.md`.

## Current phase
None. All planned phases (00–07) are complete.

- **Post-release README metrics** (D-047, D-048): pushed. Record:
  `progress/handoffs/2026-11-07-post-release-metrics.md`.
- **Repository polish** (D-049): on branch
  `chore/github-repository-polish`, not pushed. Record:
  `progress/handoffs/2026-11-07-github-polish.md`.

## Next safe action
1. Push `chore/github-repository-polish` and open a PR only after
   separate, explicit approval. The new CI workflow first runs on that
   PR; add a CI badge only after it passes.
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

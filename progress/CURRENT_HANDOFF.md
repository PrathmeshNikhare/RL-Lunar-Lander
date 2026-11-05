# Current Handoff

## State
- **Phases 00–04** are complete. Phase 04 is `74f4f36` and `153eccc`.
  Nothing is pushed.
- **Phase 05** (controlled evaluation) is complete: `e481fdf`, `6bb2bb6`,
  and "Phase 05 (3/3)" (local, not pushed).
  - Code is committed in `e481fdf` (1/2) and `6bb2bb6` (2/3).
  - The 12 runs were re-evaluated with trajectories from the clean
    `e481fdf`; every `episodes.csv` is identical to Phase 04.
  - The comparison ran from the clean `6bb2bb6`, writing to
    `results/phase05/`.
  - Experiment Analyst and Test Engineer reviews: REVISE. All required
    fixes are applied (see the handoff).

Full record: `progress/handoffs/2026-11-05-phase-05.md`.

## Current phase
`06-visualization`: not started. Phases 00–05 are complete.

## Next safe action
1. Run `/start-phase 06-visualization` and present the plan only.
2. Wait for explicit user approval before editing.

## Environment notes
- Always use `.\.venv\Scripts\python.exe`. The machine's default `py`/`python`
  is 3.14.
- Do not run pytest with a blanket `-W error::DeprecationWarning`: Box2D's SWIG
  import crashes the interpreter (see D-013).
- The package is not installed. pytest uses `pythonpath = .`, and the scripts
  add the repository root to `sys.path` (D-015). Run commands from the
  repository root.
- A 1M-step baseline run takes about 4–6 minutes on this machine (CPU,
  16 envs). Launch it in the background and never from tests.
- Before designing rewards or metrics, read `docs/environment-notes.md`. Key
  points:
  - the terminal ±100 is *assigned*, not added;
  - "at rest" (+100) is not the same as "on the pad", and the pad is
    x ∈ [−0.2, 0.2];
  - leg flags are unreliable at termination;
  - `angle` is not wrapped;
  - there is no termination at the top of the screen or for height;
  - seeding also drives the terrain and per-step engine dispersion.
- Reward modes, formulas, terminal handling and logging:
  `docs/reward-wrapper.md`. γ for shaping is always `ppo.gamma` (D-019).
- Run selection: by `metadata.json` (`lunar_lander_rl.comparison`), never
  by folder glob. The `model.zip` hash is not a policy fingerprint.
- Metrics and outcome rules: `docs/07-metrics.md`. Per-step data is in each
  run's newest `evaluations/*/trajectories.npz` (local only).
- `comparison.compare` refuses an existing output folder. Write new results
  to a new folder.
- For Phase 06: plot from `results/phase05/`. Never put shaped training
  totals on the same axis as evaluation returns.

## Constraints
- Windows 11, PowerShell, Python 3.12.
- Intermediate portfolio scope.
- One phase at a time.
- The original environment reward is the evaluation ground truth.
- Do not invent results or claim reward hacking without evidence.
- Do not push to `origin` without user approval.

# Current Handoff

## State
- **Phases 00–01** are complete (`70e3445`, `ce62dd6`).
- **Phase 02** (baseline PPO) is implemented and verified.
  - The code is committed as `0f125a5` "Phase 02 (1/2)".
  - The results, record and progress files are committed as "Phase 02 (2/2)".
    Phase 02 is complete. Nothing is pushed.
- **Official baseline:** `runs/baseline/20261103T174958Z_seed42` (local,
  git-ignored), trained from the clean `0f125a5`. Mean original return
  237.56 (sample std 52.37) over 100 episodes, from **one seed**.
- **Committed copies:** `results/baseline/`. Write-up:
  `experiments/records/2026-11-03-baseline-seed42.md`.

- **Phase 03** (reward wrapper) is implemented and verified: four reward
  modes, pure reward functions, `RewardComponentsWrapper`, per-episode
  component logging. 202 tests pass. Committed locally as "Phase 03: reward
  wrapper". Nothing is pushed.

Full records: `progress/handoffs/2026-11-03-phase-02.md`,
`progress/handoffs/2026-11-04-phase-03.md`.

## Current phase
`04-reward-experiments`: not started. Phases 00–03 are complete.

## Next safe action
1. Run `/start-phase 04-reward-experiments` and present the plan only.
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
- For Phase 04: choose and justify reward magnitudes (relative to the
  environment's built-in shaping and typical step rewards); write the four
  experiment configs; check behaviour with videos and metrics, and do not
  assume hovering emerges.

## Constraints
- Windows 11, PowerShell, Python 3.12.
- Intermediate portfolio scope.
- One phase at a time.
- The original environment reward is the evaluation ground truth.
- Do not invent results or claim reward hacking without evidence.
- Do not push to `origin` without user approval.

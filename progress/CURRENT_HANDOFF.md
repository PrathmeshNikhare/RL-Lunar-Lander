# Current Handoff

## State
- **Phases 00–03** are complete (`70e3445`, `ce62dd6`, Phase 02 1/2 and
  2/2, `a71d7d0`). Nothing is pushed.
- **Phase 04** (four reward experiments) is complete. It is committed as
  `74f4f36` "Phase 04 (1/2)" and "Phase 04 (2/2)" (local, not pushed).
  - The configs are committed in `74f4f36` "Phase 04 (1/2)".
  - 12 runs (4 configs × seeds 42/142/242) were trained from the clean
    `74f4f36`.
  - Each run was evaluated on the original reward (100 episodes) and
    recorded on video (3 clips).
  - All three specialist reviews returned REVISE. Every required fix is
    applied (see the handoff).
  - The user reviewed the upright_slow_bonus clips. Seeds 142 and 242 hover;
    seed 42 lands on the pad, late.
- The local interrupted run `runs/time_penalty/20261103T223642Z_seed242` is
  not a result (D-033).

Full record: `progress/handoffs/2026-11-05-phase-04.md`.

## Current phase
`05-evaluation`: not started. Phases 00–04 are complete.

## Next safe action
1. Run `/start-phase 05-evaluation` and present the plan only.
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
- For Phase 05:
  - select runs by `metadata.json` `status: completed`, not by folder glob;
  - the `model.zip` hash is not a policy fingerprint, so use the inner
    `policy.pth` or the evaluation CSV;
  - do not compare undiscounted shaped totals across policies.

## Constraints
- Windows 11, PowerShell, Python 3.12.
- Intermediate portfolio scope.
- One phase at a time.
- The original environment reward is the evaluation ground truth.
- Do not invent results or claim reward hacking without evidence.
- Do not push to `origin` without user approval.

# Handoff — 2026-11-07 — Post-release README metrics

## Summary
After Phase 07 was accepted (`68280d8`), the owner asked for README changes.
They are outside the phase plan and are recorded as D-047 and D-048.

- **Removed:** the "How this project was built" and "License" sections
  (D-047). There is still no `LICENSE` file, so all rights stay reserved.
- **Added to Results:**
  - "Detailed evaluation metrics": crash, rest off pad, out of bounds,
    hover, first leg contact and engine use.
  - "Early training to evaluation": the original reward early in
    training, late in training and in evaluation, with a "How it was done"
    list.
  - "Training metrics": task performance, PPO health (KL, clip fraction,
    explained variance, entropy, value loss) and sample efficiency (time
    to a 200 rolling mean, wall-clock time, hardware, seed spread).
- **Measured, not prescribed:** the owner's checklist named expected
  ranges, and several do not hold. The README reports the measured values
  and how many fall inside each range (D-048).
- **New code:**
  - `lunar_lander_rl/ppo_diagnostics.py` and
    `scripts/extract_ppo_diagnostics.py` copy SB3 `train/*` and
    `rollout/*` scalars from the git-ignored TensorBoard logs into
    `results/phase07/ppo_diagnostics.csv`, with a `.json` provenance
    record. Both outputs are write-once.
  - The event files stay ignored: their names contain the host name.
  - Tests: `tests/test_ppo_diagnostics.py` (4).

## Commands run
| Command | Result |
|---|---|
| `pytest -q` | 341 passed |
| `ruff check .`, `ruff format --check .`, `mypy --strict lunar_lander_rl scripts tests` | Clean (50 source files) |
| `scripts\extract_ppo_diagnostics.py` into the session scratchpad | 5,148 rows from 12 runs. The README numbers were computed from this copy; the committed file is regenerated after the code commit |
| Scratch table scripts over `results/phase04`, `results/phase05` and the diagnostics copy | The README tables. Cross-checks: late-training means equal `training_vs_evaluation.csv`; each run's final rolling mean equals TensorBoard `rollout/ep_rew_mean` (max difference 1e-5) |
| `scripts\extract_ppo_diagnostics.py --comparison results\phase05\comparison.json --output results\phase07\ppo_diagnostics.csv`, run at `82eab58` with the doc edits stashed | 5,148 rows; provenance `dirty: false`; SHA-256 identical to the scratch copy behind the README numbers; no host names or local paths in either file |
| Relative link check | 55 links, 0 broken |

## Review
- **Experiment Analyst: REVISE.**
  - Major: "4 of 9" runs never reached 200 was wrong; it is 3. The
    unsourced "commonly quoted" comparison was dropped. Explained variance
    is on each configuration's own training reward.
  - Minor items fixed: 61 logged of 62 updates, "≈ −ln 4", the KL time
    trend and the bands' dependence on settings, the section renamed
    "Early training to evaluation", the deterministic-versus-stochastic
    caveat, "on average", and causal wording.
- **Documentation Reviewer: REVISE.**
  - Major: the "4 of 9" count; the upright_slow_bonus run was described
    as "went backwards" (now "improved least"); the training-versus-
    evaluation cause was untested (now hedged, about equal for the
    baseline).
  - Minor items and nits fixed: "in evaluation" for potential_shaping,
    "computed post hoc", 51–63%, contact medians to one decimal, an exact
    TensorBoard match, the wall-clock wording, the PPO table split in two,
    the bold removed, the solved threshold per episode.
- The fixes were checked in this session, not re-reviewed.

## Known issues / follow-ups
- The rolling means, early window and threshold crossings were computed by
  scratch scripts. No committed code produces them; the README says so.
- The CPU and RAM were read from the machine on 2026-11-07 and are not in
  the run metadata.
- Not pushed. `origin/main` is at `31d02ca`.

## Git checkpoint
The owner approved the commit ("commit"). Code: `82eab58` (1/2). Then
"Post-release (2/2)": the CSV, README, docs and progress files.

## Next action
Push only with separate approval.

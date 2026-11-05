# Metrics and Operational Definitions

Every metric is derived from raw evaluation records: `episodes.csv` and
`trajectories.npz` in a run's `evaluations/<timestamp>/` folder. Evaluation
uses the original environment reward only, with no reward wrapper. The
code is in `lunar_lander_rl/metrics.py`, and hand-calculated tests are in
`tests/test_metrics.py`. Environment facts come from
`docs/environment-notes.md`.

**Never conflate training shaped return with original evaluation return.**
Training-time returns from `reward_components.csv` are reported in their own
table (`training_vs_evaluation.csv`).

## Per-episode values

| Metric | Definition |
|---|---|
| `evaluation_return` (`original_return`) | Sum of the original environment rewards over the episode |
| `episode_length` (`length`) | Number of environment steps |
| `outcome` | Exactly one label per episode (below) |
| `first_contact_step` | First step (1-based) whose observation has either leg flag set. Empty if no leg ever touches |
| `max_hover_steps` | Longest run of consecutive hover steps (below) |
| `hover` | `max_hover_steps >= hover.min_consecutive_steps` |
| `main_engine_steps`, `side_engine_steps` | Number of steps with action 2 (main engine), or action 1 or 3 (side engines). This is the engine-use proxy for fuel: fuel is unlimited, and its cost appears only through the reward (0.3 and 0.03 per firing step) |

## Outcome (decision D-036)

| Outcome | Rule | Environment meaning |
|---|---|---|
| `success` | `terminated` and final reward exactly +100, with \|final x\| ≤ 0.2 | The lander came to rest (Box2D put the body to sleep) on the pad, x ∈ [−0.2, 0.2] |
| `rest_off_pad` | `terminated` and final reward +100, with \|final x\| > 0.2 | Came to rest outside the pad |
| `crash` | `terminated` and final reward exactly −100, with \|final x\| < 1 | The lander body touched the ground (`game_over`) |
| `out_of_bounds` | `terminated` and final reward −100, with \|final x\| ≥ 1 | Left the screen sideways |
| `timeout` | `truncated` and not `terminated` | Hit the 1000-step time limit |

- **The terminal ±100 is assigned, not added,** so the final reward identifies
  the terminal condition.
- If both terminal conditions hold on one step, the environment's second
  check wins (+100). The rule follows the reward, so it follows the
  environment.
- **`terminated` takes precedence over `truncated`** when both are true: the
  environment's own outcome happened on the last allowed step.
- **A terminated episode whose final reward is not ±100 is an error,** not a
  silent label.
- **Leg-contact flags are not used.** They are unreliable at termination.
- **Precision:** `final_x` is a float32 observation widened to float64.
  float32(0.2) is 0.2000000030, which is above 0.2 and so counts as off the
  pad (`tests/test_metrics.py`).
- **Validation:** the comparison assigns an outcome to every evaluation
  episode and fails on any episode that does not fit a rule.
- `success_rate`, `crash_rate` and the other rates are the episodes with
  that outcome divided by the episodes evaluated.

## Hover diagnostic (decision D-037)

**This is a project-defined diagnostic, not an environment signal.** Its
thresholds are in `configs/evaluation_metrics.yaml` and were fixed before the
Phase 05 data was computed.

- **Hover step:** in the observation after a step, both leg flags are 0,
  `y ≥ 0.1` and speed `‖(vx, vy)‖ ≤ 0.2`, in scaled observation units.
  - `y = 0.1` is about 0.67 world units above pad level.
  - The speed threshold is the same "slow" as the upright bonus (D-029).
- **Hover episode:** at least **100 consecutive** hover steps (2 s of
  simulated time at 50 FPS).
- The reset observation (step 0) is not assessed.
- **Precision:** observations are float32, and NumPy compares them with the
  thresholds at float32 precision, so a speed of float32(0.2) counts as
  within `max_speed`.
- The diagnostic is reported separately from the outcome. A hovering
  episode can still end in success, if it lands later, or in a timeout.
- **Known limits:**
  - a slow final descent above y = 0.1 can count as hover steps;
  - the baseline's hover rate is the false-positive reference;
  - angle is not part of the rule.

## Summaries

- **Per run** (`run_summary.csv`): mean and sample std (ddof = 1) of return
  and length, the count and rate of each outcome, the hover count and rate,
  the mean first-contact step over episodes with contact (with their count),
  and mean engine steps.
- **Per configuration** (`config_summary.csv`): for each per-run metric, the
  mean, sample std, min and max across the training seeds, with `n_seeds`.
  - Pooled episode counts over all seeds are in `comparison.json`
    (`pooled_counts`).
  - Episodes from one policy are not independent samples of a
    configuration, so the seed is the unit of replication.
  - With three seeds, no significance tests are reported.
- **Training versus evaluation** (`training_vs_evaluation.csv`):
  - late-training mean original return and mean shaped total, over the
    window defined in D-035, for the stochastic training policy;
  - beside them, the mean original evaluation return.
  - Do not compare shaped totals across reward configurations; they are on
    different scales.

## Reproduce

```powershell
$py = ".\.venv\Scripts\python.exe"
& $py scripts\evaluate.py --run-dir runs\<experiment>\<run>   # writes trajectories.npz
& $py scripts\compare_runs.py --config configs\evaluation_metrics.yaml --output results\phase05
```

Runs are selected from `metadata.json`: `status: completed`, a source commit
matching `run_selection.source_commit`, and a selected training seed. Each
run's newest evaluation that has `trajectories.npz` is used. The comparison
refuses duplicates, a non-original-reward evaluation, mismatched evaluation
settings, trajectories that disagree with `episodes.csv`, and an output folder
that already exists. Missing runs are listed in `comparison.json`.

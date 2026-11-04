# Reward Experiments (Phase 04)

Four configurations are trained with the same PPO settings, budget and
evaluation seeds. Only `experiment_id` and `reward` differ between them,
which `tests/test_experiment_configs.py` enforces. Formulas, units and
terminal handling are in `docs/reward-wrapper.md`.

| ID | Config | Reward mode | Parameters |
|---|---|---|---|
| A | `configs/baseline.yaml` | `original` | none |
| B | `configs/time_penalty.yaml` | `time_penalty` | `coefficient` 0.05 |
| C | `configs/upright_slow_bonus.yaml` | `upright_slow_bonus` | `bonus` 1.0, `max_abs_angle` 0.2 rad, `max_speed` 0.2 |
| D | `configs/potential_shaping.yaml` | `potential_based` | `alpha` 100, weights 1 / 1 / 1, γ = `ppo.gamma` (0.999) |

## Why these values

The magnitudes are set against the original reward, measured on the Phase 02
baseline (`results/baseline/evaluation_summary.json`, seed 42, 100 episodes):

- The mean return was 237.56 over a mean of 391.8 steps.
- The final-step ±100 contributed a net +93 per episode (95 × +100,
  2 × −100). The terminal reward replaces that step's reward.
- That leaves about (237.56 − 93) / 390.8 ≈ **0.37** original reward per
  non-terminal step, on average. This average mixes large shaping steps with
  small ones, so it is a scale reference, not a typical value.
- Firing the main engine costs 0.3 per step. The environment's built-in
  shaping uses coefficient 100 on distance, speed and |angle|.

**B (time penalty) 0.05.** That is about 13% of the average step reward:
about −20 over a typical 390-step landing and −50 over a full 1000-step
timeout. It is large enough to matter but cannot outweigh the +100 for
landing. The spec's example value, 0.01 (about 3%), was considered and judged
likely to have no visible effect.

**C (upright-and-slow bonus) 1.0, 0.2 rad, 0.2.** The bonus is about 3× the
average step reward. Over the hundreds of steps left in an episode it can
outweigh the +100 landing reward, so it is a real incentive to delay. That is
the point of the probe.
- 0.2 rad is about 11°.
- Speed 0.2 in scaled units corresponds to roughly 1 unit/s horizontally or
  1.5 units/s vertically (scale factors in `docs/environment-notes.md`).
- The condition can be met while hovering anywhere, and also while resting on
  the ground before the body falls asleep.

**D (potential shaping) α = 100, unit weights.** Φ then has the same feature
scale as the environment's own shaping term. If behaviour stays the same
under a term this large, that says something, and a weak α could show no
difference for trivial reasons.
- Expected cost: large single-step values, for example F ≈ +250 on a crash
  far from the pad with |Φ| ≈ 2.5. The discounted sum still telescopes to
  −αΦ(s₀), but the learner sees more variance.
- α = 10 was the gentler alternative.

These values were fixed before training (decision D-032). The protocol
forbids tuning one experiment while leaving the others untouched, so poor
results are reported, not re-tuned.

## Seeds and budget

- **Training seeds:** 42, 142 and 242 for every configuration, so 12 runs.
  With 16 environments, the environment seed blocks are 42–57, 142–157 and
  242–257, which do not overlap.
- **Budget:** 1,000,000 requested timesteps per run. SB3 trains 1,015,808 in
  whole rollouts.
- **Evaluation:** 100 deterministic episodes, seeds 10000–10099, identical for
  every run, using the original reward only.
- **Videos:** 3 episodes, seeds 20000–20002.

## Commands

Train from a clean commit, so each run's metadata records `dirty: false`.
Runs go one after another, because parallel runs would compete for the CPU.

```powershell
$py = ".\.venv\Scripts\python.exe"
foreach ($seed in 42, 142, 242) {
  foreach ($exp in "baseline", "time_penalty", "upright_slow_bonus", "potential_shaping") {
    & $py scripts\train.py --config "configs\$exp.yaml" --seed $seed --quiet
  }
}
```

Then evaluate each new run and record its videos. The run directories are
listed in the experiment records. Do not loop over `runs\baseline` blindly,
because it also holds the Phase 02 runs.

```powershell
& $py scripts\evaluate.py --run-dir runs\<experiment>\<timestamp>_seed<N>
& $py scripts\record_video.py --run-dir runs\<experiment>\<timestamp>_seed<N>
```

## Outputs

- Local, git-ignored: `runs/<experiment>/<timestamp>_seed<N>/`.
- Committed copies: `results/phase04/<experiment>/seed<N>/`:
  - `run_config.json`, `run_metadata.json`;
  - `evaluation_summary.json`, `evaluation_episodes.csv`;
  - `video_episodes.csv`;
  - `reward_components.csv`, the per-training-episode original reward,
    components and shaped total.
- Write-ups: `experiments/records/2026-11-04-<experiment>.md`.

Phase 04 reports raw per-run tallies only. Success, crash and hover rules,
and the comparison across seeds, belong to Phase 05.

## Late-training window

The records summarise training-time episodes (from `reward_components.csv`)
over episodes **ending at or after 900,000 timesteps**.

- This window was chosen after training, as a descriptive choice. It was
  not pre-registered.
- It covers the last 115,808 steps (about 7 rollouts) under the stochastic
  training policy.
- It averages per episode, so the number of episodes differs between runs
  (117–320).
- Episodes still unfinished when training stops are not included.
- For runs that were still improving, the values shift with the window
  start. For example, time_penalty seed 142's mean original return is 179.3,
  192.2 and 215.1 for windows starting at 800k, 900k and 950k.
- Treat these as descriptions of late training, not as evaluation results.

# Training Diagnostics

Task performance, PPO health and sample efficiency for the 12 runs
(4 reward configurations × 3 seeds) summarised in the
[README](../README.md#training-diagnostics). The tables report the values
that were measured. Where the author's metrics checklist gave reference
ranges, the tables also show how many values fall inside them; those
ranges are rules of thumb, not cited sources. Values are given per seed
(42 / 142 / 242) or as mean ± sample std over the 3 seeds. All comparisons are descriptive.

## Task performance

Gymnasium's "solved" threshold is 200 points per
episode.
- **Evaluation:** all 9 baseline, time_penalty and potential_shaping runs
  averaged above 200 in deterministic evaluation, from 236.0 to 269.9.
  The 3 upright_slow_bonus runs averaged 167.7, −34.0 and −124.7.
- **Training:** the final training-time mean is about equal to evaluation
  for the baseline (244.0 against 243.9). It is lower for the shaped
  configurations. Two possible reasons, neither tested: the training
  policy samples its actions, and in the shaped runs it optimises a
  different reward.

| Configuration | Success rate (evaluation) | Crash rate, −100 endings (evaluation) | Mean episode length, evaluation (steps) | Training reward, final 100-episode mean | Training episode length, final 100-episode mean (`rollout/ep_len_mean`) |
|---|---|---|---|---|---|
| baseline | 97.0% ± 5.2 | 0.7% (2/300) | 392 / 437 / 356 | 244.0 ± 14.7 | 314 / 393 / 364 |
| time_penalty | 98.7% ± 2.3 | 0.0% (0/300) | 379 / 279 / 311 | 207.8 ± 57.4 | 975 / 788 / 306 |
| upright_slow_bonus | 30.7% ± 52.3 | 0.3% (1/300) | 776 / 999 / 1000 | 85.1 ± 39.6 | 993 / 960 / 990 |
| potential_shaping | 98.3% ± 1.5 | 1.0% (3/300) | 355 / 380 / 440 | 181.5 ± 56.7 | 948 / 888 / 404 |

- **Success:** the lander came to rest (+100) with |x| ≤ 0.2, between the
  flags.
- **Crash:** any episode ending with −100. That covers body-ground contact
  and leaving the screen; the second never happened.
- **Training columns:** these are the original reward and episode length
  of the training episodes. SB3's `Monitor` sits inside the reward
  wrapper, so `rollout/ep_rew_mean` and `rollout/ep_len_mean` also measure
  the original reward. For the shaped configurations, that is not the
  reward PPO optimised.
- **Long training episodes in working configurations:** four seeds end
  training with mean lengths of 788–975 steps: time_penalty seeds 42 and
  142, and potential_shaping seeds 42 and 142. With the deterministic
  evaluation policy, the same seeds average 279–380 steps.

## PPO health

These are SB3's `train/*` values, one per logged policy
update.
- Each run made 62 updates. SB3 writes an update's values at the next
  rollout, so the last update is never logged. That leaves 61 logged
  updates per run (first at 32,768 timesteps) and 183 per configuration.
- "Start" means after the first update.
- Settings: clip range 0.2, 4 epochs per update, no `target_kl`.

### Update size

| Configuration | `approx_kl`: range (median) | `approx_kl` in 0.005–0.02 | `clip_fraction`: range (median) | `clip_fraction` in 0.1–0.3 |
|---|---|---|---|---|
| baseline | 0.0025–0.0110 (0.0051) | 93/183 | 0.006–0.110 (0.040) | 2/183 |
| time_penalty | 0.0026–0.0124 (0.0051) | 93/183 | 0.003–0.108 (0.041) | 2/183 |
| upright_slow_bonus | 0.0016–0.0130 (0.0056) | 115/183 | 0.003–0.122 (0.031) | 3/183 |
| potential_shaping | 0.0024–0.0105 (0.0052) | 97/183 | 0.004–0.101 (0.040) | 2/183 |

### Critic and exploration

| Configuration | `explained_variance`: final (best during training) | `entropy_loss`: start → final | `value_loss`: start → final (lowest) |
|---|---|---|---|
| baseline | 0.56 / 0.79 / 0.93 (0.99–1.00) | −1.38 → −0.80 / −0.77 / −0.75 | ~4,900 → 440 / 196 / 107 (2.8–6.4) |
| time_penalty | 1.00 / 0.88 / 0.64 (0.99–1.00) | −1.38 → −0.87 / −0.65 / −0.72 | ~5,000 → 5 / 235 / 355 (2.4–5.6) |
| upright_slow_bonus | 0.76 / 0.79 / 0.47 (0.88–0.95) | −1.38 → −1.03 / −1.02 / −0.96 | ~5,000 → 50 / 560 / 657 (49–61) |
| potential_shaping | 1.00 / 1.00 / 0.98 (1.00) | −1.38 → −0.79 / −0.96 / −0.76 | ~7,000 → 41 / 36 / 136 (3.8–6.3) |

- **Approximate KL:** no logged update went above 0.02; the largest was
  0.013.
  - 51–63% of updates fell in the checklist's 0.005–0.02 band, and the
    rest were below it, meaning smaller policy changes per update.
  - The in-band updates cluster early in training. Per configuration,
    61–65 of the 93–115 in-band updates are among each seed's first 30
    updates.
- **Clip fraction:** the median is 0.03–0.04. Only 2–3 updates per
  configuration reached the checklist's 0.1–0.3 band. This is consistent
  with the small KL values. Both bands depend on batch size, epochs and
  learning rate.
- **Explained variance:** every run's critic reached 0.88–1.00 at some
  point in training.
  - The value fluctuates, and "final" is a single update. It is above 0.80
    in 6 of the 12 runs.
  - The critic fits each configuration's own training reward. For
    potential_shaping that reward includes a state-based shaping term, so
    the values are not a like-for-like comparison across configurations.
- **Entropy loss:** SB3 reports the negative mean entropy.
  - After the first update, every run is at about −1.38. That is close to
    −ln 4 ≈ −1.386, the value for a uniform choice among the 4 actions.
  - Every run ends closer to zero, but none ends near zero: the policies
    keep some randomness.
  - On average, the bonus runs stay the most random.
- **Value loss:** it falls from about 5,000–7,000 to single digits at its
  lowest (about 50 for the bonus runs).
  - It does not settle: the final values are often much higher than each
    run's lowest point.
  - Like explained variance, it is measured on each configuration's own
    training reward, so the configurations are not directly comparable.
- **Curves:** the tables give the start, end and range of each series.
  The curves themselves are in TensorBoard; no figure of them is
  committed.

## Sample efficiency and cost

| Configuration | Training rolling mean first ≥ 200 (timesteps) | Wall-clock per run (min) |
|---|---|---|
| baseline | 937k / 970k / 778k | 5.0 / 4.6 / 4.3 |
| time_penalty | never / 1,008k / 943k | 4.7 / 4.5 / 4.8 |
| upright_slow_bonus | never / never / never | 4.9 / 4.5 / 9.3 |
| potential_shaping | never / never / 963k | 4.8 / 4.6 / 10.9 |

- **Timesteps to the 200 threshold:** the first timestep at which the
  100-episode rolling mean of the training reward reached 200.
  - The rolling mean is taken over training episodes in completion order,
    across all 16 environments.
  - TensorBoard shows the same crossings at the next rollout boundary.
  - Each run had 1,015,808 timesteps (62 rollouts of 16 × 1,024 steps).
- **Staying above 200:** every run that reached 200 stayed at or above it
  to the end, except time_penalty seed 242, which dipped once and stayed
  from 946k. For time_penalty seed 142 the crossing came only about 7.5k
  steps before the end.
- **Runs that never reached 200:** 3 of the 9 runs in the working
  configurations: time_penalty seed 42, and potential_shaping seeds 42
  and 142. All 3 averaged above 200 in deterministic evaluation.
- **Training policy only:** this measures the stochastic training policy.
  Evaluation ran only at the end of training, so it is unknown when the
  deterministic policy first passed 200.
- **Hardware:** Intel Core i7-12650H laptop CPU, 16 GB RAM, no GPU. Each
  run used 16 environments in one process and 10 PyTorch threads.
  - The thread count is recorded in each run's metadata.
  - The CPU and RAM were read from the training machine on 2026-11-07;
    they are not in the run metadata.
- **Wall-clock:** two of the three seed-242 runs made on 2026-11-04 took
  about twice as long as the others. The cause was not recorded.
- **Seed-to-seed spread:** every ± here and in the README is the sample standard
  deviation over the 3 training seeds. No configuration was run with more
  seeds.
- **Sources:**
  - PPO diagnostics:
    [`results/phase07/ppo_diagnostics.csv`](../results/phase07/ppo_diagnostics.csv),
    copied from the local TensorBoard logs by
    `scripts\extract_ppo_diagnostics.py`.
  - Rolling means and threshold crossings: computed post hoc from
    `reward_components.csv` in [`results/phase04/`](../results/phase04/).
    Each run's final rolling mean matches TensorBoard's own
    `rollout/ep_rew_mean`.
  - Wall-clock times: each run's `run_metadata.json`.

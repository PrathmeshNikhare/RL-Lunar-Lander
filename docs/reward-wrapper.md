# Reward Wrapper and Reward Modes

This page describes how training rewards are built (Phase 03). Evaluation always
uses the **original** environment reward, never these modes (D-009).

- Pure calculations: `lunar_lander_rl/reward_functions.py`
- Wrapper: `lunar_lander_rl/reward_wrapper.py` (`RewardComponentsWrapper`)
- Per-episode logging during training: `lunar_lander_rl/reward_logging.py`
- Config schema and validation: `lunar_lander_rl/config.py`

The parameter values on this page are examples. The values used in
experiments are chosen in Phase 04 and recorded in each run's `config.json`.

## Notation

`s` is the observation before a step and `s'` the observation returned by
that step. All values are in the environment's **scaled observation units**
(`docs/environment-notes.md`), not physical units:

- `dist(s) = ||(x, y)||`, the distance from the pad centre;
- `speed(s) = ||(vx, vy)||`;
- `wrap(angle)` maps the angle to [−π, π]. LunarLander's `angle` is
  cumulative and not wrapped, so without this a lander that has turned a full
  circle would not count as upright.

`r_env` is the reward returned by `LunarLander-v3`. It includes the
environment's own built-in shaping term and the terminal ±100, which replaces
that step's reward rather than being added to it.

## Modes

Each step's training reward is `r_total = r_env + sum(components)`. The
config's `reward` section selects the mode. Each mode requires exactly its own
keys; unknown or missing keys, non-numbers, NaN/inf and out-of-range values
raise `ConfigError`.

| Mode | Keys (validation) | Component added each step |
|---|---|---|
| `original` | none | none: `r_total = r_env` exactly |
| `time_penalty` | `coefficient` > 0 | `time_penalty = −coefficient` |
| `upright_slow_bonus` | `bonus` > 0; `max_abs_angle` in (0, π] rad; `max_speed` > 0 | `upright_slow_bonus = bonus · I(\|wrap(angle')\| ≤ max_abs_angle ∧ speed(s') ≤ max_speed)` |
| `potential_based` | `alpha` > 0; `distance_weight`, `speed_weight`, `angle_weight` ≥ 0, at least one > 0 | `potential_shaping = alpha · (γ · Φ(s') − Φ(s))` |

```yaml
reward: {mode: original}
reward: {mode: time_penalty, coefficient: 0.01}
reward: {mode: upright_slow_bonus, bonus: 1.0, max_abs_angle: 0.1, max_speed: 0.1}
reward: {mode: potential_based, alpha: 1.0, distance_weight: 1.0, speed_weight: 1.0, angle_weight: 1.0}
```

All components are applied on **every** step, including the final one.

### Time penalty

A constant cost per step. It changes the objective: it is not
policy-invariant and can change the optimal policy, for example towards
ending episodes sooner. Ending sooner includes crashing; whether that happens
is measured in Phase 04, not assumed.

### Upright-and-slow bonus (deliberately flawed)

The indicator uses `s'`. It ignores position, height and leg contact on
purpose. Because the bonus can be collected on every step in any upright,
slow state, including hovering away from the pad, it may reward delaying the
landing. Phase 04 tests whether that actually happens; this page makes no
claim that it does.

The bonus is also paid on the final step when `s'` qualifies, which includes
the at-rest +100 step and, rarely, a slow crash. Angular velocity is not part
of the condition.

### Potential-based shaping

```
Φ(s) = −( distance_weight · dist(s) + speed_weight · speed(s) + angle_weight · |wrap(angle)| )
F    = alpha · (γ · Φ(s') − Φ(s))        with Φ(s') := 0 if the step terminated
```

- **Discount.** `γ` is always `ppo.gamma` (0.999 in the baseline). It cannot
  be set separately in the reward section (D-019).
- **Φ ≤ 0.** With positive weights, Φ is 0 only at the pad centre, at rest
  and upright.
- **Boundedness.** Before computing Φ, the wrapper clips positions and
  velocities to the environment's declared `observation_space` bounds
  (positions ±2.5, velocities ±10). The angle is not clipped: wrapping
  already bounds it to [−π, π], and clipping it at ±2π first would make
  every rotation beyond a full turn read as upright. Φ is therefore bounded
  below by `−(w_d·√12.5 + w_v·√200 + w_a·π)`. The bonus mode uses the
  unclipped observation; its thresholds are far inside the bounds.
- **Overlap with the environment.** `r_env` already contains a
  difference-of-potentials term on similar features (coefficient 100, no
  discount). This mode adds a second, separately logged term on top of it.

#### Terminal handling

| Step outcome | `Φ(s')` used in F | Reason |
|---|---|---|
| Non-terminal | `Φ(s')` | Ordinary transition |
| `terminated` (crash, out of bounds, at rest) | 0 | Absorbing state; Ng et al. require a constant (here 0) potential there |
| `truncated` (1000-step limit) | real `Φ(s')` | Not absorbing. SB3 bootstraps the value of the final observation (`TimeLimit.truncated`), so the episode is treated as continuing |
| `reset` | `Φ(s0)` stored from the reset observation | First step uses `F = alpha·(γΦ(s1) − Φ(s0))` |

- Because Φ ≤ 0 and Φ(terminal) = 0, the final step of a terminated episode
  gets `F = −alpha·Φ(s) ≥ 0`, even after a crash. This is expected: it is what
  makes the discounted sum telescope to `−alpha·Φ(s0)` (tested in
  `tests/test_reward_functions.py`). It does not depend on the policy.
- **Staying still earns a small positive F.** If `s' = s`, then
  `F = alpha·(1 − γ)·|Φ(s)|`, which is ≥ 0 and grows with the distance from
  the goal. With γ = 0.999 and unit weights it is at most about 0.02 per
  step. Over a whole episode the discounting cancels it (the telescoping
  above), but it is a per-step reward for a static state and is noted here
  for that reason.
- In training, the wrapper sits inside `DummyVecEnv`, so it sees the true
  final observation before the automatic reset.

#### Assumptions behind policy invariance

The guarantee comes from Ng, Harada & Russell (1999), "Policy invariance under
reward transformations". It holds only if all of these are true:

1. F has the form `γΦ(s') − Φ(s)` (times a constant `alpha`), with Φ a
   function of the state only. Φ uses the observation, which is a function
   of the simulator state. It does not depend on the action or on time.
   The policy also acts on observations, so the argument treats the
   observation as the (Markov) state. LunarLander hides some state, such as
   the terrain shape and the engine-dispersion RNG, so this is an
   approximation.
2. The `γ` in F is the discount of the objective being optimised (`ppo.gamma`).
3. Φ is the same constant (0) at every absorbing state.
4. Time-limit truncation is treated as non-terminal, consistently with SB3's
   bootstrapping.

Even then, the result is only about the **optimal policy** of the
`γ`-discounted problem. It does not promise that PPO learns faster, that a
finite 1M-step run reaches the same policy, or anything about the
undiscounted original return used in evaluation. The 1000-step limit also
makes the real task time-dependent, which the Markov model ignores. The time
penalty and the bonus are not of this form, so no invariance is claimed for
them.

## What is logged

| Where | Content |
|---|---|
| `info["reward_components"]` (every step) | `original`, each component, `total` |
| `info["reward_episode"]` (final step) | the same keys summed over the episode, plus `length` |
| `monitor/<rank>.monitor.csv`, `rollout/ep_rew_mean` | **original** reward only. The wrapper is applied *outside* SB3's `Monitor` |
| `reward_components.csv` | one row per finished training episode: `timesteps`, `env_rank`, `length`, `original`, each component, `total` |
| TensorBoard `reward/<name>` | mean of each per-episode value for each rollout |

The logged per-episode `potential_shaping` total is an **undiscounted** sum.
It does not telescope to `−alpha·Φ(s0)`, and it grows with episode length
through the `(1 − γ)` term above. A larger shaping total in a longer episode
is therefore not, by itself, evidence of reward hacking.

`original` mode is wrapped too, with no components, so every mode produces the
same logs. A test trains PPO both with and without the wrapper in `original`
mode and checks that the policy weights are identical.

## Contract

- The observation, `terminated`, `truncated` and the inner environment's info
  keys are passed through unchanged. The wrapper adds keys to a copy of info.
- The returned reward is a Python `float`. The original terminal ±100 is a
  Python `int`.
- The wrapper reads only what `reset()`/`step()` return and the declared
  `observation_space`. It does not access the simulator (`env.unwrapped`,
  the lander body, contacts).

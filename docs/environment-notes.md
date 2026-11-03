# Environment Notes — Gymnasium `LunarLander-v3`

Reference for later phases (reward design, metrics, evaluation). Facts come
from the installed source,
`.venv\Lib\site-packages\gymnasium\envs\box2d\lunar_lander.py` (gymnasium
1.4.0, see `requirements-lock.txt`), and its docstring, which is the source of
the official documentation page:
<https://gymnasium.farama.org/environments/box2d/lunar_lander/>.
Line numbers refer to that installed file. Anything marked
**observed** was measured in this repository and is not a guarantee.

The project uses the default configuration: discrete actions, `gravity=-10.0`,
no wind. It is created with `lunar_lander_rl.environment.make_env()`.

## Registration and wrappers

| Item | Value |
|---|---|
| ID | `LunarLander-v3` (`gymnasium/envs/__init__.py`) |
| `max_episode_steps` | 1000, enforced by the `TimeLimit` wrapper added by `gym.make` |
| `reward_threshold` | 200 ("solved" if an episode scores at least 200) |
| Simulation rate | `FPS = 50`, so one step is 1/50 s of simulated time |
| Wrappers from `make` | `TimeLimit` → `OrderEnforcing` → `PassiveEnvChecker` → `LunarLander` |

## Observation (8 × `float32`)

The order matches `OBSERVATION_LABELS` in `lunar_lander_rl/environment.py`.
Values are normalised, not physical. The scale factors come from the
docstring's "Notes". Multiply an observation value by the factor to get the
unit in the last column.

| Index | Name | Space bounds | Meaning (source, `step()` L624–633) | Scale factor | Unit after scaling |
|---|---|---|---|---|---|
| 0 | `x` | ±2.5 | Horizontal position relative to pad centre, divided by half the viewport width | 10 | units |
| 1 | `y` | ±2.5 | Height above pad level (helipad + leg length), divided by half the viewport height | 6.666 | units |
| 2 | `vx` | ±10 | Horizontal velocity, scaled | 5 | units/s |
| 3 | `vy` | ±10 | Vertical velocity, scaled; negative means descending | 7.5 | units/s |
| 4 | `angle` | ±2π | Lander body angle, radians | 1 | rad |
| 5 | `angular_velocity` | ±10 | `20 * angularVelocity / FPS` | 2.5 | rad/s |
| 6 | `left_leg_contact` | {0, 1} | 1.0 if the left leg touches the ground | — | flag |
| 7 | `right_leg_contact` | {0, 1} | 1.0 if the right leg touches the ground | — | flag |

- The landing pad is always centred at `(x, y) = (0, 0)`. It spans world
  x 8.0 to 12.0 (`helipad_x1`/`helipad_x2`, from L345–347) on a 20-unit-wide
  world. In observation units that is **x ∈ [−0.2, +0.2]**, confirmed at
  runtime.
- `angle` is the raw, cumulative Box2D body angle and is **not wrapped** to
  ±π. A spinning lander can exceed ±π, and the environment's `-100·|angle|`
  shaping term grows with it. This is relevant to reward thresholds and
  potentials in Phases 03/04.
- The bounds are those of the `Box` space. The source comments that the
  "useful range is -1 .. +1, but spikes can be higher". For `x`, episodes
  terminate at `|x| >= 1` (see Termination).
- v3 widened the bounds from v2: positions ±1.5 → ±2.5, velocities ±5 → ±10,
  angle ±π → ±2π.
- Sign conventions for `x`, `vx` and `angle` follow Box2D world coordinates.
  Verify the sign convention for `angle` (counter-clockwise positive is
  expected from Box2D) empirically before using it in a reward threshold.

## Actions (`Discrete(4)`)

| Action | Meaning |
|---|---|
| 0 | Do nothing |
| 1 | Fire left orientation engine |
| 2 | Fire main engine |
| 3 | Fire right orientation engine |

Fuel is unlimited. Engine use is penalised through the reward only.

## Reward (original, per step)

From `step()` (L637–652). The environment defines a shaping value:

```
shaping = -100 * sqrt(x² + y²)
          -100 * sqrt(vx² + vy²)
          -100 * |angle|
          + 10 * left_leg_contact + 10 * right_leg_contact
```

On every non-terminal step:

```
reward = (shaping_t - shaping_{t-1}) - 0.30 * main_engine_power - 0.03 * side_engine_power
```

`reset()` clears `prev_shaping` (L337) and then calls `self.step(0)`
internally (L447). The observation returned by `reset` therefore comes from
that internal no-op step, and `prev_shaping` is already set, so the agent's
first `step` includes a shaping difference. Engine power is 1.0 when firing in
discrete mode (L536, L584), so the costs are 0.3 for the main engine and 0.03
for a side engine.

**Reward type:** non-terminal rewards are `numpy.float64`. The terminal ±100 is
a Python `int` (observed at runtime). Reward wrappers should convert with
`float(reward)`.

The base environment already contains its own difference-of-potential term.
This matters for Phase 03/04: the project's potential-based shaping is added
on top of the original reward and must not be confused with this built-in term.

## Termination and truncation

From `step()` (L654–660):

```python
terminated = False
if self.game_over or abs(state[0]) >= 1.0:
    terminated = True
    reward = -100
if not self.lander.awake:
    terminated = True
    reward = +100
```

| Outcome | Condition | `terminated` | `truncated` | Final-step reward |
|---|---|---|---|---|
| Crash | Lander *body* contacts the ground (`game_over`, set by the contact detector, L68) | True | False | exactly −100 |
| Out of bounds | `abs(x) >= 1.0` | True | False | exactly −100 |
| At rest | Box2D puts the lander body to sleep (`not awake`) | True | False | exactly +100 |
| Time limit | 1000 steps without termination | False | True | normal step reward |

Points for later phases:

- **The terminal reward replaces the step reward; it is not added to it.**
  The docstring calls it "an additional reward", but the code assigns it.
- **The checks are ordered.** If a crash or out-of-bounds condition and the
  at-rest condition hold on the same step, the second `if` wins (+100).
- **"At rest" is not the same as "on the pad".** The +100 terminal can occur
  anywhere the lander comes to rest. Success/crash metrics (Phase 05) must
  define their exact rule from these semantics.
- **The base `LunarLander` always returns `truncated=False`.** Truncation comes
  only from the `TimeLimit` wrapper.
- **Among contacts, only lander-body contact sets `game_over`.** The body's
  collision mask allows contact with the ground only (L384–385). Leg contact
  does not end the episode, although the at-rest termination usually follows
  leg contact.
- **Nothing terminates the episode at the top or for height.** A lander that
  climbs or hovers ends only at `|x| >= 1`, at rest, on a crash, or by
  truncation at 1000 steps. This matters when looking for hovering behaviour.
- **Leg-contact flags at termination are not a reliable success signal.**
  **Observed:** with Gymnasium's built-in `heuristic` controller, seeds 0–2
  all ended with an at-rest `+100` inside the pad (`|x| <= 0.15`). The
  seed-0 episode's final observation had **both** leg flags `0`; seeds 1 and
  2 had both flags `1`. That is three episodes, not a statistic, but it shows
  a rule such as "+100 and both legs in contact" can misclassify a landing.
- **Observed:** for random-policy episodes with seeds 0–4, the last
  `terminated` step's reward was exactly `-100.0`, consistent with assignment
  rather than addition. This is a sample of five episodes, not a statistic.

## Seeding

- Gymnasium seeds the environment through `env.reset(seed=...)`. The
  environment's `np_random` then drives three things:
  - the terrain heights (L344), which differ between seeds;
  - the random initial force (L395–396);
  - a random engine dispersion applied on **every** engine-firing step (L525).

  The dynamics are therefore stochastic within an episode, not only at
  reset. Reproducing an evaluation episode needs the environment seed as well
  as a deterministic policy.
- Random actions additionally need `env.action_space.seed(seed)`.
- **Observed:** with the same seed and the same action sequence, the
  trajectories were identical (`tests/test_environment.py`). The v3
  changelog mentions a fix for non-deterministic world destruction.

## Random rollout tool

```powershell
.\.venv\Scripts\python.exe scripts\random_rollout.py --episodes 3 --seed 0
.\.venv\Scripts\python.exe scripts\random_rollout.py --episodes 2 --seed 0 --render human
```

- The tool prints the spaces, the first N transitions (`--print-steps`) and a
  per-episode summary with `terminated` and `truncated` reported separately.
- Episode `i` uses seed `seed + i`.
- `--max-steps` adds an outer bound (default: the registered 1000).
- `human` mode opens a pygame window that renders at 50 FPS.

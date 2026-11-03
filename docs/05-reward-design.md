# Reward Design Specification

## A. Baseline

Use the exact reward returned by LunarLander-v3. Do not reimplement it.

## B. Time penalty

Illustrative starting formula:

`r_total = r_env - c_time`

Start with a small configurable `c_time` (for example 0.01), then validate
the magnitude relative to the environment reward. A constant per-step cost
can change the optimal policy; it is an experimental objective, not a
policy-invariant shaping term.

## C. Deliberately flawed upright-and-slow bonus

Illustrative formula:

`r_total = r_env + b * I(upright_and_slow)`

Use a configurable bonus and clearly documented thresholds. Define upright
and slow using observation components and thresholds; confirm their signs and
scales from the environment documentation. Avoid accessing hidden simulator
state. The purpose is to test whether a repeated state bonus incentivizes
hovering or otherwise delays task completion.

Do not assume the hack will appear. Log bonus accumulation, episode duration,
landing success and video evidence.

## D. Potential-based shaping

Use:

`F(s, s') = alpha * (gamma * Phi(s') - Phi(s))`

`r_total = r_env + F(s, s')`

Define and justify `Phi` before implementation. Consider horizontal distance,
height, velocities and angle, but avoid an unbounded or poorly scaled potential.
Document terminal-state handling and the episodic assumptions required for
policy invariance. Add hand-calculated tests.

## Evaluation invariant

For all four policies, evaluate in the original environment with reward
wrappers disabled. Report original episode return and task outcomes. Keep
training reward and evaluation return as separate fields.

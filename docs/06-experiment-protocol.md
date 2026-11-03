# Experiment Protocol

## Configurations

A: baseline original reward
B: baseline plus time penalty
C: baseline plus flawed upright-and-slow bonus
D: baseline plus potential-based shaping

## Fairness

- Keep environment ID and package versions fixed.
- Keep PPO architecture and primary hyperparameters fixed initially.
- Use the same planned timestep budget.
- Run at least three training seeds where compute permits.
- Evaluate each saved policy on at least 100 episodes where practical.
- Use matched evaluation seeds when feasible.
- Do not tune one experiment extensively while leaving others untouched.
- Record any deviation and its reason.

## Run metadata

Each run records: experiment ID, timestamp, seed, environment ID, package
versions, timesteps, PPO hyperparameters, reward configuration, output paths,
training duration and source revision.

## Reporting

Show mean and variability, not just the best episode. Retain per-episode raw
data. Explain sample size and limitations. Do not infer causality from one run.

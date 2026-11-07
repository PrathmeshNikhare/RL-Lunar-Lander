# RL Concepts Required for This Project

> Original project specification, written before implementation and kept
> as the design contract. For what was built and measured, see the
> [README](../README.md), [`docs/release-summary.md`](release-summary.md)
> and [`docs/03-architecture.md`](03-architecture.md).

Understand these before or during implementation:

- MDP: states, actions, transition dynamics, rewards and discount factor.
- Observation: the eight values exposed by LunarLander.
- Action: one of four discrete controls.
- Policy: mapping from observations to action probabilities.
- Return: discounted or undiscounted sum of rewards over an episode.
- Exploration/exploitation: balancing trying actions and using learned policy.
- Policy gradient: optimizing a parameterized stochastic policy.
- PPO: policy optimization with a clipped probability-ratio objective.
- Episode termination vs truncation: environment terminal state vs time limit.

Do not describe PPO clipping as a guarantee that every update is harmless.
Explain it as a mechanism that discourages excessively large policy updates.

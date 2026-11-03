# RL Concepts Required for This Project

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

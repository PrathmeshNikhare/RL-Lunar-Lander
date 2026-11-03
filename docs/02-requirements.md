# Requirements

## Functional

- FR-01: Construct LunarLander-v3 with configurable render mode and seed.
- FR-02: Run a random-policy smoke test and inspect observation/action spaces.
- FR-03: Train PPO with configurable timesteps, seed and hyperparameters.
- FR-04: Save model, configuration and training logs per run.
- FR-05: Support four named reward configurations.
- FR-06: Log original reward and custom reward components separately.
- FR-07: Evaluate a saved model with reward shaping disabled.
- FR-08: Calculate episode return, success, crash and episode length.
- FR-09: Record representative evaluation episodes.
- FR-10: Generate comparison plots from retained raw metrics.

## Non-functional

- NFR-01: Reproducible configuration and seed recording.
- NFR-02: Clear module boundaries and typed public functions.
- NFR-03: Deterministic unit tests for reward calculations.
- NFR-04: No long training jobs in default tests.
- NFR-05: Windows 11 / Python 3.12 setup documentation.
- NFR-06: No fabricated metrics or unverified media links.
- NFR-07: Existing experiment artifacts are not overwritten silently.

## Acceptance

A new user can install dependencies, run a smoke test, train/evaluate an
agent, reproduce the reward experiments and understand the results from the
README and experiment report.

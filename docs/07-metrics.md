# Metrics and Operational Definitions

Define and test each metric before comparing experiments.

- `evaluation_return`: sum of original environment rewards for an episode.
- `episode_length`: number of environment steps.
- `success`: use the environment's documented successful landing signal /
  terminal reward semantics; document the exact rule used in code.
- `crash`: define from documented termination/reward semantics and validate
  against recorded episodes.
- `mean_return`: arithmetic mean across evaluation episodes.
- `success_rate`: successful episodes divided by evaluated episodes.
- `crash_rate`: crashed episodes divided by evaluated episodes.
- `mean_episode_length`: mean number of steps.
- `hover_indicator`: a project-defined diagnostic, not an environment truth.
  Specify thresholds, consecutive duration and detection logic.
- `fuel_cost`: report engine-use proxy or reward fuel penalties only if its
  measurement is available and clearly defined.

Never conflate training shaped return with original evaluation return.

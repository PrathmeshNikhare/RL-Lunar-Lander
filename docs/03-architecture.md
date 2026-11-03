# Architecture

## Logical flow

Environment factory → optional reward wrapper → PPO trainer → checkpoint/logs.
Evaluation loads a checkpoint into a separate environment with custom reward
modifications disabled. Analysis reads saved per-episode metrics and produces
plots. Video recording is an evaluation concern, not a training concern.

## Proposed modules

- `environment.py`: environment construction and seeding.
- `reward_functions.py`: pure reward/potential calculations.
- `reward_wrapper.py`: Gymnasium wrapper and reward-component accounting.
- `train.py`: configuration-driven PPO training.
- `evaluate.py`: policy evaluation using original environment reward.
- `record_video.py`: record selected evaluation episodes.

## Design rules

- Avoid circular imports.
- Keep pure reward calculations independently testable.
- Do not reach into Box2D internals for reward shaping.
- Keep evaluation code independent of custom training reward.
- Store a copy of the effective config with each run.
- Use unique run directories; fail safely if an output path already exists.

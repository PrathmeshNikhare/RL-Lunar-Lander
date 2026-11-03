"""SB3 callback that logs per-episode reward components during training.

For every training episode that ends, it reads ``info["reward_episode"]``
(written by ``RewardComponentsWrapper``) and:

- appends one row to ``reward_components.csv`` in the run directory
  (columns: ``timesteps``, ``env_rank``, then ``episode_fields(reward)``);
- records the mean of each value under ``reward/<name>`` in TensorBoard.

The CSV is created with mode ``"x"``, so an existing file is never
overwritten. Rows are flushed after each step that wrote any, so an
interrupted run keeps its rows.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import IO, Any

from stable_baselines3.common.callbacks import BaseCallback

from lunar_lander_rl.config import RewardConfig
from lunar_lander_rl.reward_wrapper import EPISODE_INFO_KEY, LENGTH_KEY, episode_fields


class RewardComponentsLogger(BaseCallback):
    def __init__(self, path: Path, reward: RewardConfig) -> None:
        super().__init__()
        self._path = path
        self._fields = ("timesteps", "env_rank", *episode_fields(reward))
        self._handle: IO[str] | None = None
        self._writer: csv.DictWriter[str] | None = None

    def _on_training_start(self) -> None:
        self._handle = self._path.open("x", encoding="utf-8", newline="")
        self._writer = csv.DictWriter(self._handle, fieldnames=list(self._fields))
        self._writer.writeheader()

    def _on_step(self) -> bool:
        assert self._writer is not None and self._handle is not None
        infos: list[dict[str, Any]] = self.locals["infos"]
        wrote = False
        for rank, info in enumerate(infos):
            episode = info.get(EPISODE_INFO_KEY)
            if episode is None:
                continue
            wrote = True
            self._writer.writerow(
                {"timesteps": self.num_timesteps, "env_rank": rank, **episode}
            )
            for name, value in episode.items():
                if name != LENGTH_KEY:  # SB3 already logs rollout/ep_len_mean
                    self.logger.record_mean(f"reward/{name}", value)
        if wrote:
            self._handle.flush()
        return True

    def close(self) -> None:
        """Close the CSV; safe to call more than once or before training starts."""
        if self._handle is not None:
            self._handle.close()
            self._handle = None
            self._writer = None

    def _on_training_end(self) -> None:
        self.close()

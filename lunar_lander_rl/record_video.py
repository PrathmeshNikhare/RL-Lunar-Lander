"""Record videos of selected evaluation episodes (original reward, no wrappers).

Video recording is an evaluation concern (docs/03-architecture.md). It reuses
:func:`lunar_lander_rl.evaluate.run_episodes` on an ``rgb_array`` environment
wrapped with Gymnasium's ``RecordVideo``, which needs moviepy. Each video is
written next to an ``episodes.csv`` and a ``summary.json``, so every clip can
be matched to its seed, return and outcome. Layout:
``<session>/episodes.csv``, ``<session>/summary.json``,
``<session>/mp4/episode-episode-<i>.mp4``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Final, cast

from gymnasium.wrappers import RecordVideo

from lunar_lander_rl.artifacts import MODEL_FILENAME, VIDEOS_SUBDIR, create_session_dir
from lunar_lander_rl.config import EpisodeSetConfig
from lunar_lander_rl.environment import LunarLanderEnv, make_env
from lunar_lander_rl.evaluate import (
    EnvFactory,
    PolicyFn,
    load_run_policy,
    run_episodes,
    write_session,
)

VIDEO_NAME_PREFIX: Final[str] = "episode"
# RecordVideo warns about "overwriting" if its folder already exists, so clips go
# into a subfolder it creates itself inside the (new, unique) session directory.
VIDEO_SUBDIR: Final[str] = "mp4"
RENDER_MODE: Final = "rgb_array"


def video_env_factory(video_dir: Path) -> EnvFactory:
    """Return a factory for an ``rgb_array`` env that records every episode to ``video_dir``."""

    def factory() -> LunarLanderEnv:
        env = RecordVideo(
            make_env(render_mode=RENDER_MODE),
            video_folder=str(video_dir),
            episode_trigger=lambda _episode: True,
            name_prefix=VIDEO_NAME_PREFIX,
            disable_logger=True,
        )
        return cast(LunarLanderEnv, env)

    return factory


def record_episodes(
    policy: PolicyFn,
    settings: EpisodeSetConfig,
    session_dir: Path,
    extra: dict[str, Any],
) -> list[Path]:
    """Record ``settings.episodes`` episodes into ``session_dir``; return the video paths."""
    clip_dir = session_dir / VIDEO_SUBDIR
    results = run_episodes(
        policy,
        settings.episodes,
        settings.seed_start,
        env_factory=video_env_factory(clip_dir),
    )
    write_session(session_dir, results, settings, {**extra, "render_mode": RENDER_MODE})
    return sorted(clip_dir.glob(f"{VIDEO_NAME_PREFIX}-episode-*.mp4"))


def record_run_videos(
    run_dir: Path,
    *,
    model_filename: str = MODEL_FILENAME,
    episodes: int | None = None,
    seed_start: int | None = None,
    timestamp: str | None = None,
) -> list[Path]:
    """Record videos of a run's saved model into ``run_dir/videos/<timestamp>/``."""
    loaded = load_run_policy(
        run_dir,
        "video",
        model_filename=model_filename,
        episodes=episodes,
        seed_start=seed_start,
    )
    session_dir = create_session_dir(run_dir / VIDEOS_SUBDIR, timestamp)
    return record_episodes(
        loaded.policy, loaded.settings, session_dir, loaded.provenance
    )

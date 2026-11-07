# Portfolio Release Checklist

Checked in Phase 07 (2026-11-07). Each item names its evidence.

- [x] **README explains the problem, approach, setup, experiments and
  results.** `README.md`: results table generated from
  `results/phase05/`; figures from `results/phase06/`.
- [x] **Architecture diagram matches the implementation.**
  `docs/03-architecture.md` (Mermaid plus a plain-text version) and the
  module table, checked against `lunar_lander_rl/`.
- [x] **Installation tested on Windows 11 / Python 3.12.** Fresh clone and
  fresh venv from `requirements-lock.txt`: all checks and 337 tests pass
  (`docs/release-summary.md`). The project path must be short (under about
  100 characters) unless Windows long paths are enabled.
- [x] **Training and evaluation commands are reproducible.** Commands are
  in `docs/training-and-evaluation.md`, `docs/reward-experiments.md`,
  `docs/07-metrics.md` and `docs/visualization.md`. Every artifact records
  its config, seeds, package versions and git revision.
- [x] **Reward equations and limitations are documented.**
  `docs/reward-wrapper.md`; README "Limitations and future work".
- [x] **Evaluation uses the original environment reward.** `evaluate.py`
  builds the environment without the wrapper. Every evaluation summary
  says "original environment reward (no wrappers)"; checked by `compare`
  and tested.
- [x] **Results include multiple seeds or explain why not.** 3 training
  seeds per configuration, with the seed as the unit of replication
  (D-040).
- [x] **Graphs are generated from retained data.** `figures_manifest.json`
  holds the input hashes, and regenerated figures are byte-identical.
- [x] **Videos are real outputs from this project.** The clips are copied
  from recorded sessions, matched byte-for-byte to the committed
  `video_episodes.csv`, with SHA-256 in `videos.csv`. Only the upright
  clips have been reviewed by a person, as stated.
- [x] **No secrets, local paths or unnecessary large artifacts.** Git audit
  in `docs/release-summary.md`.
- [x] **Tests and static checks pass.** 337 tests; ruff and
  `mypy --strict` are clean.
- [x] **Known limitations and future work are honest and specific.** README
  section, and the Limitations sections of the experiment records.

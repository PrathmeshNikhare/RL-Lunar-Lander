# Release Summary

**Project:** reward engineering on Gymnasium `LunarLander-v3` with
Stable-Baselines3 PPO.
**Release:** the end of Phase 07 (publication), prepared on 2026-11-07.

## What the project delivers

- A typed Python package (`lunar_lander_rl/`) with command-line scripts:
  - environment exploration;
  - configuration-driven PPO training with per-episode reward-component
    logs;
  - evaluation and video recording on the original reward;
  - a controlled comparison;
  - figures, traces and clips built from committed data.
- **Four reward configurations**, each trained with 3 seeds (12 runs, 1M
  timesteps each): baseline, time penalty, a deliberately flawed
  upright-and-slow bonus, and potential-based shaping.
- **A controlled evaluation:** 100 identical deterministic episodes per run,
  on the original reward, with outcome rules defined from the environment's
  terminal semantics.
- **Committed evidence:**
  - `results/phase04/`: per-run copies;
  - `results/phase05/`: the comparison tables;
  - `results/phase06/`: 8 figures, per-step traces and 12 clips;
  - `experiments/records/`: write-ups for Phase 02 (baseline), Phase 04
    (one per configuration), Phase 05 and Phase 06.

## Results (from `results/phase05/`; details in the README)

| Configuration | Mean return (± sample std, 3 seeds) | Success, pooled |
|---|---|---|
| baseline | 243.9 ± 9.5 | 291/300 |
| time_penalty | 259.2 ± 12.5 | 296/300 |
| upright_slow_bonus | 3.0 ± 149.7 | 92/300 (206 timeouts) |
| potential_shaping | 245.8 ± 8.6 | 295/300 |

- **Upright bonus:** the flawed bonus was exploited. Late-training shaped
  totals were 950–979 while evaluation returns were −125 to 168. Two seeds
  mostly stay airborne. The third mostly lands, but sometimes rests on the
  pad without the +100 termination.
- **The other three:** 3 seeds cannot separate them.
- Analyses:
  - `experiments/records/2026-11-05-controlled-evaluation.md`;
  - `experiments/records/2026-11-07-visual-analysis.md`.

## Requirements traceability (`docs/02-requirements.md`)

| Requirement | Implementation | Tests |
|---|---|---|
| FR-01 Construct env with render mode and seed | `environment.py` | `test_environment.py`, `test_environment_setup.py` |
| FR-02 Random-policy smoke test | `rollout.py`, `scripts/random_rollout.py` | `test_random_rollout_cli.py` |
| FR-03 Configurable PPO training | `train.py`, `config.py`, `configs/*.yaml` | `test_train_smoke.py`, `test_config.py` |
| FR-04 Save model, config and logs per run | `artifacts.py`, `train.py` | `test_artifacts.py`, `test_train_smoke.py` |
| FR-05 Four named reward configurations | `reward_functions.py`, `reward_wrapper.py`, `configs/` | `test_reward_functions.py`, `test_reward_wrapper.py`, `test_experiment_configs.py` |
| FR-06 Log original and custom components separately | `reward_logging.py` (`reward_components.csv`) | `test_reward_logging.py` |
| FR-07 Evaluate with reward shaping disabled | `evaluate.py` | `test_evaluate.py` |
| FR-08 Return, success, crash, length | `metrics.py`, `comparison.py` | `test_metrics.py`, `test_comparison.py` |
| FR-09 Record representative episodes | `record_video.py`, `video_selection.py` | `test_record_video.py`, `test_video_selection.py` |
| FR-10 Comparison plots from retained raw metrics | `figures.py`, `plotting.py`, `traces.py` | `test_figures.py`, `test_traces.py` |
| NFR-01 to NFR-07 | Seeds and provenance in every artifact; `mypy --strict`; deterministic reward tests; smoke-only training in tests; `docs/setup-windows.md`; data-derived figures and labels; write-once outputs | `test_artifacts.py`, `test_scripts_cli.py` and the above |

## Verification

| Check | Result |
|---|---|
| `pytest -q` (project venv) | 337 passed |
| `ruff check .`, `ruff format --check .` | Clean |
| `mypy --strict lunar_lander_rl scripts tests` | No issues (47 files) |
| Clean install: `git clone` of the local repository at `31d02ca`, fresh `py -3.12` venv (3.12.5), `pip install --upgrade pip`, `requirements-lock.txt` | Install OK (177 s); `pip check` clean; `check_environment.py` all checks passed; 337 tests passed (52 s); ruff and `mypy --strict` clean; figures regenerated in the clone byte-identical to the committed PNGs |
| Clean-install conditions | The successful clone sat at a short path: the session's scratch folder was mapped to drive `P:` with `subst`, and `venv` printed a harmless warning about this. `31d02ca` precedes the Phase 07 changes, which touch only Markdown and one comment-only example config, so the code and tests walked through are those being released. The README wording itself was checked by the reviewers, not by a second install |
| Clean-install pitfall found | The first attempt, at a 111-character path, failed: a PyTorch wheel file exceeds the Windows 260-character path limit. Documented in `docs/setup-windows.md` and the README (use a short path or enable long paths). That attempt's log is invalid after the failure, including a false "hashes identical" line printed after an error |
| Figures regenerated from committed data | Byte-identical to the committed PNGs (Phase 06 and Phase 07) |
| README and docs relative links | All resolve (checked by script) |

**Git audit** (before the Phase 07 commit, which adds only Markdown files):
- 227 tracked files, 6.9 MB in total. The largest is
  `results/phase06/component_traces.csv` (1.0 MB). Nothing else exceeds
  500 KB.
- `runs/` (models, TensorBoard, trajectories), `.venv` and the caches are
  ignored. 12 small MP4 clips (1.1 MB) are committed on purpose (D-042).
- No API keys, tokens or password patterns appear in tracked files.
- No local paths, usernames or machine names appear in tracked files or
  anywhere in the history.
- Commit author metadata contains the author's name and email address.
- History up to `31d02ca` (Phase 06) is already on GitHub (`origin/main`).
- No `LICENSE` file is included, by the owner's choice: all rights are
  reserved.

## Known limitations

These are stated in the README and the records:
- three seeds;
- one magnitude per reward term;
- one hyperparameter set;
- post-hoc descriptive analyses, labelled as such;
- per-step trajectories kept local;
- only the upright-bonus clips have been reviewed by a person;
- one machine.

## Phase history

| Phase | Outcome | Handoff |
|---|---|---|
| 00 Foundation | Environment, dependencies, tooling | `progress/handoffs/2026-11-03-phase-00.md` |
| 01 Environment | Exploration and environment notes | `progress/handoffs/2026-11-03-phase-01.md` |
| 02 Baseline PPO | Training and evaluation pipeline, baseline run | `progress/handoffs/2026-11-03-phase-02.md` |
| 03 Reward wrapper | Reward terms, wrapper, logging | `progress/handoffs/2026-11-04-phase-03.md` |
| 04 Experiments | 12 runs | `progress/handoffs/2026-11-05-phase-04.md` |
| 05 Evaluation | Controlled comparison | `progress/handoffs/2026-11-05-phase-05.md` |
| 06 Visualization | Figures, traces, clips, analysis | `progress/handoffs/2026-11-07-phase-06.md` |
| 07 Publication | README, architecture, release checks | `progress/handoffs/2026-11-07-phase-07.md` |

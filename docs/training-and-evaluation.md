# Training, Evaluation and Video Recording

Commands are run from the project root in PowerShell (see `setup-windows.md`).
The three steps are separate programs. Evaluation and video recording only
load a saved model; they never reuse the training environment.

## 1. Train

```powershell
# Full baseline: 1,000,000 timesteps (launch explicitly; never run by tests)
.\.venv\Scripts\python.exe scripts\train.py --config configs\baseline.yaml

# Short run, e.g. for timing or debugging
.\.venv\Scripts\python.exe scripts\train.py --config configs\baseline.yaml --total-timesteps 20000
```

| Option | Meaning |
|---|---|
| `--config` | YAML experiment config (required) |
| `--total-timesteps` | Override `training.total_timesteps` |
| `--seed` | Override `training.seed` |
| `--runs-root` | Parent directory for runs (default `runs`) |
| `--quiet` | Hide SB3's per-rollout progress tables |

- The config is validated before anything starts. Unknown keys, missing keys,
  wrong types and out-of-range values are rejected with the offending key
  named.
- Cross-field checks are also applied:
  - `batch_size` must divide `n_steps × n_envs`;
  - `checkpoint_freq` must be a multiple of `n_envs`;
  - the training, evaluation and video seed ranges must not overlap.
- Overrides are validated again and recorded in the saved effective config.
- SB3 collects whole rollouts (`n_steps × n_envs` steps each), so
  `num_timesteps` in `metadata.json` can be slightly above the requested value.

## 2. Evaluate (original reward only)

```powershell
.\.venv\Scripts\python.exe scripts\evaluate.py --run-dir runs\baseline\<run>
.\.venv\Scripts\python.exe scripts\evaluate.py --run-dir runs\baseline\<run> --model checkpoints\ppo_500000_steps.zip
```

- Evaluation builds a fresh `LunarLander-v3` with `make_env()`, which has no
  reward wrappers.
- Only the `evaluation` section of the run's `config.json` is read (D-020), so
  old runs stay evaluable after later phases add config keys.
- Actions come from the saved model's `predict` (deterministic by default).
  Episode `i` uses seed `evaluation.seed_start + i`.
- `--episodes` and `--seed-start` override the run's config.
- Results go to a new `evaluations/<timestamp>/` folder:
  - `episodes.csv`, one row per episode: `episode`, `seed`,
    `original_return`, `length`, `terminated`, `truncated`, `final_reward`,
    `final_x`, `final_y`, `final_left_leg_contact`, `final_right_leg_contact`.
  - `summary.json`: settings, the env id and time limit, the model file name
    and its SHA-256, descriptive statistics, versions and git revision. Standard deviations are sample standard deviations (ddof = 1).
    The counts of final ±100 rewards are raw outcome tallies.
- Success and crash rates are **not** computed yet. Their definitions belong
  to Phase 05; see `environment-notes.md` for why "+100" is not the same as
  "landed on the pad".

## 3. Record videos

```powershell
.\.venv\Scripts\python.exe scripts\record_video.py --run-dir runs\baseline\<run>
```

- Records `video.episodes` episodes (seeds from `video.seed_start`) with
  Gymnasium's `RecordVideo`, which uses moviepy.
- Clips are written to `videos/<timestamp>/mp4/`.
- An `episodes.csv` and `summary.json` are written alongside the clips, so
  each clip can be matched to its seed and return.

## Run directory layout

```
runs/<experiment_id>/<YYYYmmddTHHMMSSZ>_seed<N>/
  config.json                 effective config (written before training)
  metadata.json               status, timings, timesteps, seeds, torch threads,
                              effective_hyperparameters (read back from the model,
                              incl. implicit SB3 defaults), versions, argv, and the
                              git revision at launch (commit, dirty, diff_sha256, untracked)
  model.zip                   final model
  monitor/<rank>.monitor.csv  per-episode training return/length (original reward)
  reward_components.csv       per training episode: timesteps, env_rank, length,
                              original, each custom component, total (shaped)
  tensorboard/ppo_1/          SB3 scalars (+ reward/<component> means):
                              tensorboard --logdir runs
  checkpoints/ppo_<steps>_steps.zip
  evaluations/<timestamp>/    episodes.csv, summary.json
  videos/<timestamp>/         episodes.csv, summary.json, mp4/*.mp4
```

- Every directory and file above is created exclusively. Re-running never
  overwrites an earlier run, evaluation or video session. A name collision
  raises `FileExistsError`.
- `runs/` is git-ignored. Small summaries are copied into `results/`, and
  experiment write-ups go in `experiments/records/` (D-018).

## Baseline configuration (`configs/baseline.yaml`)

- **PPO values:** taken from the RL Baselines3 Zoo `LunarLander-v3` entry
  (`hyperparams/ppo.yml`, read 2026-11-03):
  - `n_envs 16`, `n_steps 1024`, `batch_size 64`, `n_epochs 4`;
  - `gamma 0.999`, `gae_lambda 0.98`, `ent_coef 0.01`;
  - 1e6 timesteps.
- **Other values:** SB3 2.9.2 defaults, written out explicitly.
- **Network:** `net_arch [64, 64]` (D-007).
- **Discount coupling:** potential-based shaping always uses `ppo.gamma`
  (D-019); there is no separate reward-level `gamma`.
- **Reward modes:** the `reward` section selects one of four modes. See
  `docs/reward-wrapper.md` for the keys, formulas and logging.
- **Phase 04 experiments:** the four configs, chosen values, seeds and run
  commands are in `docs/reward-experiments.md`.

### Seeds
| Purpose | Seeds |
|---|---|
| Training (SB3 global RNGs) | `training.seed` (42) |
| Training environments (first reset) | 42 … 57 (`seed + i`, 16 envs) |
| Evaluation episodes | 10000 … 10099 |
| Video episodes | 20000 … 20002 |

Re-running with the same config, code and package versions on the same machine
is expected to reproduce training, but bit-for-bit reproducibility across
machines or library versions is not guaranteed.

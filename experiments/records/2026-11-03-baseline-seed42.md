# Experiment Record: Baseline PPO, seed 42

- **Experiment ID:** `baseline`
- **Reward configuration:** `original`, the unmodified LunarLander-v3 reward
- **Date:** 2026-11-03
- **Git commit:** `0f125a5` ("Phase 02 (1/2)"). The working tree was clean at
  launch (`dirty: false`, no untracked files).
- **Environment and package versions:**
  - LunarLander-v3 with default physics and discrete actions;
  - gymnasium 1.4.0, Box2D 2.3.10, stable-baselines3 2.9.2, torch 2.15.0
    (CPU, 10 threads), numpy 2.5.4;
  - Python 3.12.5 on Windows 11.
- **Training seed:** 42. The 16 vectorised environments were seeded 42–57 at
  their first reset.
- **Evaluation seeds:** 10000–10099 (100 episodes, deterministic actions).
  Video seeds: 20000–20002.
- **Timesteps:** 1,000,000 requested, 1,015,808 trained (62 rollouts of
  16 × 1024 steps).
- **PPO hyperparameters:**
  - `MlpPolicy` `[64, 64]` with Tanh activation and orthogonal init;
  - `n_envs` 16, `n_steps` 1024, `batch_size` 64, `n_epochs` 4;
  - `gamma` 0.999, `gae_lambda` 0.98, `ent_coef` 0.01;
  - `learning_rate` 3e-4, `clip_range` 0.2, `vf_coef` 0.5,
    `max_grad_norm` 0.5, `normalize_advantage` true, `target_kl` none.

  Source: RL Zoo `LunarLander-v3` plus SB3 defaults. The full snapshot is in
  `results/baseline/run_metadata.json`.
- **Reward parameters:** none, because the original reward is used.
- **Checkpoint path:** `runs/baseline/20261103T174958Z_seed42/model.zip`
  (SHA-256 `1f93ba1a3fee8fde…`, full value in
  `results/baseline/evaluation_summary.json`), plus checkpoints every 100k
  steps. These files are git-ignored and exist locally only.
- **Logs path:** `runs/baseline/20261103T174958Z_seed42/`, which holds
  `monitor/`, `tensorboard/`, `evaluations/20261103T175458Z/` and
  `videos/20261103T175515Z/`. These are local and git-ignored. The small
  copies committed to the repository are in `results/baseline/`.

## Training
- **Duration:** 259.4 s wall-clock (status `completed`).
- **Training return:** SB3's `rollout/ep_rew_mean` (mean of the last 100
  training episodes, stochastic policy, original reward) went from −113 at
  98k steps to 41.9 at 393k, 129 at 688k and 215 at 983k steps. These values
  come from the training log. They are not evaluation results.
- **Notes:**
  - An earlier run with the same seed and config
    (`runs/baseline/20261103T173151Z_seed42`) was trained from uncommitted,
    pre-review code. It is **provisional and not used** (decision D-023).
  - Its logged training curve and its 100-episode evaluation (mean 237.56)
    were identical to this run's.
  - A 32,768-step timing run (`runs/baseline/20261103T172838Z_seed0`) is also
    not a result.

## Evaluation (original reward only)
| Metric | Value |
|---|---|
| Number of episodes | 100 |
| Mean return | **237.56** |
| Return variability | sample std 52.37; median 246.73; min −29.53; max 296.63 |
| Episodes with return ≥ 200 | 92 / 100 |
| Mean episode length | 391.8 steps (sample std 116.1; min 326; max 1000) |
| Terminated / truncated | 97 / 3 |
| Final reward +100 (at rest) / −100 | 95 / 2 |
| Success rate | Not computed. The definition is deferred to Phase 05 |
| Crash rate | Not computed. The definition is deferred to Phase 05 |

The mean return is above the registered `reward_threshold` of 200. This
result comes from **one training seed**; the variability between training
seeds is unknown until Phase 04/05.

Per-episode data: `results/baseline/evaluation_episodes.csv`. Full summary:
`results/baseline/evaluation_summary.json`.

## Behavioural observations
The following are observations from the per-episode data, with no
interpretation of causes.

- **Two episodes ended with −100** (seeds 10058 and 10060; returns −7.0 and
  −29.5; about 350 steps). Their final observations show both legs in
  contact and `x` of 0.154 and 0.195.
- **Three episodes were truncated at 1000 steps** (seeds 10027, 10045 and
  10091; returns 87.9–106.7).
  - Each final observation is at ground level (`y` ≈ −0.04) with only the
    right leg in contact, at `x` 0.26–0.29, just outside the pad
    (`|x| ≤ 0.2`).
  - The lander had not come to rest by the time limit.
- **95 episodes ended at rest (+100).** In 91 of them `|x| ≤ 0.2`; 4 came to
  rest outside the pad.
  - In 3 of those 4, the final leg-contact flags were both 0. This is
    consistent with the caveat in `docs/environment-notes.md` that leg flags
    are unreliable at termination.
- **Videos:**
  - 3 clips: `runs/baseline/20261103T174958Z_seed42/videos/20261103T175515Z/mp4/episode-episode-{0,1,2}.mp4`
    (local, git-ignored).
  - Their per-episode data (seeds 20000–20002): returns 268.4, 272.9 and
    234.7, all ending at rest (+100), in 370, 392 and 440 steps.
  - **The clips have not been reviewed visually yet,** so no claims are made
    about the flight behaviour they show.

## Deviations and limitations
- There is one training seed. The experiment protocol asks for at least three
  where compute permits; that is planned for Phase 04.
- Success and crash metrics are not defined yet (Phase 05). The ±100 counts
  above are raw tallies of the final reward, not outcome labels.
- Reproducibility was observed on one machine only. Two runs with the same
  seed and config gave identical training curves and identical evaluation
  results. Results on other hardware or library versions were not tested.
- `moviepy` 2.2.1 required pillow 11.3.0 (downgraded from 13.0.0). This
  affects rendering only, not training (decision D-021).

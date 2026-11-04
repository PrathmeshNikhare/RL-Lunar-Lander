# Experiment Record: Phase 04 B — Time penalty, seeds 42/142/242

- **Experiment ID:** `time_penalty` (`configs/time_penalty.yaml`)
- **Reward configuration:** original reward minus `coefficient` 0.05 per
  step during training. The formula is in `docs/reward-wrapper.md`, and the
  reason for the value in `docs/reward-experiments.md`.
- **Date:** 2026-11-04
- **Git commit:** `74f4f36`. Every run recorded `dirty: false` with no
  untracked files. The other settings are shared with all Phase 04 runs (see
  `2026-11-04-baseline.md`).
- **Committed copies:** `results/phase04/time_penalty/seed{42,142,242}/`.

## Runs
| Seed | Run directory | Training time | Model SHA-256 (prefix) |
|---|---|---|---|
| 42 | `runs/time_penalty/20261103T215919Z_seed42` | 282.0 s | `cd077f0bddaf61fa` |
| 142 | `runs/time_penalty/20261103T221837Z_seed142` | 267.7 s | `b8a7823d14e1c98c` |
| 242 | `runs/time_penalty/20261104T181910Z_seed242` | 286.7 s | `87ecd58799e60c11` |

**The first seed-242 attempt was interrupted:**
`runs/time_penalty/20261103T223642Z_seed242`.
- It was killed at about 200k steps when the session running the sweep
  ended. It has checkpoints but no `metadata.json` and no `model.zip`.
- It is kept locally, is **not a result** and is not used.
- Seed 242 was then retrained from scratch from the same commit.
- The interrupted run's 100k and 200k checkpoints have byte-identical
  `policy.pth` and `policy.optimizer.pth` to the retrained run's checkpoints
  at the same steps.
- Phase 05 tooling should select runs by `metadata.json` with
  `status: completed`. That excludes this run, which a `*_seed242` glob would
  otherwise pick up.

## Evaluation (original reward only)
| Seed | Mean return | Sample std | Median | Min | Mean length | Terminated / truncated | Final +100 / −100 | Return ≥ 200 |
|---|---|---|---|---|---|---|---|---|
| 42 | 245.57 | 32.85 | 244.9 | 115.0 | 378.6 | 96 / 4 | 96 / 0 | 96 |
| 142 | 269.93 | 21.21 | 269.0 | 230.2 | 279.0 | 100 / 0 | 100 / 0 | 100 |
| 242 | 262.25 | 20.89 | 262.5 | 219.0 | 311.0 | 100 / 0 | 100 / 0 | 100 |

## Observations
- **Seed 42 had 4 truncated episodes** (returns 115.0–173.7).
  - Each final observation is at ground level (y between −0.039 and −0.001),
    and 4 of them show at least one leg in contact.
  - The largest final |x| was 0.277, outside the pad (|x| ≤ 0.2).
  - These episodes did not come to rest within 1000 steps. This is the same
    pattern as the baseline seed-42 truncations.
- **Evaluation episodes were shorter than the baseline's** for the same seed:
  379, 279 and 311 mean steps, against 392, 437 and 356. With three seeds this
  is an observation, not a tested effect.
- **Training episodes ran much longer than evaluation episodes.** For seeds
  42 and 142, the training-time episodes ending at or after 900k steps had
  mean lengths of 977 and 834 steps, against 379 and 279 in evaluation (deterministic
  actions).
  - Over those training episodes the penalty averaged −48.9 and −41.7. For
    seed 42 that is close to the −50 maximum, not the "−20 over a typical
    landing" assumed in the rationale.
  - 96%, 65% and 12% of those episodes (seeds 42, 142, 242) reached 1000
    steps. The baseline figures are 7%, 20% and 2%.
  - The cause was not investigated; the training policy is stochastic. Note
    it for Phase 05.
- **Videos:** 3 clips per run. Every clip ended at rest (+100), with returns
  235–299. **They have not been reviewed visually yet.**

## Limitations
- There are three seeds and one coefficient. No claim is made that the
  penalty causes faster landings in general.

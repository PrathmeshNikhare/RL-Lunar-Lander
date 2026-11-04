# Experiment Record: Phase 04 C — Upright-and-slow bonus, seeds 42/142/242

- **Experiment ID:** `upright_slow_bonus` (`configs/upright_slow_bonus.yaml`)
- **Reward configuration:** original reward + `bonus` 1.0 for each step in
  which |wrap(angle)| ≤ 0.2 rad and speed ‖(vx, vy)‖ ≤ 0.2 (scaled units).
  - The bonus deliberately ignores position, height and legs (D-029). It was
    designed as a probe for reward hacking.
  - Formula: `docs/reward-wrapper.md`. Reason for the values:
    `docs/reward-experiments.md`.
- **Date:** 2026-11-04
- **Git commit:** `74f4f36`. Every run recorded `dirty: false` with no
  untracked files. The other settings are shared with all Phase 04 runs (see
  `2026-11-04-baseline.md`).
- **Committed copies:**
  `results/phase04/upright_slow_bonus/seed{42,142,242}/`.

## Runs
| Seed | Run directory | Training time | Model SHA-256 (prefix) |
|---|---|---|---|
| 42 | `runs/upright_slow_bonus/20261103T220405Z_seed42` | 293.8 s | `edde6a982b2ad55a` |
| 142 | `runs/upright_slow_bonus/20261103T222308Z_seed142` | 270.5 s | `bab2ff5b90d70b32` |
| 242 | `runs/upright_slow_bonus/20261104T182401Z_seed242` | 560.5 s | `9e3d7b77eb58aa52` |

Seed 242 took about 2× as long as the other seeds. It ran after the
interrupted sweep, and the cause was not investigated. Seed 142, which has
similarly long episodes, took 270.5 s, so the longer episodes alone do not
explain it.

## Evaluation (original reward only)
| Seed | Mean return | Sample std | Median | Min | Max | Mean length | Terminated / truncated | Final +100 / −100 | Return ≥ 200 |
|---|---|---|---|---|---|---|---|---|---|
| 42 | 167.71 | 46.98 | 170.7 | 7.0 | 248.7 | 776.1 | 93 / 7 | 92 / 1 | 22 |
| 142 | −34.00 | 38.20 | −36.8 | −83.6 | 176.3 | 998.8 | 1 / 99 | 1 / 0 | 0 |
| 242 | −124.74 | 38.35 | −135.7 | −181.3 | −5.6 | 1000.0 | 0 / 100 | 0 / 0 | 0 |

## Training-time reward components
These are training episodes (stochastic policy) ending at or after 900k
steps, from `reward_components.csv`.

| Seed | Episodes | Mean length | Original | Bonus | Shaped total |
|---|---|---|---|---|---|
| 42 | 117 | 994 | 127.3 | 823.1 | 950.4 |
| 142 | 125 | 953 | 80.4 | 879.0 | 959.5 |
| 242 | 119 | 991 | 43.4 | 935.7 | 979.1 |

In all three seeds, the bonus is most of the training return late in
training: 86.6%, 91.6% and 95.6% of the mean shaped total (seeds 42, 142,
242). Of these training episodes, 99%, 91% and 98% reached 1000 steps.

## Measured behaviour
The evidence here is the evaluation's final observations and episode
lengths, plus the project owner's viewing of the video clips of all three
seeds (2026-11-05). Each seed's clips exist in two byte-identical recording
sessions (D-034).

- **Seed 242: hovering in the air, confirmed visually.**
  - The project owner watched the three seed-242 clips and reported that
    the lander hovers.
  - In evaluation, 41 of 100 episodes end above y = 1.4 (about the spawn
    height) and 82 above y = 1.0.
  - All 100 episodes reached the 1000-step time limit.
  - At truncation the lander's height y was 0.272–1.781 (mean 1.277), with
    no leg contact in any episode.
  - Final |x| was at most 0.200.
  - The original return was negative in every episode (−181.3 to −5.6).
- **Seed 142: hovering above the surface, confirmed visually.**
  - The project owner watched the seed-142 clips. The lander hovers well
    above the surface and looks stable, with both legs at almost the same
    level.
  - 99 of 100 evaluation episodes reached the time limit.
  - At truncation y was −0.001 to 0.681 (mean 0.370). 4 of the 99 ended at
    ground level (y < 0.05), and 3 had leg contact. That is lower than seed
    242's final heights (mean 1.277).
  - One episode ended at rest (+100), with return 176.3.
- **Seed 42: lands on the pad, late.**
  - The project owner watched the seed-42 clips: in all three, the lander
    lands cleanly between the two flags.
  - Those clips are long (551, 906 and 971 steps).
  - 93 episodes terminated, 92 of them at rest (+100), with a mean length
    of 776 steps (median 799, minimum 388) against 392 for the baseline
    seed 42.
  - All 7 truncated episodes end at ground level (y −0.001 to 0.034, |x| ≤
    0.100, 4 with leg contact). They do not show hovering in the air.
  - The final states cannot tell a slow descent apart from staying on the
    ground and keeping the body awake.
    - Staying awake on the ground can still earn the bonus, which
      `docs/reward-experiments.md` itself notes.
    - Loitering on the ground would also fit the low original return,
      through fuel spent after touchdown.
  - Only 22 episodes reached a return of 200, against 92 for the baseline
    seed 42.
  - The viewing report does not say how the long durations split between
    descent and time on the ground, and time to first leg contact was not
    measured.
- **Video clips** (seeds 20000–20002):
  - seed 42: returns 164.9, 249.7, 192.4; lengths 906, 551, 971; all ended
    at rest;
  - seed 142: all three ran 1000 steps, returns −37.3, −21.9, −32.4;
  - seed 242: all three ran 1000 steps, returns −145.8, −91.3, −112.9.

## Interpretation (limited)
- **Reward hacking observed in two of three seeds (142 and 242).** The
  trained policy hovers instead of landing, which keeps earning the bonus
  until the time limit. That is the hack this probe was designed to expose.
  - Evidence: the video clips (both seeds), truncation counts and end
    positions in evaluation, and a bonus earned on most training steps,
    86.6–95.6% of the training return (from `reward_components.csv`).
  - Evaluation does not log angle or speed. "Upright and slow" is therefore
    shown by the training-time bonus and the clips, not measured in
    evaluation.
  - Hover rates rest on 100 evaluation episodes per seed and 3 viewed clips.
    A formal hover definition and rate belong to Phase 05.
- **The outcome depends on the seed.** Seed 42 lands on the pad, in the
  clips and in 92 of 100 evaluation episodes, but slowly: mean 776 steps,
  and 22 of 100 reach a return of 200.
  - Whether the extra time is spent descending or on the ground (a milder
    form of the same incentive) is open. Phase 05 can measure time to first
    leg contact.
- Phase 05 defines the hover, success and crash rules and does the
  cross-seed comparison.

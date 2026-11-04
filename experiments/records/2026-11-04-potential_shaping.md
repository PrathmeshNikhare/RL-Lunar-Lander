# Experiment Record: Phase 04 D — Potential-based shaping, seeds 42/142/242

- **Experiment ID:** `potential_shaping` (`configs/potential_shaping.yaml`)
- **Reward configuration:** original + F with F = α(γΦ(s′) − Φ(s)).
  - α = 100, unit weights, γ = `ppo.gamma` = 0.999.
  - Φ(s′) = 0 only when the episode is `terminated` (D-027).
  - Definitions are in `docs/reward-wrapper.md`.
- **Date:** 2026-11-04
- **Git commit:** `74f4f36`. Every run recorded `dirty: false` with no
  untracked files. The other settings are shared with all Phase 04 runs (see
  `2026-11-04-baseline.md`).
- **Committed copies:** `results/phase04/potential_shaping/seed{42,142,242}/`.

## Runs
| Seed | Run directory | Training time | Model SHA-256 (prefix) |
|---|---|---|---|
| 42 | `runs/potential_shaping/20261103T220904Z_seed42` | 287.2 s | `f054a8848d7ed7d7` |
| 142 | `runs/potential_shaping/20261103T222742Z_seed142` | 274.0 s | `dd58523929254886` |
| 242 | `runs/potential_shaping/20261104T183327Z_seed242` | 651.0 s | `486039dec8c25897` |

Seed 242 took about 2.3× as long as the other seeds. It ran alone, after the
interrupted sweep. The cause was not investigated, and the run's results are
unaffected (`status: completed`, 1,015,808 steps).

## Evaluation (original reward only)
| Seed | Mean return | Sample std | Median | Min | Mean length | Terminated / truncated | Final +100 / −100 | Return ≥ 200 |
|---|---|---|---|---|---|---|---|---|
| 42 | 250.01 | 44.94 | 256.8 | −5.9 | 354.8 | 100 / 0 | 97 / 3 | 97 |
| 142 | 251.58 | 21.12 | 250.4 | 211.9 | 380.1 | 100 / 0 | 100 / 0 | 100 |
| 242 | 235.95 | 29.80 | 239.1 | 90.2 | 440.4 | 98 / 2 | 98 / 0 | 96 |

## Observations
- **Shaping total per training episode.** In `reward_components.csv`, the
  shaping component averaged +219.9, +220.4 and +220.4 per episode (episodes
  ending at or after 900k steps). It is almost the same across the three
  seeds.
  - The spawn potential gives a smaller reference value. Φ(s₀) measured at
    reset for the training env seeds (42–57, 142–157, 242–257) and the
    evaluation seeds averages −1.96 (−2.39 to −1.51), so −αΦ(s₀) is about
    196 (151 to 239). The Reward Engineer measured this with the repository's
    `potential()`.
  - **Only the discounted sum telescopes exactly to −αΦ(s₀),** and only for
    episodes that end by termination. The undiscounted sum is
    ΣF = −αΦ(s₀) + α(1−γ)·Σₜ₌₁…T₋₁|Φ(sₜ)| − αγ|Φ(s_T)|·[truncated].
    - With α(1−γ) = 0.1, the middle term is positive and grows with episode
      length.
    - Many late training episodes ran the full 1000 steps (see below).
    - The ≈ +24 gap between 220 and 196 is therefore consistent with the
      γ < 1 term, net of the truncation term. That size is inferred from the
      gap, not measured per step.
  - The logged undiscounted total is **not** the quantity that potential
    shaping leaves unchanged. Shaped totals should not be compared across
    policies.
  - This is a sanity check on the shaping term, not evidence of policy
    invariance.
- **Training episodes truncated far more often than evaluation episodes.**
  Of the training episodes ending at or after 900k steps, 91%, 87% and 20%
  (seeds 42, 142, 242) reached 1000 steps.
  - The baseline figures are 7%, 20% and 2%.
  - In evaluation (deterministic actions) there were 0, 0 and 2 truncations.
  - The cause was not investigated.
- **Seed 42 had three −100 episodes** (seeds 10041, 10058 and 10098; returns
  27.1, −5.9 and 60.0). Their final |x| ≤ 0.141.
- **Seed 242 had two truncated episodes** (returns 90.2 and 97.5).
  - Both end at ground level (y = −0.022) with leg contact, at |x| up to
    0.203.
  - They did not come to rest within 1000 steps.
- **Videos:** 3 clips per run. Every clip ended at rest (+100), with returns
  237–292. **They have not been reviewed visually yet.**

## Limitations
- Policy invariance holds for the optimal policy under the assumptions in
  `docs/reward-wrapper.md`. Similar evaluation returns after 1M PPO steps
  neither prove nor disprove it.

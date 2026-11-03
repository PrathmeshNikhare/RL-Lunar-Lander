# Decision Log

| ID | Decision | Rationale | Status |
|---|---|---|---|
| D-001 | Intermediate scope | Focus on reward engineering, not algorithm breadth | Accepted |
| D-002 | Portfolio objective | Emphasize reproducibility, evidence and clear README | Accepted |
| D-003 | Local Windows 11 development | User-selected environment | Accepted |
| D-004 | Python 3.12 target | Conservative compatibility choice for Box2D ecosystem | Accepted |
| D-005 | Gymnasium LunarLander-v3 | Current environment target | Accepted |
| D-006 | Stable-Baselines3 PPO / MlpPolicy | Avoid implementing PPO from scratch | Accepted |
| D-007 | Initial network [64, 64] | Small baseline architecture | Accepted |
| D-008 | Four reward configurations | Baseline, time penalty, flawed bonus, potential shaping | Accepted |
| D-009 | Original reward for all evaluations | Fair comparison of actual task performance | Accepted |
| D-010 | Phase-gated Claude Code workflow | Prevent uncontrolled implementation and preserve review gates | Accepted |
| D-011 | 2026-11-03: Dependencies in `requirements.txt` / `requirements-dev.txt` / `requirements-lock.txt` (ranges + exact freeze) | Alternative `pyproject.toml` + `[dev]` extra would need an empty package before Phase 01; revisit when a package layout exists | Accepted (user, Phase 00 plan) |
| D-012 | 2026-11-03: Dev tools pytest, ruff (<0.18), mypy `--strict` | Supports typed-module rule; ruff capped because pre-1.0 minors change rules | Accepted (user, Phase 00 plan) |
| D-013 | 2026-11-03: Ignore only the two Box2D SWIG DeprecationWarning patterns in `pytest.ini` | Upstream and not actionable; as errors they crash the interpreter (access violation) instead of failing a test; a blanket ignore would hide real warnings | Accepted (Test Engineer M1; user review at checkpoint) |
| D-014 | 2026-11-03: Phase checkpoints committed locally; no push to `origin` without explicit approval | Outward-facing action; user chose local-only | Accepted (user, Phase 00 plan) |
| D-015 | 2026-11-03: Root-level package `lunar_lander_rl/`, not installed; pytest `pythonpath = .`; scripts add the repo root to `sys.path` | Keeps D-011 (no `pyproject.toml`). Alternative: `src/` layout + `pip install -e .`, which is more robust but needs packaging metadata now. Trade-off: commands must run from the repo root, and each script carries a small path bootstrap. Revisit if packaging is introduced | Accepted (user, Phase 01 plan) |
| D-016 | 2026-11-03: Exploration-tool parameters via argparse with documented defaults; YAML config + validation deferred to Phase 02 (training) | Avoids adding PyYAML before a config consumer exists | Accepted (user, Phase 01 plan) |
| D-017 | 2026-11-03: Baseline PPO hyperparameters from the RL Zoo `LunarLander-v3` entry (n_envs 16, n_steps 1024, batch 64, n_epochs 4, gamma 0.999, gae_lambda 0.98, ent_coef 0.01, 1e6 steps). Other values are SB3 2.9.2 defaults written out explicitly. `DummyVecEnv` on CPU | A documented source tuned for this env and budget. Alternative: SB3 defaults with 1 env, simpler but less likely to reach 200 in 1M steps | Accepted (user, Phase 02 plan) |
| D-018 | 2026-11-03: `runs/` (models, checkpoints, TensorBoard, monitor, videos) stays git-ignored. Small result copies (config, metadata, evaluation summary, per-episode CSVs) go in `results/<experiment>/`; write-ups go in `experiments/records/` | Large or binary artifacts stay out of Git (CLAUDE.md), while per-episode raw data is still retained (experiment protocol) | Accepted (user, Phase 02 plan) |
| D-019 | 2026-11-03: Phase 03 potential-based shaping must use the same discount as PPO (`ppo.gamma`, 0.999 in the baseline) | Policy invariance of F = γΦ(s') − Φ(s) relies on the γ the learner optimises | Accepted (Phase 02 plan; to be enforced in Phase 03) |
| D-020 | 2026-11-03: Evaluation and video tools read only the `evaluation`/`video` section of a run's `config.json` | Keeps earlier runs evaluable when later phases add config keys (Architecture Reviewer M1) | Accepted (review fix) |
| D-021 | 2026-11-03: Add `moviepy` 2.2.1 for `RecordVideo`. This forces pillow 13.0.0 → 11.3.0; `pip check` is clean | Gymnasium's standard video path. The downgrade affects rendering only. Alternative: GIFs via Pillow with no new dependency | Accepted (user, Phase 02 plan) |
| D-022 | 2026-11-03: YAML experiment configs loaded with `yaml.safe_load` (PyYAML), strictly validated; the effective config is saved as JSON | Matches the harness `experiments.yaml.example`. Alternative: TOML via `tomllib`, with no dependency | Accepted (user, Phase 02 plan) |
| D-023 | 2026-11-03: The official baseline was trained from the clean commit `0f125a5` (Phase 02 1/2). The earlier run from uncommitted code (`runs/baseline/20261103T173151Z_seed42`) is kept locally, labelled provisional, and not used | Reproducible provenance for the reference run (RL Engineer M1). Alternative: keep the provisional run and document the gap | Accepted (user) |
| D-024 | 2026-11-03: Phase 02 evaluation records raw per-episode outcomes (return, length, flags, final reward, final x/y, legs). Success and crash labels are defined in Phase 05 | `+100` ≠ "on the pad", and leg flags are unreliable at termination (docs/environment-notes.md) | Accepted (user, Phase 02 plan) |

Add new decisions with date, alternatives, trade-offs and user approval status.

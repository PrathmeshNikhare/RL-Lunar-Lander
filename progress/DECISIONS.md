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

Add new decisions with date, alternatives, trade-offs and user approval status.

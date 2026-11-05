# Phase Log

For each phase, append:
- date;
- phase identifier;
- plan approval reference/summary;
- files changed;
- checks run and results;
- specialist review;
- acceptance decision;
- Git checkpoint;
- handoff path.

---

## 2026-11-03 — 00-foundation

- **Plan approval:** The user approved the plan as presented ("approve"), with
  the recommended defaults: three requirements files, pytest/ruff/mypy, the
  README encoding fix in scope, and a local commit only.
- **Files changed:** `requirements.txt`, `requirements-dev.txt`,
  `requirements-lock.txt`, `pytest.ini`, `scripts/check_environment.py`,
  `tests/test_environment_setup.py`, `docs/setup-windows.md`, `README.md`,
  `progress/*`.
- **Checks:**
  - venv Python 3.12.5, pip 26.3.
  - `pip check` clean.
  - `check_environment.py` exit 0.
  - pytest 10 passed.
  - ruff check/format clean.
  - `mypy --strict` clean.
  - A fresh install from `requirements-lock.txt` passed the same checks.
  - `.venv` is git-ignored.
- **Specialist review:**
  - Test Engineer: accept after fixing M1 (narrow SWIG warning filter). L4 (64-bit check) was added.
  - Documentation Reviewer: revise on M1 (lock-file install path unverified), resolved by the fresh install. L1 and L3 were applied.
  - Deferred, non-blocking items are listed in the handoff.
- **Acceptance decision:** Accepted. All acceptance criteria are evidenced, and the user approved the checkpoint commit ("commit").
- **Git checkpoint:** "Phase 00: foundation and local setup" (local commit on `main`, not pushed).
- **Handoff:** `progress/handoffs/2026-11-03-phase-00.md`

---

## 2026-11-03 — 01-environment

- **Plan approval:** The user approved the plan as presented ("approve"), with
  the recommended defaults:
  - root-level package, not installed (D-015);
  - argparse parameters (D-016);
  - the human-render check done by the user.

  The user separately approved applying the RL Engineer doc fixes ("apply the
  remaining doc fixes").
- **Files changed:**
  - `lunar_lander_rl/__init__.py`, `lunar_lander_rl/environment.py`,
    `lunar_lander_rl/rollout.py`
  - `scripts/random_rollout.py`
  - `tests/test_environment.py`, `tests/test_random_rollout_cli.py`
  - `docs/environment-notes.md`, `pytest.ini`
  - `progress/*`
- **Checks:**
  - pytest: 37 passed (10 Phase 00 + 22 environment/rollout + 5 command-line).
  - ruff check/format: clean.
  - `mypy --strict`: clean on 8 files.
  - Headless rollout of 3 episodes: exit 0, `terminated=True` after
    66/99/106 steps.
  - The user's manual human-render run: "The lander tumbled to the left once
    and to the right once." Its output matched the headless run.
  - `pip check`: clean.
- **Specialist review:**
  - RL Engineer: ACCEPT with doc revisions M1–M5. All were verified
    independently and applied. N1 (negative `--seed`) is deferred.
  - Test Engineer: ACCEPT with M1 recommended. M1 was added; L1–L5 and the
    command-line smoke test were applied.
- **Acceptance decision:** Accepted. All acceptance criteria are evidenced,
  and the user approved the checkpoint commit ("commit").
- **Git checkpoint:** "Phase 01: environment exploration" (local commit on
  `main`, not pushed).
- **Handoff:** `progress/handoffs/2026-11-03-phase-01.md`

---

## 2026-11-03 — 02-baseline-ppo

- **Plan approval:** The user approved the plan as presented ("approve"):
  - RL Zoo hyperparameters (D-017);
  - YAML (D-022);
  - moviepy (D-021);
  - full run in this phase (D-018).

  The user then approved the full run ("run it"). After the reviews, they
  chose "Commit, then re-run" for the official baseline (D-023).
- **Files changed:** see `progress/handoffs/2026-11-03-phase-02.md` (package
  modules, scripts, config, tests, docs, requirements, `results/baseline/`,
  experiment record, progress files).
- **Checks:**
  - pytest 123 passed; ruff and `mypy --strict` clean; `pip check` clean.
  - Timing run: 32,768 steps in 10.4 s.
  - Official 1M run from `0f125a5`: 1,015,808 steps in 259.4 s, status
    completed, `dirty: false`.
  - Evaluation: 100 episodes, mean original return 237.56, sample std 52.37,
    92/100 ≥ 200.
  - 3 videos recorded.
- **Specialist review:** RL Engineer, Architecture Reviewer and Test Engineer
  all recommended REVISE (minor/light). All required fixes were applied and
  are covered by new tests. Accepted remainders are listed in the handoff.
- **Acceptance decision:** Accepted. All acceptance criteria are evidenced,
  and the user approved the checkpoint commit ("commit").
- **Git checkpoint:** `0f125a5` "Phase 02 (1/2)" (code), then "Phase 02 (2/2)"
  (results, record, progress). Both are local commits on `main`, not pushed.
- **Handoff:** `progress/handoffs/2026-11-03-phase-02.md`

---

## 2026-11-04 — 03-reward-wrapper

- **Plan approval:** The user approved the plan with all six recommended
  decisions ("approve"): D-025 to D-030.
- **Files changed:** see `progress/handoffs/2026-11-04-phase-03.md` (three new
  modules, config/train changes, four test files new or extended, reward
  docs, progress files).
- **Checks:** pytest 202 passed (~15 s); ruff and `mypy --strict` clean. No
  long training (largest test run: 2,048 steps).
- **Specialist review:** Reward Engineer REVISE (M1 angle clipping, fixed;
  L1–L4 documented). Test Engineer REVISE (M1–M3 and L4/L5/L7 fixed; L6, L8,
  L9 accepted). Architecture Reviewer ACCEPT (L1–L5 applied).
- **Acceptance decision:** Accepted. All acceptance criteria are evidenced,
  and the user approved the checkpoint commit ("commit").
- **Git checkpoint:** "Phase 03: reward wrapper" (local commit on `main`, not
  pushed).
- **Handoff:** `progress/handoffs/2026-11-04-phase-03.md`

---

## 2026-11-04/05 — 04-reward-experiments

- **Plan approval:** the configs and protocol were committed as `74f4f36`
  (D-031, D-032). The user's approval wording was not preserved. The user
  approved retraining the three missing seed-242 runs (D-033) and the
  evaluation, copy, review and progress steps (D-034).
- **Files changed:** see `progress/handoffs/2026-11-05-phase-04.md`
  (`results/phase04/`, four records, docs window section, progress files).
- **Checks:**
  - Training: 12/12 runs `completed` (1,015,808 steps, `74f4f36`,
    `dirty: false`). The first sweep was interrupted after 9 runs and 3 were
    retrained.
  - Evaluation and video: 48 commands, all exit 0.
  - pytest 223 passed; ruff and `mypy --strict` clean.
- **Specialist review:**
  - Reward Engineer REVISE (M1 telescoping explanation, M2 seed-42
    over-reading; fixed).
  - Experiment Analyst REVISE (H1 21 hand-typed table values, regenerated by
    script; M1–M3 fixed; L2–L4 accepted).
  - RL Engineer: mechanics ACCEPT, phase REVISE (M1 duplicate sessions
    documented; M2 phase-gate items in progress).
- **Videos:** the user reviewed the upright_slow_bonus clips. Seeds 142 and
  242 hover; seed 42 lands between the flags.
- **Acceptance decision:** Accepted. The acceptance criteria are evidenced,
  and the user approved the checkpoint commit ("commit").
- **Git checkpoint:** `74f4f36` "Phase 04 (1/2)" (configs), then "Phase 04
  (2/2)" (results, records, progress). Both are local commits on `main`,
  not pushed.
- **Handoff:** `progress/handoffs/2026-11-05-phase-04.md`

---

## 2026-11-05 — 05-evaluation

- **Plan approval:** the user approved the plan and its 5 recommended
  decisions ("go ahead"): D-036, D-037, D-038 and D-040. D-039 was added
  during the phase.
- **Files changed:** see `progress/handoffs/2026-11-05-phase-05.md`.
- **Checks:**
  - pytest 296 passed; ruff and `mypy --strict` clean.
  - Re-evaluation: 12/12 runs exit 0, and every `episodes.csv` is
    byte-identical to Phase 04.
  - The comparison ran three times with identical tables. The committed
    output is clean at `6bb2bb6`.
- **Specialist review:**
  - Test Engineer REVISE: (a) and (b) fixed, plus most recommended items.
  - Experiment Analyst REVISE (minor): M1–M4 fixed; L1–L8 addressed.
- **Acceptance decision:** Accepted. The acceptance criteria are evidenced,
  and the user approved the checkpoint commit ("commit") after D-039 was
  flagged for review.
- **Git checkpoint:** `e481fdf` (1/2), `6bb2bb6` (2/3), then "Phase 05
  (3/3)" (results, report, progress). All are local commits on `main`, not
  pushed.
- **Handoff:** `progress/handoffs/2026-11-05-phase-05.md`

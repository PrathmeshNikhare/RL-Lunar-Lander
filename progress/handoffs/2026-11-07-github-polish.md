# Handoff — 2026-11-07 — Repository polish

## Summary
A GitHub portfolio polish, outside the phase plan (D-049). The user approved
the Stage 1 audit with the recommended options. No code, scripts, tests,
configs, results, experiment records or spec-doc names changed.

- **README:**
  - added static stack badges (no test, CI or license badges), a
    Highlights list, a Contents line and a Documentation index;
  - moved the long training-metrics tables to `docs/training-diagnostics.md`,
    leaving a short summary that keeps the out-of-band values and the
    caveats;
  - removed the "owner" wording, and described the CI workflow as
    configured (it has not run yet).
- **D-047 still applies:** there is no "How this project was built" section
  and no license statement or `LICENSE` file. The harness appears only as
  a row in the Documentation index.
- **CI:** `.github/workflows/ci.yml` runs on `windows-latest` with Python
  3.12. It installs `requirements-lock.txt`, then runs the environment
  check, ruff, ruff format, `mypy --strict` and pytest.
- **`.gitignore`:** now ignores `.env`, `.vscode/`, `.idea/` and
  `.claude/settings.local.json`.
- **Progress files:**
  - `STATUS.md`, `CURRENT_HANDOFF.md` and `PHASE_LOG.md` said that
    `origin/main` was at `31d02ca` with later work unpushed. `git fetch`
    showed `origin/main` at `e42672a`, and they are corrected.
  - D-049 is added to `DECISIONS.md`.

## Repository state
- Branch: `chore/github-repository-polish`, created from `e42672a`; same as
  `main` and `origin/main` before this work.
- Commit: one local commit, "Repository polish: README, training
  diagnostics doc, CI, gitignore".
- Working tree: clean after the commit.
- Active phase: none. All phases 00–07 are complete.
- `backup/pre-github-polish` (local and remote, `e42672a`) is untouched.

## Files changed
| Path | Change |
|---|---|
| `README.md` | Badges, highlights, contents, training-diagnostics summary, documentation index, wording |
| `docs/training-diagnostics.md` | New. The README training-metrics section, moved verbatim except for headings, intro, link prefixes and one "this README" phrase |
| `.github/workflows/ci.yml` | New. Windows / Python 3.12 tests and static checks |
| `.gitignore` | 4 entries |
| `progress/STATUS.md`, `progress/CURRENT_HANDOFF.md`, `progress/PHASE_LOG.md`, `progress/DECISIONS.md` | Real pushed state, the polish entry and D-049 |
| `progress/handoffs/2026-11-07-github-polish.md` | This file |

## Verification evidence
| Command/check | Result | Notes |
|---|---|---|
| `pytest -q` | 341 passed | Before and after the change |
| `ruff check .`, `ruff format --check .`, `mypy --strict lunar_lander_rl scripts tests` | Clean | |
| Relative-link and anchor check (session scratchpad script) | 0 broken | All tracked `.md` files |
| Moved-text check (script) | All 133 non-blank lines are accounted for: 12 intended edits, the rest verbatim | Every numeric token of the original block is present in the new doc |
| `git diff --name-only -- lunar_lander_rl scripts tests configs results experiments` | Empty | Scope check |
| Public GitHub API | No description, no topics; Wiki and Projects enabled | Read-only |

## Review
- Reviewer: Documentation Reviewer.
- Findings:
  - missing handoff file (major);
  - wrong sign on entropy;
  - summary dropped the out-of-band values and some caveats;
  - stale `PHASE_LOG`;
  - CI described as if it had already run;
  - one long line.
- Resolutions: all fixed in this change.
- Recommendation: REVISE. Every finding was then fixed.

## Decisions
D-049 in `progress/DECISIONS.md`.

## Known issues / blockers
- The CI workflow has not run on GitHub. Wheel availability for every
  pinned package on a GitHub runner is unverified. Add a CI badge only
  after a passing run.
- Badges and Mermaid diagrams have not been checked as rendered on
  GitHub.
- GitHub settings are recommendations only and were not applied (no `gh`
  CLI, and the change is outward-facing):
  - description: "Reward engineering on Gymnasium LunarLander-v3 with
    Stable-Baselines3 PPO: four reward designs, a documented reward hack,
    evaluated on the original reward."
  - topics: `reinforcement-learning`, `reward-shaping`, `reward-hacking`,
    `ppo`, `stable-baselines3`, `gymnasium`, `lunar-lander`, `pytorch`,
    `python`;
  - disable Wiki and Projects.

## Next action
The user reviews the branch. With approval, push it and open a PR to
`main`.

## Approval state
User approval is required for the push, the PR and the merge.

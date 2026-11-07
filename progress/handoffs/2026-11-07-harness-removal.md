# Handoff — 2026-11-07 — Harness removal

## Summary
At the user's request, the AI development harness is removed from the
repository, so that it presents the finished project (D-050).

- **Removed:**
  - `CLAUDE.md`;
  - `AGENTS.md`;
  - `phases/` (8 phase plans);
  - `.claude/` (6 specialist-agent prompts and 6 slash commands).
- **Why `.claude/` went too:** it held only development-agent
  configuration. Its commands read `phases/` and `CLAUDE.md`, so they would
  no longer work.
- **Content preserved:**
  - **`CLAUDE.md`:** its rules for the finished project are now design
    rules in `docs/03-architecture.md`: no re-implemented PPO or Box2D,
    separate terminated/truncated handling, tunables in validated configs,
    CPU only, relative paths, no secrets. Its other rules were already
    covered in `docs/` and the README.
  - **`phases/`:** the plans repeat `docs/02-requirements.md` and
    `docs/06-experiment-protocol.md`. Their outcomes are in
    `progress/PHASE_LOG.md` and the dated handoffs.
  - **`AGENTS.md`:** it described reviewer roles only. Review verdicts
    remain in the progress records.
- **Kept:** `progress/` (decision log, phase log, handoffs) and `templates/`.
- **References updated:** the README documentation index and repository
  map, and D-044 (marked superseded by D-050).

## Repository state
- Branch: `chore/github-repository-polish`. It has two local commits on top
  of `e42672a`: `d441161` (polish) and the harness-removal commit.
- `main`, `origin/main` and `backup/pre-github-polish` are unchanged at
  `e42672a`. The removed files remain in their history.
- Not pushed.

## Verification evidence
| Command/check | Result |
|---|---|
| `pytest -q` | 341 passed |
| `ruff check .`, `ruff format --check .`, `mypy --strict lunar_lander_rl scripts tests` | Clean |
| Relative-link and anchor check | 0 broken |
| `git diff --name-only` over `lunar_lander_rl scripts tests configs results experiments` | Empty |

## Known issues
- `lunar_lander_rl/config.py` lines 39 and 613 have comments that cite
  `CLAUDE.md`. They are not links. They are left unchanged because
  implementation files are out of scope; changing them is a two-word
  comment edit if the user approves it.
- Decisions D-018, D-034 and D-044 name the removed files as historical
  records, not instructions. D-050 explains where those rules now live.
- Claude Code sessions in this checkout no longer load project instructions,
  agents or slash commands.

## Next action
The user reviews the branch. With approval, push it and open a PR to `main`.

## Approval state
User approval is required for the push, the PR and the merge.

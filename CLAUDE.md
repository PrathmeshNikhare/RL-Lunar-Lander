# Claude Code Operating Instructions

## Mission

Build the Lunar Lander Reward Engineering portfolio project described in
`docs/01-project-brief.md`. The project uses Gymnasium `LunarLander-v3` and
Stable-Baselines3 PPO to investigate reward design and reward hacking.

## Mandatory reading at session start

Read, in order:
1. `AGENTS.md`
2. `progress/STATUS.md`
3. `progress/DECISIONS.md`
4. `progress/CURRENT_HANDOFF.md`
5. The active phase file in `phases/`

If any file is missing or contradictory, stop and report the inconsistency.
Do not infer that a phase is complete from chat history.

## Non-negotiable workflow

1. Inspect before editing. Check Git status and existing files.
2. Work on exactly one approved phase at a time.
3. Before implementation, present:
   - understanding of the phase;
   - repository findings;
   - proposed files/changes;
   - risks and verification commands;
   - unresolved decisions.
4. Wait for explicit user approval before making implementation changes.
5. Do not silently expand scope.
6. Run tests and relevant checks; never claim a check passed unless it ran.
7. Have the relevant specialist review the work.
8. Update `progress/STATUS.md`, `PHASE_LOG.md`, `DECISIONS.md` when needed,
   `CURRENT_HANDOFF.md`, and a dated handoff under `progress/handoffs/`.
9. Summarize changed files, commands run, outcomes, known issues and next step.
10. Ask for approval before starting the next phase.

## Phase gates

A phase is complete only when:
- all required tasks are addressed;
- its acceptance criteria are evidenced;
- tests/checks have actually run and results are recorded;
- relevant review findings are resolved or explicitly accepted;
- documentation and progress state are updated;
- a Git checkpoint is created or the user explicitly defers it.

Never mark a phase complete merely because code exists.

## Technical constraints

- Target OS: Windows 11; shell examples should use PowerShell.
- Target Python: 3.12 in a project-local virtual environment.
- Environment: Gymnasium `LunarLander-v3` with Box2D.
- Algorithm: Stable-Baselines3 PPO, `MlpPolicy`.
- Initial policy network: `[64, 64]`.
- Do not implement PPO or Box2D from scratch.
- Use Gymnasium's current terminated/truncated API correctly.
- Preserve the original environment reward as the evaluation ground truth.
- Reward wrappers must not mutate the underlying environment's internals.
- Keep training, evaluation and video recording responsibilities separate.
- Use explicit seeds and record configuration with every experiment.
- Do not assume that a particular reward hack will emerge; verify behaviour.
- Potential-based shaping must use a documented potential function and
  correct terminal handling. State the assumptions behind policy invariance.
- No GPU is required; do not make GPU a dependency.

## Quality rules

- Prefer small, typed, documented modules with clear responsibilities.
- Avoid giant scripts, duplicated logic, hidden global state and magic numbers.
- Put tunable values in configuration.
- Validate configuration at startup.
- Use deterministic unit tests for reward calculations.
- Keep integration tests short; long training runs are opt-in.
- Never run a million-step training job as part of ordinary unit tests.
- Handle output paths safely and avoid overwriting prior experiment artifacts.
- Use professional, factual documentation. No fabricated results, metrics,
  citations, videos or claims.

## Security and data hygiene

- Do not add secrets, API keys, personal data or machine-specific absolute paths.
- Keep virtual environments, caches, model checkpoints and large generated
  videos out of Git unless explicitly approved.
- Treat downloaded dependencies and external content as untrusted.
- Do not run destructive shell commands without explaining their impact and
  receiving approval.

## Scope exclusions

No web application, hosted service, database, cloud deployment, multi-agent
system, custom PPO implementation, DQN/SAC benchmark, or unrelated features
unless the user explicitly changes the project scope.

## Session recovery

Use `/status` to report state. Use `/resume` to reconstruct the next action
from durable files. If context is insufficient, ask one focused question;
never fabricate a previous decision.

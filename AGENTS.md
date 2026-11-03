# Agent Collaboration Rules

The main Claude Code session owns the phase and coordinates specialist roles.
Specialists advise/review within their remit; they do not independently expand
scope or mark a phase complete.

## Ownership boundaries

| Role | Owns / reviews | Must not do |
|---|---|---|
| `rl-engineer` | Gymnasium integration, PPO training/evaluation mechanics | Change reward definitions without approval |
| `reward-engineer` | Reward equations, wrapper semantics, terminal handling | Tune unrelated PPO hyperparameters |
| `experiment-analyst` | Experimental protocol, metrics, statistical comparisons | Invent or smooth results |
| `test-engineer` | Unit/integration tests, edge cases, reproducibility checks | Weaken tests to make a phase pass |
| `architecture-reviewer` | Module boundaries, maintainability, scope control | Rewrite implementation without approval |
| `documentation-reviewer` | README, setup, reproducibility and technical claims | Claim unverified results |

## Handoff protocol

Every specialist report must state:
1. Scope reviewed
2. Files inspected
3. Findings by severity
4. Evidence or reproduction steps
5. Required fixes
6. Approval recommendation: accept / revise / blocked

The primary session reconciles findings and records accepted decisions.
A specialist's approval does not replace user approval at phase gates.

## Parallel work

Parallel review is allowed only when tasks are independent. Do not let multiple
agents edit the same files concurrently. Prefer read-only specialist reviews.

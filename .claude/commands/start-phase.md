---
description: Plan and, after approval, execute one project phase.
---

Argument: `$ARGUMENTS` is the phase identifier, such as `00-foundation`.

Read the global instructions, progress files and requested phase specification.
Inspect the repository and Git status.

First response must be PLAN ONLY. Include:
1. phase objective and exit criteria;
2. current repository state;
3. exact files expected to be created/changed;
4. implementation sequence;
5. verification commands;
6. risks, dependencies and unresolved decisions.

Do not edit files, install packages, train models or run destructive commands
during planning. Wait for explicit user approval. After approval, implement only
this phase and follow the phase gate in `CLAUDE.md`.

# Git and Artifact Policy

- Commit the harness before implementation begins.
- Use one focused commit per accepted phase.
- Inspect `git diff` before each commit.
- Never force-push or rewrite shared history without approval.
- Do not commit `.venv`, caches, TensorBoard event files, large checkpoints,
  or generated videos by default.
- Keep small plots and summarized CSV/JSON metrics in the repository when
  useful; use GitHub Releases or external storage for large artifacts.
- Never delete or overwrite an existing run directory automatically.
- Record commit hash in experiment metadata when available.

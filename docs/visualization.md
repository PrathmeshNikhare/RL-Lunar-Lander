# Figures, traces and clips (Phase 06)

Phase 06 turns the committed Phase 04/05 results into figures, one
per-step trace per run and one video clip per run. The analysis is in
`experiments/records/2026-11-07-visual-analysis.md`.

## Pipeline

| Step | Script | Needs local `runs/`? | Output (write-once) |
|---|---|---|---|
| 1. Traces | `scripts/extract_traces.py` | yes: `trajectories.npz` (D-038) | `results/phase06/component_traces.csv` + `.json` |
| 2. Clips | `scripts/collect_videos.py` | yes: video sessions | `results/phase06/videos/` (`*.mp4`, `videos.csv`, `videos.json`) |
| 3. Figures | `scripts/make_figures.py` | no: committed files only | `results/phase06/figures/` (`*.png`, `figures_manifest.json`) |

Steps 1 and 2 read the git-ignored `runs/` folder (`--runs-root`, default
`runs`). It is produced by training and evaluating the 12 Phase 04 runs
(`docs/training-and-evaluation.md`, `docs/reward-experiments.md`) and the
Phase 05 re-evaluation (`docs/07-metrics.md`). Step 3 needs only the
repository.

The commands below produced the committed outputs (they match the `argv`
in each provenance record). Each script refuses an output that already
exists, so in a clone that already has `results/phase06/` they stop with an
error; write a re-run to a new path instead. Every output records the
source revision, package versions and command line.

```powershell
$py = ".\.venv\Scripts\python.exe"
& $py scripts\extract_traces.py --config configs\visualization.yaml --output results\phase06\component_traces.csv
& $py scripts\collect_videos.py --config configs\visualization.yaml --output results\phase06\videos
& $py scripts\make_figures.py  --config configs\visualization.yaml --output results\phase06\figures
```

To check reproducibility, regenerate the figures into another folder and
compare the PNGs (they carry no timestamps or software metadata). Empty
output from `Compare-Object` means every hash matches. `figures_manifest.json`
is not compared, because its `created_at` and `argv` differ between runs.

```powershell
$py = ".\.venv\Scripts\python.exe"
& $py scripts\make_figures.py --config configs\visualization.yaml --output $env:TEMP\phase06_rerun
Compare-Object (Get-FileHash results\phase06\figures\*.png).Hash (Get-FileHash $env:TEMP\phase06_rerun\*.png).Hash
```

## Configuration (`configs/visualization.yaml`)

- Input paths, all relative to the project root.
- `training_curve.rolling_episodes`: trailing window for the training
  curves (100). Episodes are ordered by the step at which they ended; only
  full windows are plotted.
- `selection.trace_episode` / `selection.video_episode`: the one evaluation
  episode and the one video episode used for every run (both 0). They were
  fixed before any trace or clip was viewed, not chosen by outcome.

Run selection, evaluation settings and the hover thresholds are not repeated
in this file. They are read from `results/phase05/comparison.json`.

## How the traces are built

Evaluation uses the original reward only (D-009). To show what each run's
*training* reward would have added over an evaluation episode, step 1
replays the recorded observations, original rewards and terminal flags
through the unchanged `RewardComponentsWrapper`, using a replay environment
(D-041). No reward formula is re-implemented. A test checks that the replay
is bit-identical to the live wrapper in all four reward modes, for both
terminated and truncated episodes.

Before writing, each trace is checked:
- against `results/phase05/episode_metrics.csv`: env seed, length,
  terminal flags and the exact original return;
- each run's `config.json` against its committed `run_config.json`.

CSV rows hold `experiment_id`, `training_seed`, `step`, the action, the
8 observation values *after* the step, `original`, the custom component's
name and value (empty for the baseline) and `training_reward`. The run
directory, evaluation session, episode index and env seed of each trace are
in `component_traces.json`.

## How the clips are chosen

For each run, the clip of the configured video episode is copied from the
newest video session whose `episodes.csv` is byte-identical to the committed
`results/phase04/<experiment>/seed<N>/video_episodes.csv` (D-034). The clip's
label in `videos.csv` comes from that CSV and the D-036 outcome rule. Clips
are not labelled from viewing.

## Figure rules

- Figures read only committed files, and `figures_manifest.json` records
  each input's SHA-256.
- Evaluation figures show the original reward only. No figure plots a
  shaped training total (original + custom term). Custom terms appear only
  in figure 2 (per training episode) and the middle row of figure 8 (over
  one evaluation episode), on their own axes and labelled "training reward
  only".
- Reference lines come from data or configuration, never typed in:
  - the 200 reward threshold;
  - the time limit, from the Gymnasium registry;
  - the pre-registered hover threshold, from `comparison.json`;
  - the post-hoc hover line, computed as the largest baseline value and
    labelled post-hoc.
- Colours are the fixed categorical palette in `lunar_lander_rl/plotting.py`
  (`CATEGORICAL`, `SEED_STYLES`), checked for colour-vision separation
  before use. Configurations use slots 1–4 and outcomes slots 1–5, so
  the two share colours. For that reason the outcome figure labels its
  configuration groups in neutral text. Seeds use slots 5–7 and also
  differ by line style.

# Testing Strategy

> Original project specification, written before implementation and kept
> as the design contract. For what was built and measured, see the
> [README](../README.md), [`docs/release-summary.md`](release-summary.md)
> and [`docs/03-architecture.md`](03-architecture.md).

## Unit tests

- Reward functions return expected values for hand-calculated inputs.
- Potential shaping matches its equation.
- Terminal transition handling follows the documented convention.
- Wrapper preserves observation and info contracts.
- Reward components sum to reported shaped reward.
- Invalid configuration is rejected with a useful error.

## Integration tests

- Environment reset/step smoke test.
- Wrapper reset/step smoke test.
- Very short PPO learn/save/load smoke test.
- Evaluation returns the expected metric schema.
- Video recording smoke test only when optional media dependencies are present.

## Manual checks

- Random agent visibly behaves as expected.
- Baseline agent is evaluated with the original reward.
- Reward-hacking claim is supported by video and metrics.
- All experiment runs have metadata and distinct output directories.

Default test suite must not launch full training.

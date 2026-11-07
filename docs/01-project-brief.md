# Project Brief

> Original project specification, written before implementation and kept
> as the design contract. For what was built and measured, see the
> [README](../README.md), [`docs/release-summary.md`](release-summary.md)
> and [`docs/03-architecture.md`](03-architecture.md).

## Problem

An RL agent optimizes the reward signal it receives. If the reward is poorly
specified, the agent may maximize the score while failing the intended task.
This project demonstrates the relationship between reward design and learned
behaviour in Gymnasium LunarLander-v3.

## Objective

Train PPO agents under four reward configurations, compare their behaviour
and evaluate every policy against the original task reward.

## In scope

- Local Python project on Windows 11
- Gymnasium LunarLander-v3 and Box2D
- Stable-Baselines3 PPO / MlpPolicy
- Baseline, time penalty, flawed bonus and potential shaping
- Reproducible training/evaluation
- Metrics, graphs, videos and GitHub documentation

## Out of scope

Custom PPO, additional RL algorithms, web UI, database, cloud deployment,
robotics hardware and unrelated features.

## Portfolio story

The repository should tell a clear experimental story:
baseline learning → reward modification → unintended incentive → improved
reward formulation → controlled evaluation and lessons learned.

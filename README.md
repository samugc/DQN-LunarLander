# DeepRL-LunarLander

[![CI](https://github.com/samugc/DeepRL-LunarLander/actions/workflows/ci.yml/badge.svg)](https://github.com/samugc/DeepRL-LunarLander/actions/workflows/ci.yml)

Deep reinforcement learning experiments on **LunarLander-v3** with Double DQN, dueling networks and prioritized experience replay.

Originally developed as MSc Neural Networks coursework and later consolidated into a single tested implementation.

## Algorithms

The canonical agent combines:

- **Double DQN** for decoupled action selection and target evaluation.
- **Dueling architecture** with separate value and advantage streams.
- **Prioritized Experience Replay (PER)**.
- **Importance-sampling weights** for non-uniform replay.
- **Huber loss** applied per transition.
- **Target-network updates** for stable bootstrap targets.

A simpler uniform-replay Double-DQN implementation is retained under `academic/` as historical reference.

## Historical evaluation

| Policy | Evaluation | Mean return |
|---|---|---:|
| Dueling + Double DQN + PER | 30 greedy episodes, one trained policy | **268.32** |

This is a historical evaluation of a **single trained policy**. The 30 episodes are not 30 independent training seeds, so no confidence interval or algorithm-superiority claim is made.

The historical score predates a correction to the per-sample Huber-loss weighting in the cleaned implementation.

## Architecture

```text
state
  ↓
shared MLP
  ├── value stream ───────┐
  └── advantage stream ──┤
                         ↓
                    Q(s, a)
                         ↓
Double-DQN target + PER
```

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Short integration smoke test:

```bash
python -m dqn.run train --episodes 2 --checkpoint runs/smoke.keras
python -m dqn.run evaluate --episodes 2 --checkpoint runs/smoke.keras
```

A two-episode run checks the pipeline only; it is not evidence that the agent learned.

## Repository structure

```text
dqn/                 canonical agent, replay memory, config and CLI
academic/            historical baseline implementation
evidence/            preserved evaluation/provenance records
tests/               loss, replay, update and serialization tests
VALIDATION.md        executed checks and known limits
```

## Engineering improvements

- one canonical agent/replay/configuration path;
- correct per-transition Huber weighting before PER importance weights;
- detached bootstrap targets;
- separate handling of terminated and truncated episodes;
- configuration/checkpoint metadata;
- model-serialization and synthetic-update tests.

## Limitations

- The historical result is from one trained policy, not a multi-seed experiment.
- The cleaned implementation needs new multi-seed runs before comparative claims.
- Historical checkpoints could not be mapped confidently to the canonical variant.
- Historical variants use different hyperparameters, so this is not a controlled ablation.

See [`VALIDATION.md`](VALIDATION.md) for executed checks and validation scope.

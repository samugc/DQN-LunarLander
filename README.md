# DQN LunarLander

Deep Q-learning experiments on `LunarLander-v3`, developed during MSc Neural
Networks coursework (2025–26), in a group context. Samuel García confirms
implementing the studied variants; DQN, Double DQN, dueling and PER are existing
methods, not algorithms invented here.

## Experiment story
A reference DQN motivated a uniform-replay Double-DQN baseline, followed by a
variant with dueling value/advantage heads, proportional prioritized replay,
importance weights, Huber loss and periodic target-network updates.
`academic/uniform_double_dqn.py` preserves the baseline implementation excerpt.
`dqn/` is the canonical, cleaned Dueling/Double/PER implementation.
Hyperparameters and training runs differ; this is **not a controlled ablation**.

## Historical evidence
The original notebook records a greedy evaluation mean of **268.32 over 30
episodes** for a single trained policy. See `evidence/` for exact cell outputs,
policy descriptions and source hashes. These are not 30 independent training
seeds. No confidence interval across trained policies or relative improvement
claim is made. Existing checkpoints could not be confidently associated with
this variant, so none is bundled.

## Important correction
The historical Huber call received rank-one tensors: Keras reduced the last
axis before replay weights were applied. The cleaned version uses shape
`(batch, 1)` to retain one loss per transition. Targets are explicitly detached.
**The historical score predates this correction and does not validate it.**
Tests cover weighted-loss shape/value, terminal targets, one synthetic gradient
update, target copying, replay updates and model serialization.

## Run
Python 3.11/3.12 recommended. From the repository root:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m dqn.run train --episodes 2 --checkpoint runs/smoke.keras
python -m dqn.run evaluate --episodes 2 --checkpoint runs/smoke.keras
```
Box2D installation may require SWIG and platform build tools. A two-episode run
checks integration; it is not evidence that the agent learned. The CLI records
per-episode returns, configuration and evaluation checkpoint hashes.
Long training is explicit, e.g. `--episodes 1000`; it was not run in this cleanup.
Compatibility ranges are not the original environment lock.

## 2026 portfolio extension
Extracted one canonical agent/replay/config, removed discarded notebooks and
unidentified weights, fixed per-sample Huber semantics, added a minimal CLI,
checks and provenance. The CLI uses a fixed episode count and final checkpoint;
it does not reproduce the historical best-moving-average selection procedure.
Termination disables bootstrapping; time-limit truncation only ends rollout.

## Limitations and interview review
Explain Bellman targets, online selection versus target evaluation, centered
advantage, PER sampling bias, importance weights, termination versus truncation
and selection on training returns. The corrected version needs multi-seed
training before any performance claim. Review the AI-assisted 2026 extraction
and loss fix before presenting the new implementation as personally mastered.

Executed checks and limits: [VALIDATION.md](VALIDATION.md).

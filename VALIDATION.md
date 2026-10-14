# Cleanup validation (2026-10-14)

Python 3.12, TensorFlow 2.21.0, Keras 3.15.1, NumPy 2.5.4.
Seven unittest checks passed: PER tree/ring sampling, empty-buffer rejection,
per-transition Huber values, terminal targets, synthetic gradient update,
target copying and Keras serialization roundtrip.
No long training or multi-seed performance evaluation was performed.
The 268.32 historical mean predates the corrected loss.

Gymnasium 1.4.0 / Box2D 2.3.10 CLI integration also passed: two training-mode
episodes (166 steps, below the 5,000-step learning warmup), checkpoint save,
reload and two greedy evaluation episodes. This is an untrained-agent smoke
check, not a learning result. NumPy action counts are converted to Python ints
for Keras compatibility. Generated weights and smoke outputs are excluded.

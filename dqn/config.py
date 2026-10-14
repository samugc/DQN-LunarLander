# =========================
# Core DQN hyperparameters
# =========================
LEARNING_RATE = 1e-4
BATCH_SIZE = 128
GAMMA = 0.99

# Replay buffer
MEMORY_SIZE = 200_000
WARMUP_STEPS = 5_000  # pasos antes de empezar a entrenar

# Training frequency
TRAIN_FREQ = 2        # entrenar cada TRAIN_FREQ pasos
GRADIENT_STEPS = 1    # número de updates por llamada a learn()

# Target network update
TARGET_UPDATE_INTERVAL = 1000  # hard update cada N pasos

# Epsilon-greedy (schedule por pasos)
EPS_START = 1.0
EPS_END = 0.10
EPS_DECAY_STEPS = 60_000  # lineal hasta EPS_END
EPS_FINAL = 0.02           # opcional: bajar más tras cierto punto

# PER (Prioritized Experience Replay)
PER_ALPHA = 0.6
PER_BETA_START = 0.4
PER_BETA_FRAMES = 200_000
PER_EPS = 1e-3

# Entrenamiento / evaluación
MAX_EPISODES_FOR_TRAINING = 1000
TRAINING_GOAL = 280
EPISODES_TO_CHECK_TRAINING_GOAL = 10
EPISODES_TO_EVALUATE_MODEL_PERFORMANCE = 30

SEED = 123

"""Original baseline excerpt; not a controlled ablation of dqn.agent.
Imported only for study, not by the canonical CLI.
"""
import random
import math
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Learning rate
LEARNING_RATE = 0.0005
# Batch size
BATCH_SIZE = 128
# Discount factor gamma
GAMMA = 0.99
# Size of the replay memory (replay buffer) that saves transitions
MEMORY_SIZE = 100000
# Values related to the exploration rate used by the epsilon-greedy policy
EXPLORATION_RATE_MAX = 1
EXPLORATION_RATE_MIN = 0.01
EXPLORATION_RATE_DECAY = 0.995
# Maximum number of episodes for training
MAX_EPISODES_FOR_TRAINING = 600
# Goal score to be achieved during the training process
TRAINING_GOAL = 200
# Number of episodes used to check the goal score during the training process
EPISODES_TO_CHECK_TRAINING_GOAL = 10
# Number of episodes used to evaluate the model performance after training
EPISODES_TO_EVALUATE_MODEL_PERFORMANCE = 30
# Target network update rate
TAU = 0.01

class ReplayMemory:
    def __init__(self, number_of_observations):
        # Inicialización de arrays (igual que antes)
        self.states = np.zeros((MEMORY_SIZE, number_of_observations))
        self.states_next = np.zeros((MEMORY_SIZE, number_of_observations))
        self.actions = np.zeros(MEMORY_SIZE, dtype=np.int32)
        self.rewards = np.zeros(MEMORY_SIZE)
        self.terminal_states = np.zeros(MEMORY_SIZE, dtype=bool)
        self.current_size = 0
        self.ptr = 0  # <--- NUEVO: Puntero para saber dónde escribir

    def store_transition(self, state, action, reward, state_next, terminal_state):
        # Usamos el puntero 'ptr' para saber en qué índice guardar
        index = self.ptr 

        self.states[index] = state
        self.states_next[index] = state_next
        self.actions[index] = action
        self.rewards[index] = reward
        self.terminal_states[index] = terminal_state

        # Actualizamos el puntero (si llega al final, vuelve a 0)
        self.ptr = (self.ptr + 1) % MEMORY_SIZE

        # El tamaño actual crece hasta llenarse, pero no pasa de MEMORY_SIZE
        self.current_size = min(self.current_size + 1, MEMORY_SIZE)

    def sample_memory(self, batch_size):
        # Generamos la muestra usando el tamaño actual válido
        batch = np.random.choice(self.current_size, batch_size)
        states = self.states[batch]
        states_next = self.states_next[batch]
        rewards = self.rewards[batch]
        actions = self.actions[batch]
        terminal_states = self.terminal_states[batch]
        return states, actions, rewards, states_next, terminal_states

class DQN:

    def __init__(self, number_of_observations, number_of_actions):
        # Initialize variables and create neural model
        self.number_of_actions = number_of_actions
        self.number_of_observations = number_of_observations
        self.scores = []
        self.memory = ReplayMemory(number_of_observations)

        # Create models
        self.model = self._build_model()
        self.target_model = self._build_model()

        # Initialize target model weights
        self.target_model.set_weights(self.model.get_weights())

    def _build_model(self):

        # Neural model
        model = keras.models.Sequential()
        model.add(keras.layers.InputLayer(shape=(self.number_of_observations,)))
        model.add(keras.layers.Dense(64, activation="relu",kernel_initializer="he_normal"))
        model.add(keras.layers.Dense(64, activation="relu",kernel_initializer="he_normal"))
        model.add(keras.layers.Dense(self.number_of_actions, activation="linear"))
        model.compile(loss=keras.losses.Huber(), optimizer=keras.optimizers.AdamW(learning_rate=LEARNING_RATE, clipnorm=1.0))

        return model

    def update_target_model(self):
        # target_param = τ * policy_param + (1 - τ) * target_param
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()

        new_weights = []
        for i in range(len(weights)):
            new_layer_weights = TAU * weights[i] + (1 - TAU) * target_weights[i]
            new_weights.append(new_layer_weights)

        self.target_model.set_weights(new_weights)

    def remember(self, state, action, reward, next_state, terminal_state):
        # Store a tuple (s, a, r, s') for experience replay
        state = np.reshape(state, [1, self.number_of_observations])
        next_state = np.reshape(next_state, [1, self.number_of_observations])
        self.memory.store_transition(state, action, reward, next_state, terminal_state)

    def select(self, state, exploration_rate):
        # Generate an action for a given state using epsilon-greedy policy
        if np.random.rand() < exploration_rate:
            return random.randrange(self.number_of_actions)
        else:
            state = np.reshape(state, [1, self.number_of_observations])
            q_values = self.model(state).numpy()
            return np.argmax(q_values[0])

    def select_greedy_policy(self, state):
        # Generate an action for a given state using greedy policy
        state = np.reshape(state, [1, self.number_of_observations])
        q_values = self.model(state).numpy()
        return np.argmax(q_values[0])
    def _train_step(self, states, actions, rewards, next_states, terminal_states):
        gamma = tf.convert_to_tensor(GAMMA, dtype=tf.float32)

        # 1. Double DQN: El modelo principal elige la mejor acción siguiente
        next_q_values = self.model(next_states)
        best_next_actions = tf.argmax(next_q_values, axis=1)

        # 2. El modelo target evalúa el valor de esa acción
        next_target_q_values = self.target_model(next_states)
        # Seleccionamos los Q-values correspondientes a las mejores acciones elegidas
        indices = tf.stack([tf.range(tf.shape(best_next_actions)[0], dtype=tf.int64), best_next_actions], axis=1)
        max_next_q = tf.gather_nd(next_target_q_values, indices)

        # 3. Calculamos el target: Reward + Gamma * Q_next (si no es terminal)
        target_q_values = rewards + (1 - terminal_states) * gamma * max_next_q

        # 4. Gradient Descent solo sobre las acciones tomadas
        with tf.GradientTape() as tape:
            q_values = self.model(states)

            # Máscara para obtener solo los Q-values de las acciones que realmente tomamos
            action_masks = tf.one_hot(actions, self.number_of_actions)
            q_taken = tf.reduce_sum(q_values * action_masks, axis=1)

            # Huber Loss entre el valor predicho y el calculado
            loss = tf.keras.losses.Huber()(target_q_values, q_taken)

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

    def learn(self):
        if self.memory.current_size < BATCH_SIZE: return

        # Muestreo (igual que tenías, pero asegúrate de que los tipos sean correctos)
        states, actions, rewards, next_states, terminal_states = self.memory.sample_memory(BATCH_SIZE)

        # Conversión a tensores para TensorFlow
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.int32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        # Terminal states como float (0.0 o 1.0) para la fórmula matemática
        terminal_states = tf.convert_to_tensor(terminal_states, dtype=tf.float32)

        self._train_step(states, actions, rewards, next_states, terminal_states)

    def add_score(self, score):
        # Add the obtained score to a list to be presented later
        self.scores.append(score)

    def delete_scores(self):
        # Delete the scores
        self.scores = []




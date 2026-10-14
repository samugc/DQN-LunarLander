import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from .config import *
from .replay import PrioritizedReplayMemory

class DQN:
    def __init__(self, number_of_observations, number_of_actions):
        self.number_of_actions = int(number_of_actions)
        self.number_of_observations = int(number_of_observations)
        self.scores = []

        # PER memory
        self.memory = PrioritizedReplayMemory(number_of_observations)

        # Networks
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model(hard=True)

        self.optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=10.0)
        self.huber = keras.losses.Huber(reduction="none")

    def _build_model(self):
        inputs = keras.Input(shape=(self.number_of_observations,))
        x = keras.layers.LayerNormalization()(inputs)
        x = keras.layers.Dense(256, activation="relu", kernel_initializer="he_normal")(x)
        x = keras.layers.Dense(256, activation="relu", kernel_initializer="he_normal")(x)

        # Dueling heads
        v = keras.layers.Dense(128, activation="relu", kernel_initializer="he_normal")(x)
        v = keras.layers.Dense(1, activation="linear")(v)

        a = keras.layers.Dense(128, activation="relu", kernel_initializer="he_normal")(x)
        a = keras.layers.Dense(self.number_of_actions, activation="linear")(a)

        # a: (batch, n_actions)
        # Convertimos a (batch, n_actions, 1) para poder usar pooling
        a_expanded = keras.layers.Reshape((self.number_of_actions, 1))(a)

        # mean over actions -> (batch, 1)
        a_mean = keras.layers.GlobalAveragePooling1D()(a_expanded)

        # volvemos a (batch, 1) explícito para restar por broadcast
        a_mean = keras.layers.Reshape((1,))(a_mean)

        adv_centered = keras.layers.Subtract()([a, a_mean])  # (batch, n_actions)
        q = keras.layers.Add()([v, adv_centered])

        return keras.Model(inputs=inputs, outputs=q)

    def update_target_model(self, hard=False):
        if hard:
            self.target_model.set_weights(self.model.get_weights())
        else:
            # si algún día queréis volver a soft update
            tau = 0.005
            w = self.model.get_weights()
            tw = self.target_model.get_weights()
            self.target_model.set_weights([tau*wi + (1-tau)*twi for wi, twi in zip(w, tw)])

    def remember(self, state, action, reward, next_state, done):
        state = np.asarray(state, dtype=np.float32)
        next_state = np.asarray(next_state, dtype=np.float32)
        self.memory.store_transition(state, action, reward, next_state, done)

    def select(self, state, epsilon):
        if np.random.rand() < epsilon:
            return random.randrange(self.number_of_actions)
        q = self.model(np.expand_dims(state, axis=0)).numpy()[0]
        return int(np.argmax(q))

    def select_greedy_policy(self, state):
        q = self.model(np.expand_dims(state, axis=0)).numpy()[0]
        return int(np.argmax(q))

    @tf.function
    def _train_step(self, states, actions, rewards, next_states, dones, weights):
        # Double DQN:
        # - online chooses action
        # - target evaluates it
        next_q_online = self.model(next_states)
        best_next_actions = tf.argmax(next_q_online, axis=1, output_type=tf.int32)

        next_q_target = self.target_model(next_states)
        batch_indices = tf.range(tf.shape(states)[0], dtype=tf.int32)
        gather_idx = tf.stack([batch_indices, best_next_actions], axis=1)
        max_next_q = tf.gather_nd(next_q_target, gather_idx)

        targets = tf.stop_gradient(rewards + (1.0 - dones) * GAMMA * max_next_q)

        with tf.GradientTape() as tape:
            q_values = self.model(states)
            action_masks = tf.one_hot(actions, self.number_of_actions)
            q_taken = tf.reduce_sum(q_values * action_masks, axis=1)

            per_sample_loss = self.huber(targets[:, None], q_taken[:, None])
            loss = tf.reduce_mean(per_sample_loss * weights)

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        td_errors = targets - q_taken
        return td_errors

    def learn(self, total_steps):
        if self.memory.current_size < max(WARMUP_STEPS, BATCH_SIZE):
            return

        # beta annealing
        beta = min(1.0, PER_BETA_START + total_steps * (1.0 - PER_BETA_START) / PER_BETA_FRAMES)

        for _ in range(GRADIENT_STEPS):
            (states, actions, rewards, next_states, dones, tree_indices, weights) = \
                self.memory.sample_memory(BATCH_SIZE, beta)

            states = tf.convert_to_tensor(states, dtype=tf.float32)
            actions = tf.convert_to_tensor(actions, dtype=tf.int32)
            rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
            next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
            dones = tf.convert_to_tensor(dones, dtype=tf.float32)
            weights = tf.convert_to_tensor(weights, dtype=tf.float32)

            td_errors = self._train_step(states, actions, rewards, next_states, dones, weights)
            td_errors_np = td_errors.numpy()
            self.memory.update_priorities(tree_indices, td_errors_np)

    def add_score(self, score):
        self.scores.append(score)

    def delete_scores(self):
        self.scores = []

    def average_score(self, number_of_episodes):
        index = len(self.scores) - number_of_episodes
        return float(np.mean(self.scores[max(0, index):]))



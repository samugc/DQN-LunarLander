import numpy as np
from .config import MEMORY_SIZE, PER_ALPHA, PER_EPS

class SumTree:
    """
    SumTree para muestreo O(log N) en PER (prioridades proporcionales).
    """
    def __init__(self, capacity: int):
        if capacity < 2:
            raise ValueError("SumTree capacity must be at least two")
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float32)
        self.write = 0
        self.n_entries = 0

    @property
    def total(self) -> float:
        return float(self.tree[0])

    def _propagate(self, idx: int, change: float):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def update(self, idx: int, priority: float):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def add(self, priority: float):
        idx = self.write + self.capacity - 1
        self.update(idx, priority)
        self.write = (self.write + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)
        return idx

    def _retrieve(self, idx: int, s: float) -> int:
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def get(self, s: float):
        idx = self._retrieve(0, s)
        data_idx = idx - (self.capacity - 1)
        return idx, data_idx, self.tree[idx]


class PrioritizedReplayMemory:
    def __init__(self, number_of_observations: int):
        self.capacity = MEMORY_SIZE
        self.tree = SumTree(self.capacity)

        self.states = np.zeros((self.capacity, number_of_observations), dtype=np.float32)
        self.next_states = np.zeros((self.capacity, number_of_observations), dtype=np.float32)
        self.actions = np.zeros((self.capacity,), dtype=np.int32)
        self.rewards = np.zeros((self.capacity,), dtype=np.float32)
        self.dones = np.zeros((self.capacity,), dtype=np.float32)

        self.max_priority = 1.0  # para nuevas transiciones

    @property
    def current_size(self) -> int:
        return self.tree.n_entries

    def store_transition(self, state, action, reward, next_state, done):
        # índice circular controlado por tree.write
        data_idx = self.tree.write

        self.states[data_idx] = state
        self.next_states[data_idx] = next_state
        self.actions[data_idx] = action
        self.rewards[data_idx] = reward
        self.dones[data_idx] = float(done)

        self.tree.add(self.max_priority)

    def sample_memory(self, batch_size: int, beta: float):
        if self.current_size == 0 or batch_size < 1:
            raise ValueError("Replay requires stored transitions and a positive batch size")
        total = self.tree.total

        # Fallback: si el árbol aún no tiene masa válida, muestreo uniforme
        if (not np.isfinite(total)) or (total <= 0.0):
            max_n = self.current_size
            indices = np.random.choice(max_n, batch_size, replace=False) if max_n >= batch_size \
                      else np.random.choice(max_n, batch_size, replace=True)
            tree_indices = np.zeros(batch_size, dtype=np.int32)
            weights = np.ones(batch_size, dtype=np.float32)
            return (
                self.states[indices],
                self.actions[indices],
                self.rewards[indices],
                self.next_states[indices],
                self.dones[indices],
                tree_indices,
                weights,
            )

        indices = np.zeros(batch_size, dtype=np.int32)
        tree_indices = np.zeros(batch_size, dtype=np.int32)
        priorities = np.zeros(batch_size, dtype=np.float32)

        segment = total / batch_size

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)

            # Protege contra rangos inválidos
            if (not np.isfinite(a)) or (not np.isfinite(b)) or (b <= a):
                s = np.random.uniform(0.0, total)
            else:
                s = np.random.uniform(a, b)

            t_idx, d_idx, p = self.tree.get(s)
            tree_indices[i] = t_idx
            indices[i] = d_idx
            priorities[i] = p

        # Probabilidades robustas
        probs = priorities / (total + 1e-8)
        probs = np.clip(probs, 1e-8, 1.0)  # evita 0

        # IS weights robustos
        weights = (self.current_size * probs) ** (-beta)
        weights = np.clip(weights, 1e-8, 1e8).astype(np.float32)
        weights /= (weights.max() + 1e-8)

        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
            tree_indices,
            weights,
        )

    def update_priorities(self, tree_indices, td_errors):
        td_errors = np.asarray(td_errors, dtype=np.float32)

        # Evita NaN/inf y controla explosiones
        td_errors = np.nan_to_num(td_errors, nan=0.0, posinf=100.0, neginf=-100.0)

        # CLIP fuerte del TD error para que los outliers no dominen el replay
        td_errors = np.clip(td_errors, -10.0, 10.0)

        priorities = (np.abs(td_errors) + PER_EPS) ** PER_ALPHA

        # cap razonable de prioridades
        priorities = np.clip(priorities, 1e-4, 10.0).astype(np.float32)

        for t_idx, p in zip(tree_indices, priorities):
            t_idx = int(t_idx)
            if t_idx == 0:
                # Cuando usamos fallback de sample_memory, tree_indices puede venir a 0
                continue
            self.tree.update(t_idx, float(p))
            self.max_priority = max(self.max_priority, float(p))

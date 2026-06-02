"""
mhn/modern_hopfield.py
Modern Hopfield Network (Ramsauer et al. 2020).
Regola di update = softmax attention → equivalente a cross-attention dei Transformer.
"""

import numpy as np


class ModernHopfieldNetwork:

    def __init__(self, beta: float = 1.0):
        """
        beta: temperatura inversa.
              beta alto → convergenza rapida, bacini più stretti.
              beta basso → recupero morbido, bacini più larghi.
        """
        self.beta = float(beta)
        self.memories = None   # shape: (n_memories, dim)

    # ── Memorizzazione ────────────────────────────────────────────────────────
    def store(self, patterns: np.ndarray):
        """patterns: (n_memories, dim), valori {-1, +1} o continui."""
        self.memories = np.asarray(patterns, dtype=np.float64)

    def _check_memories(self):
        if self.memories is None:
            raise ValueError("No memories stored. Call store(patterns) first.")

    # ── Update rule (un passo) ────────────────────────────────────────────────
    def _step(self, q: np.ndarray) -> np.ndarray:
        """
        q^(t+1) = X^T · softmax(β · X · q^(t))
        Numericamente stabile: sottrae il massimo prima dell'exp.
        """
        self._check_memories()
        q = np.asarray(q, dtype=np.float64)

        scores = self.beta * (self.memories @ q)   # (n_memories,)
        scores = scores - np.max(scores)           # stabilità numerica
        attn = np.exp(scores)
        attn = attn / np.sum(attn)

        return self.memories.T @ attn              # (dim,)

    # ── Recupero continuo ─────────────────────────────────────────────────────
    def retrieve(self, query: np.ndarray, n_steps: int = 20) -> np.ndarray:
        """Restituisce lo stato continuo dopo n_steps di update."""
        self._check_memories()
        q = np.asarray(query, dtype=np.float64).copy()
        for _ in range(n_steps):
            q = self._step(q)
        return q

    # ── Recupero con tracciamento ─────────────────────────────────────────────
    def retrieve_tracked(
        self,
        query: np.ndarray,
        n_steps: int = 20,
        tol: float = 1e-8
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """
        Recupero con convergenza anticipata.
        Restituisce:
            (stato_finale, energy_trace, n_steps_effettivi)
        """
        self._check_memories()
        q = np.asarray(query, dtype=np.float64).copy()
        energies = [self.energy(q)]

        for step in range(n_steps):
            q_new = self._step(q)
            energies.append(self.energy(q_new))

            if np.max(np.abs(q_new - q)) < tol:
                return q_new, np.array(energies, dtype=np.float64), step + 1

            q = q_new

        return q, np.array(energies, dtype=np.float64), n_steps

    # ── Recupero binarizzato {-1, +1} ─────────────────────────────────────────
    def retrieve_binary(self, query: np.ndarray, n_steps: int = 20) -> np.ndarray:
        """Recupero con output binarizzato via np.sign."""
        q = self.retrieve(query, n_steps)
        out = np.sign(q)
        out[out == 0] = 1.0
        return out

    # ── Energia ───────────────────────────────────────────────────────────────
    def energy(self, query: np.ndarray) -> float:
        """
        E(q) = -(1/beta) * log Σ_i exp(beta * x_i^T q) + 1/2 ||q||²
        Versione stabile con log-sum-exp.
        """
        self._check_memories()
        q = np.asarray(query, dtype=np.float64)

        raw_scores = self.beta * (self.memories @ q)
        m = np.max(raw_scores)
        log_sum_exp = m + np.log(np.sum(np.exp(raw_scores - m)))

        return -(1.0 / self.beta) * log_sum_exp + 0.5 * np.dot(q, q)

    # ── Pattern più vicino (nearest memory) ───────────────────────────────────
    def nearest_memory(self, query: np.ndarray) -> tuple[int, np.ndarray]:
        """
        Trova il pattern memorizzato più vicino in distanza L2.
        Utile per valutare il recupero senza binarizzazione.
        """
        self._check_memories()
        q = np.asarray(query, dtype=np.float64)
        dists = np.linalg.norm(self.memories - q, axis=1)
        idx = int(np.argmin(dists))
        return idx, self.memories[idx]

    # ── Metriche di recupero ──────────────────────────────────────────────────
    def recovery_rate(
        self,
        memories: np.ndarray,
        corruption_rate: float = 0.10,
        n_trials: int = 200,
        n_steps: int = 20,
        seed: int = 0
    ) -> float:
        """
        Tasso di recupero corretto:
        - Corrompe ogni pattern con bit-flip al tasso indicato
        - Recupera con la MHN
        - Conta i recuperi esatti (Hamming = 0)
        """
        self._check_memories()
        rng = np.random.default_rng(seed)
        n_mems = len(memories)
        correct = 0

        for _ in range(n_trials):
            idx = rng.integers(n_mems)
            original = np.asarray(memories[idx], dtype=np.float64)
            corrupted = original.copy()
            mask = rng.random(len(original)) < corruption_rate
            corrupted[mask] *= -1.0

            recovered = self.retrieve_binary(corrupted, n_steps)
            if np.array_equal(recovered, original):
                correct += 1

        return correct / n_trials

    # ── Repr ──────────────────────────────────────────────────────────────────
    def __repr__(self):
        n = len(self.memories) if self.memories is not None else 0
        d = self.memories.shape[1] if self.memories is not None else 0
        return f"ModernHopfieldNetwork(beta={self.beta}, memories={n}, dim={d})"
"""
experiments/exp1_correction.py
Esperimento 1: tasso di recupero corretto al variare del corruption rate.
Confronto Hopfield classico vs MHN su tutti e tre i livelli di complessità LTLf.
"""
import numpy as np
import json
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from classical.hopfield_bindings import ClassicalHopfield
from mhn.modern_hopfield import ModernHopfieldNetwork

# ── Configurazione ────────────────────────────────────────────────────────────
DATA_DIR         = os.path.join(os.path.dirname(__file__), "..", "dataset", "data")
RESULTS_DIR      = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Configurazione ─────────────────────────────────────────────────
CORRUPTION_RATES = [0.05, 0.10, 0.15, 0.20, 0.25]
LEVELS           = ["simple", "medium", "complex"]
N_TRIALS         = 200
MHN_BETA         = 5.0
MHN_STEPS        = 20
CLASSIC_STEPS    = 30
SEED             = 42

# Ciascun modello viene testato sulla propria scala naturale
N_STORE_CLASSIC  = 4    # ≈ 0.138 × 32, dentro capacità teorica
N_STORE_MHN      = 50   # capacità esponenziale MHN


def recovery_rate_classic(net: ClassicalHopfield, memories: np.ndarray,
                           rate: float, n_trials: int, rng) -> float:
    correct = 0
    n = len(memories)
    for _ in range(n_trials):
        idx      = rng.integers(n)
        original = memories[idx].astype(np.float64)
        corrupted = original.copy()
        corrupted[rng.random(len(original)) < rate] *= -1.0
        recovered, _, _ = net.retrieve(corrupted, max_steps=CLASSIC_STEPS)
        if np.array_equal(recovered, original):
            correct += 1
    return correct / n_trials


def recovery_rate_mhn(mhn: ModernHopfieldNetwork, memories: np.ndarray,
                      rate: float, n_trials: int, rng) -> float:
    correct = 0
    n = len(memories)
    for _ in range(n_trials):
        idx      = rng.integers(n)
        original = memories[idx].astype(np.float64)
        corrupted = original.copy()
        corrupted[rng.random(len(original)) < rate] *= -1.0
        recovered = mhn.retrieve_binary(corrupted, n_steps=MHN_STEPS)
        if np.array_equal(recovered, original):
            correct += 1
    return correct / n_trials

def run_experiment():
    rng     = np.random.default_rng(SEED)
    results = {}

    for level in LEVELS:
        print(f"\n── Livello: {level.upper()} ──")
        traces = np.load(os.path.join(DATA_DIR, f"valid_{level}.npy"))

        # ── Classica: seleziona i 4 pattern più ortogonali ───────────────────────────
        # Invece di campionare casualmente, scegliamo i pattern
        # con correlazione media minima (massima separazione nello spazio)
        all_c = traces.astype(np.float64)
        corr = np.abs(all_c @ all_c.T) / all_c.shape[1]  # matrice correlazione
        np.fill_diagonal(corr, 1.0)

        # Greedy: parti dal pattern più "centrale", aggiungi quello meno correlato
        selected = [0]
        for _ in range(N_STORE_CLASSIC - 1):
            avg_corr = corr[selected].mean(axis=0)
            avg_corr[selected] = 1.0  # esclude già selezionati
            selected.append(int(np.argmin(avg_corr)))

        mem_c = all_c[selected]
        net = ClassicalHopfield()
        net.store(mem_c)

        # ── MHN: N_STORE_MHN pattern ──────────────────────────────────────────
        idx_m   = rng.choice(len(traces), size=N_STORE_MHN, replace=False)
        mem_m   = traces[idx_m].astype(np.float64)
        mhn     = ModernHopfieldNetwork(beta=MHN_BETA)
        mhn.store(mem_m)

        results[level] = {"classic": {}, "mhn": {}}

        print(f"  Classic memorizza {N_STORE_CLASSIC} pattern | "
              f"MHN memorizza {N_STORE_MHN} pattern")

        for rate in CORRUPTION_RATES:
            rr_c = recovery_rate_classic(net, mem_c, rate, N_TRIALS, rng)
            rr_m = recovery_rate_mhn(mhn, mem_m, rate, N_TRIALS, rng)

            results[level]["classic"][rate] = rr_c
            results[level]["mhn"][rate]     = rr_m

            print(f"  rate={int(rate*100):2d}%  |  "
                  f"Classic ({N_STORE_CLASSIC} pat): {rr_c:.2%}  |  "
                  f"MHN ({N_STORE_MHN} pat): {rr_m:.2%}")

    # salvataggio invariato...
    out_path = os.path.join(RESULTS_DIR, "exp1_correction.json")
    serializable = {
        lvl: {
            model: {str(k): v for k, v in rates.items()}
            for model, rates in models.items()
        }
        for lvl, models in results.items()
    }
    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2)

    print(f"\n  ✅ Risultati salvati in {out_path}")
    return results


if __name__ == "__main__":
    results = run_experiment()

    # Stampa tabella riassuntiva
    print("\n── Tabella riassuntiva (recovery rate) ──")
    header = f"{'Level':<10} {'Model':<10}" + "".join(
        f"  {int(r*100)}%" for r in CORRUPTION_RATES
    )
    print(header)
    print("─" * len(header))
    for level in LEVELS:
        for model in ["classic", "mhn"]:
            row = f"{level:<10} {model:<10}"
            for rate in CORRUPTION_RATES:
                row += f"  {results[level][model][rate]:.2%}"
            print(row)
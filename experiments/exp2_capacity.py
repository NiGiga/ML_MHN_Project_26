"""
experiments/exp2_capacity.py
Esperimento 2: recovery rate vs numero di pattern memorizzati.
Mostra la soglia di capacità 0.138*N per la classica
e la crescita esponenziale della MHN.
"""
import numpy as np
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from classical.hopfield_bindings import ClassicalHopfield
from mhn.modern_hopfield import ModernHopfieldNetwork

DATA_DIR    = os.path.join(os.path.dirname(__file__), "..", "dataset", "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

N          = 32            # dimensione vettori
DIM        = N             # alias per chiarezza
SEED       = 42
N_TRIALS   = 500           # query per punto
RATE       = 0.10          # corruption rate fisso per questo esperimento
MHN_BETA   = 5.0
MHN_STEPS  = 20
CLASSIC_STEPS = 30

# Scale di memorizzazione per i due modelli
# Classica: da 1 a 10 (soglia teorica ≈ 4)
N_CLASSIC  = [1, 2, 3, 4, 5, 6, 7, 8, 10]
# MHN: da 5 a 200 (capacità esponenziale)
N_MHN      = [5, 10, 20, 30, 50, 75, 100, 150, 200]


def select_orthogonal(traces, k):
    """Seleziona k pattern con correlazione media minima."""
    all_t = traces.astype(np.float64)
    corr  = np.abs(all_t @ all_t.T) / all_t.shape[1]
    np.fill_diagonal(corr, 1.0)
    selected = [0]
    for _ in range(k - 1):
        avg_corr = corr[selected].mean(axis=0)
        avg_corr[selected] = 1.0
        selected.append(int(np.argmin(avg_corr)))
    return all_t[selected]


def measure_recovery(model, memories, rate, n_trials, rng, mode="mhn"):
    correct = 0
    n = len(memories)
    for _ in range(n_trials):
        idx      = rng.integers(n)
        original = memories[idx].copy()
        corrupted = original.copy()
        corrupted[rng.random(len(original)) < rate] *= -1.0

        if mode == "classic":
            recovered, _, _ = model.retrieve(corrupted, max_steps=CLASSIC_STEPS)
        else:
            recovered = model.retrieve_binary(corrupted, n_steps=MHN_STEPS)

        if np.array_equal(recovered, original):
            correct += 1
    return correct / n_trials


def run_experiment():
    rng    = np.random.default_rng(SEED)
    # Usa il livello "complex" come benchmark principale
    # (risultati più stabili dall'Exp 1)
    traces = np.load(os.path.join(DATA_DIR, "valid_complex.npy"))

    results = {"classic": {}, "mhn": {}, "config": {
        "N": N, "corruption_rate": RATE,
        "theoretical_capacity": round(0.138 * N, 2),
        "n_classic": N_CLASSIC, "n_mhn": N_MHN,
    }}

    # ── Classica ─────────────────────────────────────────────────────────────
    print("── Classica: recovery rate vs n_patterns ──")
    print(f"   (N={N}, corruption={int(RATE*100)}%, soglia teorica≈{0.138*N:.1f})")
    print(f"   {'n_pat':>6}  {'rate':>8}  {'note'}")

    for k in N_CLASSIC:
        if k <= len(traces):
            memories = select_orthogonal(traces, k)
        else:
            memories = traces[:k].astype(np.float64)

        net = ClassicalHopfield()
        net.store(memories)
        rr  = measure_recovery(net, memories, RATE, N_TRIALS, rng, mode="classic")
        results["classic"][k] = rr

        note = "← soglia teorica" if k == 4 else (
               "⚠ oltre capacità" if k > 4 else "")
        print(f"   {k:>6}  {rr:>8.2%}  {note}")

    # ── MHN ──────────────────────────────────────────────────────────────────
    print("\n── MHN: recovery rate vs n_patterns ──")
    print(f"   (beta={MHN_BETA}, corruption={int(RATE*100)}%)")
    print(f"   {'n_pat':>6}  {'rate':>8}")

    for k in N_MHN:
        idx      = rng.choice(len(traces), size=min(k, len(traces)), replace=False)
        memories = traces[idx].astype(np.float64)

        mhn = ModernHopfieldNetwork(beta=MHN_BETA)
        mhn.store(memories)
        rr  = measure_recovery(mhn, memories, RATE, N_TRIALS, rng, mode="mhn")
        results["mhn"][k] = rr

        print(f"   {k:>6}  {rr:>8.2%}")

    # Salvataggio
    out_path = os.path.join(RESULTS_DIR, "exp2_capacity.json")
    with open(out_path, "w") as f:
        json.dump({k: {str(kk): vv for kk, vv in v.items()}
                   if isinstance(v, dict) else v
                   for k, v in results.items()}, f, indent=2)

    print(f"\n  ✅ Risultati salvati in {out_path}")
    return results


if __name__ == "__main__":
    results = run_experiment()

    print("\n── Riassunto capacità ──")
    print(f"  Soglia teorica classica: 0.138 × {N} = {0.138*N:.1f} pattern")
    print(f"\n  Classica:")
    for k, rr in results["classic"].items():
        bar = "█" * int(rr * 20)
        print(f"    {k:>3} pat | {bar:<20} {rr:.1%}")
    print(f"\n  MHN (beta={MHN_BETA}):")
    for k, rr in results["mhn"].items():
        bar = "█" * int(rr * 20)
        print(f"    {k:>3} pat | {bar:<20} {rr:.1%}")
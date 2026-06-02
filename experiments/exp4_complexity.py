"""
experiments/exp4_complexity.py
Esperimento 4: recovery rate vs complessità della formula LTLf.
Domanda: le formule più complesse sono più difficili da recuperare?
Confronto su tutti i livelli con MHN a beta=5 e classica ortogonale.
"""
import numpy as np
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from classical.hopfield_bindings import ClassicalHopfield
from mhn.modern_hopfield import ModernHopfieldNetwork

DATA_DIR    = os.path.join(os.path.dirname(__file__), "..", "dataset", "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

CORRUPTION_RATES = [0.05, 0.10, 0.15, 0.20, 0.25]
LEVELS           = ["simple", "medium", "complex"]
N_STORE_CLASSIC  = 4
N_STORE_MHN      = 50
N_TRIALS         = 300
MHN_BETA         = 5.0
MHN_STEPS        = 20
CLASSIC_STEPS    = 30
SEED             = 42
N_BOOTSTRAP      = 10    # ripetizioni con seed diversi → distribuzione dei risultati


def select_orthogonal(traces, k):
    all_t = traces.astype(np.float64)
    corr  = np.abs(all_t @ all_t.T) / all_t.shape[1]
    np.fill_diagonal(corr, 1.0)
    selected = [0]
    for _ in range(k - 1):
        avg_corr = corr[selected].mean(axis=0)
        avg_corr[selected] = 1.0
        selected.append(int(np.argmin(avg_corr)))
    return all_t[selected]


def measure_recovery_classic(memories, rate, n_trials, rng):
    net = ClassicalHopfield()
    net.store(memories)
    correct = 0
    n = len(memories)
    for _ in range(n_trials):
        idx = rng.integers(n)
        orig = memories[idx].copy()
        corr = orig.copy()
        corr[rng.random(len(orig)) < rate] *= -1.0
        rec, _, _ = net.retrieve(corr, max_steps=CLASSIC_STEPS)
        if np.array_equal(rec, orig):
            correct += 1
    return correct / n_trials


def measure_recovery_mhn(memories, rate, n_trials, rng):
    mhn = ModernHopfieldNetwork(beta=MHN_BETA)
    mhn.store(memories)
    correct = 0
    n = len(memories)
    for _ in range(n_trials):
        idx = rng.integers(n)
        orig = memories[idx].copy()
        corr = orig.copy()
        corr[rng.random(len(orig)) < rate] *= -1.0
        rec = mhn.retrieve_binary(corr, n_steps=MHN_STEPS)
        if np.array_equal(rec, orig):
            correct += 1
    return correct / n_trials


def run_experiment():
    results = {level: {"classic": {}, "mhn": {}} for level in LEVELS}

    for level in LEVELS:
        print(f"\n── Livello: {level.upper()} ──")
        traces = np.load(os.path.join(DATA_DIR, f"valid_{level}.npy"))

        for rate in CORRUPTION_RATES:
            # Bootstrap: N_BOOTSTRAP ripetizioni con seed diversi
            classic_runs, mhn_runs = [], []

            for boot in range(N_BOOTSTRAP):
                rng = np.random.default_rng(SEED + boot * 100)

                # Classica: pattern ortogonali
                mem_c = select_orthogonal(traces, N_STORE_CLASSIC)
                rr_c  = measure_recovery_classic(mem_c, rate, N_TRIALS, rng)
                classic_runs.append(rr_c)

                # MHN: campione casuale
                idx   = rng.choice(len(traces), size=N_STORE_MHN, replace=False)
                mem_m = traces[idx].astype(np.float64)
                rr_m  = measure_recovery_mhn(mem_m, rate, N_TRIALS, rng)
                mhn_runs.append(rr_m)

            # Statistiche bootstrap: media ± std
            c_mean, c_std = np.mean(classic_runs), np.std(classic_runs)
            m_mean, m_std = np.mean(mhn_runs),     np.std(mhn_runs)

            results[level]["classic"][rate] = {"mean": c_mean, "std": c_std,
                                                "runs": classic_runs}
            results[level]["mhn"][rate]     = {"mean": m_mean, "std": m_std,
                                                "runs": mhn_runs}

            print(f"  rate={int(rate*100):2d}%  |  "
                  f"Classic: {c_mean:.2%} ±{c_std:.2%}  |  "
                  f"MHN: {m_mean:.2%} ±{m_std:.2%}")

    # Salvataggio
    out_path = os.path.join(RESULTS_DIR, "exp4_complexity.json")
    serializable = {}
    for level in LEVELS:
        serializable[level] = {}
        for model in ["classic", "mhn"]:
            serializable[level][model] = {
                str(k): v for k, v in results[level][model].items()
            }
    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2)

    print(f"\n  ✅ Risultati salvati in {out_path}")
    return results


if __name__ == "__main__":
    results = run_experiment()

    # Tabella finale: effetto della complessità a corruption=15%
    rate_focus = 0.15
    print(f"\n── Effetto complessità LTLf (corruption={int(rate_focus*100)}%) ──")
    print(f"  {'Livello':<10} {'Classic mean':>14} {'Classic std':>12} "
          f"{'MHN mean':>12} {'MHN std':>10}")
    print("  " + "─" * 60)
    for level in LEVELS:
        rc = results[level]["classic"][rate_focus]
        rm = results[level]["mhn"][rate_focus]
        print(f"  {level:<10} {rc['mean']:>14.2%} {rc['std']:>12.2%} "
              f"{rm['mean']:>12.2%} {rm['std']:>10.2%}")

    # Tendenza: la complessità aiuta o danneggia?
    print(f"\n── Tendenza recovery rate al crescere della complessità ──")
    for model in ["classic", "mhn"]:
        vals = [results[l][model][rate_focus]["mean"] for l in LEVELS]
        trend = "↑ migliora" if vals[2] > vals[0] else "↓ peggiora"
        print(f"  {model:<10}: simple={vals[0]:.1%}  medium={vals[1]:.1%}  "
              f"complex={vals[2]:.1%}  {trend}")
"""
experiments/exp3_comparison.py
Esperimento 3: confronto diretto Classic vs MHN su stessa scala.
Misura: accuracy, n_steps a convergenza, profilo energetico medio.
Entrambi i modelli operano su N_STORE pattern ortogonali
per un confronto equo punto a punto.
"""

import numpy as np
import json
import os
import sys
import csv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from classical.hopfield_bindings import ClassicalHopfield
from mhn.modern_hopfield import ModernHopfieldNetwork

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset", "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

N_STORE = 4
N_TRIALS = 300
CORRUPTION = 0.15
MHN_BETAS = [1.0, 2.0, 5.0, 10.0]
CLASSIC_STEPS = 50
MHN_STEPS = 30
SEED = 42
LEVELS = ["simple", "medium", "complex"]


def select_orthogonal(traces, k):
    all_t = traces.astype(np.float64)
    corr = np.abs(all_t @ all_t.T) / all_t.shape[1]
    np.fill_diagonal(corr, 1.0)
    selected = [0]
    for _ in range(k - 1):
        avg_corr = corr[selected].mean(axis=0)
        avg_corr[selected] = 1.0
        selected.append(int(np.argmin(avg_corr)))
    return all_t[selected]


def corrupt_pattern(x, corruption, rng):
    y = x.copy()
    mask = rng.random(len(y)) < corruption
    y[mask] *= -1.0
    return y


def pad_energy_profiles(profiles):
    if not profiles:
        return [], [], []
    max_len = max(len(p) for p in profiles)
    padded = []
    for p in profiles:
        if len(p) < max_len:
            p = p + [p[-1]] * (max_len - len(p))
        padded.append(p)
    arr = np.array(padded, dtype=np.float64)
    return arr.mean(axis=0).tolist(), arr.std(axis=0).tolist(), max_len


def run_experiment():
    rng = np.random.default_rng(SEED)
    results = {}

    for level in LEVELS:
        print(f"\n{'─' * 55}")
        print(
            f"  Livello: {level.upper()}  |  {N_STORE} pattern  |  "
            f"corruption={int(CORRUPTION * 100)}%"
        )
        print(f"{'─' * 55}")

        traces = np.load(os.path.join(DATA_DIR, f"valid_{level}.npy"))
        memories = select_orthogonal(traces, N_STORE)

        results[level] = {}

        # =========================
        # Hopfield classica
        # =========================
        net = ClassicalHopfield()
        net.store(memories)

        c_correct = 0
        c_steps = []
        c_energy_profiles = []
        c_hamming = []

        for _ in range(N_TRIALS):
            idx = rng.integers(N_STORE)
            original = memories[idx].copy()
            corrupted = corrupt_pattern(original, CORRUPTION, rng)

            recovered, energy_trace, steps = net.retrieve(
                corrupted, max_steps=CLASSIC_STEPS
            )

            if np.array_equal(recovered, original):
                c_correct += 1

            c_steps.append(int(steps))
            c_energy_profiles.append(list(energy_trace))
            c_hamming.append(float(np.mean(recovered != original)))

        c_rate = c_correct / N_TRIALS
        c_steps_mean = float(np.mean(c_steps))
        c_steps_std = float(np.std(c_steps))
        c_energy_start = float(np.mean([e[0] for e in c_energy_profiles if len(e) > 0]))
        c_energy_end = float(np.mean([e[-1] for e in c_energy_profiles if len(e) > 0]))
        c_hamming_mean = float(np.mean(c_hamming))
        c_energy_mean_profile, c_energy_std_profile, c_profile_len = pad_energy_profiles(c_energy_profiles)

        print(f"\n  CLASSICA")
        print(f"    Accuracy:          {c_rate:.2%}")
        print(f"    Steps a conv.:     {c_steps_mean:.1f} ± {c_steps_std:.1f}")
        print(f"    Hamming finale:    {c_hamming_mean:.4f}")
        print(f"    Energia iniziale:  {c_energy_start:.2f}")
        print(f"    Energia finale:    {c_energy_end:.2f}")

        results[level]["classic"] = {
            "accuracy": c_rate,
            "steps_mean": c_steps_mean,
            "steps_std": c_steps_std,
            "hamming_mean": c_hamming_mean,
            "energy_start": c_energy_start,
            "energy_end": c_energy_end,
            "energy_profile_mean": c_energy_mean_profile,
            "energy_profile_std": c_energy_std_profile,
            "profile_len": c_profile_len,
        }

        # =========================
        # MHN sweep su beta
        # =========================
        print(f"\n  MHN (sweep beta)")
        results[level]["mhn"] = {}

        best_beta = None
        best_acc = -1.0

        for beta in MHN_BETAS:
            mhn = ModernHopfieldNetwork(beta=beta)
            mhn.store(memories)

            m_correct = 0
            m_steps_list = []
            m_energy_profiles = []
            m_hamming = []

            for _ in range(N_TRIALS):
                idx = rng.integers(N_STORE)
                original = memories[idx].copy()
                corrupted = corrupt_pattern(original, CORRUPTION, rng)

                recovered, energy_trace, steps = mhn.retrieve_tracked(
                    corrupted, n_steps=MHN_STEPS
                )

                recovered_bin = np.sign(recovered)
                recovered_bin[recovered_bin == 0] = 1

                if np.array_equal(recovered_bin, original):
                    m_correct += 1

                m_steps_list.append(int(steps))
                m_energy_profiles.append(list(energy_trace))
                m_hamming.append(float(np.mean(recovered_bin != original)))

            m_rate = m_correct / N_TRIALS
            m_steps_mean = float(np.mean(m_steps_list))
            m_steps_std = float(np.std(m_steps_list))
            m_energy_start = float(np.mean([e[0] for e in m_energy_profiles if len(e) > 0]))
            m_energy_end = float(np.mean([e[-1] for e in m_energy_profiles if len(e) > 0]))
            m_hamming_mean = float(np.mean(m_hamming))
            m_energy_mean_profile, m_energy_std_profile, m_profile_len = pad_energy_profiles(m_energy_profiles)

            print(
                f"    beta={beta:>4.1f}  |  "
                f"acc={m_rate:.2%}  |  "
                f"steps={m_steps_mean:.1f}±{m_steps_std:.1f}  |  "
                f"hamming={m_hamming_mean:.4f}  |  "
                f"E_start={m_energy_start:.2f}  "
                f"E_end={m_energy_end:.2f}"
            )

            results[level]["mhn"][f"beta_{beta}"] = {
                "beta": beta,
                "accuracy": m_rate,
                "steps_mean": m_steps_mean,
                "steps_std": m_steps_std,
                "hamming_mean": m_hamming_mean,
                "energy_start": m_energy_start,
                "energy_end": m_energy_end,
                "energy_profile_mean": m_energy_mean_profile,
                "energy_profile_std": m_energy_std_profile,
                "profile_len": m_profile_len,
            }

            if m_rate > best_acc:
                best_acc = m_rate
                best_beta = beta

        results[level]["mhn_best_beta"] = best_beta
        results[level]["mhn_best_accuracy"] = best_acc

        print(f"\n    → Best beta su {level}: {best_beta} (accuracy={best_acc:.2%})")

    # =========================
    # Salvataggio JSON
    # =========================
    out_json = os.path.join(RESULTS_DIR, "exp3_comparison.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    # =========================
    # Salvataggio CSV riassuntivo
    # =========================
    out_csv = os.path.join(RESULTS_DIR, "exp3_comparison.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "level", "model", "beta", "accuracy",
            "steps_mean", "steps_std", "hamming_mean",
            "energy_start", "energy_end", "delta_energy"
        ])

        for level in LEVELS:
            r = results[level]["classic"]
            writer.writerow([
                level, "classic", "",
                r["accuracy"], r["steps_mean"], r["steps_std"],
                r["hamming_mean"], r["energy_start"], r["energy_end"],
                r["energy_end"] - r["energy_start"]
            ])

            for _, rr in results[level]["mhn"].items():
                writer.writerow([
                    level, "mhn", rr["beta"],
                    rr["accuracy"], rr["steps_mean"], rr["steps_std"],
                    rr["hamming_mean"], rr["energy_start"], rr["energy_end"],
                    rr["energy_end"] - rr["energy_start"]
                ])

    print(f"\n✅ Risultati salvati in:")
    print(f"   - {out_json}")
    print(f"   - {out_csv}")

    return results


if __name__ == "__main__":
    results = run_experiment()

    print("\n── Tabella confronto (livello COMPLEX, corruption=15%) ──")
    lvl = "complex"
    print(f"  {'Modello':<18} {'Accuracy':>10} {'Steps':>10} {'Ham':>10} {'ΔE':>10}")
    print("  " + "─" * 65)

    r = results[lvl]["classic"]
    delta_e = r["energy_end"] - r["energy_start"]
    print(
        f"  {'Classic':<18} {r['accuracy']:>10.2%} "
        f"{r['steps_mean']:>10.1f} {r['hamming_mean']:>10.4f} {delta_e:>10.2f}"
    )

    for _, r in results[lvl]["mhn"].items():
        delta_e = r["energy_end"] - r["energy_start"]
        beta_val = r["beta"]
        label = f"MHN β={beta_val}"
        print(
            f"  {label:<18} {r['accuracy']:>10.2%} "
            f"{r['steps_mean']:>10.1f} {r['hamming_mean']:>10.4f} {delta_e:>10.2f}"
        )

    print(
        f"\nBest beta su {lvl}: "
        f"{results[lvl]['mhn_best_beta']} "
        f"(accuracy={results[lvl]['mhn_best_accuracy']:.2%})"
    )
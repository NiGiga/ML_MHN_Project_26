"""
tests/test_sanity.py
Test di sanità per ClassicalHopfield e ModernHopfieldNetwork.
Esegui con: pytest tests/ -v
"""
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from classical.hopfield_bindings import ClassicalHopfield

# ── Fixture: 4 pattern ortogonali da manuale ─────────────────────────────────
@pytest.fixture
def orthogonal_patterns():
    """
    4 pattern binari {-1,+1} di dimensione 32, quasi ortogonali.
    Generati con seed fisso per riproducibilità.
    """
    rng = np.random.default_rng(seed=7)
    patterns = rng.choice([-1.0, 1.0], size=(4, 32))
    return patterns.astype(np.float64)


# ── Test 1: recupero perfetto senza corruzione ────────────────────────────────
def test_perfect_recall(orthogonal_patterns):
    """
    Un pattern memorizzato deve essere recuperato esattamente
    quando presentato senza corruzione.
    """
    net = ClassicalHopfield()
    net.store(orthogonal_patterns)

    for i, pat in enumerate(orthogonal_patterns):
        recovered, _, steps = net.retrieve(pat.copy(), max_steps=30)
        hamming = np.mean(recovered != pat)
        assert hamming == 0.0, (
            f"Pattern {i}: recupero imperfetto, hamming={hamming:.2%} in {steps} step"
        )
    print("\n  ✅ Recupero perfetto: OK")


# ── Test 2: recupero da query corrotta (20% bit-flip) ────────────────────────
def test_corrupted_recall(orthogonal_patterns):
    """
    Un pattern corrotto al 20% deve essere recuperato correttamente
    per almeno 3 pattern su 4 (soglia conservativa).
    """
    rng = np.random.default_rng(seed=42)
    net = ClassicalHopfield()
    net.store(orthogonal_patterns)

    correct = 0
    for pat in orthogonal_patterns:
        corrupted = pat.copy()
        mask = rng.random(len(pat)) < 0.20
        corrupted[mask] *= -1.0

        recovered, _, _ = net.retrieve(corrupted, max_steps=30)
        if np.array_equal(recovered, pat):
            correct += 1

    print(f"\n  Recupero con 20% corruzione: {correct}/4")
    assert correct >= 3, f"Troppi fallimenti: {correct}/4 corretti"


# ── Test 3: energia non cresce durante il recupero ───────────────────────────
def test_energy_decreasing(orthogonal_patterns):
    """
    L'energia deve essere monotonicamente non crescente durante il recupero.
    È la proprietà fondamentale della rete di Hopfield.
    """
    rng = np.random.default_rng(seed=1)
    net = ClassicalHopfield()
    net.store(orthogonal_patterns)

    pat = orthogonal_patterns[0].copy()
    corrupted = pat.copy()
    mask = rng.random(len(pat)) < 0.15
    corrupted[mask] *= -1.0

    _, energy_trace, steps = net.retrieve(corrupted, max_steps=50)

    for t in range(1, steps):
        assert energy_trace[t] <= energy_trace[t-1] + 1e-9, (
            f"Energia crescente allo step {t}: "
            f"{energy_trace[t-1]:.4f} → {energy_trace[t]:.4f}"
        )
    print(f"\n  Energia monotona su {steps} step: ✅")


# ── Test 4: capacità di stoccaggio (soglia teorica ~0.138 * N) ───────────────
def test_storage_capacity():
    """
    La rete di Hopfield classica con N=32 neuoni dovrebbe memorizzare
    correttamente fino a ~0.138 * 32 ≈ 4 pattern.
    Con 8 pattern (oltre capacità) il recupero deve degradare.
    """
    rng = np.random.default_rng(seed=99)
    N = 32
    theoretical_capacity = int(0.138 * N)  # ≈ 4

    # Test dentro capacità
    patterns_ok = rng.choice([-1.0, 1.0], size=(theoretical_capacity, N)).astype(np.float64)
    net = ClassicalHopfield()
    net.store(patterns_ok)

    correct_in = 0
    for pat in patterns_ok:
        rec, _, _ = net.retrieve(pat.copy(), max_steps=30)
        if np.array_equal(rec, pat):
            correct_in += 1

    print(f"\n  Dentro capacità ({theoretical_capacity} pattern): {correct_in}/{theoretical_capacity} corretti")
    assert correct_in >= theoretical_capacity - 1, "Troppi fallimenti dentro la capacità teorica"

    # Test oltre capacità (il recupero deve degradare — non è un errore)
    patterns_over = rng.choice([-1.0, 1.0], size=(theoretical_capacity * 3, N)).astype(np.float64)
    net2 = ClassicalHopfield()
    net2.store(patterns_over)

    correct_over = 0
    for pat in patterns_over:
        rec, _, _ = net2.retrieve(pat.copy(), max_steps=30)
        if np.array_equal(rec, pat):
            correct_over += 1

    rate_over = correct_over / len(patterns_over)
    print(f"  Oltre capacità ({theoretical_capacity*3} pattern): {correct_over}/{theoretical_capacity*3} corretti ({rate_over:.0%})")
    # Non assertiamo il fallimento — lo documentiamo solo


# ── Test 5: caricamento dataset LTLf ─────────────────────────────────────────
def test_dataset_loading():
    """
    Verifica che il dataset generato sia caricabile e coerente.
    """
    data_path = os.path.join(os.path.dirname(__file__), "..", "dataset", "data")
    valid = np.load(os.path.join(data_path, "valid_traces.npy"))

    assert valid.shape == (1500, 32), f"Shape inattesa: {valid.shape}"
    assert np.all(np.isin(valid, [-1.0, 1.0])), "Valori fuori {-1, +1}"

    for level in ["simple", "medium", "complex"]:
        arr = np.load(os.path.join(data_path, f"valid_{level}.npy"))
        assert arr.shape[1] == 32
        assert len(arr) == 500, f"{level}: attese 500 tracce, trovate {len(arr)}"

        for rate in [5, 10, 15, 20, 25]:
            fname = f"corrupted_{level}_{rate:02d}pct.npy"
            c = np.load(os.path.join(data_path, fname))
            assert c.shape == arr.shape, f"Shape mismatch: {fname}"

    print("\n  ✅ Dataset: tutti i file presenti e coerenti")

# ═══════════════════════════════════════════════════════════════════════════════
# Test MHN
# ═══════════════════════════════════════════════════════════════════════════════
from mhn.modern_hopfield import ModernHopfieldNetwork


def test_mhn_perfect_recall(orthogonal_patterns):
    """MHN deve recuperare esattamente i pattern memorizzati."""
    mhn = ModernHopfieldNetwork(beta=5.0)
    mhn.store(orthogonal_patterns)

    for i, pat in enumerate(orthogonal_patterns):
        recovered = mhn.retrieve_binary(pat.copy(), n_steps=20)
        hamming = np.mean(recovered != pat)
        assert hamming == 0.0, f"MHN pattern {i}: hamming={hamming:.2%}"
    print("\n  ✅ MHN recupero perfetto: OK")


def test_mhn_energy_decreasing(orthogonal_patterns):
    """Energia MHN deve essere monotonicamente non crescente."""
    rng = np.random.default_rng(seed=3)
    mhn = ModernHopfieldNetwork(beta=5.0)
    mhn.store(orthogonal_patterns)

    pat       = orthogonal_patterns[0].copy()
    corrupted = pat.copy()
    corrupted[rng.random(len(pat)) < 0.15] *= -1.0

    _, energy_trace, steps = mhn.retrieve_tracked(corrupted, n_steps=50)

    for t in range(1, len(energy_trace)):
        assert energy_trace[t] <= energy_trace[t-1] + 1e-6, (
            f"MHN energia crescente step {t}: "
            f"{energy_trace[t-1]:.4f} → {energy_trace[t]:.4f}"
        )
    print(f"\n  ✅ MHN energia monotona su {steps} step: OK")


def test_mhn_vs_classical_capacity(orthogonal_patterns):
    """
    MHN con beta alto deve recuperare più pattern della rete classica
    quando siamo vicini alla capacità teorica (~0.138*N).
    """
    rng = np.random.default_rng(seed=55)
    N   = 32
    n_patterns = int(0.138 * N) + 1   # leggermente oltre la capacità classica

    extra = rng.choice([-1.0, 1.0], size=(n_patterns, N)).astype(np.float64)

    # Classica
    net = ClassicalHopfield()
    net.store(extra)
    correct_classic = sum(
        np.array_equal(net.retrieve(p.copy(), 30)[0], p) for p in extra
    )

    # MHN con beta alto
    mhn = ModernHopfieldNetwork(beta=10.0)
    mhn.store(extra)
    correct_mhn = sum(
        np.array_equal(mhn.retrieve_binary(p.copy(), 20), p) for p in extra
    )

    print(f"\n  Classica: {correct_classic}/{n_patterns} | MHN: {correct_mhn}/{n_patterns}")
    # MHN deve fare almeno uguale o meglio
    assert correct_mhn >= correct_classic, \
        f"MHN ({correct_mhn}) peggio della classica ({correct_classic})"
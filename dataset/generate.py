"""
dataset/generate.py  —  versione 2 (API spot corrette)
"""

import spot
import numpy as np
import json, os, itertools
from collections import deque
from tqdm import tqdm

# ── Configurazione ───────────────────────────────────────────────────────────
N_PROPS          = 4
TRACE_LEN        = 8
N_PER_LEVEL      = 500
CORRUPTION_RATES = [0.05, 0.10, 0.15, 0.20, 0.25]
PROP_NAMES       = [f"p{i}" for i in range(N_PROPS)]
OUTPUT_DIR       = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

COMPLEXITY_LEVELS = {
    "simple":  {"tree_size": (2, 5)},
    "medium":  {"tree_size": (5, 9)},
    "complex": {"tree_size": (9, 14)},
}

# ── Generazione formule ──────────────────────────────────────────────────────
def generate_formulas(level: str, n: int = 80) -> list[str]:
    cfg = COMPLEXITY_LEVELS[level]
    # ltl_priorities: SOLO operatori temporali
    # boolean_priorities: SOLO operatori booleani — ma causa errore con LTL output
    # Soluzione: nessuna priorità personalizzata, usa solo tree_size per controllare la complessità
    gen = spot.randltl(
        PROP_NAMES,
        ltl_count=n * 6,
        seed=42,
        tree_size=cfg["tree_size"],
        simplify=1,
    )
    formulas = []
    for f in gen:
        s = str(f)
        if s not in ("0", "1", "true", "false") and len(s) > 2:
            formulas.append(s)
        if len(formulas) >= n:
            break
    return formulas


# ── Valutazione simbolo su etichetta BDD ────────────────────────────────────
def symbol_satisfies_bdd(bdict, ap_list, sym_tuple, bdd_cond) -> bool:
    """
    Verifica se un simbolo (tupla di 0/1) soddisfa un BDD di transizione.
    Usa spot.bdd_format_formula per leggere l'etichetta come stringa
    e valutarla direttamente come assegnamento.
    """
    label = spot.bdd_format_formula(bdict, bdd_cond)
    if label == "1":
        return True
    if label == "0":
        return False

    # Costruisce un dizionario ap_name → valore booleano
    assignment = {str(ap): bool(sym_tuple[i]) for i, ap in enumerate(ap_list)}

    # Valuta l'espressione con sostituzione diretta
    # Gestisce: !ap, ap & ap, ap | ap, ap, 1, 0
    expr = label
    # Sostituisce ogni proposizione con True/False (in ordine decrescente di lunghezza)
    for ap_name in sorted(assignment.keys(), key=len, reverse=True):
        expr = expr.replace(ap_name, str(assignment[ap_name]))
    # Normalizza la notazione spot → Python
    expr = expr.replace("!", " not ").replace("&", " and ").replace("|", " or ")
    try:
        return bool(eval(expr))
    except Exception:
        return False


# ── BFS sampling su DFA spot ─────────────────────────────────────────────────
def sample_traces_from_aut(aut, trace_len: int, max_traces: int,
                           rng: np.random.Generator) -> list:
    """
    BFS/random walk su un automa spot per raccogliere tracce accettanti.
    Usa l'API Python ufficiale di spot (bdd_format_formula, aut.out, aut.acc).
    """
    bdict   = aut.get_dict()
    ap_list = list(aut.ap())
    n_ap    = len(ap_list)
    init    = aut.get_init_state_number()

    # Tutti i simboli possibili: 2^n_ap tuple di 0/1
    # Limitiamo a N_PROPS proposizioni; padding con 0 se l'automa ne ha di meno
    all_syms = list(itertools.product([0, 1], repeat=n_ap))

    traces = []
    # BFS con randomizzazione per diversificare le tracce
    queue = deque()
    queue.append((init, []))
    visited_paths = set()

    while queue and len(traces) < max_traces * 4:
        state, path = queue.popleft()

        if len(path) == trace_len:
            # Controlla accettazione: usiamo la condizione di accettazione Büchi/Reachability
            # Per DFA/safety: lo stato finale deve essere "accettante"
            try:
                is_acc = aut.state_is_accepting(state)
            except Exception:
                # Fallback: controlla se qualche transizione uscente è accepting
                is_acc = any(bool(t.acc) for t in aut.out(state))
            if is_acc:
                key = tuple(x for step in path for x in step)
                if key not in visited_paths:
                    visited_paths.add(key)
                    traces.append(path)
            continue

        # Esplora le transizioni in ordine casuale
        out_edges = list(aut.out(state))
        rng.shuffle(out_edges)

        for t in out_edges:
            # Prova tutti i simboli che soddisfano la guardia BDD
            syms_shuffled = list(all_syms)
            rng.shuffle(syms_shuffled)
            for sym in syms_shuffled:
                if symbol_satisfies_bdd(bdict, ap_list, sym, t.cond):
                    # Padda il simbolo a N_PROPS elementi
                    padded = list(sym) + [0] * (N_PROPS - n_ap)
                    new_path = path + [padded[:N_PROPS]]
                    queue.append((t.dst, new_path))
                    break   # una transizione per simbolo basta per la BFS

        if len(traces) >= max_traces:
            break

    return traces[:max_traces]


# ── Pipeline formula → tracce ────────────────────────────────────────────────
def formula_to_traces(formula_str: str, n_traces: int,
                      rng: np.random.Generator) -> list:
    """
    Traduce formula LTL in automa DFA deterministico e campiona tracce.
    """
    try:
        # spot.translate: produce DBA deterministico, completo
        aut = spot.translate(
            formula_str,
            "deterministic", "complete", "state-based"
        )
    except Exception as e:
        return []

    traces = sample_traces_from_aut(aut, TRACE_LEN, n_traces, rng)
    return traces


# ── Conversione traccia → vettore {-1, +1} ───────────────────────────────────
def trace_to_vector(trace: list) -> np.ndarray:
    """Traccia (lista di liste 0/1) → vettore piatto {-1, +1} shape (TRACE_LEN*N_PROPS,)"""
    arr = np.full((TRACE_LEN, N_PROPS), -1.0, dtype=np.float32)
    for t, step in enumerate(trace[:TRACE_LEN]):
        for p, val in enumerate(step[:N_PROPS]):
            arr[t, p] = 1.0 if val == 1 else -1.0
    return arr.flatten()


# ── Corruzione ────────────────────────────────────────────────────────────────
def corrupt_trace(vec: np.ndarray, rate: float,
                  rng: np.random.Generator) -> np.ndarray:
    corrupted = vec.copy()
    mask = rng.random(vec.shape) < rate
    corrupted[mask] *= -1.0
    return corrupted


# ── Validazione ───────────────────────────────────────────────────────────────
def validate_dataset(arr: np.ndarray):
    print("\n── Validazione dataset ──")
    assert arr.ndim == 2
    assert arr.shape[1] == TRACE_LEN * N_PROPS, f"dim errata: {arr.shape[1]}"
    assert np.all(np.isin(arr, [-1.0, 1.0])), "Valori fuori {-1, +1}"
    unique = np.unique(arr, axis=0)
    print(f"  Totali:  {len(arr)}")
    print(f"  Unici:   {len(unique)}")
    print(f"  Ratio+1: {(arr == 1.0).mean():.2%}")
    print("  ✅ OK")


# ── Build principale ──────────────────────────────────────────────────────────
def build_dataset():
    rng = np.random.default_rng(seed=0)
    all_valid   = {}
    all_corrupt = {}
    metadata    = {
        "dim": TRACE_LEN * N_PROPS, "trace_len": TRACE_LEN,
        "n_props": N_PROPS, "props": PROP_NAMES,
        "corruption_rates": CORRUPTION_RATES, "levels": {}
    }

    for level in ["simple", "medium", "complex"]:
        print(f"\n{'─'*50}")
        print(f"  Livello: {level.upper()}")
        print(f"{'─'*50}")

        formulas = generate_formulas(level, n=150)
        print(f"  Formule generate: {len(formulas)}")

        valid_vecs, formula_log = [], []

        for fstr in tqdm(formulas, desc=f"  Sampling {level}"):
            traces = formula_to_traces(fstr, n_traces=25, rng=rng)
            for tr in traces:
                vec = trace_to_vector(tr)
                if not np.all(vec == vec[0]):   # scarta vettori costanti
                    valid_vecs.append(vec)
                    formula_log.append(fstr)
            if len(valid_vecs) >= N_PER_LEVEL * 2:
                break

        if len(valid_vecs) == 0:
            print(f"  ⚠️  Nessuna traccia valida per {level}!")
            all_valid[level] = np.zeros((0, TRACE_LEN * N_PROPS), dtype=np.float32)
            all_corrupt[level] = {}
            continue

        # Deduplica e tronca
        arr = np.unique(np.array(valid_vecs, dtype=np.float32), axis=0)
        arr = arr[:N_PER_LEVEL]
        print(f"  Tracce valide uniche: {len(arr)}")
        all_valid[level] = arr

        # Versioni corrupted
        corrupt_by_rate = {}
        for rate in CORRUPTION_RATES:
            corrupt_by_rate[rate] = np.array(
                [corrupt_trace(v, rate, rng) for v in arr], dtype=np.float32
            )
        all_corrupt[level] = corrupt_by_rate

        metadata["levels"][level] = {
            "n_traces": len(arr),
            "sample_formulas": list(set(formula_log))[:5],
        }

    # Salvataggio
    print("\n  Salvataggio file...")
    levels_with_data = [l for l in ["simple","medium","complex"] if len(all_valid[l]) > 0]
    valid_all = np.concatenate([all_valid[l] for l in levels_with_data], axis=0)
    np.save(os.path.join(OUTPUT_DIR, "valid_traces.npy"), valid_all)

    for level in levels_with_data:
        np.save(os.path.join(OUTPUT_DIR, f"valid_{level}.npy"), all_valid[level])
        for rate in CORRUPTION_RATES:
            fname = f"corrupted_{level}_{int(rate*100):02d}pct.npy"
            np.save(os.path.join(OUTPUT_DIR, fname), all_corrupt[level][rate])

    with open(os.path.join(OUTPUT_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n  ✅ Dataset completato!")
    print(f"  Shape: {valid_all.shape}  (dim = {TRACE_LEN}×{N_PROPS} = {TRACE_LEN*N_PROPS})")
    print(f"  File in: {OUTPUT_DIR}/")
    return valid_all, all_corrupt, metadata


if __name__ == "__main__":
    valid_all, all_corrupt, metadata = build_dataset()
    validate_dataset(valid_all)
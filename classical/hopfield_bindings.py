"""
classical/hopfield_bindings.py
Wrapper ctypes per hopfield.so
"""
import ctypes
import numpy as np
import os

_lib_path = os.path.join(os.path.dirname(__file__), "hopfield.so")
_lib = ctypes.CDLL(_lib_path)

# ── Firma delle funzioni C ────────────────────────────────────────────────────
_lib.alloc_weights.restype  = ctypes.POINTER(ctypes.c_double)
_lib.alloc_weights.argtypes = [ctypes.c_int]

_lib.free_weights.restype  = None
_lib.free_weights.argtypes = [ctypes.POINTER(ctypes.c_double)]

_lib.hebb_learning.restype  = None
_lib.hebb_learning.argtypes = [
    ctypes.POINTER(ctypes.c_double),  # W
    ctypes.POINTER(ctypes.c_double),  # patterns
    ctypes.c_int,                     # n_patterns
    ctypes.c_int,                     # n
]

_lib.retrieve_with_energy.restype  = ctypes.c_int
_lib.retrieve_with_energy.argtypes = [
    ctypes.POINTER(ctypes.c_double),  # W
    ctypes.POINTER(ctypes.c_double),  # query (modified in-place)
    ctypes.c_int,                     # n
    ctypes.c_int,                     # max_steps
    ctypes.POINTER(ctypes.c_double),  # energy_trace
    ctypes.POINTER(ctypes.c_int),     # order
]

_lib.compute_energy.restype  = ctypes.c_double
_lib.compute_energy.argtypes = [
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_int,
]

_lib.hamming_distance.restype  = ctypes.c_double
_lib.hamming_distance.argtypes = [
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_int,
]


# ── Classe Python ─────────────────────────────────────────────────────────────
class ClassicalHopfield:
    def __init__(self):
        self.W    = None
        self.n    = None
        self._ptr = None   # puntatore C grezzo

    def store(self, patterns: np.ndarray):
        """
        patterns: np.ndarray shape (n_patterns, n), dtype float64, valori {-1, +1}
        """
        patterns = np.ascontiguousarray(patterns, dtype=np.float64)
        n_patterns, n = patterns.shape
        self.n = n

        self._ptr = _lib.alloc_weights(n)
        _lib.hebb_learning(
            self._ptr,
            patterns.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_int(n_patterns),
            ctypes.c_int(n),
        )
        # Copia in numpy per uso Python
        self.W = np.frombuffer(
            (ctypes.c_double * (n*n)).from_address(
                ctypes.addressof(self._ptr.contents)
            ), dtype=np.float64
        ).reshape(n, n).copy()

    def retrieve(self, query: np.ndarray, max_steps: int = 20):
        """
        Recupera il pattern più vicino a query.
        Restituisce (stato_finale, energy_trace, n_steps)
        """
        assert self.n is not None, "Chiama store() prima di retrieve()"
        n = self.n
        q = np.ascontiguousarray(query, dtype=np.float64).copy()

        energy_trace = np.zeros(max_steps, dtype=np.float64)
        rng = np.random.default_rng()
        order = np.ascontiguousarray(
            rng.permutation(n).astype(np.int32)
        )

        steps = _lib.retrieve_with_energy(
            self._ptr,
            q.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_int(n),
            ctypes.c_int(max_steps),
            energy_trace.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            order.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        )
        return q, energy_trace[:steps], steps

    def energy(self, state: np.ndarray) -> float:
        s = np.ascontiguousarray(state, dtype=np.float64)
        return _lib.compute_energy(
            self._ptr,
            s.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_int(self.n),
        )

    def __del__(self):
        if self._ptr is not None:
            _lib.free_weights(self._ptr)
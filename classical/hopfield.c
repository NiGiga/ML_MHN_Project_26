/*
 * classical/hopfield.c
 * Rete di Hopfield classica con regola di Hebb e update asincrono.
 * Compilazione: gcc -O2 -fopenmp -shared -fPIC -o hopfield.so hopfield.c
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#endif

/* ── Allocazione matrice pesi ─────────────────────────────────────────────── */
double* alloc_weights(int n) {
    double *W = (double*)calloc(n * n, sizeof(double));
    if (!W) { fprintf(stderr, "OOM: alloc_weights\n"); exit(1); }
    return W;
}

void free_weights(double *W) { free(W); }

/* ── Regola di Hebb ───────────────────────────────────────────────────────── */
/*
 * W_ij = (1/n) * sum_k [ xi_k_i * xi_k_j ]   con W_ii = 0
 * patterns: array piatto (n_patterns * n), row-major
 */
void hebb_learning(double *W, const double *patterns,
                   int n_patterns, int n) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i == j) { W[i*n + j] = 0.0; continue; }
            double s = 0.0;
            for (int k = 0; k < n_patterns; k++)
                s += patterns[k*n + i] * patterns[k*n + j];
            W[i*n + j] = s / n;
        }
    }
}

/* ── Calcolo campo locale ─────────────────────────────────────────────────── */
static inline double local_field(const double *W, const double *state,
                                  int i, int n) {
    double h = 0.0;
    for (int j = 0; j < n; j++)
        h += W[i*n + j] * state[j];
    return h;
}

/* ── Update asincrono (sequenziale casuale) ───────────────────────────────── */
/*
 * Aggiorna i neuroni uno alla volta in ordine casuale.
 * Restituisce 1 se lo stato è cambiato, 0 se convergenza raggiunta.
 */
int async_update(const double *W, double *state, int n,
                 int *order, int max_steps) {
    double *prev = (double*)malloc(n * sizeof(double));
    int converged = 0;

    for (int step = 0; step < max_steps && !converged; step++) {
        memcpy(prev, state, n * sizeof(double));

        /* Fisher-Yates shuffle dell'ordine di aggiornamento */
        for (int i = n-1; i > 0; i--) {
            int j = order[i];           /* ordine passato dall'esterno */
            double h = local_field(W, state, j, n);
            state[j] = (h >= 0.0) ? 1.0 : -1.0;
        }

        /* Controlla convergenza */
        converged = 1;
        for (int i = 0; i < n; i++) {
            if (prev[i] != state[i]) { converged = 0; break; }
        }
    }

    free(prev);
    return converged;
}

/* ── Energia della configurazione ────────────────────────────────────────── */
double compute_energy(const double *W, const double *state, int n) {
    double E = 0.0;
    #pragma omp parallel for reduction(-:E) schedule(static)
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            E -= 0.5 * W[i*n + j] * state[i] * state[j];
    return E;
}

/* ── Distanza di Hamming normalizzata ─────────────────────────────────────── */
double hamming_distance(const double *a, const double *b, int n) {
    int diff = 0;
    for (int i = 0; i < n; i++)
        if (a[i] != b[i]) diff++;
    return (double)diff / n;
}

/* ── Recupero completo con tracciamento energia ───────────────────────────── */
/*
 * Esegue il recupero e riempie energy_trace (array di max_steps double).
 * Restituisce il numero di step effettivi fino a convergenza.
 * query viene modificato in-place → contiene lo stato finale recuperato.
 */
int retrieve_with_energy(const double *W, double *query,
                          int n, int max_steps,
                          double *energy_trace, int *order) {
    int steps = 0;
    for (int step = 0; step < max_steps; step++) {
        energy_trace[step] = compute_energy(W, query, n);

        double *prev = (double*)malloc(n * sizeof(double));
        memcpy(prev, query, n * sizeof(double));

        /* Update asincrono: un'epoca = n aggiornamenti */
        for (int i = n-1; i >= 0; i--) {
            int j = order[i];
            double h = local_field(W, query, j, n);
            query[j] = (h >= 0.0) ? 1.0 : -1.0;
        }

        steps++;
        int conv = 1;
        for (int i = 0; i < n; i++)
            if (prev[i] != query[i]) { conv = 0; break; }
        free(prev);
        if (conv) break;
    }
    return steps;
}
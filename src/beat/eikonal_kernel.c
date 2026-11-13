/* Fast Iterative Method (FIM) for the anisotropic eikonal equation on simplices.
 *
 * Solves sqrt(grad(t)^T V grad(t)) = 1 at the vertices of a mesh of triangles (cell_size 3)
 * or tetrahedra (cell_size 4). `metric` holds V^{-1} per cell as 6 symmetric components
 * (xx, yy, zz, xy, xz, yz); coordinates are always 3D.
 *
 * The local update of vertex x in a cell with the other vertices y_i minimises
 *   t(y) + ||y - x||_{V^{-1}}
 * over the convex hull of the y_i, with t linear on it. The interior stationary point has a
 * closed form; if it is not feasible the minimum lies on the boundary (edges, then vertices).
 *
 * Jacobi form of FIM (Jeong & Whitaker 2008; Fu, Kirby & Whitaker 2013): every pass computes
 * all updates from the current t before applying them, so the update phases parallelise with
 * OpenMP without races and the result does not depend on the number of threads.
 */
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#define BEAT_EIKONAL_FAR 1e300

static inline double mdot(const double *M, const double *a, const double *b)
{
    return a[0] * (M[0] * b[0] + M[3] * b[1] + M[4] * b[2]) +
           a[1] * (M[3] * b[0] + M[1] * b[1] + M[5] * b[2]) +
           a[2] * (M[4] * b[0] + M[5] * b[1] + M[2] * b[2]);
}

/* Minimum over the segment [y0, y1]; e_i = y_i - x */
static inline double edge_update(const double *M, const double *e0, const double *e1, double t0,
                                 double t1)
{
    double c0 = mdot(M, e0, e0);
    double best = fmin(t0 + sqrt(c0), t1 + sqrt(mdot(M, e1, e1)));
    double E[3] = {e1[0] - e0[0], e1[1] - e0[1], e1[2] - e0[2]};
    double G = mdot(M, E, E), g = mdot(M, E, e0);
    double dt = t1 - t0;
    double a = dt * dt / G;
    if (a < 1.0) {
        double r = c0 - g * g / G;
        if (r < 0.0)
            r = 0.0;
        double s = sqrt(r / (1.0 - a));
        double mu = -(g + s * dt) / G;
        if (mu > 0.0 && mu < 1.0) {
            double v = t0 + mu * dt + s;
            if (v < best)
                best = v;
        }
    }
    return best;
}

/* Minimum over the triangle [y0, y1, y2] */
static inline double face_update(const double *M, const double *e0, const double *e1,
                                 const double *e2, double t0, double t1, double t2)
{
    double A[3] = {e1[0] - e0[0], e1[1] - e0[1], e1[2] - e0[2]};
    double B[3] = {e2[0] - e0[0], e2[1] - e0[1], e2[2] - e0[2]};
    double G11 = mdot(M, A, A), G12 = mdot(M, A, B), G22 = mdot(M, B, B);
    double g1 = mdot(M, A, e0), g2 = mdot(M, B, e0), c = mdot(M, e0, e0);
    double d1 = t1 - t0, d2 = t2 - t0;
    double det = G11 * G22 - G12 * G12;
    double I11 = G22 / det, I12 = -G12 / det, I22 = G11 / det;
    double a = d1 * (I11 * d1 + I12 * d2) + d2 * (I12 * d1 + I22 * d2);
    if (a < 1.0) {
        double gg = g1 * (I11 * g1 + I12 * g2) + g2 * (I12 * g1 + I22 * g2);
        double r = c - gg;
        if (r < 0.0)
            r = 0.0;
        double s = sqrt(r / (1.0 - a));
        double h1 = g1 + s * d1, h2 = g2 + s * d2;
        double m1 = -(I11 * h1 + I12 * h2), m2 = -(I12 * h1 + I22 * h2);
        if (m1 >= 0.0 && m2 >= 0.0 && m1 + m2 <= 1.0)
            return t0 + m1 * d1 + m2 * d2 + s;
    }
    double b = edge_update(M, e0, e1, t0, t1);
    b = fmin(b, edge_update(M, e0, e2, t0, t2));
    return fmin(b, edge_update(M, e1, e2, t1, t2));
}

/* Minimum over all cells of v of the local update */
static double vertex_update(int32_t v, int32_t cell_size, const double *x, const int32_t *cells,
                            const double *metric, const int32_t *v2c_ptr, const int32_t *v2c,
                            const double *t)
{
    double best = BEAT_EIKONAL_FAR;
    const double *xv = x + 3 * v;
    for (int32_t k = v2c_ptr[v]; k < v2c_ptr[v + 1]; ++k) {
        int32_t c = v2c[k];
        const int32_t *cv = cells + cell_size * c;
        const double *M = metric + 6 * c;
        double e[3][3], tt[3];
        int n = 0;
        for (int j = 0; j < cell_size; ++j) {
            int32_t w = cv[j];
            if (w == v || t[w] >= BEAT_EIKONAL_FAR)
                continue;
            e[n][0] = x[3 * w] - xv[0];
            e[n][1] = x[3 * w + 1] - xv[1];
            e[n][2] = x[3 * w + 2] - xv[2];
            tt[n] = t[w];
            ++n;
        }
        double u;
        if (n == 3)
            u = face_update(M, e[0], e[1], e[2], tt[0], tt[1], tt[2]);
        else if (n == 2)
            u = edge_update(M, e[0], e[1], tt[0], tt[1]);
        else if (n == 1)
            u = tt[0] + sqrt(mdot(M, e[0], e[0]));
        else
            continue;
        if (u < best)
            best = u;
    }
    return best;
}

int beat_eikonal_max_threads(void)
{
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

/* Run FIM to convergence. The non-fixed neighbours of `seeds` form the first active list.
 * `t` is updated in place (values only ever decrease). Returns 0 when the active list
 * empties, 1 if `max_iter` passes were not enough, 2 if memory ran out.
 * stats[0] += passes, stats[1] += vertex updates. */
int beat_eikonal_fim(int32_t nv, int32_t cell_size, const double *x, const int32_t *cells,
                     const double *metric, const int32_t *v2c_ptr, const int32_t *v2c,
                     const int32_t *nb_ptr, const int32_t *nb, const uint8_t *fixed, double *t,
                     const int32_t *seeds, int32_t nseeds, double tol, int64_t max_iter,
                     int32_t num_threads, int64_t *stats)
{
#ifdef _OPENMP
    if (num_threads > 0)
        omp_set_num_threads(num_threads);
#else
    (void)num_threads;
#endif
    size_t n = nv > 0 ? (size_t)nv : 1;
    int32_t *L = malloc(sizeof(int32_t) * n), *Ln = malloc(sizeof(int32_t) * n);
    int32_t *C = malloc(sizeof(int32_t) * n);
    double *q = malloc(sizeof(double) * n), *qc = malloc(sizeof(double) * n);
    uint8_t *inL = calloc(n, 1), *conv = malloc(n), *cand = calloc(n, 1);
    int status = 0;
    if (!L || !Ln || !C || !q || !qc || !inL || !conv || !cand) {
        status = 2;
        goto done;
    }
    int32_t nL = 0;
    for (int32_t i = 0; i < nseeds; ++i) {
        int32_t s = seeds[i];
        for (int32_t k = nb_ptr[s]; k < nb_ptr[s + 1]; ++k) {
            int32_t w = nb[k];
            if (!fixed[w] && !inL[w]) {
                inL[w] = 1;
                L[nL++] = w;
            }
        }
    }
    int64_t it = 0, nupd = 0;
    while (nL > 0 && it < max_iter) {
        ++it;
        /* (1) updates of the active vertices from the current t */
#pragma omp parallel for schedule(static)
        for (int32_t i = 0; i < nL; ++i)
            q[i] = vertex_update(L[i], cell_size, x, cells, metric, v2c_ptr, v2c, t);
        /* (2) apply them; converged when the value dropped by less than tol */
#pragma omp parallel for schedule(static)
        for (int32_t i = 0; i < nL; ++i) {
            int32_t v = L[i];
            double p = t[v];
            double qi = q[i] < p ? q[i] : p;
            t[v] = qi;
            conv[i] = (p - qi) < tol;
        }
        nupd += nL;
        for (int32_t i = 0; i < nL; ++i)
            if (conv[i])
                inL[L[i]] = 0;
        /* (3) candidates: de-duplicated, non-active neighbours of converged vertices */
        int32_t nc = 0;
        for (int32_t i = 0; i < nL; ++i) {
            if (!conv[i])
                continue;
            int32_t v = L[i];
            for (int32_t k = nb_ptr[v]; k < nb_ptr[v + 1]; ++k) {
                int32_t w = nb[k];
                if (fixed[w] || inL[w] || cand[w])
                    continue;
                cand[w] = 1;
                C[nc++] = w;
            }
        }
#pragma omp parallel for schedule(static)
        for (int32_t j = 0; j < nc; ++j)
            qc[j] = vertex_update(C[j], cell_size, x, cells, metric, v2c_ptr, v2c, t);
        nupd += nc;
        /* (4) next list: the unconverged plus candidates that improve by more than tol */
        int32_t nn = 0;
        for (int32_t i = 0; i < nL; ++i)
            if (!conv[i])
                Ln[nn++] = L[i];
        for (int32_t j = 0; j < nc; ++j) {
            int32_t w = C[j];
            cand[w] = 0;
            if (qc[j] < t[w] - tol) {
                t[w] = qc[j];
                inL[w] = 1;
                Ln[nn++] = w;
            }
        }
        int32_t *tmp = L;
        L = Ln;
        Ln = tmp;
        nL = nn;
    }
    stats[0] += it;
    stats[1] += nupd;
    status = nL == 0 ? 0 : 1;
done:
    free(L);
    free(Ln);
    free(C);
    free(q);
    free(qc);
    free(inL);
    free(conv);
    free(cand);
    return status;
}

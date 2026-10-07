/* Point loop that beat.c_backend appends to the C code that gotranx generates for one cell.
 *
 * BEAT_SCHEME is defined to the per-cell scheme, which has the gotranx signature
 *
 *     void scheme(const double *states, const double t, const double dt,
 *                 const double *parameters, double *values)
 *
 * states and out are (n_states, n_points) row-major arrays: state j of point i is at
 * j * n_points + i. params is either one vector shared by all points (params_stride == 0) or a
 * (n_params, n_points) array (params_stride == n_points). out may be the same array as states.
 * num_threads is only used when compiled with -DBEAT_OPENMP; 0 means the OpenMP default.
 */
#include <stdlib.h>

#ifdef BEAT_OPENMP
#include <omp.h>
#endif

#ifndef BEAT_SCHEME
#error "BEAT_SCHEME must be defined to the name of the per-cell scheme"
#endif

#define BEAT_STACK_SIZE 256

void beat_scheme_vec(int n_states, long n_points, const double *states, double t, double dt,
                     const double *params, int n_params, long params_stride, double *out,
                     int num_threads)
{
#ifdef BEAT_OPENMP
    int threads = num_threads > 0 ? num_threads : omp_get_max_threads();
#pragma omp parallel num_threads(threads)
#else
    (void)num_threads;
#endif
    {
        double stack_buffer[3 * BEAT_STACK_SIZE];
        double *heap_buffer = NULL;
        double *buffer = stack_buffer;
        if (n_states > BEAT_STACK_SIZE || n_params > BEAT_STACK_SIZE) {
            heap_buffer = malloc(sizeof(double) * (2 * (size_t)n_states + (size_t)n_params));
            buffer = heap_buffer;
        }
        double *x = buffer;
        double *y = buffer + n_states;
        double *p = buffer + 2 * n_states;

#ifdef BEAT_OPENMP
        /* Dynamic, since adaptive schemes make some points much more expensive than others */
#pragma omp for schedule(dynamic, 256)
#endif
        for (long i = 0; i < n_points; i++) {
            for (int j = 0; j < n_states; j++) {
                x[j] = states[j * n_points + i];
            }
            const double *point_params = params;
            if (params_stride > 0) {
                for (int k = 0; k < n_params; k++) {
                    p[k] = params[k * params_stride + i];
                }
                point_params = p;
            }
            BEAT_SCHEME(x, t, dt, point_params, y);
            for (int j = 0; j < n_states; j++) {
                out[j * n_points + i] = y[j];
            }
        }
        free(heap_buffer);
    }
}

/* The number of threads that beat_scheme_vec uses for the given num_threads argument */
int beat_max_threads(int num_threads)
{
#ifdef BEAT_OPENMP
    return num_threads > 0 ? num_threads : omp_get_max_threads();
#else
    (void)num_threads;
    return 1;
#endif
}

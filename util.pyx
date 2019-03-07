import numpy as np
cimport numpy as np
cimport cython


# Finite-difference method from http://www.cosy.sbg.ac.at/events/parnum05/book/horak1.pdf
# Parallel Numerical Solution of 2-D Heat Equation, Verena Horak & Peter Gruber (Parallel Numerics ’05, 47-56)

@cython.boundscheck(False)
def diffuse(initial_condition, int nstep, double c = 1.0, double delta_t_factor = 0.5):
    """Evolve `initial_condition` according to the 2D heat (diffusion) equation under periodic boundary conditions."""

    # Short circut
    if nstep == 0:
        return initial_condition
    
    mat_np = initial_condition.astype(np.float, copy = True)
    mat_new_np = np.empty(shape = initial_condition.shape, dtype = np.float)

    cdef double [:, :] mat = mat_np
    cdef double [:, :] mat_new = mat_new_np

    assert initial_condition.shape[0] == initial_condition.shape[1]
    cdef Py_ssize_t side_length = initial_condition.shape[0]

    cdef double delta_s = 1.0 / (side_length + 1)
    cdef delta_t = delta_t_factor * ((delta_s) ** 2 / (2 * c))

    cdef double multiplier = c * (delta_t / (delta_s * delta_s))

    cdef Py_ssize_t i = 0
    cdef Py_ssize_t j = 0
    cdef Py_ssize_t tstep = 0

    with nogil:
        for tstep in range(nstep):
            for i in range(side_length):
                for j in range(side_length):
                    mat_new[i, j] = mat[i, j] + multiplier * \
                                        (
                                            mat[(i + 1) % side_length, j] + \
                                            mat[(i - 1) % side_length, j] - \
                                            4 * mat[i, j] + \
                                            mat[i, (j + 1) % side_length] + \
                                            mat[i, (j - 1) % side_length]
                                        )
            mat[:] = mat_new

    # Sanity Check
    assert np.sum(mat_new_np) - np.sum(initial_condition) < 0.0000001
    assert np.max(mat_new_np) <= np.max(initial_condition)
    assert np.min(mat_new_np) >= np.min(initial_condition)
    assert not np.any(np.isnan(mat_new_np))

    return mat_new_np

# cython: language_level=3

import numpy as np
cimport numpy as np
cimport cython

from libc.math cimport fabs, round

#@cython.boundscheck(False)

def sads_familiarity(chem_weight = 0.0):
    def sads_familiarity_internal(scenes):
        assert 0 <= chem_weight <= 1
        maxfam = scenes[0].shape[0] *  scenes[0].shape[1]
        def func(scene, fambuf):
            sads_hsv_metric(
                scenes,
                scene,
                fambuf,
                chem_weight
            )

        func.max_familiarity = maxfam

        return func
    return sads_familiarity_internal


cdef void sads_hsv_metric(const np.uint8_t [:, :, :, :] familiar_scenes,
                          const np.uint8_t [:, :, :] scene,
                          np.float_t [:] fambuf,
                          const double chem_weight):

    cdef np.float_t diff = 0.0
    cdef np.float_t thispx = 0.0

    cdef Py_ssize_t fam_idex, i, j
    cdef Py_ssize_t xdim = scene.shape[0]
    cdef Py_ssize_t ydim = scene.shape[1]
    cdef np.float_t maxfam = xdim * ydim

    for fam_idex in range(len(familiar_scenes)):
        diff = 0.0
        for i in range(xdim):
            for j in range(ydim):
                if scene[i, j, 0] == familiar_scenes[fam_idex, i, j, 0]:
                    # Same chemical, difference is concentration difference:
                    thispx = abs(scene[i, j, 1] - familiar_scenes[fam_idex, i, j, 1])
                    # At most 255
                else:
                    # Since they are different chemicals, all stimulus increases
                    # the difference between the two scenes, so we sum their
                    # concentrations:
                    thispx = scene[i, j, 1] + familiar_scenes[fam_idex, i, j, 1]
                    # At most 2*255
                # Since this can be twice as large as usual,
                thispx *= 0.5
                # Thus when the chemical is the same, the difference can be
                # at most 127, half of the maximum. But when the chemicals are
                # different, it can go up to a full 255.
                # Thus,
                #   same chem + same concentration: +0 to difference
                #   same chem + diff concentration: +0.5*diff
                #   diff chem + zero concentration: +0 diff
                # Weight
                thispx *= chem_weight
                thispx += (1 - chem_weight) * abs(scene[i, j, 2] - familiar_scenes[fam_idex, i, j, 2]) # SADS for V
                # Normalize
                thispx /= 255.
                diff += thispx
        fambuf[fam_idex] = maxfam - diff


def set_HS_where_equal(const np.int_t [:, :] labels,
                       np.uint8_t [:, :, :] image,
                       const np.uint8_t [:] H,
                       const np.uint8_t [:] S):
    cdef Py_ssize_t i, j
    cdef np.int_t label
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            label =  labels[i, j]
            if label > 0: # 0 is no label
                image[i, j, 0] = H[label - 1]
                image[i, j, 1] = S[label - 1]


def downscale_chem(const np.uint8_t [:, :, :] image,
                   const Py_ssize_t factor_rows,
                   const Py_ssize_t factor_cols):
    cdef np.ndarray concentrations_np = np.empty(shape = 256, dtype = np.int)
    cdef np.int_t [:] concentrations = concentrations_np
    cdef double avg_value = 0

    out_size = (int(image.shape[0] // factor_rows), int(image.shape[1] // factor_cols), image.shape[2])
    out_np = np.empty(shape = out_size, dtype = np.uint8)
    cdef np.uint8_t [:, :, :] out = out_np

    cdef Py_ssize_t n_row_blocks = out_size[0]
    cdef Py_ssize_t n_col_blocks = out_size[1]
    cdef Py_ssize_t block_i, block_j, i, j

    for block_i in range(n_row_blocks):
        for block_j in range(n_col_blocks):
            concentrations[:] = 0
            avg_value = 0
            for i in range(factor_rows):
                for j in range(factor_cols):
                    concentrations[
                        image[block_i*factor_rows + i, block_j*factor_cols + j, 0]
                    ] += image[block_i*factor_rows + i, block_j*factor_cols + j, 1]
                    avg_value += image[block_i*factor_rows + i, block_j*factor_cols + j, 2]
            avg_value /= factor_rows*factor_cols
            avg_value = round(avg_value)
            out[block_i, block_j, 2] = <np.uint8_t>avg_value
            out[block_i, block_j, 0] = np.argmax(concentrations_np)
            out[block_i, block_j, 1] = <np.uint8_t>round(concentrations[out[block_i, block_j, 0]] / factor_rows*factor_cols)

    return out_np


@cython.boundscheck(False)
def ssds(a_np, b_np):
    cdef double [:, :] a = a_np
    cdef double [:, :] b = b_np

    cdef double diff = 0.0

    cdef Py_ssize_t i, j

    for i in range(a_np.shape[0]):
        for j in range(a_np.shape[1]):
            diff += (a[i, j] - b[i, j])**2

    return diff

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

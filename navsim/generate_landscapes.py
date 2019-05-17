import random
import numpy as np
import matplotlib.pyplot as plt

from navsim.util import diffuse


def random_squares(shape, s, n, value = 1):
    """Generate a matrix with a white background with black squares superimposed at random locations."""

    assert s % 2== 0, "Side length must be even"
    mat = np.zeros(shape = shape, dtype = np.int)
    s2 = s // 2
    for i in range(n):
        x, y = random.randrange(0, shape[0]), random.randrange(0, shape[1])
        mat[x - s2:x + s2, y - s2:y + s2] = value
    return mat

def random_matrix_bw_balance(shape, proportion = 0.5, threshold = 0.06, max_iter = 100, func = random_squares, **kwargs):
    """Using another generator, generate a matrix with a desired balance between black and white pixels."""
    total_pixels = shape[0] * shape[1]

    assert 0 < threshold < 1
    assert 0 < proportion < 1

    for _ in range(max_iter):
        mat = func(shape, **kwargs)

        mat_proportion = np.sum(mat) / total_pixels

        if (mat_proportion > proportion - threshold) and (mat_proportion < proportion + threshold):
            return mat

    raise RuntimeError("Couldn't generate a matrix within the desired range")

def checkerboard(shape, checkersize):
    out = np.zeros(shape = (shape, shape))
    for i in range(checkersize):
        for j in range(checkersize):
            out[i::checkersize*2, j::checkersize*2] = 1.0
            out[(i + checkersize)::checkersize*2, (j + checkersize)::checkersize*2] = 1.0
    return out

def image_from_prob_mat(prob_mat):
    """Generate a BW image from a probability matrix."""
    rand = np.random.random(size = prob_mat.shape)
    out = np.zeros(shape = prob_mat.shape)
    out[rand < prob_mat] = 1

    return out

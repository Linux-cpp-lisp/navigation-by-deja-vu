import random
import numpy as np
import matplotlib.pyplot as plt

from navsim.util import diffuse

import skimage

def random_squares(shape, s, n, value = 1):
    """Generate a matrix with a white background with black squares superimposed at random locations."""

    assert s % 2== 0, "Side length must be even"
    mat = np.zeros(shape = shape, dtype = np.int)
    s2 = s // 2
    for i in range(n):
        x, y = random.randrange(0, shape[0]), random.randrange(0, shape[1])
        mat[x - s2:x + s2, y - s2:y + s2] = value
    return mat

def random_squares_rot(shape, s, n):
    """Generate a matrix with a white background with black squares superimposed at random locations."""

    square = np.empty(shape = (s, s), dtype = np.float)
    square.fill(1)

    squares = [skimage.transform.rotate(square, ang, resize = True).astype(np.int) for ang in np.linspace(0., 360., 120)]

    mat = np.zeros(shape = tuple(e + 4 * s for e in shape), dtype = np.int)
    for i in range(n):
        x, y = random.randrange(2 * s, shape[0] + 2 * s), random.randrange(2 * s, shape[1] + 2 * s)
        sq = random.choice(squares)
        mat[x - (sq.shape[0] // 2):x + (sq.shape[0] - (sq.shape[0] // 2)),
            y - (sq.shape[1] // 2):y + (sq.shape[1] - (sq.shape[1] // 2))] += sq

    mat[mat >= 1] = 1

    out = mat[2 * s:-(2 * s), 2 * s:-(2 * s)]
    assert out.shape == shape
    return out

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

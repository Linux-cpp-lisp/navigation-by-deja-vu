import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import skimage
import itertools
import math
import sys

from NavBySceneFamiliarity import NavBySceneFamiliarity

landscape = np.load("../../data/landscape-diffuse-150.npy")

#landscape = np.zeros((500, 500), dtype = np.float32)
#landscape[0::100, :] = 1
#landscape[1::100, :] = 1
#landscape[2::100, :] = 1

nsf = NavBySceneFamiliarity(landscape, (40, 2), 2.0, sensor_pixel_dimensions = [2, 4], n_test_angles = 180, n_sensor_levels = 6,
                            max_distance_to_training_path = 12)

def sin_training_path(curveness, start_x, l, arclen = 2.0):
    # Assume derivative never goes above 4
    x = np.linspace(start_x, start_x + l, 4 * np.floor(l / arclen))
    y = x - 0.5 * l * curveness * np.sin((x - 0.5 * l - start_x) * np.pi / (0.5 * l))
    dists = np.sqrt((x[1:] - x[:-1])**2 + (y[1:] - y[:-1])**2)

    i = np.searchsorted(np.cumsum(dists), arclen * np.arange(np.floor(np.sum(dists) / arclen)))

    path = np.vstack((x[i], y[i])).T

    return path

path = sin_training_path(0.0, 100, 300, arclen = 1.)
print(len(path))
#path = np.vstack((x, np.full(len(x), 185.))).T

nsf.train_from_path(path)

nsf.position = path[0] + [3.0, -2.0]
nsf.angle = 0.

#fig, anim = nsf.animate(frames = 20)
#anim.save("test-curveness0-1.mp4")
#plt.show(fig)

fig = nsf.quiver_plot(n_box = 10, max_distance = 20.0)
plt.show(fig)

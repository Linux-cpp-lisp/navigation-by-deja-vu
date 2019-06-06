import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import skimage
import itertools
import math
import sys

from navsim import NavBySceneFamiliarity

landscape = np.load("../../data/test_set/landscape-diffuse-450.npy")

#landscape = np.zeros((500, 500), dtype = np.float32)
#landscape[0::100, :] = 1
#landscape[1::100, :] = 1
#landscape[2::100, :] = 1

nsf = NavBySceneFamiliarity(landscape, (40, 2), 2.0, sensor_pixel_dimensions = [2, 4], n_test_angles = 20, n_sensor_levels = 9,
                            max_distance_to_training_path = 80, saccade_degrees = 60, sensor_real_area = (14., "$\mathrm{mm}$"))

def sin_training_path(curveness, start_x, l, arclen = 2.0):
    # Assume derivative never goes above 4
    x = np.linspace(start_x, start_x + l, 4 * int(np.floor(l / arclen)))
    y = x - 0.5 * l * curveness * np.sin((x - 0.5 * l - start_x) * np.pi / (0.5 * l))
    dists = np.sqrt((x[1:] - x[:-1])**2 + (y[1:] - y[:-1])**2)

    i = np.searchsorted(np.cumsum(dists), arclen * np.arange(np.floor(np.sum(dists) / arclen)))

    path = np.vstack((x[i], y[i])).T

    return path

path = sin_training_path(0.5, 100, 300, arclen = 1.)
print(len(path))
#path = np.vstack((x, np.full(len(x), 185.))).T

nsf.train_from_path(path)

nsf.position = path[1] + [10.0, 9.0]
d = path[2] - path[1]
nsf.angle = np.arctan2(d[1], d[0])

fig, anim = nsf.animate(frames = 250)
#anim.save("sacadde-60deg-4levels.mp4")
plt.show(fig)

#fig = nsf.quiver_plot(n_box = 40, max_distance = 20.0, n_test_angles = 24)
#plt.show(fig)

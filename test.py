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

nsf = NavBySceneFamiliarity(landscape, (30, 20), 2.0, sensor_pixel_dimensions = [2, 4], n_test_angles = 180, n_sensor_levels = 6,
                            max_distance_to_training_path = 12)

x = np.linspace(50, 450, 300)
l = 400
path = np.vstack((x, x - 0.5 * l * 0.05 * np.sin((x - 0.5 * l - np.min(x)) * np.pi / (0.5 * l)))).T
#path = np.vstack((x, np.full(len(x), 185.))).T

nsf.train_from_path(path)

nsf.position = path[150] + [1.0, -1.0]
nsf.angle = 0.

fig, anim = nsf.animate(frames = 200)
anim.save("test-curveness0-1.mp4")

fig = nsf.quiver_plot(n_box = 8)

plt.show(fig)

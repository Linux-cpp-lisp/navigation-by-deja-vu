import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import skimage
import itertools
import math

from NavBySceneFamiliarity import NavBySceneFamiliarity

landscape = np.load("../../notebooks/blobbyness/a-landscape.npy").astype(np.float32)

print(np.max(landscape))

nsf = NavBySceneFamiliarity(landscape, (2, 40), 4.0, sensor_pixel_dimensions = [4, 2], n_test_angles = 160, n_sensor_levels = 4)
path = np.vstack((np.linspace(50, 130, 100), np.linspace(50, 130, 100))).T

nsf.train_from_path(path)

#plt.matshow(nsf._mask)

fig, anim = nsf.animate(path[0], 0)

#anim.save("test2.mp4")

plt.show(fig)

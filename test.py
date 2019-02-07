import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import skimage
import itertools
import math

from NavBySceneFamiliarity import NavBySceneFamiliarity

landscape = np.load("../../notebooks/blobbyness/a-landscape.npy").astype(np.float32)

print(np.max(landscape))

nsf = NavBySceneFamiliarity(landscape, 12, 2.0, pixel_scale_factor = 2, n_test_angles = 160)
path = np.vstack((np.linspace(30, 170, 100), np.linspace(30, 130, 100))).T

nsf.train_from_path(path)

plt.matshow(nsf._mask)

fig, anim = nsf.animate(path[0], 0)

#anim.save("test2.mp4")

plt.show(fig)

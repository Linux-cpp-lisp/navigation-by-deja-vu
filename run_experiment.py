import numpy as np
import matplotlib.pyplot as plt

import itertools
import math

import logging
logging.basicConfig()
logger = logging.getLogger('experiments')
logger.setLevel(logging.INFO)

from NavBySceneFamiliarity import NavBySceneFamiliarity

import sys, os, glob, time

from mpi4py import MPI

# Run as:
# run_experiment.py landscape_dir/ output_dir/

landscape_dir = sys.argv[1]
output_dir = sys.argv[2]

def sin_training_path(curveness, start_x, l, arclen = 2.0):
    # Assume derivative never goes above 4
    x = np.linspace(start_x, start_x + l, 4 * int(np.floor(l / arclen)))
    y = x - 0.5 * l * curveness * np.sin((x - 0.5 * l - start_x) * np.pi / (0.5 * l))
    dists = np.sqrt((x[1:] - x[:-1])**2 + (y[1:] - y[:-1])**2)

    i = np.searchsorted(np.cumsum(dists), arclen * np.arange(np.floor(np.sum(dists) / arclen)))

    path = np.vstack((x[i], y[i])).T

    return path


def run_experiment(id_str, landscape, training_path,
                   sensor_dimensions,
                   step_size,
                   n_quiver_box = 20,
                   **kwargs):

    nsf = NavBySceneFamiliarity(landscape, sensor_dimensions, step_size, **kwargs)
    nsf.train_from_path(training_path)

    nsf.position = training_path[1]
    nsf.angle = 0.

    fig, anim = nsf.animate(frames = 150)
    anim.save("nav-%s.mp4" % id_str)

    quiv = nsf.quiver_plot(n_box = 20)
    quiv.savefig("quiver-%s.png" % id_str)

    return nsf.navigation_error


sensor_dim = (40, 2)
sensor_pixel_dimensions = [2, 4]
step_size = 2.0

landscapes = [os.path.abspath(p) for p in glob.glob(landscape_dir + "/*.npy")]
n_paths = 2

#curvenesses = np.linspace(0., 1., n_paths)
curveness = [1.]

comm = MPI.COMM_WORLD

os.chdir(output_dir)

if comm.rank == 0:
    logger.info("Starting...")
    start_time = time.time()

assert len(landscapes) >= comm.size

part_landscapes = np.array_split(landscapes, comm.size)
my_landscapes = part_landscapes[comm.rank]
my_errs = np.empty(shape = (len(my_landscapes), n_paths))
my_diff_times = np.empty(shape = len(my_landscapes), dtype = np.int)

for i, landscape_file in enumerate(my_landscapes):
    landscape = np.load(landscape_file)
    diffuse_time = landscape_file[:-4].split("-")[-1]
    my_diff_times[i] = diffuse_time
    margin = 1.5 * np.max(np.multiply(sensor_dim, sensor_pixel_dimensions)) // 2

    for j, curveness in enumerate(curvenesses):
        tpath = sin_training_path(curveness, margin, np.min(landscape.shape) - 2 * margin, arclen = step_size * 0.5)
        id_str = "diffuse-%s-curve-%0.2f" % (diffuse_time, curveness)
        logger.info("Task %i running experiment '%s'" % (comm.rank, id_str))
        my_errs[i, j] = run_experiment(id_str,
                                       landscape,
                                       tpath,
                                       sensor_dim,
                                       step_size,
                                       sensor_pixel_dimensions = sensor_pixel_dimensions,
                                       n_test_angles = 180,
                                       n_sensor_levels = 6,
                                       max_distance_to_training_path = 12)


gathered = comm.gather((my_diff_times, my_errs), root = 0)

if comm.rank == 0:
    err_surface = np.vstack(tuple(e[1] for e in gathered))
    diffuse_times = np.concatenate(tuple(e[0] for e in gathered))
    logger.debug(err_surface.shape)
    np.savez("navigation_errors.npz", errors=err_surface, diffuse_times=diffuse_times)
    logger.info("Done!")
    logger.info("It took about %i seconds" % int(time.time() - start_time))

import numpy as np
import scipy.io
import matplotlib.pyplot as plt

import itertools
import math
import shutil

# -------- GLOBAL SETTINGS --------

OUTFILE_NAME = 'output.npz'

FRAME_FACTOR = 1.2
TRAINING_SCENE_ARCLEN_FACTOR = 0.5
QUIVER_DIST_FACTOR = 10
N_QUIVER_BOX = 40

# --------- EXPERIMENTAL VARIABLES ----
# ---- TEST VARS ----
# variable_dict = {
#     'landscape_class' : ["test_set"], # At this point, just checkerboard = 1
#     'landscape_diffuse_time' : [450], #
#     'training_path_curve' : [1.0],
#     'saccade_degrees' : [90.],
#     'n_sensor_levels' : [4, 8],
#     'sensor_dimensions' : [(40, 1)],
#     'sensor_pixel_dimensions' : [(2, 4)],
#     'start_offset' : [np.array([3.5, 3.5])]
# }

# --- REAL VARS ---
variable_dict = {
    'landscape_class' : ["irreg"], #Don't do irreg yet # At this point, just checkerboard = 1
    'landscape_diffuse_time' : [0, 125, 450, 750, 1400], # These will be different and higher
    'training_path_curve' : [0.0, 0.5, 1.0],
    'sensor_dimensions' : [(40, 4, 2, 2),
                           (40, 2, 2, 4),
                           (40, 1, 2, 8),
                           (20, 1, 4, 8),
                           (10, 1, 8, 8)],
    # These are chosen to hold total sensor area constant
    # want at least some blurring to avoid weird effects at the highest resultion,
    # so our sensor area (in px) will be 80x8 (corresponding to 14mm^2 real world)

    'n_sensor_levels' : [2, 4, 8, 16],
    'saccade_degrees' : [30., 60., 90.],
    # Sensor is 80px wide, so 1/16th width is 5px. We'll divide that a little
    # in each direction. (3.5 is the side length of a right isoceles with
    # a hypotenuse of ~5.)
    'start_offset' : [np.array([0., 0.]), np.array([3.5, 3.5])] # Units are px
}

defaults = {
    'angular_resolution' : 3, #degrees
    'max_distance_to_training_path' : 80,
    'sensor_real_area' : (14., "$\mathrm{mm}$"),
    'step_size' : 2.0,
}

short_names = {
    'training_path_curve' : ('crv', '{0:0.2f}'),
    'saccade_degrees' : ('sacc', '{0:0.0f}'),
    'n_sensor_levels' : ('grays', '{0}'),
    'sensor_dimensions' : ('sensor', '{0[0]}x{0[1]}at{0[2]}x{0[3]}'),
    'start_offset' : ('ofst', '{0[0]:0.0f}x{0[0]:0.0f}'),
    'landscape_class' : ('lclass', '{0}'),
    'landscape_diffuse_time' : ('diff', '{0}')
}

result_variables = {
    'path_coverage' : np.float,
    'rmsd_error' : np.float,
    'completed_frames' : np.int,
    'stop_status' : np.int
}

assert VARIABLE_DIRECTORY_NESTING_LEVELS < len(variable_dict)

# ---------------------- RUN STUFF ----------------------------------------


import logging
logging.basicConfig()
logger = logging.getLogger('experiments')
logger.setLevel(logging.INFO)

from navsim import NavBySceneFamiliarity, StopNavigationException

import sys, os, glob, time
import pickle

from mpi4py import MPI

# Run as:
# run_experiment.py landscape_dir/

landscape_dir = os.path.abspath(sys.argv[1])

# ------ FUNCTIONS ----------

def sin_training_path(curveness, start_x, l, arclen = 2.0):
    # Assume derivative never goes above 4
    x = np.linspace(start_x, start_x + l, 4 * int(np.floor(l / arclen)))
    y = x - 0.5 * l * curveness * np.sin((x - 0.5 * l - start_x) * np.pi / (0.5 * l))
    dists = np.sqrt((x[1:] - x[:-1])**2 + (y[1:] - y[:-1])**2)

    i = np.searchsorted(np.cumsum(dists), arclen * np.arange(np.floor(np.sum(dists) / arclen)))

    path = np.vstack((x[i], y[i])).T

    return path

loaded_landscapes = dict()
def make_nsf(params):
    trial = dict(defaults)
    trial.update(params)

    sres = trial.pop('sensor_dimensions')
    trial['sensor_dimensions'] = sres[0:2]
    trial['sensor_pixel_dimensions'] = sres[2:5]

    # - Load/create stuff
    # Memoize landscapes
    landscape_class = trial.pop("landscape_class")
    landscape_diff_time = trial.pop("landscape_diffuse_time")
    lkey = (landscape_class, landscape_diff_time)
    if lkey in loaded_landscapes:
        landscape = loaded_landscapes[lkey]
    else:
        landscape = np.load(landscape_dir + ("/%s/landscape-diffuse-%i.npy" % lkey))
        loaded_landscapes[lkey] = landscape

    trial['landscape'] = landscape
    # Training Path
    margin = 1.5 * np.max(np.multiply(trial['sensor_dimensions'], trial['sensor_pixel_dimensions'])) // 2
    tpath = sin_training_path(trial.pop('training_path_curve'),
                              margin,
                              np.min(landscape.shape) - 2 * margin,
                              arclen = trial['step_size'] * TRAINING_SCENE_ARCLEN_FACTOR)

    angular_resolution = trial.pop('angular_resolution')
    trial['n_test_angles'] = int(trial['saccade_degrees'] / angular_resolution)

    start_offset = trial.pop("start_offset")

    nsf = NavBySceneFamiliarity(**trial)
    nsf.train_from_path(tpath)
    nsf.position = tpath[1] + start_offset,
    nsf.angle = np.arctan2(*(list(tpath[2] - tpath[1])[::-1])) % (2 * np.pi) # y then x for arctan2

    return nsf


def run_experiment(nsf,
                   frames = None):
    if frames is None:
        frames = int(FRAME_FACTOR * TRAINING_SCENE_ARCLEN_FACTOR * len(nsf.training_path))

    my_status = None
    completed_frames = 0
    try:
        for _ in range(frames):
            nsf.step_forward()
            completed_frames += 1
    except StopNavigationException as e:
        my_status = e

    my_status = (0 if my_status is None else my_status.get_code())

    return {
        'path_coverage' : nsf.percent_recapitulated,
        'rmsd_error' : nsf.navigation_error,
        'completed_frames' : completed_frames,
        'stop_status' : my_status
    }


for k in variable_dict:
    variable_dict[k] = np.asarray(variable_dict[k])

n_variables = len(variable_dict)
variables = list(variable_dict.keys())
variables.sort() # get a consistant variable ordering. Just in case we're running on old python
trials = list(itertools.product(*[variable_dict[k] for k in variables]))
n_trials_by_variable = [len(variable_dict[k]) for k in variables]
n_trials = len(trials)

# Housekeeping

comm = MPI.COMM_WORLD

assert n_trials >= comm.size, "n_trials %i < comm size %i" % (n_trials, comm.size)

if comm.rank == 0:
    logger.info("Starting...")
    logger.info("Running a total of %i trials over %i processes" % (n_trials, comm.size))
    # Log start time
    start_time = time.time()

# Distribute Trials
part_trials = np.array_split(trials, comm.size)
my_trials = part_trials[comm.rank]

logger.debug("Task %i has %i trials" % (comm.rank, len(my_trials)))

# Set up result arrays
my_variable_values = [np.empty(
                        shape = (len(my_trials),) + variable_dict[v].shape[1:],
                        dtype = variable_dict[v].dtype
                      ) for v in variables]
my_results = {}
for resvar in result_variables:
    my_results[resvar] = np.empty(shape = len(my_trials), dtype = result_variables[resvar])

for i, trial_vals in enumerate(my_trials):
    # - Set up dict for convinience
    trial = dict(zip(variables, trial_vals))

    if i % 50 == 0:
        logger.info("Task %i running its trial %i/%i" % (comm.rank, i + 1, len(my_trials)))

    nsf = make_nsf(trial)
    trial_result = run_experiment(nsf)

    for vi, val in enumerate(trial_vals):
        my_variable_values[vi][i] = val

    for resvar in result_variables.keys():
        my_results[resvar][i] = trial_result[resvar]


gathered = comm.gather((my_variable_values, my_results), root = 0)

if comm.rank == 0:
    all_results = dict([(resvar, np.concatenate(tuple(e[1][resvar] for e in gathered))) for resvar in result_variables.keys()])
    all_vars = [np.concatenate(tuple(e[0][i] for e in gathered)) for i in range(len(variables))]

    for resarr in all_results.values():
        assert len(all_vars[0]) == len(resarr)

    to_save = dict(zip(variables, all_vars))
    to_save.update(all_results)

    np.savez("output.npz", **to_save)
    # Also save the same data in MATLAB format for convinience
    scipy.io.savemat("output.mat", to_save)

    logger.info("Done! Finished %i trials" % len(all_vars[0]))
    logger.info("It took about %i minutes" % int((time.time() - start_time) / 60.))

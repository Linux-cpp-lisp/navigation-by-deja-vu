import numpy as np
import scipy.io
import matplotlib.pyplot as plt

import itertools
import math
import shutil

from PIL import Image

# -------- GLOBAL SETTINGS --------

FRAME_FACTOR = 3.0
TRAINING_SCENE_ARCLEN_FACTOR = 1.0 # Set to 1:1
# Essentially, 2x training over the same path.
# IN the future, when multiple training paths are added, this will go up to
# 1.0 and I'll explicitly have trials with two overlapping training paths.
#TRAINING_SCENE_ARCLEN_FACTOR = 1.0

N_CONSECUTIVE_SCENES = 0.05 # 5% of training path length. Seems reasonable.

# --------- EXPERIMENTAL VARIABLES ----
# ---- TEST VARS ----
import os, sys
# Run as:
# run_experiment.py [test|syn|sand] landscape_dir/

defaults = {
    'angular_resolution' : 3, #degrees
    'max_distance_to_training_path' : 400, # Enough that not possible on 1000x1000 landscape to go out of bounds.
    'sensor_real_area' : (14., "$\mathrm{mm}$"),
}

result_variables = {
    'path_coverage' : np.float,
    'rmsd_error' : np.float,
    'completed_frames' : np.int,
    'stop_status' : np.int,
    'n_captures' : np.int,
    'percent_forgiving' : np.float
}

# ---------------------- RUN STUFF ----------------------------------------

import logging
logging.basicConfig()
logger = logging.getLogger('experiments')
logger.setLevel(logging.INFO)

from navsim import NavBySceneFamiliarity, StopNavigationException
from navsim.generate_landscapes import image_from_prob_mat

import  glob, time
import pickle

from mpi4py import MPI

# ------ FUNCTIONS ----------

def sin_training_path(curveness, start_x, l, arclen = 2.0):
    # Assume derivative never goes above 4
    x = np.linspace(start_x, start_x + l, 4 * int(np.floor(l / arclen)))
    y = x - 0.5 * l * curveness * np.sin((x - 0.5 * l - start_x) * np.pi / (0.5 * l))
    dists = np.sqrt((x[1:] - x[:-1])**2 + (y[1:] - y[:-1])**2)

    i = np.searchsorted(np.cumsum(dists), arclen * np.arange(np.floor(np.sum(dists) / arclen)))

    path = np.vstack((x[i], y[i])).T

    return path

def chop_path_to_len(path, length):
    """
    Chop a path given as a list of points down to a given length, taking
    points off the front and back alternatingly.
    """
    lens = np.linalg.norm(path[1:] - path[:-1], axis = 1)
    assert np.sum(lens) >= length
    start = 0
    end = len(path)
    for _ in range(len(path)):
        if np.sum(lens[start:end]) <= length:
            break
        start += 1
        if np.sum(lens[start:end]) <= length:
            break
        end -= 1
    assert np.sum(lens[start:end]) <= length
    return path[start:end]


loaded_landscapes = dict()
def make_nsf(params, landscape_dir):
    trial = dict(defaults)
    trial.update(params)

    sres = trial.pop('sensor_dimensions')
    trial['sensor_dimensions'] = sres[0:2]
    trial['sensor_pixel_dimensions'] = sres[2:5]

    sensor_pixel_depth = trial['sensor_dimensions'][1] * trial['sensor_pixel_dimensions'][1]
    sensor_pixel_width = trial['sensor_dimensions'][0] * trial['sensor_pixel_dimensions'][0]
    # Don't do this anymore
    # trial['step_size'] = trial['step_size'] * sensor_pixel_depth

    # Memoize landscapes
    landscape_class = trial.pop("landscape_class")
    landscape_name = trial.pop("landscape_name")
    lkey = (landscape_class, landscape_name)
    if lkey in loaded_landscapes:
        landscape = loaded_landscapes[lkey]
    else:
        landscape = np.asarray(Image.open(landscape_dir + ("/%s/%s" % lkey)))
        landscape = landscape / np.max(landscape) # Turn 256 into 0-1
        # Memoize
        loaded_landscapes[lkey] = landscape

    l_noise = trial.pop('landscape_noise_factor', 0)
    assert 0 <= l_noise <= 1
    if l_noise != 0:
        # # 0.5 constant is the probability for white noise.
        # # Treating landscape as a probability landscape.
        # probmat = (1 - l_noise)*landscape + (l_noise * 0.5)
        # # Draw fresh each trial to emphasize the spirit of noise, rather than
        # # doing a static pattern.
        # landscape = image_from_prob_mat(probmat)

        # For greyscale, just to weighted sum
        landscape = (1 - l_noise)*landscape + (l_noise * 0.5)

    trial['landscape'] = landscape

    # Training Path
    margin = 0.2 * np.min(landscape.shape) # Slightly larger margin for some play
    tpath = sin_training_path(trial.pop('training_path_curve'),
                              margin,
                              np.min(landscape.shape) - 2 * margin,
                              arclen = trial['step_size'] * TRAINING_SCENE_ARCLEN_FACTOR)
    margin = 0.25 * np.min(landscape.shape) # Crop to the real margin
    tpath = chop_path_to_len(tpath, np.sqrt(2 * (np.min(landscape.shape) - 2 * margin)**2))

    angular_resolution = trial.pop('angular_resolution')
    trial['n_test_angles'] = int(trial['saccade_degrees'] / angular_resolution)

    start_offset = trial.pop("start_offset")

    nsf = NavBySceneFamiliarity(**trial)
    nsf.train_from_path(tpath)

    nsf.angle = np.arctan2(*(list(tpath[2] - tpath[1])[::-1])) % (2 * np.pi) # y then x for arctan2
    nsf.angle += np.deg2rad(start_offset[1])
    offset = [
        np.cos(nsf.angle + 0.5*np.pi) * start_offset[0] * sensor_pixel_width,
        np.sin(nsf.angle + 0.5*np.pi) * start_offset[0] * sensor_pixel_width,
    ]
    nsf.position = tpath[1] + offset


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
        'stop_status' : my_status,
        'percent_forgiving' : nsf.percent_recapitulated_forgiving(n_consecutive_scenes = N_CONSECUTIVE_SCENES),
        'n_captures' : nsf.n_captures(n_consecutive_scenes = N_CONSECUTIVE_SCENES)
     }

if __name__ == '__main__':
    mode = sys.argv[1]
    landscape_dir = os.path.abspath(sys.argv[2])

    if mode == 'test':
        variable_dict = {
            'landscape_class' : ['sand2020'],
            'landscape_name' : os.listdir(landscape_dir + '/' + 'sand2020'),
            # 'landscape_noise_factor' : np.repeat([0.0, 0.25], 2),
            'training_path_curve' : [0.0, 0.5],
            'sensor_dimensions' : [(40, 1, 2, 8)],
            'n_sensor_levels' : [4],
            'saccade_degrees' : [60.,],
            'start_offset' : [(0., 0.)], # Units are px
            # Step size as a fraction of sensor depth
            'step_size' : [1.0]
        }
    elif mode == 'sand':
        # --- REAL VARS ---
        variable_dict = {
            'landscape_class' : ['sand2020'],
            'landscape_name' : os.listdir(landscape_dir + '/' + 'sand2020'),
            # 'landscape_noise_factor' : np.repeat([0.0, 0.25, 0.5, 0.75, 1.0], 3), # Run 3 trials at each noise factor, since noise is generated randomly each time.
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
            'saccade_degrees' : [10., 30., 60., 90.],
            # Fraction of sensor width and an angle offset
            'start_offset' : [(0., 0.), (0.1, 0.), (-0.25, 15), (0.5, -30)], # Units are (frac, deg)
            # Step size as a fraction of sensor depth
            'step_size' : [1.]
        }
    else:
        raise ValueError("Invalid mode '%s'" % mode)

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

        nsf = make_nsf(trial, landscape_dir)
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

        to_save['variable_dict'] = variable_dict

        datestring = time.strftime("%Y-%m-%d")
        np.savez("output-%s-%s.npz" % (mode, datestring), **to_save)
        # Also save the same data in MATLAB format for convinience
        scipy.io.savemat("output-%s-%s.mat" % (mode, datestring), to_save)

        logger.info("Done! Finished %i trials" % len(all_vars[0]))
        logger.info("It took about %i minutes" % int((time.time() - start_time) / 60.))
        logger.info("Each trial added about %.2fs of wall-clock time" % ((time.time() - start_time) / len(all_vars[0])))

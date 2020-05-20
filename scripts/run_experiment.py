import numpy as np
RNG = np.random.default_rng()

import scipy.io
import matplotlib.pyplot as plt

import itertools
import math
import shutil
from tqdm import tqdm

from PIL import Image
import skimage.measure
import skimage.filters.rank

# -------- GLOBAL SETTINGS --------

PX_PER_MM = 450 / 33

FRAME_FACTOR = 3.0
#TRAINING_SCENE_ARCLEN_FACTOR = 1/3 # Set to 1:1
# Essentially, 2x training over the same path.
# IN the future, when multiple training paths are added, this will go up to
# 1.0 and I'll explicitly have trials with two overlapping training paths.
#TRAINING_SCENE_ARCLEN_FACTOR = 1.0

N_CONSECUTIVE_SCENES = 0.05 # 5% of training path length. Seems reasonable.

LANDSCAPE_THRESHOLD = 200

# --------- EXPERIMENTAL VARIABLES ----
# ---- TEST VARS ----
import os, sys
# Run as:
# run_experiment.py [test|syn|sand] landscape_dir/

defaults = {
    'n_test_angles' : 10,
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

from navsim import NavBySceneFamiliarity, StopNavigationException, sads_familiarity
from navsim.util import set_HS_where_equal
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

def add_chemistry(landscape, grainlabels, grainprops,
                  min_grain_diameter = 2,
                  n_chemicals = 2,
                  concentration_range = (255, 256)):
    n_grains = len(grainprops)
    grainchems = RNG.integers(n_chemicals, size = n_grains, dtype = np.uint8) * (255 // n_chemicals) # Evenly space hue
    grainsats = RNG.integers(concentration_range[0], concentration_range[1], size = n_grains, dtype = np.uint8) # Saturation (concentration) in given range
    for grain in range(n_grains):
        if grainprops[grain].equivalent_diameter < min_grain_diameter:
            grainsats[grain] = 0

    set_HS_where_equal(
        grainlabels,
        landscape,
        grainchems,
        grainsats
    )


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
    min_chem_grain_diameter = trial.pop('min_chem_grain_diameter', 2)
    lkey = (landscape_class, landscape_name)
    if lkey in loaded_landscapes:
        landscape, grainlabels, grainprops = loaded_landscapes[lkey]
    else:
        landscape = np.asarray(Image.open(landscape_dir + ("/%s/%s" % lkey)).convert('HSV'))
        # Threshold on value
        for_labeling = (landscape[:, :, 2] >= LANDSCAPE_THRESHOLD).astype(np.uint8)
        neighborhood_width = min_chem_grain_diameter // 2
        if not neighborhood_width == 0:
            if neighborhood_width % 2 == 0:
                neighborhood_width -= 1
            neighborhood_width = int(neighborhood_width)
            for_labeling = skimage.filters.rank.modal(
                for_labeling,
                np.ones((neighborhood_width, neighborhood_width), dtype = np.uint8)
            )
        grainlabels = skimage.measure.label(for_labeling)
        grainprops = skimage.measure.regionprops(grainlabels)
        # Memoize
        loaded_landscapes[lkey] = (landscape, grainlabels, grainprops)

    n_chemicals = trial.pop("n_chemicals", 1)
    if n_chemicals >= 1:
        landscape = landscape.copy()
        add_chemistry(
            landscape, grainlabels, grainprops,
            min_grain_diameter = min_chem_grain_diameter,
            n_chemicals = n_chemicals,
            concentration_range = trial.pop("concentration_range", (127, 128))
        )

    trial['landscape'] = landscape

    ldims = landscape.shape[:2]

    # Training Path
    margin = 0.2 * np.min(ldims) # Slightly larger margin for some play
    tpath_arclen = trial['step_size'] / (trial['n_test_angles'])
    tpath = sin_training_path(trial.pop('training_path_curve'),
                              margin,
                              np.min(ldims) - 2 * margin,
                              arclen = tpath_arclen)
    margin = 0.25 * np.min(ldims) # Crop to the real margin
    tpath = chop_path_to_len(tpath, np.sqrt(2 * (np.min(ldims) - 2 * margin)**2))

    start_offset = trial.pop("start_offset")

    nsf = NavBySceneFamiliarity(
        familiarity_model = sads_familiarity(chem_weight = trial.pop('chem_weight', 0.)),
        **trial
    )
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
        frames = int(FRAME_FACTOR * nsf.training_path_length / nsf.step_size)

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
            # == Environmental properties ==
            'landscape_class' : ['sand2020'],
            'landscape_name' : ['landscape-0.png'],
            'training_path_curve' : [0.5],
            # 'landscape_noise_factor' : np.repeat([0.0, 0.25, 0.5, 0.75, 1.0], 3), # Run 3 trials at each noise factor, since noise is generated randomly each time.
            'n_chemicals' : [2],
            'min_chem_grain_diameter' : np.array([0.25]) * PX_PER_MM,
            'chem_weight' : [0., 0.5],
            # == Agent properties ==
            # One pectine width: 20x5 gives about 7.35mm wide
            # and 1x12 gives about 0.88mm depth.
            'sensor_dimensions' : [(44, 1, 6, 12)],
            'mask_middle_n' : [2],
            # These are chosen to hold total sensor area constant
            # want at least some blurring to avoid weird effects at the highest resultion,
            # so our sensor area (in px) will be 80x8 (corresponding to 14mm^2 real world)

            'n_sensor_levels' : [4,],

            # Step size/glimpse frequency properties
            'step_size' : [1. * PX_PER_MM], # In milimeters
            'saccade_degrees' : [35.], # Degrees, from data (35+-15)
            'n_test_angles' : [15], # All odd so that 0 included

            # == Other ==
            # Fraction of sensor width and an angle offset
            'start_offset' : [(0., 0.),], # Units are (frac, deg)
        }
    elif mode.startswith('sand'):
        # --- REAL VARS ---
        # landscape_name
        # os.listdir(landscape_dir + '/' + 'sand2020')
        if mode == 'sand0:
            landscape_names = ['landscape-0.png']
        elif mode == 'sand1':
            landscape_names = ['landscape-1.png']
        else:
            raise ValueError("Invalid Mode!")

        variable_dict = {
            # == Environmental properties ==
            'landscape_class' : ['sand2020'],
            'landscape_name' : landscape_names,
            'training_path_curve' : [0.0, 0.5, 1.0],
            # 'landscape_noise_factor' : np.repeat([0.0, 0.25, 0.5, 0.75, 1.0], 3), # Run 3 trials at each noise factor, since noise is generated randomly each time.
            'n_chemicals' : [0, 1, 2, 4],
            'min_chem_grain_diameter' : np.array([0.25]) * PX_PER_MM,
            'chem_weight' : [0., 0.25, 0.5, 0.75, 1.0],
            # == Agent properties ==
            # One pectine width: 20x5 gives about 7.35mm wide
            # and 1x12 gives about 0.88mm depth.
            'sensor_dimensions' : [(44, 1, 6, 12),
                                   (44, 2, 6, 6),
                                   (44, 4, 6, 3),
                                   (44, 6, 6, 2)],
            'mask_middle_n' : [2],
            # These are chosen to hold total sensor area constant
            # want at least some blurring to avoid weird effects at the highest resultion,
            # so our sensor area (in px) will be 80x8 (corresponding to 14mm^2 real world)

            'n_sensor_levels' : [2, 4, 8, 16],

            # Step size/glimpse frequency properties
            'step_size' : [1.3 * PX_PER_MM], # In milimeters
            'saccade_degrees' : [30., 40., 50.], # Degrees, from data (35+-15)
            'n_test_angles' : [15, 31], # All odd so that 0 included

            # == Other ==
            # Fraction of sensor width and an angle offset
            'start_offset' : [(0., 0.), (0.1, -5.), (-0.1, 7.), (-0.25, 15), (0.5, -30)], # Units are (frac, deg)
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
        # scipy.io.savemat("output-%s-%s.mat" % (mode, datestring), to_save)

        logger.info("Done! Finished %i trials" % len(all_vars[0]))
        logger.info("It took about %i minutes" % int((time.time() - start_time) / 60.))
        logger.info("Each trial added about %.2fs of wall-clock time" % ((time.time() - start_time) / len(all_vars[0])))

import numpy as np
RNG = np.random.default_rng()

import scipy.io
import matplotlib.pyplot as plt

import itertools
import math
import shutil
import json
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
# run_experiment.py path/to/trial_file.json landscape_dir/

defaults = {
    'n_test_angles' : 10,
    'max_distance_to_training_path' : 450, # Enough that not possible on 2000x2000 landscape to go out of bounds.
    'sensor_px_per_mm' : PX_PER_MM
}

float_formatstr = "{:6f}"
result_variables = {
    'path_coverage' : float_formatstr,
    'rmsd_error' : float_formatstr,
    'completed_frames' : "{:d}",
    'stop_status' : "{:d}",
    'n_captures' : "{:d}",
    'percent_forgiving' : float_formatstr
}

variable_formats = {
    'landscape_class' : "{}",
    'landscape_name' : "{}",
    'training_path_curve' : "{:4f}",
    'landscape_noise_factor' : "{:4f}",
    'n_chemicals' : "{:d}",
    'min_chem_grain_diameter' : "{:4f}",
    'chem_weight' : "{:4f}",
    'sensor_dimensions' : "{0[0]:d};{0[1]:d};{0[2]:d};{0[3]:d}",
    'mask_middle_n' : "{:d}",
    'n_sensor_levels' : "{:d}",
    'step_size' : "{:4f}",
    'saccade_degrees' : "{:4f}",
    'n_test_angles' : "{:d}",
    'start_offset' : "{0[0]:4f};{0[1]:4f}",
    'landscape_flip_vertical' : '{:d}', # Bools as ints
    'landscape_flip_horizontal' : '{:d}'
}

# ---------------------- RUN STUFF ----------------------------------------

import logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)
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

    # == Memoize landscapes
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

    # ==  Add Chemistry
    n_chemicals = trial.pop("n_chemicals", 1)
    if n_chemicals >= 1:
        landscape = landscape.copy()
        add_chemistry(
            landscape, grainlabels, grainprops,
            min_grain_diameter = min_chem_grain_diameter,
            n_chemicals = n_chemicals,
            concentration_range = trial.pop("concentration_range", (127, 128))
        )

    # == Landscape flips
    landscape = landscape[
        ::(-1 if trial.pop("landscape_flip_vertical", False) else 1),
        ::(-1 if trial.pop("landscape_flip_horizontal", False) else 1)
    ]

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
    trial_file = sys.argv[1]
    mode = os.path.basename(trial_file)
    if mode.endswith('.json'):
        mode = mode[:-5] # Drop extension
    landscape_dir = os.path.abspath(sys.argv[2])

    # Load trial variables
    with open(trial_file, 'r') as f:
        variable_dict = json.load(f)
    # Remove comments
    for k in list(variable_dict.keys()):
        if k.startswith("_comment"):
            del variable_dict[k]
        elif not isinstance(variable_dict[k], list):
            raise ValueError("Variable %s must have a list of values!" % k)

    # make ndarrays
    for k in variable_dict.keys():
        variable_dict[k] = np.asarray(variable_dict[k])

    n_variables = len(variable_dict)
    variables = list(variable_dict.keys())
    variables.sort() # get a consistant variable ordering. Just in case we're running on old python
    result_vars_list = list(result_variables.keys())
    result_vars_list.sort() # get a consistant variable ordering. Just in case we're running on old python
    trials = list(itertools.product(*[variable_dict[k] for k in variables]))
    n_trials_by_variable = [len(variable_dict[k]) for k in variables]
    n_trials = len(trials)

    # Housekeeping

    comm = MPI.COMM_WORLD

    assert n_trials >= comm.size, "n_trials %i < comm size %i" % (n_trials, comm.size)

    outdir = None
    if comm.rank == 0:
        logger.info("Starting...")
        logger.info("Running a total of %i trials over %i processes" % (n_trials, comm.size))
        # == Making output dir
        run_number = 0
        datestring = time.strftime("%Y-%m-%d")
        while True:
            outdir = "output-%s-%s-run%i" % (mode, datestring, run_number)
            if os.path.exists(outdir):
                run_number += 1
            else:
                break
        os.makedirs(outdir)
        # Log start time
        start_time = time.time()

    outdir = comm.bcast(outdir, root = 0)

    # 1 indicates line buffering: https://stackoverflow.com/questions/3167494/how-often-does-python-flush-to-a-file
    outfile = open(outdir + ("/task-%i.csv" % comm.rank), 'w', buffering = 1)

    # == Write headers
    print(
        ", ".join(variables + result_vars_list),
        file = outfile
    )

    # == Distribute Trials
    part_trials = np.array_split(trials, comm.size)
    my_trials = part_trials[comm.rank]

    logger.debug("Task %i has %i trials" % (comm.rank, len(my_trials)))

    for i, trial_vals in enumerate(my_trials):
        # - Set up dict for convinience
        trial = dict(zip(variables, trial_vals))

        if i % 50 == 0:
            logger.info("Task %i running its trial %i/%i" % (comm.rank, i + 1, len(my_trials)))

        nsf = make_nsf(trial, landscape_dir)
        trial_result = run_experiment(nsf)
        outstr = ", ".join([variable_formats[v].format(trial[v]) for v in variables])
        outstr += ", "
        outstr += ", ".join([result_variables[v].format(trial_result[v]) for v in result_vars_list])
        print(outstr, file = outfile)

    outfile.close()
    comm.barrier()

    if comm.rank == 0:
        logger.info("Done! Finished %i trials" % len(trials))
        logger.info("It took about %i minutes" % int((time.time() - start_time) / 60.))
        logger.info("Each trial added about %.2fs of wall-clock time" % ((time.time() - start_time) / len(trials)))

import numpy as np
import scipy.io
import matplotlib.pyplot as plt

import itertools
import math
import shutil

# -------- GLOBAL SETTINGS --------

sensor_dim = (40, 2)
sensor_pixel_dimensions = [2, 4]
step_size = 2.0

OUTFILE_NAME = 'output.npz'
VARIABLE_DIRECTORY_NESTING_LEVELS = 3
FRAME_FACTOR = 1.2
TRAINING_SCENE_ARCLEN_FACTOR = 0.5

# --------- EXPERIMENTAL VARIABLES ----

variable_dict = {
    'landscape_class' : ["test_set"], # At this point, just checkerboard = 1
    'landscape_diffuse_time' : [150, 450], #
    'training_path_curve' : [0.0, 1.0],
    'saccade_degrees' : [90.],
    'n_sensor_levels' : [6],
    'sensor_dimensions' : [(40, 1)],
    'sensor_pixel_dimensions' : [(2, 4)],
}

defaults = {
    'n_test_angles' : 25,
    'max_distance_to_training_path' : 80,
    'sensor_real_area' : (14., "$\mathrm{mm}$"),
    'step_size' : 2.0,
}

short_names = {
    'training_path_curve' : ('curve', '%0.2f'),
    'saccade_degrees' : ('saccade', '%i'),
    'n_sensor_levels' : ('greys', '%i'),
    'sensor_pixel_dimensions' : ('pxdim', '{0[0]}x{0[1]}'),
    'sensor_dimensions' : ('sensor', '{0[0]}x{0[1]}')
}

assert VARIABLE_DIRECTORY_NESTING_LEVELS < len(variable_dict)

# ---------------------- RUN STUFF ----------------------------------------


import logging
logging.basicConfig()
logger = logging.getLogger('experiments')
logger.setLevel(logging.INFO)

from navsim import NavBySceneFamiliarity

import sys, os, glob, time
import pickle

from mpi4py import MPI

# Run as:
# run_experiment.py landscape_dir/ output_dir/

landscape_dir = os.path.abspath(sys.argv[1])
output_dir = os.path.abspath(sys.argv[2])

# ------ FUNCTIONS ----------

def sin_training_path(curveness, start_x, l, arclen = 2.0):
    # Assume derivative never goes above 4
    x = np.linspace(start_x, start_x + l, 4 * int(np.floor(l / arclen)))
    y = x - 0.5 * l * curveness * np.sin((x - 0.5 * l - start_x) * np.pi / (0.5 * l))
    dists = np.sqrt((x[1:] - x[:-1])**2 + (y[1:] - y[:-1])**2)

    i = np.searchsorted(np.cumsum(dists), arclen * np.arange(np.floor(np.sum(dists) / arclen)))

    path = np.vstack((x[i], y[i])).T

    return path


def run_experiment(save_to, id_str, training_path, nsf_params,
                   n_quiver_box = 20, frames = None,
                   starting_pos = None, starting_angle = None):

    nsf_params = dict(nsf_params)
    nsf_params.update(defaults)

    nsf = NavBySceneFamiliarity(**nsf_params)
    nsf.train_from_path(training_path)

    nsf.position = starting_pos
    nsf.angle = starting_angle

    fig, anim = nsf.animate(frames = frames)
    anim.save(save_to + "/nav-%s.mp4" % id_str)

    quiv_dat = nsf.quiver_plot_data(n_box = n_quiver_box)
    quiv = nsf.plot_quiver(**quiv_dat)
    pickle.dump(quiv_dat, save_to + "/quiv-%s.pkl" % id_str)
    quiv.savefig(save_to + "/quiv-%s.png" % id_str)

    return nsf.navigation_error, nsf.percent_recapitulated, nsf.stopped_with_exception


for k in variable_dict:
    variable_dict[k] = np.asarray(variable_dict[k])

n_variables = len(variable_dict)
variables = list(variable_dict.keys())
variables.sort() # get a consistant variable ordering. Just in case we're running on old python
trials = list(itertools.product(*[variable_dict[k] for k in variables]))
n_trials_by_variable = [len(variable_dict[k]) for k in variables]
n_trials = len(trials)

id_str_vars = list(short_names.keys())

for v in id_str_vars:
    if not n_trials_by_variable[variables.index(v)] > 1:
        id_str_vars.remove(v)

remove_anyway = ['landscape_class', 'landscape_diffuse_time']
for ra in remove_anyway:
    if ra in id_str_vars:
        id_str_vars.remove(ra)


# Housekeeping

comm = MPI.COMM_WORLD

assert n_trials >= comm.size, "n_trials %i < comm size %i" % (n_trials, comm.size)

if comm.rank == 0:
    logger.info("Making output directories...")
    # Deal with output dir
    if os.path.isdir(output_dir):
        if os.path.exists(output_dir + "/" + OUTFILE_NAME):
            logger.info("Clearing old output directory...")
            shutil.rmtree(output_dir)
        elif not os.listdir(output_dir):
            pass
        else:
            logger.error("Bad output dir")
            comm.abort()
    else:
        logger.info("Creating output dir...")
        os.mkdir(output_dir)

    # Create trial dirs
    os.mkdir(output_dir + "/trials/")
    # Split over landscapes
    for landscape_class in variable_dict['landscape_class']:
        cdir = output_dir + "/trials/landscape_class_%s" % landscape_class
        os.mkdir(cdir)
        for diff_time in variable_dict['landscape_diffuse_time']:
            os.mkdir(cdir + "/landscape_diffuse_time_%i" % diff_time)


    logger.info("Starting...")
    logger.info("Running a total of %i trials over %i processes" % (n_trials, comm.size))
    # Log start time
    start_time = time.time()

os.chdir(output_dir)

# Distribute Trials
part_trials = np.array_split(trials, comm.size)
my_trials = part_trials[comm.rank]

# Set up result arrays
my_variable_values = [np.empty(shape = len(my_trials), dtype = variable_dict[v].dtype) for v in variables]
my_naverrs = np.empty(shape = len(my_trials))
my_coverages = np.empty(shape = len(my_trials))

for i, trial_vals in enumerate(my_trials):
    # - Set up dict for convinience
    trial = dict(zip(variables, trial_vals))

    id_str = "_".join("%s-%s" % (short_names[v][0], short_names[v][1].format(trial[v])) for v in id_str_vars)
    logger.info("Task %i running experiment '%s'" % (comm.rank, id_str))

    # - Save variable values for this trial
    my_variable_values.append(trial_vals)

    # - Load/create stuff
    landscape_class = trial.pop("landscape_class")
    landscape_diff_time = trial.pop("landscape_diffuse_time")
    landscape = np.load(landscape_dir + ("/%s/landscape-diffuse-%i.npy" % (landscape_class, landscape_diff_time)))
    trial['landscape'] = landscape
    # Training Path
    margin = 1.5 * np.max(np.multiply(trial['sensor_dimensions'], trial['sensor_pixel_dimensions'])) // 2
    tpath = sin_training_path(trial.pop('training_path_curve'),
                              margin,
                              np.min(landscape.shape) - 2 * margin,
                              arclen = step_size * TRAINING_SCENE_ARCLEN_FACTOR)

    frames = int(FRAME_FACTOR * TRAINING_SCENE_ARCLEN_FACTOR * len(tpath))

    my_naverr, my_coverage, my_status = run_experiment(
                                       "/trials/landscape_class_%s/landscape_diffuse_time_%i/" % (landscape_class, landscape_diff_time),
                                       id_str,
                                       training_path = tpath,
                                       nsf_params = trial,
                                       frames = frames,
                                       starting_pos = tpath[1],
                                       starting_angle = np.arctan2(*list(tpath[2] - tpath[1])) % 2 * np.pi)

    for vi, val in enumerate(trial_vals):
        my_variable_values[vi][i] = val

    my_naverrs[i] = my_naverr
    my_coverages[i] = my_coverage


gathered = comm.gather((my_variable_values, my_naverrs, my_coverages), root = 0)

if comm.rank == 0:
    all_naverrs = np.vstack(tuple(e[1] for e in gathered))
    all_coverage = np.vstack(tuple(e[2] for e in gathered))
    all_vars = [np.vstack(tuple(e[0][i] for e in gathered)) for i in range(len(variables))]

    to_save = dict(zip(variables, all_vars))
    to_save["navigation_errors"] = all_naverrs
    to_save["path_coverages"] = all_coverage

    np.savez("output.npz", **to_save)
    # Also save the same data in MATLAB format for convinience
    scipy.io.savemat("output.mat", to_save)

    logger.info("Done!")
    logger.info("It took about %i minutes" % int((time.time() - start_time) / 60.))
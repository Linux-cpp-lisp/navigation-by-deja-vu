import numpy as np
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
    'landscape_class' : [1], # At this point, just checkerboard = 1
    'landscape_diffuse_time' : [], #
    'training_path_curve' :
    'saccade_degrees' : [],
    'n_sensor_levels' : [],
    'sensor_pixel_dimensions' :,

}

short_names = {
    'training_path_curve' : ('curve', '%0.2f'),
    'saccade_degrees' : ('saccade', '%i'),
    'n_sensor_levels' : ('greys', '%i'),
    'sensor_pixel_dimensions' : ('pxdim', '%ix%i'),
    'sensor_dimensions' : ('sensor', '%ix%i')
}

assert VARIABLE_DIRECTORY_NESTING_LEVELS < len(variable_dict)

# ---------------------- RUN STUFF ----------------------------------------


import logging
logging.basicConfig()
logger = logging.getLogger('experiments')
logger.setLevel(logging.INFO)

from NavBySceneFamiliarity import NavBySceneFamiliarity

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

    return nsf.navigation_error



#landscapes = [os.path.abspath(p) for p in glob.glob(landscape_dir + "/*.npy")]
n_variables = len(variable_dict)
variables = list(variable_dict.keys())
variables.sort() # get a consistant variable ordering. Just in case we're running on old python
trials = list(itertools.product(*[variable_dict[k] for k in variables]))
n_trials_by_variable = [len(variable_dict[k]) for k in variables]
n_trials = len(trials)

id_str_vars = variables[:]
id_str_vars.remove('landscape_class')
id_str_vars.remove('landscape_diffuse_time')
for i, v in enumerate(variables):
    if not n_trials_by_variable[i] > 1:
        id_str_vars.remove(v)


# Housekeeping

comm = MPI.COMM_WORLD

assert n_trials >= comm.size

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
            sys.exit(1)
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
my_results = np.empty(shape = len(my_trials))
my_variable_values = []

for i, trial_vals in enumerate(my_trials):
    # - Set up dict for convinience
    trial = dict(zip(variables, trial_vals))

    # - Save variable values for this trial
    my_variable_values.append(trial_vals)

    # - Load/create stuff
    landscape_class = trial.pop("landscape_class")
    landscape_diff_time = trial.pop("landscape_diffuse_time")
    landscape = np.load(landscape_dir + ("/%s/landscape-diffuse-%i.npy" % (landscape_class, landscape_diff_time)))
    # Training Path
    margin = 1.5 * np.max(np.multiply(trial['sensor_dimensions'], trial['sensor_pixel_dimensions'])) // 2
    tpath = sin_training_path(trial.pop('training_path_curve'),
                              margin,
                              np.min(landscape.shape) - 2 * margin,
                              arclen = step_size * TRAINING_SCENE_ARCLEN_FACTOR)

    frames = FRAME_FACTOR * TRAINING_SCENE_ARCLEN_FACTOR * len(tpath)

    id_str = "_".join("%s-%s" % (short_names[v][0], short_names[v][1] % trial[v]) for v in id_str_vars)

    logger.info("Task %i running experiment '%s'" % (comm.rank, id_str))
    my_naverr, my_coverage, my_status = run_experiment(
                                       "/trials/landscape_class_%s/landscape_diffuse_time_%i/" % (landscape_class, landscape_diff_time),
                                       id_str,
                                       training_path = tpath,
                                       nsf_params = trial,
                                       frames = frames,
                                       starting_pos = ,
                                       starting_angle = )


gathered = comm.gather((my_diff_times, my_errs), root = 0)

if comm.rank == 0:
    err_surface = np.vstack(tuple(e[1] for e in gathered))
    diffuse_times = np.concatenate(tuple(e[0] for e in gathered))
    logger.debug(err_surface.shape)
    np.savez("output.npz", errors=err_surface, diffuse_times=diffuse_times)
    logger.info("Done!")
    logger.info("It took about %i minutes" % int((time.time() - start_time) / 60.))

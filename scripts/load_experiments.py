import numpy as np
import pandas as pd

import glob

loading_col_dtypes = {
    'landscape_class' : str,
    'landscape_name' : str,
    'training_path_curve' : np.float32,
    'landscape_noise_factor' : np.float32,
    'n_chemicals' : np.int,
    'min_chem_grain_diameter' : np.float32,
    'chem_weight' : np.float32,
    'mask_middle_n' : np.int,
    'n_sensor_levels' : np.int,
    'step_size' : np.float32,
    'saccade_degrees' : np.float32,
    'n_test_angles' : np.int,
    'landscape_flip_vertical' : np.bool, # Bools as ints
    'landscape_flip_horizontal' : np.bool
}
loading_col_converters = {
    'sensor_dimensions' : lambda s: [int(e) for e in s.split(';')],
    'start_offset' : lambda s: [float(e) for e in s.split(';')],
}


def load_runs(runs):
    results = []
    for run in runs:
        for f in glob.glob(run + "/task-*.csv"):
            results.append(dict(
                pd.read_csv(
                    f,
                    sep = ',',
                    header = 0,
                    skipinitialspace=True,
                    dtype = loading_col_dtypes,
                    converters = loading_col_converters
                )
            ))

    data = {
        k : np.concatenate(tuple(r[k] for r in results)) for k in results[0].keys()
    }

    for k in loading_col_converters.keys():
        data[k] = np.vstack(data[k])

    return data

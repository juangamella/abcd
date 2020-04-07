import os
from utils import graph_utils
import numpy as np


def collect_dags(batch_folder, current_data, n_boot, save_dags=False):
    # === DEFINE PATHS FOR FILES WHICH WILL HOLD THE TEMPORARY DATA
    samples_path = os.path.join(batch_folder, 'samples.csv')
    interventions_path = os.path.join(batch_folder, 'interventions.csv')
    dags_path = os.path.join(batch_folder, 'TEMP_DAGS/')

    # === SAVE DATA, THEN CALL R CODE WITH DATA TO GET DAG SAMPLES
    graph_utils._write_data(current_data, samples_path, interventions_path)
    graph_utils.run_gies_boot(n_boot, samples_path, interventions_path, dags_path, delete=True)
    amats, dags = graph_utils._load_dags(dags_path, delete=True)
    if len(dags) != n_boot:
        raise RuntimeError('Correct number of DAGs not saved, check R code')

    if save_dags:
        for d, amat in enumerate(amats):
            np.save(os.path.join(batch_folder, 'dag%d.npy' % d), amat)

    return dags






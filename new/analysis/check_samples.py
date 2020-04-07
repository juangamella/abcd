import os
import yaml
import xarray as xr
import numpy as np
from config import DATA_FOLDER


def count_samples(dataset, strat_names, ks, bs, ns):
    dataset_folder = os.path.join(DATA_FOLDER, dataset)
    counts_fn = os.path.join(dataset_folder, 'counts.netcdf')
    if os.path.exists(counts_fn):
        return xr.open_dataarray(counts_fn)

    dataset_config = yaml.load(open(os.path.join(dataset_folder, 'config.yaml')))
    dags_folder = os.path.join(dataset_folder, 'dags')
    n_nodes = dataset_config['n_nodes']
    n_dags = dataset_config['n_dags']

    counts = xr.DataArray(
        np.zeros([len(strat_names), len(ks), len(bs), len(ns), n_dags, n_nodes]),
        dims=['strategy', 'k', 'b', 'n', 'dag', 'intervened_node'],
        coords={
            'strategy': strat_names,
            'k': ks,
            'b': bs,
            'n': ns,
            'dag': list(range(n_dags)),
            'intervened_node': list(range(n_nodes))
        }
    )

    for dag_num in range(n_dags):
        dag_folder = os.path.join(dags_folder, 'dag%d' % dag_num)
        for strat_fn in filter(os.path.isdir, map(lambda f: os.path.join(dag_folder, f), os.listdir(dag_folder))):
            strat_name, n_str, b_str, k_str = strat_fn.split(',')
            strat_name = os.path.basename(strat_name)
            n = int(n_str[2:])
            b = int(b_str[2:])
            k = int(k_str[2:]) if k_str[2:] != 'None' else None

            samples_folder = os.path.join(strat_fn, 'samples')
            for samples_fn in os.listdir(samples_folder):
                if samples_fn.endswith('.csv'):
                    intervened_node = int(samples_fn.partition('=')[2][:-4])
                    if intervened_node != -1:
                        full_samples_fn = os.path.join(samples_folder, samples_fn)
                        nsamples = sum(1 for line in open(full_samples_fn))
                        counts.loc[dict(strategy=strat_name, n=n, b=b, k=k, dag=dag_num, intervened_node=intervened_node)] = nsamples

    counts.to_netcdf(os.path.join(dataset_folder, 'counts.netcdf'))
    return counts





import glob
import os

import pandas as pd

from Experiments.cate_error.cate_error import cate_error_test

iters = 1
n_repeats = 1
print_progress = True
k_est_mean = 10
k_est_linear = 40

custom_iters = None

datasets = [
    'dense_continuous',
    # 'sine',
    # 'exp',
    # 'friedman',
]

methods = [
    'lcm_mean',
    # 'malts',
    # 'lcm_linear',
    'linear_prog_mean',
    # 'linear_prog_linear',
    # 'ensemble_prog_mean',
    # 'ensemble_prog_linear',
    # 'doubleml',
    # 'bart',
    # 'causal_forest',
    # 'causal_forest_dml'
]

for data in datasets:
    dataset_config = {'n_train': 0}
    if 'dense' in data:
        n_splits = 2
        dataset_config['num_samples'] = 500
        dataset_config['std'] = 1.5
        if 'continuous' in data:
            dataset_config['imp_c'] = 16
            dataset_config['unimp_c'] = 200
            dataset_config['imp_d'] = 0
            dataset_config['unimp_d'] = 0
        elif 'discrete' in data:
            dataset_config['imp_c'] = 0
            dataset_config['unimp_c'] = 0
            dataset_config['imp_d'] = 150
            dataset_config['unimp_d'] = 100
        elif 'mixed' in data:
            dataset_config['imp_c'] = 5
            dataset_config['unimp_c'] = 10
            dataset_config['imp_d'] = 15
            dataset_config['unimp_d'] = 10
    if data == 'sine':
        n_splits = 10
        dataset_config['num_samples'] = 5000
        dataset_config['imp_c'] = 2
        dataset_config['unimp_c'] = 98
    if data == 'exp':
        n_splits = 10
        dataset_config['num_samples'] = 5000
        dataset_config['imp_c'] = 4
        dataset_config['unimp_c'] = 96
    if data == 'friedman':
        n_splits = 5
        dataset_config['num_samples'] = 2500
        dataset_config['imp_c'] = 0
        dataset_config['unimp_c'] = 0

    if 'acic' not in data:
        cate_error_test(dataset=data, n_splits=n_splits,
                        dataset_config=dataset_config,
                        methods=methods, n_repeats=n_repeats,
                        k_est_mean=k_est_mean, k_est_linear=k_est_linear,
                        print_progress=print_progress,
                        iters=iters, custom_iters=custom_iters)
    else:
        for acic_file in all_acic_2019_files:
            dataset_config['acic_file'] = acic_file
            n_splits = 4
            cate_error_test(dataset=data, n_splits=n_splits,
                            dataset_config=dataset_config,
                            methods=methods, n_repeats=n_repeats,
                            k_est_mean=k_est_mean, k_est_linear=k_est_linear,
                            print_progress=print_progress, iters=iters,
                            custom_iters=custom_iters)
import glob
import os

import pandas as pd

from Experiments.cate_error.cate_error import cate_error_test

iters = 1
print_progress = True
k_est_mean = 15
k_est_linear = 60

datasets = [
    'dense_continuous',
    # 'dense_discrete',
    # 'dense_mixed',
    # 'polynomials',
    # 'sine',
    # 'non_linear_mixed',
    # 'test',
    # 'poly_no_interaction',
    # 'poly_interaction',
    # 'exp_log_interaction',
    # 'friedman',
    # 'ihdp',
    # 'acic_2018',
    # 'acic_2019',
    # 'news'
]

all_acic_2018_files = [f.replace('.csv', '') for f in set([c.split('/')[-1].replace('_cf', '') for c in
                                                           glob.glob(f"{os.getenv('ACIC_2018_FOLDER')}/*.csv")])]
n_samples_per_split = 1000
# all_acic_2019_files = list(range(1, 9))
all_acic_2019_files = [3]


methods_config = {
    'linear_coef_matching': {'double_model': [False], 'n_repeats': 1, 'params': None,
                             'methods': [['linear_pruned', False]]},
    # 'tree_imp_matching': True,
    # 'malts': {'methods': ['linear']},
    # 'manhatten': {'methods': ['mean', 'linear']},
    # 'manhatten_pruned': {'params': None, 'methods': ['mean', 'linear']},
    # 'propensity': None,
    # 'genmatch': None,
    'prognostic': None,
    # 'bart': None,
    # 'causal_forest': None
    # 'doubleml': None,
    # 'drlearner': None
}

for data in datasets:
    dataset_config = {'n_train': 0}
    if 'dense' in data:
        n_splits = 3
        dataset_config['num_samples'] = 3000
        if 'continuous' in data:
            dataset_config['imp_c'] = 10
            dataset_config['unimp_c'] = 190
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
    if data in ['polynomials', 'sine', 'non_linear_mixed', 'test']:
        n_splits = 5
        dataset_config['num_samples'] = 10000
        dataset_config['imp_c'] = 10
        dataset_config['unimp_c'] = 90

    if data == 'ihdp':
        n_splits = 25
        dataset_config['ihdp_file'] = 100

    if 'acic' not in data:
        cate_error_test(dataset=data, n_splits=n_splits, dataset_config=dataset_config, methods_config=methods_config,
                        k_est_mean=k_est_mean, k_est_linear=k_est_linear,
                        print_progress=print_progress, iters=iters, custom_iters=None)
    else:
        for acic_file in all_acic_2019_files:
            dataset_config['acic_file'] = acic_file
            n_splits = 2

            cate_error_test(dataset=data, n_splits=n_splits, dataset_config=dataset_config, methods_config=methods_config,
                            k_est_mean=k_est_mean, k_est_linear=k_est_linear,
                            print_progress=print_progress, iters=iters, custom_iters=None)


        # for acic_file in all_acic_2018_files:
        #     dataset_config['acic_file'] = acic_file
        #     n_splits = pd.read_csv(f"{os.getenv('ACIC_2018_FOLDER')}/{acic_file}.csv").shape[0] // n_samples_per_split
        #     n_splits = max(min(n_splits, 5), 3)
        #     print(f'Running ACIC File {acic_file} in {n_splits} splits...')
        #     cate_error_test(dataset=data, n_splits=n_splits, dataset_config=dataset_config, methods_config=methods_config,
        #                     k_est_mean=k_est_mean, k_est_linear=k_est_linear,
        #                     print_progress=print_progress, iters=iters, custom_iters=None)

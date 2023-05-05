from Experiments.cate_error.cate_error import cate_error_test

iters = 1
n_repeats = 1
print_progress = True
k_est_mean = 10
k_est_linear = 40

methods = [
    'lcm_mean',
    'lcm_linear',
    'doubleml',
    'bart',
    'causal_forest',
    'causal_forest_dml',
]

method_order = ['LCM\nMean', 'LCM\nLinear', 'Linear\nDML',
                'Causal\nForest DML', 'Causal\nForest', 'BART\nTLearner']

n_splits = 5
dataset_config = {
    'n_train': 0,
    'num_samples': 2500,
    'std': 1.5,
    'imp_c': 25,
    'unimp_c': 125,
    'imp_d': 0,
    'unimp_d': 0
}

cate_error_test(dataset='dense_continuous', n_splits=n_splits,
                dataset_config=dataset_config,
                methods=methods, n_repeats=n_repeats,
                k_est_mean=k_est_mean, k_est_linear=k_est_linear,
                print_progress=print_progress,
                iters=iters, method_order=method_order)
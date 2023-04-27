from Experiments.cate_error.cate_error import cate_error_test

iters = 1
n_repeats = 1
print_progress = True
k_est_mean = 10
k_est_linear = 40

datasets = [
    'sine',
    'exp',
]

methods = [
    'lcm_mean',
    'linear_prog_mean',
]

method_order = ['LCM', 'Linear PGM']

for data in datasets:
    dataset_config = {'n_train': 0}
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

    cate_error_test(dataset=data, n_splits=n_splits,
                    dataset_config=dataset_config,
                    methods=methods, n_repeats=n_repeats,
                    k_est_mean=k_est_mean, k_est_linear=k_est_linear,
                    print_progress=print_progress,
                    iters=iters, method_order=method_order)

# -*- coding: utf-8 -*-
"""
Script to calculate Relative % CATE Error of various methods on a specified
dataset.
"""

import json
import numpy as np
import pandas as pd
import time
import warnings

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

from Experiments.helpers import create_folder, get_data, summarize_warnings, \
    get_errors
from other_methods import bart, causalforest, prognostic, doubleml, \
    causalforest_dml, pymalts
from src.variable_imp_matching import VIM_CF


warnings.filterwarnings("ignore")
np.random.seed(0)
random_state = 0


def cate_error_test(dataset, n_splits, dataset_config, methods, n_repeats,
                    k_est_mean, k_est_linear, print_progress, iters,
                    method_order):
    save_folder = create_folder(dataset, print_progress)

    config = {'n_splits': n_splits, 'n_repeats': iters,
              'k_est_mean': k_est_mean, 'k_est_linear': k_est_linear,
              **dataset_config}
    n_splits = config['n_splits']
    with open(f'{save_folder}/config.txt', 'w') as f:
        json.dump(config, f, indent=2)

    df_err = pd.DataFrame(columns=['Method', 'True_CATE', 'Est_CATE',
                                   'Relative Error (%)', 'Iter'])
    all_times = []

    if print_progress:
        print(f'Running {dataset} for {iters} '
              f'iterations with the following configuration:')
        for k, v in dataset_config.items():
            print(f'{k}: {v}')

    total_time = time.time()

    for iter in range(0, iters):
        df_data, df_true, binary = get_data(data=dataset,
                                            config=dataset_config)
        df_data.to_csv(f'{save_folder}/df_data.csv')
        with open(f'{save_folder}/binary_cols.txt', 'w') as f:
            f.write(str(binary))
        print(f'{df_data.shape[1] - 2} covs')

        times = {}

        scaling_factor = np.abs(df_true['TE']).mean()

        split_strategy = None

        if 'lcm_mean' in methods:
            method_name = 'LCM'
            start = time.time()
            with warnings.catch_warnings(record=True) as warning_list:
                lcm = VIM_CF(outcome='Y', treatment='T', data=df_data,
                             n_splits=n_splits, n_repeats=n_repeats,
                             random_state=random_state)
                lcm.fit(model='linear', separate_treatments=True)
                lcm.create_mgs(k=k_est_mean)
                lcm.est_cate(cate_methods=['mean'], diameter_prune=None)
            times[method_name] = time.time() - start
            df_err = pd.concat([df_err,
                                get_errors(lcm.cate_df[['avg.CATE_mean']],
                                           df_true[['TE']],
                                           method_name=method_name,
                                           scale=scaling_factor,
                                           iter=iter)
                                ])
            print(f'\n{method_name} method complete: {time.time() - start}')
            summarize_warnings(warning_list, method_name)
            print()
            split_strategy = lcm.split_strategy  # save split strategy to use for all other methods

        if 'lcm_linear' in methods:
            method_name = 'LCM Linear'
            start = time.time()
            with warnings.catch_warnings(record=True) as warning_list:
                lcm = VIM_CF(outcome='Y', treatment='T', data=df_data,
                             n_splits=n_splits, n_repeats=n_repeats,
                             random_state=random_state)
                if split_strategy is not None:
                    lcm.split_strategy = split_strategy
                else:
                    split_strategy = lcm.split_strategy
                lcm.fit(model='linear', separate_treatments=True)
                lcm.create_mgs(k=k_est_linear)
                lcm.est_cate(cate_methods=['linear_pruned'], diameter_prune=None)
            times[method_name] = time.time() - start
            df_err = pd.concat([df_err,
                                get_errors(lcm.cate_df[['avg.CATE_linear_pruned']],
                                           df_true[['TE']],
                                           method_name=method_name,
                                           scale=scaling_factor,
                                           iter=iter)
                                ])
            print(f'\n{method_name} method complete: {time.time() - start}')
            summarize_warnings(warning_list, method_name)
            print()

        if 'lasso fs' in methods:
            method_name = 'LASSO FS'
            start = time.time()
            lcm.M_list = [np.where(m > 0, 1, 0) for m in lcm.M_list]
            lcm.create_mgs(k=k_est_mean)
            lcm.est_cate(cate_methods=['mean'], diameter_prune=None)
            times[method_name] = time.time() - start
            df_err = pd.concat([df_err,
                                get_errors(lcm.cate_df[['avg.CATE_mean']],
                                           df_true[['TE']],
                                           method_name=method_name,
                                           scale=scaling_factor,
                                           iter=iter)
                                ])
            print(f'\n{method_name} method complete: {time.time() - start}')
            summarize_warnings(warning_list, method_name)
            print()

        if 'oracle_fs' in methods:
            method_name = 'Oracle FS'
            start = time.time()
            lcm.M_list = [np.concatenate([np.ones(dataset_config['imp_c'],),
                                          np.zeros(dataset_config['unimp_c'],)])
                          for i in range(n_splits)]
            lcm.create_mgs(k=k_est_mean)
            lcm.est_cate(cate_methods=['mean'], diameter_prune=None)
            times[method_name] = time.time() - start
            df_err = pd.concat([df_err,
                                get_errors(lcm.cate_df[['avg.CATE_mean']],
                                           df_true[['TE']],
                                           method_name=method_name,
                                           scale=scaling_factor,
                                           iter=iter)
                                ])
            print(f'\n{method_name} method complete: {time.time() - start}')
            summarize_warnings(warning_list, method_name)
            print()

        if 'malts' in methods:
            method_name = 'MALTS'
            start = time.time()
            with warnings.catch_warnings(record=True) as warning_list:
                m = pymalts.malts_mf('Y', 'T', data=df_data,
                                     discrete=binary,
                                     k_est=k_est_mean,
                                     n_splits=n_splits, estimator='mean',
                                     smooth_cate=False,
                                     split_strategy=split_strategy,
                                     random_state=random_state)
            times[method_name] = time.time() - start
            df_err = pd.concat([df_err,
                                get_errors(m.CATE_df[['avg.CATE']],
                                           df_true[['TE']],
                                           method_name=method_name,
                                           scale=scaling_factor,
                                           iter=iter)
                                ])
            print(f'\n{method_name} method complete: {time.time() - start}')
            print(f'{method_name} complete: {time.time() - start}')
            summarize_warnings(warning_list, method_name)
            print()

        if 'linear_prog_mean' in methods:
            method_name = 'Linear PGM'
            start = time.time()
            with warnings.catch_warnings(record=True) as warning_list:
                cate_est_prog, _, _ = prognostic.prognostic_cv('Y', 'T',
                                                               df_data,
                                                               method='linear',
                                                               double=True,
                                                               k_est=k_est_mean,
                                                               est_method='mean',
                                                               diameter_prune=None,
                                                               gen_skf=split_strategy,
                                                               random_state=random_state)
            times[method_name] = time.time() - start
            df_err = pd.concat([df_err,
                                get_errors(cate_est_prog[['avg.CATE']],
                                           df_true[['TE']],
                                           method_name=method_name,
                                           scale=scaling_factor,
                                           iter=iter)
                                ])
            print(f'\n{method_name} method complete: {time.time() - start}')
            summarize_warnings(warning_list, method_name)
            print()

        if 'linear_prog_linear' in methods:
            method_name = 'Linear PGM Linear'
            start = time.time()
            with warnings.catch_warnings(record=True) as warning_list:
                cate_est_prog, _, _ = prognostic.prognostic_cv('Y', 'T',
                                                               df_data,
                                                               method='linear',
                                                               double=True,
                                                               k_est=k_est_linear,
                                                               est_method='linear_pruned',
                                                               diameter_prune=None,
                                                               gen_skf=split_strategy,
                                                               random_state=random_state)
            times[method_name] = time.time() - start
            df_err = pd.concat([df_err,
                                get_errors(cate_est_prog[['avg.CATE']],
                                           df_true[['TE']],
                                           method_name=method_name,
                                           scale=scaling_factor,
                                           iter=iter)
                                ])
            print(f'\n{method_name} method complete: {time.time() - start}')
            summarize_warnings(warning_list, method_name)
            print()

        if 'ensemble_prog_mean' in methods:
            method_name = 'Nonparametric PGM'
            start = time.time()
            with warnings.catch_warnings(record=True) as warning_list:
                cate_est_prog, c_mg, t_mg = prognostic.prognostic_cv('Y', 'T',
                                                                     df_data,
                                                                     method='ensemble',
                                                                     double=True,
                                                                     k_est=k_est_mean,
                                                                     est_method='mean',
                                                                     diameter_prune=None,
                                                                     gen_skf=split_strategy,
                                                                     random_state=random_state)
            times[method_name] = time.time() - start
            df_err = pd.concat([df_err,
                                get_errors(cate_est_prog[['avg.CATE']],
                                           df_true[['TE']],
                                           method_name=method_name,
                                           scale=scaling_factor,
                                           iter=iter)
                                ])
            print(f'\n{method_name} method complete: {time.time() - start}')
            summarize_warnings(warning_list, method_name)
            print()

        if 'ensemble_prog_linear' in methods:
            method_name = 'Nonparametric PGM Linear'
            start = time.time()
            with warnings.catch_warnings(record=True) as warning_list:
                cate_est_prog, c_mg, t_mg = prognostic.prognostic_cv('Y', 'T',
                                                                     df_data,
                                                                     method='ensemble',
                                                                     double=True,
                                                                     k_est=k_est_linear,
                                                                     est_method='linear_pruned',
                                                                     diameter_prune=None,
                                                                     gen_skf=split_strategy,
                                                                     random_state=random_state)
            times[method_name] = time.time() - start
            df_err = pd.concat([df_err,
                                get_errors(cate_est_prog[['avg.CATE']],
                                           df_true[['TE']],
                                           method_name=method_name,
                                           scale=scaling_factor,
                                           iter=iter)
                                ])
            print(f'\n{method_name} method complete: {time.time() - start}')
            summarize_warnings(warning_list, method_name)
            print()

        if 'doubleml' in methods:
            method_name = 'Linear DML'
            start = time.time()
            with warnings.catch_warnings(record=True) as warning_list:
                cate_est_doubleml = doubleml.doubleml('Y', 'T', df_data,
                                                      gen_skf=split_strategy,
                                                      random_state=random_state)
            times[method_name] = time.time() - start
            df_err = pd.concat([df_err,
                                get_errors(cate_est_doubleml[['avg.CATE']],
                                           df_true[['TE']],
                                           method_name=method_name,
                                           scale=scaling_factor,
                                           iter=iter)
                                ])
            print(f'\n{method_name} method complete: {time.time() - start}')
            summarize_warnings(warning_list, method_name)
            print()

        if 'bart' in methods:
            method_name = 'BART'
            start = time.time()
            with warnings.catch_warnings(record=True) as warning_list:
                cate_est_bart = bart.bart('Y', 'T', df_data,
                                          gen_skf=split_strategy,
                                          random_state=random_state)
            times[method_name] = time.time() - start
            df_err = pd.concat([df_err,
                                get_errors(cate_est_bart[['avg.CATE']],
                                           df_true[['TE']],
                                           method_name=method_name,
                                           scale=scaling_factor,
                                           iter=iter)
                                ])
            print(f'\n{method_name} method complete: {time.time() - start}')
            summarize_warnings(warning_list, method_name)
            print()

        if 'causal_forest' in methods:
            method_name = 'Causal Forest'
            start = time.time()
            with warnings.catch_warnings(record=True) as warning_list:
                cate_est_cf = causalforest.causalforest('Y', 'T', df_data,
                                                        gen_skf=split_strategy,
                                                        random_state=random_state)
            times[method_name] = time.time() - start
            df_err = pd.concat([df_err,
                                get_errors(cate_est_cf[['avg.CATE']],
                                           df_true[['TE']],
                                           method_name=method_name,
                                           scale=scaling_factor,
                                           iter=iter)
                                ])
            print(f'\n{method_name} method complete: {time.time() - start}')
            summarize_warnings(warning_list, method_name)
            print()

        if 'causal_forest_dml' in methods:
            method_name = 'Causal Forest DML'
            start = time.time()
            with warnings.catch_warnings(record=True) as warning_list:
                cate_est_cf = causalforest_dml.causalforest_dml('Y', 'T',
                                                                df_data,
                                                                gen_skf=split_strategy,
                                                                random_state=random_state)
            times[method_name] = time.time() - start
            df_err = pd.concat([df_err,
                                get_errors(cate_est_cf[['avg.CATE']],
                                           df_true[['TE']],
                                           method_name=method_name,
                                           scale=scaling_factor,
                                           iter=iter)
                                ])
            print(f'\n{method_name} method complete: {time.time() - start}')
            summarize_warnings(warning_list, method_name)
            print()

        df_err.loc[df_err['Iter'] == iter, 'Relative Error (%)'] = df_err.loc[df_err['Iter'] == iter,
                                                                              'Relative Error (%)'] * 100

        order = [m for m in method_order if m in df_err['Method'].unique()]
        palette = {method_order[i]: sns.color_palette()[i] for i in
                   range(len(method_order))}

        plt.figure()
        sns.set_context("paper")
        sns.set_style("darkgrid")
        sns.set(font_scale=1)
        ax = sns.boxplot(x='Method', y='Relative Error (%)',
                         data=df_err[df_err['Iter'] == iter], showfliers=False,
                         order=order, palette=palette)
        ax.yaxis.set_major_formatter(ticker.PercentFormatter())
        ax.get_figure().savefig(f'{save_folder}/boxplot_multifold_iter{iter:02d}.png')

        all_times.append(times)
        if print_progress:
            print(f'{iter} total time: {time.time() - total_time}\n')

    if print_progress:
        print('Saving all results...')
    df_err = df_err.reset_index(drop=True)
    df_err.to_csv(f'{save_folder}/df_err.csv')
    all_times = pd.DataFrame(all_times).T
    all_times['avg'] = all_times.mean(axis=1)
    all_times.to_csv(f'{save_folder}/times.csv')

    if iters > 1:
        if print_progress:
            print('Creating final lineplot...')
        sns.set_context("paper")
        sns.set_style("darkgrid")
        sns.set(font_scale=6)
        fig, ax = plt.subplots(figsize=(40, 50))
        df_err = df_err.rename(columns={'Iter': 'Iteration'})
        pp = sns.pointplot(data=df_err.reset_index(drop=True), x='Iteration', y='Relative Error (%)', hue='Method',
                           errorbar=("pi", 95), dodge=True, scale=5)
        plt.setp(pp.get_legend().get_texts(), fontsize='50')
        plt.setp(pp.get_legend().get_title(), fontsize='60')
        ax.yaxis.set_major_formatter(ticker.PercentFormatter())
        plt.tight_layout()
        fig.savefig(f'{save_folder}/{iters}_iterations_lineplot.png')

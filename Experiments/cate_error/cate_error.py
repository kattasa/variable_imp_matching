# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 01:31:42 2020
@author: Harsh
"""

import json
import numpy as np
import pandas as pd
import time
import warnings

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

from Experiments.helpers import create_folder, get_data
from other_methods import pymalts, bart, causalforest, matchit, prognostic
from src.linear_coef_matching_mf import LCM_MF
from utils import get_match_groups, get_CATES


warnings.filterwarnings("ignore")
np.random.seed(0)


def cate_error_test(dataset, n_splits, dataset_config, methods_config, k_est_mean, k_est_linear, print_progress, iters,
                    custom_iters=None):
    save_folder = create_folder(dataset, print_progress)

    config = {'n_splits': n_splits, 'n_repeats': iters, 'k_est_mean': k_est_mean, 'k_est_linear': k_est_linear,
              **dataset_config}
    with open(f'{save_folder}/config.txt', 'w') as f:
        json.dump(config, f, indent=2)
    with open(f'{save_folder}/methods_config.txt', 'w') as f:
        json.dump(methods_config, f, indent=2)
    if custom_iters is not None:
        with open(f'{save_folder}/custom_iters.txt', 'w') as f:
            json.dump(custom_iters, f, indent=2)

    df_err = pd.DataFrame(columns=['Method', 'True_CATE', 'Est_CATE', 'Relative Error (%)', 'Iter'])
    model_scores = pd.DataFrame(columns=['Method', 'Score', 'Iter'])
    all_times = []

    if print_progress:
        print(f'Running {dataset} for {iters} iterations with the following configuration:')
        for k, v in dataset_config.items():
            print(f'{k}: {v}')

    methods = list(methods_config.keys())

    total_time = time.time()

    for iter in range(0, iters):
        if custom_iters is not None:
            for k, v in custom_iters[iter].items():
                dataset_config[k] = v

        df_data, df_true, discrete, dummy_cols = get_data(data=dataset, config=dataset_config)
        df_true.to_csv(f'{save_folder}/df_true{iter}.csv')

        df_admalts_data = df_data.copy(deep=True)
        if dummy_cols is not None:
            df_data = df_data.drop(columns=dummy_cols)
            df_admalts_data = df_admalts_data.drop(columns=discrete)
            with open(f'{save_folder}/dummy_cols{iter}.txt', 'w') as f:
                f.write(str(dummy_cols))


        times = {}

        split_strategy = None
        if 'linear_coef_matching' in methods:
            for double_model in methods_config['linear_coef_matching']['double_model']:
                start = time.time()
                lcm = LCM_MF(outcome='Y', treatment='T', data=df_admalts_data, n_splits=n_splits,
                              n_repeats=methods_config['linear_coef_matching']['n_repeats'])
                init_time = time.time() - start
                if split_strategy:
                    lcm.gen_skf = split_strategy
                else:
                    split_strategy = lcm.gen_skf
                start = time.time()
                lcm.fit(params=methods_config['linear_coef_matching']['params'], double_model=double_model)
                fit_time = time.time() - start
                # model_scores = model_scores.append(
                #     pd.DataFrame([[f'{"Double" if double_model else "Single"} Model Lasso Matching']*n_splits,
                #                   lcm.modl_score_list, [iter]*n_splits]).T.rename(columns={0: 'Method', 1: 'Score', 2: 'Iter'}))
                # print('M')
                # print([df_admalts_data.columns[np.argsort(-z)[:np.sum(z != 0)]] for z in ad_m.M_list])
                # print([z[np.argsort(-z)[:np.sum(z != 0)]] for z in ad_m.M_list])
                # print(f'M Nonzero weights: {[np.sum(z != 0) for z in ad_m.M_list]}')
                for augmented in methods_config['linear_coef_matching']['augmented']:
                    for e_method in [[m, k_est_mean if m == 'mean' else k_est_linear, augmented] for m in
                                     methods_config['linear_coef_matching']['methods']]:
                        method_name = f'{"Double" if double_model else "Single"} Model Lasso Matching ' \
                                      f'{"Augmented " if e_method[2] else ""}{" ".join(e_method[0].split("_")).title()}'
                        if e_method[1] != lcm.MG_size:
                            start = time.time()
                            lcm.MG(k=e_method[1])
                            mg_time = time.time() - start
                        start = time.time()
                        lcm.CATE(cate_methods=[e_method[0]], augmented=e_method[2])
                        times[method_name] = time.time() - start + fit_time + mg_time + init_time
                        cate_df = lcm.cate_df.sort_index()
                        cate_df = cate_df.rename(columns={'avg.CATE': 'Est_CATE'})
                        cate_df['True_CATE'] = df_true['TE'].to_numpy()
                        cate_df['Relative Error (%)'] = np.abs((cate_df['Est_CATE']-cate_df['True_CATE'])/np.abs(cate_df['True_CATE']).mean())
                        cate_df['Method'] = [method_name for i in range(cate_df.shape[0])]
                        cate_df['Iter'] = iter
                        df_err = df_err.append(cate_df[['Method', 'True_CATE', 'Est_CATE', 'Relative Error (%)', 'Iter']].copy(deep=True))
                        if print_progress:
                            print(f'{method_name} method complete: {time.time() - start + fit_time + mg_time + init_time}')

        if 'malts' in methods:
            est_methods = [[m, k_est_mean if m == 'mean' else k_est_linear] for m in methods_config['malts']['methods']]
            for e_method in est_methods:
                method_name = f'MALTS Matching {e_method[0].title()}'
                start = time.time()
                m = pymalts.malts_mf('Y', 'T', data=df_data, discrete=discrete, k_tr=15, k_est=e_method[1],
                                     n_splits=n_splits, estimator=e_method[0], smooth_cate=False,
                                     gen_skf=split_strategy)
                times[method_name] = time.time() - start
                cate_df = m.CATE_df
                cate_df = cate_df.rename(columns={'avg.CATE': 'Est_CATE'})
                cate_df['True_CATE'] = df_true['TE'].to_numpy()
                cate_df['Relative Error (%)'] = np.abs(
                    (cate_df['Est_CATE'] - cate_df['True_CATE']) / np.abs(cate_df['True_CATE']).mean())
                cate_df['Method'] = [method_name for i in range(cate_df.shape[0])]
                cate_df['Iter'] = iter
                df_err = df_err.append(cate_df[['Method', 'True_CATE', 'Est_CATE', 'Relative Error (%)', 'Iter']].copy(deep=True))
                if print_progress:
                    print(f'{method_name} complete: {time.time() - start}')

        if 'manhatten' in methods:
            est_methods = [[m, k_est_mean if m == 'mean' else k_est_linear] for m in methods_config['manhatten']['methods']]
            covariates = [c for c in df_data.columns if c not in ['Y', 'T']]
            weights = np.array([1 for v in range(len(covariates))])
            for e_method in est_methods:
                method_name = f'Manhatten Matching {e_method[0].title()}'
                start = time.time()
                c_mg, t_mg, _, _ = get_match_groups(df_data, k=e_method[1], covariates=covariates, treatment='T',
                                                    M=weights,
                                                    return_original_idx=False,
                                                    check_est_df=False)
                cates = get_CATES(df_data, c_mg, t_mg, e_method[0], covariates, 'Y', 'T', None, None, weights,
                                  augmented=False, control_preds=None, treatment_preds=None, check_est_df=False)
                times[method_name] = time.time() - start
                cate_df = cates.to_frame()
                cate_df.columns = ['Est_CATE']
                cate_df['True_CATE'] = df_true['TE'].to_numpy()
                cate_df['Relative Error (%)'] = np.abs(
                    (cate_df['Est_CATE'] - cate_df['True_CATE']) / np.abs(cate_df['True_CATE']).mean())
                cate_df['Method'] = [method_name for i in range(cate_df.shape[0])]
                cate_df['Iter'] = iter
                df_err = df_err.append(cate_df[['Method', 'True_CATE', 'Est_CATE', 'Relative Error (%)', 'Iter']].copy(deep=True))
                if print_progress:
                    print(f'{method_name} complete: {time.time() - start}')

        if 'manhatten_pruned' in methods:
            start = time.time()
            man_pruned = LCM_MF(outcome='Y', treatment='T', data=df_data, n_splits=n_splits, n_repeats=1)
            init_time = time.time() - start
            man_pruned.gen_skf = split_strategy
            man_pruned.fit(params=methods_config['manhatten_pruned']['params'])
            fit_time = time.time() - start
            est_methods = [[f'{m}_pruned', k_est_mean if m == 'mean' else k_est_linear] for m in
                           methods_config['manhatten_pruned']['methods']]
            man_pruned.M_list = [(a != 0).astype(int) for a in man_pruned.M_list]
            for e_method in est_methods:
                method_name = f'Manhatten Matching {" ".join(e_method[0].split("_")).title()}'
                start = time.time()
                man_pruned.MG(k=e_method[1])
                mg_time = time.time() - start
                start = time.time()
                man_pruned.CATE(cate_methods=['mean' if e_method[0] == 'mean_pruned' else e_method[0]], augmented=False)
                times[method_name] = time.time() - start + fit_time + mg_time + init_time
                cate_df = man_pruned.cate_df
                cate_df = cate_df.rename(columns={'avg.CATE': 'Est_CATE'})
                cate_df['True_CATE'] = df_true['TE'].to_numpy()
                cate_df['Relative Error (%)'] = np.abs(
                    (cate_df['Est_CATE'] - cate_df['True_CATE']) / np.abs(cate_df['True_CATE']).mean())
                cate_df['Method'] = [method_name for i in range(cate_df.shape[0])]
                cate_df['Iter'] = iter
                df_err = df_err.append(cate_df[['Method', 'True_CATE', 'Est_CATE', 'Relative Error (%)', 'Iter']].copy(deep=True))
                if print_progress:
                    print(f'{method_name} method complete: {time.time() - start + fit_time + mg_time + init_time}')

        if 'propensity' in methods:
            method_name = 'Propensity Score Matching'
            start = time.time()
            ate_psnn, t_psnn = matchit.matchit('Y', 'T', data=df_data, method='nearest', replace=True)
            times[method_name] = time.time() - start
            df_err_psnn = pd.DataFrame()
            df_err_psnn['Method'] = [method_name for i in range(t_psnn.shape[0])]
            df_err_psnn['Relative Error (%)'] = np.abs((t_psnn['CATE'].to_numpy() - df_true['TE'].to_numpy())/np.abs(df_true['TE']).mean())
            df_err_psnn['Iter'] = iter
            df_err_psnn['True_CATE'] = df_true['TE'].to_numpy()
            df_err_psnn['Est_CATE'] = t_psnn['CATE'].to_numpy()
            df_err = df_err.append(df_err_psnn[['Method', 'True_CATE', 'Est_CATE', 'Relative Error (%)', 'Iter']])
            if print_progress:
                print(f'Propensity Score complete: {time.time() - start}')

        if 'genmatch' in methods:
            method_name = 'GanMatch'
            start = time.time()
            ate_gen, t_gen = matchit.matchit('Y', 'T', data=df_data, method='genetic', replace=True)
            times[method_name] = time.time() - start
            df_err_gen = pd.DataFrame()
            df_err_gen['Method'] = [method_name for i in range(t_gen.shape[0])]
            df_err_gen['Relative Error (%)'] = np.abs((t_gen['CATE'].to_numpy() - df_true['TE'].to_numpy())/np.abs(df_true['TE']).mean())
            df_err_gen['Iter'] = iter
            df_err_gen['True_CATE'] = df_true['TE'].to_numpy()
            df_err_gen['Est_CATE'] = t_gen['CATE'].to_numpy()
            df_err = df_err.append(df_err_gen[['Method', 'True_CATE', 'Est_CATE', 'Relative Error (%)', 'Iter']])
            if print_progress:
                print(f'GenMatch complete: {time.time() - start}')

        if 'prognostic' in methods:
            method_name = 'Prognostic Score Matching'
            start = time.time()
            cate_est_prog, _, _ = prognostic.prognostic_cv('Y', 'T', df_data, k_est=k_est_mean, gen_skf=split_strategy,
                                                           n_splits=n_splits)
            times[method_name] = time.time() - start
            df_err_prog = pd.DataFrame()
            df_err_prog['Method'] = [method_name for i in range(cate_est_prog.shape[0])]
            df_err_prog['Relative Error (%)'] = np.abs((cate_est_prog['avg.CATE'].to_numpy() -
                                                        df_true['TE'].to_numpy()) / np.abs(df_true['TE']).mean())
            df_err_prog['Iter'] = iter
            df_err_prog['True_CATE'] = df_true['TE'].to_numpy()
            df_err_prog['Est_CATE'] = cate_est_prog['avg.CATE'].to_numpy()
            df_err = df_err.append(df_err_prog[['Method', 'True_CATE', 'Est_CATE', 'Relative Error (%)', 'Iter']])
            if print_progress:
                print(f'{method_name} complete: {time.time() - start}')

        if 'bart' in methods:
            method_name = 'BART'
            start = time.time()
            cate_est_bart = bart.bart('Y', 'T', df_data, gen_skf=split_strategy, n_splits=n_splits)
            times[method_name] = time.time() - start
            df_err_bart = pd.DataFrame()
            df_err_bart['Method'] = [method_name for i in range(cate_est_bart.shape[0])]
            df_err_bart['Relative Error (%)'] = np.abs(
                (cate_est_bart['avg.CATE'].to_numpy() - df_true['TE'].to_numpy()) / np.abs(df_true['TE']).mean())
            df_err_bart['Iter'] = iter
            df_err_bart['True_CATE'] = df_true['TE'].to_numpy()
            df_err_bart['Est_CATE'] = cate_est_bart['avg.CATE'].to_numpy()
            df_err = df_err.append(df_err_bart[['Method', 'True_CATE', 'Est_CATE', 'Relative Error (%)', 'Iter']])
            if print_progress:
                print(f'{method_name} complete: {time.time() - start}')

        if 'causal_forest' in methods:
            method_name = 'Causal Forest'
            start = time.time()
            cate_est_cf = causalforest.causalforest('Y', 'T', df_data, gen_skf=split_strategy, n_splits=n_splits)
            times[method_name] = time.time() - start
            df_err_cf = pd.DataFrame()
            df_err_cf['Method'] = [method_name for i in range(cate_est_cf.shape[0])]
            df_err_cf['Relative Error (%)'] = np.abs((cate_est_cf['avg.CATE'].to_numpy() - df_true['TE'].to_numpy())/np.abs(df_true['TE']).mean())
            df_err_cf['Iter'] = iter
            df_err_cf['True_CATE'] = df_true['TE'].to_numpy()
            df_err_cf['Est_CATE'] = cate_est_cf['avg.CATE'].to_numpy()
            df_err = df_err.append(df_err_cf[['Method', 'True_CATE', 'Est_CATE', 'Relative Error (%)', 'Iter']])
            if print_progress:
                print(f'{method_name} complete: {time.time() - start}')

        df_err.loc[df_err['Iter'] == iter, 'Relative Error (%)'] = df_err.loc[df_err['Iter'] == iter,
                                                                              'Relative Error (%)'] * 100
        sns.set_context("paper")
        sns.set_style("darkgrid")
        sns.set(font_scale=6)
        fig, ax = plt.subplots(figsize=(40, 50))
        sns.boxenplot(x='Method', y='Relative Error (%)', data=df_err[df_err['Iter'] == iter])
        plt.xticks(rotation=65, horizontalalignment='right')
        ax.yaxis.set_major_formatter(ticker.PercentFormatter())
        plt.tight_layout()
        fig.savefig(f'{save_folder}/boxplot_multifold_iter{iter:02d}.png')
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
    model_scores.to_csv(f'{save_folder}/model_scores.csv')

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
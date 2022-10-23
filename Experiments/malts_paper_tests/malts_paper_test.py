# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 01:31:42 2020
@author: Harsh
"""

import json
import numpy as np
import pandas as pd
import time

import pymalts

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

import sys
sys.path.append("..")
from helpers import create_folder, get_data
sys.path.append("..")
from MALTS.amect_mf import Amect_mf
from other_methods import bart, causalforest, matchit, prognostic
from utils import get_match_groups, get_CATES

import warnings

warnings.filterwarnings("ignore")

np.random.seed(0)

datasets = [
    'dense_continuous',
    'dense_discrete',
    'dense_mixed',
    # 'polynomials',
    # 'sine',
    # 'non_linear_mixed',
    # 'test',
    # 'poly_no_interaction',
    # 'poly_interaction',
    # 'exp_log_interaction',
    # 'friedman',
    'ihdp',
    # 'acic_2018',
    # 'acic_2019'
]

admalts_params = None
admalts_prune = False
malts_methods = ['mean', 'linear']
manhatten_methods = ['mean', 'linear']
manhatten_pruned_methods = ['mean', 'linear']
prognostic_methods = ['lasso', 'rf']
methods = [
    'malts',
    'manhatten',
    # 'manhatten_pruned',
    'propensity',
    'prognostic',
    'genmatch',
    'bart',
    'causal_forest'
]

num_samples = 3000
n_repeats = 1
k_est_mean = 15
k_est_linear = 60

print_progress = True

iters = 1
include_title = False
iter_in_title = False

for data in datasets:
    plot_name = f"{data.replace('_', ' ').title()} DGP"
    save_folder = create_folder(data, print_progress)

    if data == 'ihdp':
        n_splits = 2
    elif data == 'acic_2019':
        n_splits = 4
    elif data == 'acic_2018':
        n_splits = 5
    else:
        n_splits = 6

    config = {'n_splits': n_splits, 'n_repeats': n_repeats, 'k_est_mean': k_est_mean, 'k_est_linear': k_est_linear}

    df_err = pd.DataFrame(columns=['Method', 'Relative Error (%)', 'Iter'])
    all_times = []

    for iter in range(0, iters):
        total_time = time.time()
        if data == 'dense_continuous':
            nci = 15
            ncu = 25
            ndi = 0
            ndu = 0
        elif data == 'dense_discrete':
            nci = 0
            ncu = 0
            ndi = 15
            ndu = 10
        elif data == 'dense_mixed':
            nci = 5
            ncu = 10
            ndi = 15
            ndu = 10
        else:
            nci = 10
            ncu = 90
            ndi = 0
            ndu = 0
        print(f'Imp continuous: {nci}\nImp discrete: {ndi}\nUnimp continuous: {ncu}\nUnimp discrete: {ndu}')

        df_data, df_true, discrete, config = get_data(data, num_samples, config, imp_c=nci, imp_d=ndi, unimp_c=ncu,
                                                      unimp_d=ndu)
        # import itertools
        # from sklearn.feature_selection import SelectKBest
        # from sklearn.feature_selection import f_regression
        # second_order_df = pd.DataFrame()
        # for combo in itertools.combinations_with_replacement([c for c in df_data.columns if c not in ['Y', 'T']], 2):
        #     second_order_df[f'{combo[0]}_{combo[1]}'] = df_data[combo[0]] * df_data[combo[1]]
        # print(df_data.shape)
        # second_order_sel = SelectKBest(f_regression, k=min(int(df_data.shape[0] / n_splits)-df_data.shape[1]-2, second_order_df.shape[1]))\
        #     .fit(second_order_df, df_data['Y'])
        # df_admalts_data = df_data.join(pd.DataFrame(second_order_sel.transform(second_order_df),
        #                                     columns=second_order_sel.get_feature_names_out()))
        # print(df_admalts_data.shape)

        # sin_cos_df = pd.DataFrame()
        # for col in [c for c in df_data.columns if c not in ['Y', 'T']]:
        #     sin_cos_df[f'{col}_sin'] = np.sin(df_data[col])
        #     sin_cos_df[f'{col}_cos'] = np.cos(df_data[col])
        # df_admalts_data = df_data.join(sin_cos_df)
        # print(df_admalts_data.shape)

        with open(f'{save_folder}/config.txt', 'w') as f:
            json.dump(config, f, indent=2)

        times = {}

        start = time.time()
        ad_m = Amect_mf(outcome='Y', treatment='T', data=df_data, n_splits=n_splits, n_repeats=n_repeats)
        init_time = time.time() - start
        split_strategy = ad_m.gen_skf
        ad_m.fit(params=admalts_params, prune=False)
        fit_time = time.time() - start
        print('M_C')
        print([df_data.columns[np.argsort(-z)[:np.sum(z != 0)]] for z in ad_m.M_C_list])
        print([z[np.argsort(-z)[:np.sum(z != 0)]] for z in ad_m.M_C_list])
        print('M_T')
        print([df_data.columns[np.argsort(-z)[:np.sum(z != 0)]] for z in ad_m.M_T_list])
        print([z[np.argsort(-z)[:np.sum(z != 0)]] for z in ad_m.M_T_list])
        print(f'MC Nonzero weights: {[np.sum(z != 0) for z in ad_m.M_C_list]}')
        print(f'MT Nonzero weights: {[np.sum(z != 0) for z in ad_m.M_T_list]}')
        for e_method in [['mean', k_est_mean, False], ['linear_pruned', k_est_linear, False]]:
            method_name = f'Lasso Matching {"Augmented " if e_method[2] else ""}{" ".join(e_method[0].split("_")).title()}'
            if e_method[1] != ad_m.MG_size:
                start = time.time()
                ad_m.MG(k=e_method[1])
                mg_time = time.time() - start
            start = time.time()
            ad_m.CATE(cate_methods=[e_method[0]], augmented=e_method[2])
            times[method_name] = time.time() - start + fit_time + mg_time
            cate_df = ad_m.cate_df
            cate_df['true.CATE'] = df_true['TE'].to_numpy()
            cate_df['Relative Error (%)'] = np.abs((cate_df['avg.CATE']-cate_df['true.CATE'])/np.abs(cate_df['true.CATE']).mean())
            cate_df['Method'] = [method_name for i in range(cate_df.shape[0])]
            cate_df['Iter'] = iter
            df_err = df_err.append(cate_df[['Method', 'Relative Error (%)', 'Iter']].copy(deep=True))
            if print_progress:
                print(f'{method_name} method complete: {time.time() - start + fit_time + mg_time}')

        if 'malts' in methods:
            est_methods = [[m, k_est_mean if m == 'mean' else k_est_linear] for m in malts_methods]
            for e_method in est_methods:
                method_name = f'MALTS Matching {e_method[0].title()}'
                start = time.time()
                m = pymalts.malts_mf('Y', 'T', data=df_data, discrete=discrete, k_tr=15, k_est=e_method[1],
                                     n_splits=n_splits, estimator=e_method[0], smooth_cate=False,
                                     gen_skf=split_strategy)
                times[method_name] = time.time() - start
                cate_df = m.CATE_df
                cate_df['true.CATE'] = df_true['TE'].to_numpy()
                cate_df['Relative Error (%)'] = np.abs(
                    (cate_df['avg.CATE'] - cate_df['true.CATE']) / np.abs(cate_df['true.CATE']).mean())
                cate_df['Method'] = [method_name for i in range(cate_df.shape[0])]
                cate_df['Iter'] = iter
                df_err = df_err.append(cate_df[['Method', 'Relative Error (%)', 'Iter']].copy(deep=True))
                if print_progress:
                    print(f'{method_name} complete: {time.time() - start}')

        if 'manhatten' in methods:
            est_methods = [[m, k_est_mean if m == 'mean' else k_est_linear] for m in manhatten_methods]
            covariates = [c for c in df_data.columns if c not in ['Y', 'T']]
            weights = np.array([1 for v in range(len(covariates))])
            for e_method in est_methods:
                method_name = f'Manhatten Matching {e_method[0].title()}'
                start = time.time()
                c_mg, t_mg, _, _ = get_match_groups(df_data, k=e_method[1], covariates=covariates, treatment='T',
                                                    M_C=weights,
                                                    M_T=weights,
                                                    method=None,
                                                    return_original_idx=False,
                                                    check_est_df=False)
                cates = get_CATES(df_data, c_mg, t_mg, e_method[0], covariates, 'Y', 'T', None, None, weights, weights,
                                  augmented=False, control_preds=None, treatment_preds=None, check_est_df=False)
                times[method_name] = time.time() - start
                cate_df = cates.to_frame()
                cate_df.columns = ['avg.CATE']
                cate_df['true.CATE'] = df_true['TE'].to_numpy()
                cate_df['Relative Error (%)'] = np.abs(
                    (cate_df['avg.CATE'] - cate_df['true.CATE']) / np.abs(cate_df['true.CATE']).mean())
                cate_df['Method'] = [method_name for i in range(cate_df.shape[0])]
                cate_df['Iter'] = iter
                df_err = df_err.append(cate_df[['Method', 'Relative Error (%)', 'Iter']].copy(deep=True))
                if print_progress:
                    print(f'{method_name} complete: {time.time() - start}')

        if 'manhatten_pruned' in methods:
            est_methods = [[f'{m}_pruned', k_est_mean if m == 'mean' else k_est_linear] for m in manhatten_pruned_methods]
            ad_m.M_C_list = [(a != 0).astype(int) for a in ad_m.M_C_list]
            ad_m.M_T_list = [(a != 0).astype(int) for a in ad_m.M_T_list]
            for e_method in est_methods:
                method_name = f'Manhatten Matching {" ".join(e_method[0].split("_")).title()}'
                start = time.time()
                ad_m.MG(k=e_method[1])
                mg_time = time.time() - start
                start = time.time()
                ad_m.CATE(cate_methods=['mean' if e_method[0] == 'mean_pruned' else e_method[0]], augmented=False)
                times[method_name] = time.time() - start + fit_time + mg_time
                cate_df = ad_m.cate_df
                cate_df['true.CATE'] = df_true['TE'].to_numpy()
                cate_df['Relative Error (%)'] = np.abs(
                    (cate_df['avg.CATE'] - cate_df['true.CATE']) / np.abs(cate_df['true.CATE']).mean())
                cate_df['Method'] = [method_name for i in range(cate_df.shape[0])]
                cate_df['Iter'] = iter
                df_err = df_err.append(cate_df[['Method', 'Relative Error (%)', 'Iter']].copy(deep=True))
                if print_progress:
                    print(f'{method_name} method complete: {time.time() - start + fit_time + mg_time}')

        if 'propensity' in methods:
            method_name = 'Propensity Score Matching'
            start = time.time()
            ate_psnn, t_psnn = matchit.matchit('Y', 'T', data=df_data, method='nearest', replace=True)
            times[method_name] = time.time() - start
            df_err_psnn = pd.DataFrame()
            df_err_psnn['Method'] = [method_name for i in range(t_psnn.shape[0])]
            df_err_psnn['Relative Error (%)'] = np.abs((t_psnn['CATE'].to_numpy() - df_true['TE'].to_numpy())/np.abs(df_true['TE']).mean())
            df_err_psnn['Iter'] = iter
            df_err = df_err.append(df_err_psnn)
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
            df_err = df_err.append(df_err_gen)
            if print_progress:
                print(f'GenMatch complete: {time.time() - start}')

        if 'prognostic' in methods:
            for prog_method in prognostic_methods:
                method_name = f'{"Random Forest" if prog_method == "rf" else prog_method.title()} Prognostic Score Matching'
                start = time.time()
                cate_est_prog, _, _ = prognostic.prognostic_cv('Y', 'T', df_data, method=prog_method,
                                                               k_est=k_est_mean, gen_skf=split_strategy)
                times[method_name] = time.time() - start
                df_err_prog = pd.DataFrame()
                df_err_prog['Method'] = [method_name for i in range(cate_est_prog.shape[0])]
                df_err_prog['Relative Error (%)'] = np.abs((cate_est_prog['avg.CATE'].to_numpy() - df_true['TE'].to_numpy())/np.abs(df_true['TE']).mean())
                df_err_prog['Iter'] = iter
                df_err = df_err.append(df_err_prog)
                if print_progress:
                    print(f'{method_name} complete: {time.time() - start}')

        if 'bart' in methods:
            method_name = 'BART'
            start = time.time()
            cate_est_bart = bart.bart('Y', 'T', df_data, method='new', gen_skf=split_strategy)
            times[method_name] = time.time() - start
            df_err_bart = pd.DataFrame()
            df_err_bart['Method'] = [method_name for i in range(cate_est_bart.shape[0])]
            df_err_bart['Relative Error (%)'] = np.abs(
                (cate_est_bart['avg.CATE'].to_numpy() - df_true['TE'].to_numpy()) / np.abs(df_true['TE']).mean())
            df_err_bart['Iter'] = iter
            df_err = df_err.append(df_err_bart)
            if print_progress:
                print(f'{method_name} complete: {time.time() - start}')

        if 'causal_forest' in methods:
            method_name = 'Causal Forest'
            start = time.time()
            cate_est_cf = causalforest.causalforest('Y', 'T', df_data, gen_skf=split_strategy)
            times[method_name] = time.time() - start
            df_err_cf = pd.DataFrame()
            df_err_cf['Method'] = [method_name for i in range(cate_est_cf.shape[0])]
            df_err_cf['Relative Error (%)'] = np.abs((cate_est_cf['avg.CATE'].to_numpy() - df_true['TE'].to_numpy())/np.abs(df_true['TE']).mean())
            df_err_cf['Iter'] = iter
            df_err = df_err.append(df_err_cf)
            if print_progress:
                print(f'{method_name} complete: {time.time() - start}')

        df_err.loc[df_err['Iter'] == iter, 'Relative Error (%)'] = df_err.loc[df_err['Iter'] == iter,
                                                                              'Relative Error (%)'] * 100
        sns.set_context("paper")
        sns.set_style("darkgrid")
        sns.set(font_scale=6)
        fig, ax = plt.subplots(figsize=(40, 50))
        sns.boxenplot(x='Method', y='Relative Error (%)', data=df_err[df_err['Iter'] == iter])
        if include_title:
            if iter_in_title:
                plt.title(f'CATE Errors for {plot_name} = {iter}')
            else:
                plt.title(f'CATE Errors for {plot_name}')
        plt.xticks(rotation=65, horizontalalignment='right')
        ax.yaxis.set_major_formatter(ticker.PercentFormatter())
        plt.tight_layout()
        # plt.ylim([0, 100])
        fig.savefig(f'{save_folder}/boxplot_multifold_iter{iter:02d}.png')
        all_times.append(times)

        if print_progress:
            print(f'{data} iter {iter} total time: {time.time() - total_time}\n')

    print('Creating final plot and compiling times...')
    df_err = df_err.reset_index(drop=True)
    df_err.to_csv(f'{save_folder}/df_err.csv')
    sns.set_context("paper")
    sns.set_style("darkgrid")
    sns.set(font_scale=6)
    fig, ax = plt.subplots(figsize=(40, 50))
    df_err = df_err.rename(columns={'Iter': 'Iteration'})
    pp = sns.pointplot(data=df_err.reset_index(drop=True), x='Iteration', y='Relative Error (%)', hue='Method', ci='sd',
                       dodge=True, scale=5)
    plt.setp(pp.get_legend().get_texts(), fontsize='50')
    plt.setp(pp.get_legend().get_title(), fontsize='60')
    for line, name in zip(list(ax.lines)[::iters+1], df_err['Method'].unique().tolist()):
        y = line.get_ydata()[-1]
        x = line.get_xdata()[-1]
        if not np.isfinite(y):
            y = next(reversed(line.get_ydata()[~line.get_ydata().mask]), float("nan"))
        if not np.isfinite(y) or not np.isfinite(x):
            continue
        text = ax.annotate(name, xy=(x, y), xytext=(0, 0), color=line.get_color(), xycoords=(ax.get_xaxis_transform(), ax.get_yaxis_transform()), textcoords="offset points")
        text_width = (text.get_window_extent(fig.canvas.get_renderer()).transformed(ax.transData.inverted()).width)
        if np.isfinite(text_width):
            ax.set_xlim(ax.get_xlim()[0], text.xy[0] + text_width * 1.05)
    if include_title:
        plt.title(f'CATE Error for {plot_name} for {iters} iterations')
    ax.yaxis.set_major_formatter(ticker.PercentFormatter())
    plt.tight_layout()
    fig.savefig(f'{save_folder}/{iters}_iterations_lineplot.png')
    all_times = pd.DataFrame(all_times).T
    all_times['avg'] = all_times.mean(axis=1)
    all_times.to_csv(f'{save_folder}/times.csv')

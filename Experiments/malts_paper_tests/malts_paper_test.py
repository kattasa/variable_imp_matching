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

import warnings

warnings.filterwarnings("ignore")

# np.random.seed(0)

datasets = [
    'dense_continuous',
    # 'dense_discrete',
    # 'dense_mixed',
    # 'sine',
    # 'non_linear_mixed',
    # 'drop_off',
    # 'poly_no_interaction',
    # 'poly_interaction',
    # 'exp_log_interaction',
    # 'friedman',
    # 'ihdp',
    # 'acic'
]

malts_methods = ['mean', 'linear']
prognostic_methods = ['lasso']
methods = [
    # 'malts',
    'propensity',
    'prognostic',
    # 'genmatch',
    'bart',
    'causal_forest'
]

num_samples = 2500
n_splits = 5
n_repeats = 1
k_est_mean = 15
k_est_linear = 60
augment = True

print_progress = True

iters = 1
iter_name = 'Unimportant Covariates (log_2)'

for data in datasets:
    save_folder = create_folder(data, print_progress)

    config = {'n_splits': n_splits, 'n_repeats': n_repeats, 'k_est_mean': k_est_mean, 'k_est_linear': k_est_linear}

    df_err = pd.DataFrame(columns=['Method', 'Relative Error (%)', 'Iter'])
    all_times = []

    for iter in range(0, iters):
        total_time = time.time()

        nci = 10
        ncu = 20
        ndi = 0
        ndu = 0
        print(f'Imp: {nci}\nUnimp: {ncu}')

        df_data, df_true, discrete, config = get_data(data, num_samples, config, imp_c=nci, imp_d=ndi, unimp_c=ncu,
                                                      unimp_d=ndu, augment=augment)
        if augment:
            df_augmented = df_data.copy(deep=True)
            df_data = df_data.drop(columns=['Y_new'])

        with open(f'{save_folder}/config.txt', 'w') as f:
            json.dump(config, f, indent=2)

        times = {}

        start = time.time()
        ad_m = Amect_mf(outcome='Y', treatment='T', data=df_data, n_splits=n_splits, n_repeats=n_repeats)
        ad_m.fit()
        print('M_C')
        print(ad_m.M_C_list)
        print('M_T')
        print(ad_m.M_T_list)
        print(f'MC Nonzero weights: {[np.sum(z != 0) for z in ad_m.M_C_list]}')
        print(f'MT Nonzero weights: {[np.sum(z != 0) for z in ad_m.M_T_list]}')
        for e_method in [['mean', k_est_mean], ['linear_pruned', k_est_linear]]:
            ad_m.CATE(k=e_method[1], cate_methods=[e_method[0]], outcome='Y')
            times[f'AdMALTS Lasso {e_method[0]}'] = time.time() - start
            cate_df = ad_m.cate_df
            cate_df['true.CATE'] = df_true['TE'].to_numpy()
            cate_df['Relative Error (%)'] = np.abs((cate_df['avg.CATE']-cate_df['true.CATE'])/np.abs(cate_df['true.CATE']).mean())
            cate_df['Method'] = [f'AdMALTS Lasso {e_method[0]}' for i in range(cate_df.shape[0])]
            df_err_admalts = pd.DataFrame()
            df_err_admalts['Method'] = [f'AdMALTS Lasso {e_method[0]}' for i in range(cate_df.shape[0])]
            df_err_admalts['Relative Error (%)'] = np.abs((cate_df['avg.CATE']-cate_df['true.CATE'])/np.abs(cate_df['true.CATE']).mean())
            df_err_admalts['Iter'] = iter
            df_err = df_err.append(df_err_admalts.copy(deep=True))
            if print_progress:
                print(f'AdMALTS {e_method[0]} method complete: {time.time() - start}')
        # Augmented AdMALTS
        if augment:
            ad_m.col_order = [*ad_m.covariates, ad_m.treatment, ad_m.outcome, 'Y_new']
            ad_m.data = df_augmented[ad_m.col_order].reset_index(drop=True)
            for e_method in [['mean', k_est_mean], ['linear_pruned', k_est_linear]]:
                ad_m.CATE(k=e_method[1], cate_methods=[e_method[0]], outcome='Y_new')
                times[f'Augmented AdMALTS Lasso {e_method[0]}'] = time.time() - start
                cate_df = ad_m.cate_df
                cate_df['true.CATE'] = df_true['TE'].to_numpy()
                cate_df['Relative Error (%)'] = np.abs((cate_df['avg.CATE']-cate_df['true.CATE'])/np.abs(cate_df['true.CATE']).mean())
                cate_df['Method'] = [f'Augmented AdMALTS Lasso {e_method[0]}' for i in range(cate_df.shape[0])]
                df_err_admalts = pd.DataFrame()
                df_err_admalts['Method'] = [f'Augmented AdMALTS Lasso {e_method[0]}' for i in range(cate_df.shape[0])]
                df_err_admalts['Relative Error (%)'] = np.abs((cate_df['avg.CATE']-cate_df['true.CATE'])/np.abs(cate_df['true.CATE']).mean())
                df_err_admalts['Iter'] = iter
                df_err = df_err.append(df_err_admalts.copy(deep=True))
                if print_progress:
                    print(f'Augmented AdMALTS {e_method[0]} method complete: {time.time() - start}')

        if 'malts' in methods:
            est_methods = [[m, k_est_mean if m == 'mean' else k_est_linear] for m in malts_methods]
            for e_method in est_methods:
                start = time.time()
                if augment:
                    m = pymalts.malts_mf('Y', 'T', data=df_augmented, discrete=discrete, k_tr=15, k_est=e_method[1],
                                         n_splits=n_splits, estimator=e_method[0], smooth_cate=False, augment=True)
                else:
                    m = pymalts.malts_mf('Y', 'T', data=df_data, discrete=discrete, k_tr=15, k_est=e_method[1],
                                         n_splits=n_splits, estimator=e_method[0], smooth_cate=False, augment=False)
                times['MALTS'] = time.time() - start
                cate_df = m.CATE_df
                cate_df['true.CATE'] = df_true['TE'].to_numpy()
                cate_df['Relative Error (%)'] = np.abs(
                    (cate_df['avg.CATE'] - cate_df['true.CATE']) / np.abs(cate_df['true.CATE']).mean())
                cate_df['Method'] = [f'MALTS {e_method[0]}' for i in range(cate_df.shape[0])]
                df_err_malts = pd.DataFrame()
                df_err_malts['Method'] = [f'MALTS {e_method[0]}' for i in range(cate_df.shape[0])]
                df_err_malts['Relative Error (%)'] = np.abs(
                    (cate_df['avg.CATE'] - cate_df['true.CATE']) / np.abs(cate_df['true.CATE']).mean())
                df_err_malts['Iter'] = iter
                df_err = df_err.append(df_err_malts)
                if print_progress:
                    print(f'MALTS {e_method[0]} complete: {time.time() - start}')
                # Augmented MALTS
                if augment:
                    cate_df = m.augmented_CATE_df
                    cate_df['true.CATE'] = df_true['TE'].to_numpy()
                    cate_df['Relative Error (%)'] = np.abs(
                        (cate_df['avg.CATE'] - cate_df['true.CATE']) / np.abs(cate_df['true.CATE']).mean())
                    cate_df['Method'] = [f'Augmented MALTS {e_method[0]}' for i in range(cate_df.shape[0])]
                    df_err_malts = pd.DataFrame()
                    df_err_malts['Method'] = [f'Augmented MALTS {e_method[0]}' for i in range(cate_df.shape[0])]
                    df_err_malts['Relative Error (%)'] = np.abs(
                        (cate_df['avg.CATE'] - cate_df['true.CATE']) / np.abs(cate_df['true.CATE']).mean())
                    df_err_malts['Iter'] = iter
                    df_err = df_err.append(df_err_malts)
                    if print_progress:
                        print(f'Augmented MALTS {e_method[0]} complete: {time.time() - start}')

        if 'propensity' in methods:
            start = time.time()
            ate_psnn, t_psnn = matchit.matchit('Y', 'T', data=df_data, method='nearest', replace=True)
            times['propensity'] = time.time() - start
            df_err_psnn = pd.DataFrame()
            df_err_psnn['Method'] = ['Propensity Score' for i in range(t_psnn.shape[0])]
            df_err_psnn['Relative Error (%)'] = np.abs((t_psnn['CATE'].to_numpy() - df_true['TE'].to_numpy())/np.abs(df_true['TE']).mean())
            df_err_psnn['Iter'] = iter
            df_err = df_err.append(df_err_psnn)
            if print_progress:
                print(f'Propensity Score complete: {time.time() - start}')

        if 'genmatch' in methods:
            start = time.time()
            ate_gen, t_gen = matchit.matchit('Y', 'T', data=df_data, method='genetic', replace=True)
            times['genmatch'] = time.time() - start
            df_err_gen = pd.DataFrame()
            df_err_gen['Method'] = ['GenMatch' for i in range(t_gen.shape[0])]
            df_err_gen['Relative Error (%)'] = np.abs((t_gen['CATE'].to_numpy() - df_true['TE'].to_numpy())/np.abs(df_true['TE']).mean())
            df_err_gen['Iter'] = iter
            df_err = df_err.append(df_err_gen)
            if print_progress:
                print(f'GenMatch complete: {time.time() - start}')

        if 'prognostic' in methods:
            for prog_method in prognostic_methods:
                for e_method in ['smooth']:
                    start = time.time()
                    cate_est_prog, prog_mgs = prognostic.prognostic_cv('Y', 'T', df_data, method=prog_method,
                                                                       k_est=k_est_mean, est_method=e_method,
                                                                       n_splits=n_splits)
                    times[f'prognostic_{prog_method}'] = time.time() - start
                    df_err_prog = pd.DataFrame()
                    df_err_prog['Method'] = [f'Prognostic Score {prog_method}' for i in range(cate_est_prog.shape[0])]
                    df_err_prog['Relative Error (%)'] = np.abs((cate_est_prog['avg.CATE'].to_numpy() - df_true['TE'].to_numpy())/np.abs(df_true['TE']).mean())
                    df_err_prog['Iter'] = iter
                    df_err = df_err.append(df_err_prog)
                    if print_progress:
                        print(f'Prognostic Score {prog_method} complete: {time.time() - start}')

        if 'bart' in methods:
            start = time.time()
            cate_est_bart = bart.bart('Y', 'T', df_data, n_splits=2, method='new')
            times[f'BART'] = time.time() - start
            df_err_bart = pd.DataFrame()
            df_err_bart['Method'] = [f'BART' for i in range(cate_est_bart.shape[0])]
            df_err_bart['Relative Error (%)'] = np.abs(
                (cate_est_bart['avg.CATE'].to_numpy() - df_true['TE'].to_numpy()) / np.abs(df_true['TE']).mean())
            df_err_bart['Iter'] = iter
            df_err = df_err.append(df_err_bart)
            if print_progress:
                print(f'BART complete: {time.time() - start}')

        if 'causal_forest' in methods:
            start = time.time()
            cate_est_cf = causalforest.causalforest('Y', 'T', df_data, n_splits=2)
            times['causal_forest'] = time.time() - start
            df_err_cf = pd.DataFrame()
            df_err_cf['Method'] = ['Causal Forest' for i in range(cate_est_cf.shape[0])]
            df_err_cf['Relative Error (%)'] = np.abs((cate_est_cf['avg.CATE'].to_numpy() - df_true['TE'].to_numpy())/np.abs(df_true['TE']).mean())
            df_err_cf['Iter'] = iter
            df_err = df_err.append(df_err_cf)
            if print_progress:
                print(f'Causal Forest complete: {time.time() - start}')

        df_err.loc[df_err['Iter'] == iter, 'Relative Error (%)'] = df_err.loc[df_err['Iter'] == iter,
                                                                              'Relative Error (%)'] * 100
        sns.set_context("paper")
        sns.set_style("darkgrid")
        sns.set(font_scale=6)
        fig, ax = plt.subplots(figsize=(40, 50))
        sns.boxenplot(x='Method', y='Relative Error (%)', data=df_err[df_err['Iter'] == iter])
        plt.title(f'CATE Errors for {iter_name} = {iter}')
        plt.xticks(rotation=65, horizontalalignment='right')
        ax.yaxis.set_major_formatter(ticker.PercentFormatter())
        plt.tight_layout()
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
    df_err = df_err.rename(columns={'Iter': iter_name})
    pp = sns.pointplot(data=df_err.reset_index(drop=True), x=iter_name, y='Relative Error (%)', hue='Method', ci='sd',
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
    plt.title(f'CATE Error vs {iter_name}')
    # plt.xticks(rotation=65, horizontalalignment='right')
    ax.yaxis.set_major_formatter(ticker.PercentFormatter())
    plt.tight_layout()
    fig.savefig(f'{save_folder}/{iters}_iterations_lineplot.png')
    all_times = pd.DataFrame(all_times).T
    all_times['avg'] = all_times.mean(axis=1)
    all_times.to_csv(f'{save_folder}/times.csv')

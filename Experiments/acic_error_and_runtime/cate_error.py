# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 01:31:42 2020
@author: Harsh
"""

import json
import numpy as np
import os
import pandas as pd
import time
import warnings

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

from Experiments.helpers import get_acic_data, summarize_warnings, lcm_to_malts_weights, weights_to_feature_selection
from other_methods import pymalts, bart, causalforest, prognostic, doubleml, drlearner, causalforest_dml
from src.linear_coef_matching_mf import LCM_MF
import pickle

np.random.seed(1)
random_state = 1

acic_year = os.getenv('ACIC_YEAR').replace("'", '').replace('"', '')
acic_file = os.getenv('ACIC_FILE').replace("'", '').replace('"', '')
k_est = int(os.getenv('K_EST'))
save_folder = os.getenv('SAVE_FOLDER').replace("'", '').replace('"', '')
n_splits = int(os.getenv('N_SPLITS'))
n_samples_per_split = int(os.getenv('N_SAMPLES_PER_SPLIT'))
malts_max = int(os.getenv('MALTS_MAX'))
malts_k_train = 15

df_err = pd.DataFrame(columns=['Method', 'True_CATE', 'Est_CATE', 'Relative Error (%)'])

print(f'Running {acic_year} file {acic_file}...')

total_time = time.time()

df_data, df_true, binary, categorical, dummy_cols, categorical_to_dummy = get_acic_data(year=acic_year, file=acic_file, n_train=0)
df_true.to_csv(f'{save_folder}/df_true.csv')

new_n_splits = df_data.shape[0] // n_samples_per_split
n_splits = max(min(new_n_splits, 10), n_splits)

run_malts = True
if df_data.shape[0] > malts_max:
    print(f'**Not running malts. > {malts_max} samples.')
    run_malts = False

run_bart = True
if acic_year == 'acic_2018' and acic_file == 'd09f96200455407db569ae33fe06b0d3':
    print('**Not running bart. BART fails to create predictions due to small size of treated group.')
    run_bart = False

config = {'n_splits': n_splits, 'k_est': k_est, 'run_malts': run_malts, 'run_bart': run_bart}

with open(f'{save_folder}/config.txt', 'w') as f:
    json.dump(config, f, indent=2)

df_dummy_data = df_data.copy(deep=True)
if dummy_cols is not None:
    df_data = df_data.drop(columns=dummy_cols)
    df_dummy_data = df_dummy_data.drop(columns=categorical)
    with open(f'{save_folder}/dummy_cols.txt', 'w') as f:
        f.write(str(dummy_cols))
    df_dummy_data.to_csv(f'{save_folder}/df_dummy_data.csv')
df_data.to_csv(f'{save_folder}/df_data.csv')
with open(f'{save_folder}/binary_cols.txt', 'w') as f:
    f.write(str(binary))
with open(f'{save_folder}/categorical_cols.txt', 'w') as f:
    f.write(str(categorical))


times = {}

method_name = 'LASSO Coefficient Matching'
start = time.time()
with warnings.catch_warnings(record=True) as warning_list:
    lcm = LCM_MF(outcome='Y', treatment='T', data=df_dummy_data, n_splits=n_splits, n_repeats=1, random_state=random_state)
    lcm.fit(method='linear')
    lcm.MG(k=k_est)
    lcm.CATE(cate_methods=[['mean', False]])
times[method_name] = time.time() - start
cate_df = lcm.cate_df
cate_df = cate_df.rename(columns={'avg.CATE': 'Est_CATE'})
cate_df['True_CATE'] = df_true['TE'].to_numpy()
cate_df['Relative Error (%)'] = np.abs((cate_df['Est_CATE']-cate_df['True_CATE'])/np.abs(cate_df['True_CATE']).mean())
cate_df['Method'] = [method_name for i in range(cate_df.shape[0])]
df_err = pd.concat([df_err, cate_df[['Method', 'True_CATE', 'Est_CATE', 'Relative Error (%)']].copy(deep=True)])
print(f'\n{method_name} method complete: {time.time() - start}')
summarize_warnings(warning_list, method_name)
print()

split_strategy = lcm.gen_skf  # save split strategy to use for all other methods
with open(f'{save_folder}/split.pkl', 'wb') as f:
    pickle.dump(split_strategy, f)

# lasso_malts_init = lcm_to_malts_weights(lcm, [c for c in df_data.columns if c not in ['Y', 'T']], categorical_to_dummy)
# if run_malts:
#     method_name = 'MALTS Matching'
#     start = time.time()
#     with warnings.catch_warnings(record=True) as warning_list:
#         m = pymalts.malts_mf('Y', 'T', data=df_data, discrete=binary+categorical, k_tr=malts_k_train, k_est=k_est,
#                              n_splits=n_splits, estimator='single_linear', smooth_cate=False,
#                              gen_skf=split_strategy, random_state=random_state)
#     times[method_name] = time.time() - start
#     cate_df = m.CATE_df
#     cate_df = cate_df.rename(columns={'avg.CATE': 'Est_CATE'})
#     cate_df['True_CATE'] = df_true['TE'].to_numpy()
#     cate_df['Relative Error (%)'] = np.abs(
#         (cate_df['Est_CATE'] - cate_df['True_CATE']) / np.abs(cate_df['True_CATE']).mean())
#     cate_df['Method'] = [method_name for i in range(cate_df.shape[0])]
#     df_err = pd.concat([df_err, cate_df[['Method', 'True_CATE', 'Est_CATE', 'Relative Error (%)']].copy(deep=True)])
#     print(f'{method_name} complete: {time.time() - start}')
#     summarize_warnings(warning_list, method_name)
#     # malts_m_opt = m.M_opt_list
#     print()
#
#     method_name = 'MALTS Matching with LASSO Weights'
#     start = time.time()
#     with warnings.catch_warnings(record=True) as warning_list:
#         m = pymalts.malts_mf('Y', 'T', data=df_data, discrete=binary+categorical, k_tr=malts_k_train, k_est=k_est,
#                              n_splits=n_splits, estimator='single_linear', smooth_cate=False,
#                              gen_skf=split_strategy,
#                              M_init=lasso_malts_init,
#                              random_state=random_state)
#     times[method_name] = time.time() - start
#     cate_df = m.CATE_df
#     cate_df = cate_df.rename(columns={'avg.CATE': 'Est_CATE'})
#     cate_df['True_CATE'] = df_true['TE'].to_numpy()
#     cate_df['Relative Error (%)'] = np.abs(
#         (cate_df['Est_CATE'] - cate_df['True_CATE']) / np.abs(cate_df['True_CATE']).mean())
#     cate_df['Method'] = [method_name for i in range(cate_df.shape[0])]
#     df_err = pd.concat([df_err, cate_df[['Method', 'True_CATE', 'Est_CATE', 'Relative Error (%)']].copy(deep=True)])
#     print(f'{method_name} complete: {time.time() - start}')
#     summarize_warnings(warning_list, method_name)
#     # lasso_weights_malts_m_opt = m.M_opt_list
#     print()
#
# lasso_feature_selection = weights_to_feature_selection(
#                              lasso_malts_init, [c for c in df_data.columns if c not in ['Y', 'T']])
# method_name = 'MALTS Matching with LASSO Feature Selection'
# start = time.time()
# with warnings.catch_warnings(record=True) as warning_list:
#     m = pymalts.malts_mf('Y', 'T', data=df_data, discrete=binary + categorical, k_tr=malts_k_train, k_est=k_est,
#                          n_splits=n_splits, estimator='single_linear', smooth_cate=False,
#                          gen_skf=split_strategy,
#                          trim_features=lasso_feature_selection,
#                          random_state=random_state)
# times[method_name] = time.time() - start
# cate_df = m.CATE_df
# cate_df = cate_df.rename(columns={'avg.CATE': 'Est_CATE'})
# cate_df['True_CATE'] = df_true['TE'].to_numpy()
# cate_df['Relative Error (%)'] = np.abs(
#     (cate_df['Est_CATE'] - cate_df['True_CATE']) / np.abs(cate_df['True_CATE']).mean())
# cate_df['Method'] = [method_name for i in range(cate_df.shape[0])]
# df_err = pd.concat([df_err, cate_df[['Method', 'True_CATE', 'Est_CATE', 'Relative Error (%)']].copy(deep=True)])
# print(f'{method_name} complete: {time.time() - start}')
# summarize_warnings(warning_list, method_name)
# print()
#
# # generic_malts = pymalts.malts('Y', 'T', data=df_data, discrete=binary+categorical, k=malts_k_train)
# # print('MALTS Objective Lossess:')
# # print(f'LCM: {[generic_malts.objective(w) for w in lasso_malts_init]}')
# # if run_malts:
# #     print(f'MALTS: {[generic_malts.objective(w.to_numpy().reshape(-1,)) for w in malts_m_opt]}')
# #     print(f'LASSO Initialized MALTS: {[generic_malts.objective(w.to_numpy().reshape(-1,)) for w in lasso_weights_malts_m_opt]}')
# # print(f"LASSO Feature Selection MALTS: "
# #       f"{[pymalts.malts('Y', 'T', data=df_data[lasso_feature_selection[i] + ['Y', 'T']], discrete=[c for c in binary+categorical if c in lasso_feature_selection[i]], k=malts_k_train).objective(m.M_opt_list[i].to_numpy().reshape(-1,)) for i in range(len(m.M_opt_list))]}"
# #       f"")
#
#
# method_name = 'Tree Feature Importance Matching'
# start = time.time()
# with warnings.catch_warnings(record=True) as warning_list:
#     lcm = LCM_MF(outcome='Y', treatment='T', data=df_dummy_data, n_splits=n_splits, n_repeats=1, random_state=random_state)
#     lcm.gen_skf = split_strategy
#     lcm.fit(method='tree', params={'max_depth': 4})
#     lcm.MG(k=k_est)
#     lcm.CATE(cate_methods=[['linear_pruned', False]])
# times[method_name] = time.time() - start
# cate_df = lcm.cate_df
# cate_df = cate_df.rename(columns={'avg.CATE': 'Est_CATE'})
# cate_df['True_CATE'] = df_true['TE'].to_numpy()
# cate_df['Relative Error (%)'] = np.abs((cate_df['Est_CATE']-cate_df['True_CATE'])/np.abs(cate_df['True_CATE']).mean())
# cate_df['Method'] = [method_name for i in range(cate_df.shape[0])]
# df_err = pd.concat([df_err, cate_df[['Method', 'True_CATE', 'Est_CATE', 'Relative Error (%)']].copy(deep=True)])
# print(f'{method_name} method complete: {time.time() - start}')
# summarize_warnings(warning_list, method_name)
# print()
#
# method_name = 'Equal Weighted LASSO Matching'
# start = time.time()
# with warnings.catch_warnings(record=True) as warning_list:
#     lcm = LCM_MF(outcome='Y', treatment='T', data=df_dummy_data, n_splits=n_splits, n_repeats=1, random_state=random_state)
#     lcm.gen_skf = split_strategy
#     lcm.fit(method='linear', equal_weights=True)
#     lcm.MG(k=k_est)
#     lcm.CATE(cate_methods=[['linear_pruned', False]])
# times[method_name] = time.time() - start
#
# cate_df = lcm.cate_df
# cate_df = cate_df.rename(columns={'avg.CATE': 'Est_CATE'})
# cate_df['True_CATE'] = df_true['TE'].to_numpy()
# cate_df['Relative Error (%)'] = np.abs((cate_df['Est_CATE']-cate_df['True_CATE'])/np.abs(cate_df['True_CATE']).mean())
# cate_df['Method'] = [method_name for i in range(cate_df.shape[0])]
# df_err = pd.concat([df_err, cate_df[['Method', 'True_CATE', 'Est_CATE', 'Relative Error (%)']].copy(deep=True)])
# print(f'{method_name} method complete: {time.time() - start}')
# summarize_warnings(warning_list, method_name)
# print()

method_name = 'Prognostic Score Matching'
start = time.time()
with warnings.catch_warnings(record=True) as warning_list:
    cate_est_prog, _, _ = prognostic.prognostic_cv('Y', 'T', df_dummy_data,
                                                   k_est=k_est, gen_skf=split_strategy, random_state=random_state)
times[method_name] = time.time() - start
df_err_prog = pd.DataFrame()
df_err_prog['Method'] = [method_name for i in range(cate_est_prog.shape[0])]
df_err_prog['Relative Error (%)'] = np.abs((cate_est_prog['avg.CATE'].to_numpy() - df_true['TE'].to_numpy())/np.abs(df_true['TE']).mean())
df_err_prog['True_CATE'] = df_true['TE'].to_numpy()
df_err_prog['Est_CATE'] = cate_est_prog['avg.CATE'].to_numpy()
df_err = pd.concat([df_err, df_err_prog[['Method', 'True_CATE', 'Est_CATE', 'Relative Error (%)']].copy(deep=True)])
print(f'{method_name} complete: {time.time() - start}')
summarize_warnings(warning_list, method_name)
print()

# method_name = 'DoubleML'
# start = time.time()
# with warnings.catch_warnings(record=True) as warning_list:
#     cate_est_doubleml = doubleml.doubleml('Y', 'T', df_dummy_data, gen_skf=split_strategy, random_state=random_state)
# times[method_name] = time.time() - start
# df_err_doubleml = pd.DataFrame()
# df_err_doubleml['Method'] = [method_name for i in range(cate_est_doubleml.shape[0])]
# df_err_doubleml['Relative Error (%)'] = np.abs(
#     (cate_est_doubleml['avg.CATE'].to_numpy() - df_true['TE'].to_numpy()) / np.abs(df_true['TE']).mean())
# df_err_doubleml['True_CATE'] = df_true['TE'].to_numpy()
# df_err_doubleml['Est_CATE'] = cate_est_doubleml['avg.CATE'].to_numpy()
# df_err = pd.concat([df_err, df_err_doubleml[['Method', 'True_CATE', 'Est_CATE', 'Relative Error (%)']].copy(deep=True)])
# print(f'{method_name} complete: {time.time() - start}')
# summarize_warnings(warning_list, method_name)
# print()
#
# method_name = 'DRLearner'
# start = time.time()
# with warnings.catch_warnings(record=True) as warning_list:
#     cate_est_drlearner = drlearner.drlearner('Y', 'T', df_dummy_data, gen_skf=split_strategy, random_state=random_state)
# times[method_name] = time.time() - start
#
# df_err_drlearner = pd.DataFrame()
# df_err_drlearner['Method'] = [method_name for i in range(cate_est_drlearner.shape[0])]
# df_err_drlearner['Relative Error (%)'] = np.abs(
#     (cate_est_drlearner['avg.CATE'].to_numpy() - df_true['TE'].to_numpy()) / np.abs(df_true['TE']).mean())
# df_err_drlearner['True_CATE'] = df_true['TE'].to_numpy()
# df_err_drlearner['Est_CATE'] = cate_est_drlearner['avg.CATE'].to_numpy()
# df_err = pd.concat([df_err, df_err_drlearner[['Method', 'True_CATE', 'Est_CATE', 'Relative Error (%)']].copy(deep=True)])
# print(f'{method_name} complete: {time.time() - start}')
# summarize_warnings(warning_list, method_name)
# print()
#
# if run_bart:
#     method_name = 'BART'
#     start = time.time()
#     with warnings.catch_warnings(record=True) as warning_list:
#         cate_est_bart = bart.bart('Y', 'T', df_dummy_data, gen_skf=split_strategy, random_state=random_state)
#     times[method_name] = time.time() - start
#     df_err_bart = pd.DataFrame()
#     df_err_bart['Method'] = [method_name for i in range(cate_est_bart.shape[0])]
#     df_err_bart['Relative Error (%)'] = np.abs(
#         (cate_est_bart['avg.CATE'].to_numpy() - df_true['TE'].to_numpy()) / np.abs(df_true['TE']).mean())
#     df_err_bart['True_CATE'] = df_true['TE'].to_numpy()
#     df_err_bart['Est_CATE'] = cate_est_bart['avg.CATE'].to_numpy()
#     df_err = pd.concat([df_err, df_err_bart[['Method', 'True_CATE', 'Est_CATE', 'Relative Error (%)']].copy(deep=True)])
#     print(f'{method_name} complete: {time.time() - start}')
#     summarize_warnings(warning_list, method_name)
#     print()
#
# method_name = 'Causal Forest'
# start = time.time()
# with warnings.catch_warnings(record=True) as warning_list:
#     cate_est_cf = causalforest.causalforest('Y', 'T', df_dummy_data, gen_skf=split_strategy, random_state=random_state)
# times[method_name] = time.time() - start
# df_err_cf = pd.DataFrame()
# df_err_cf['Method'] = [method_name for i in range(cate_est_cf.shape[0])]
# df_err_cf['Relative Error (%)'] = np.abs((cate_est_cf['avg.CATE'].to_numpy() - df_true['TE'].to_numpy())/np.abs(df_true['TE']).mean())
# df_err_cf['True_CATE'] = df_true['TE'].to_numpy()
# df_err_cf['Est_CATE'] = cate_est_cf['avg.CATE'].to_numpy()
# df_err = pd.concat([df_err, df_err_cf[['Method', 'True_CATE', 'Est_CATE', 'Relative Error (%)']].copy(deep=True)])
# print(f'{method_name} complete: {time.time() - start}')
# summarize_warnings(warning_list, method_name)
# print()
#
# method_name = 'Causal Forest DML'
# start = time.time()
# with warnings.catch_warnings(record=True) as warning_list:
#     cate_est_cf = causalforest_dml.causalforest_dml('Y', 'T', df_dummy_data, gen_skf=split_strategy, random_state=random_state)
# times[method_name] = time.time() - start
# df_err_cf = pd.DataFrame()
# df_err_cf['Method'] = [method_name for i in range(cate_est_cf.shape[0])]
# df_err_cf['Relative Error (%)'] = np.abs((cate_est_cf['avg.CATE'].to_numpy() - df_true['TE'].to_numpy())/np.abs(df_true['TE']).mean())
# df_err_cf['True_CATE'] = df_true['TE'].to_numpy()
# df_err_cf['Est_CATE'] = cate_est_cf['avg.CATE'].to_numpy()
# df_err = pd.concat([df_err, df_err_cf[['Method', 'True_CATE', 'Est_CATE', 'Relative Error (%)']].copy(deep=True)])
# print(f'{method_name} complete: {time.time() - start}')
# summarize_warnings(warning_list, method_name)
# print()

df_err.loc[:, 'Relative Error (%)'] = df_err.loc[:, 'Relative Error (%)'] * 100

sns.set_context("paper")
sns.set_style("darkgrid")
sns.set(font_scale=6)
fig, ax = plt.subplots(figsize=(40, 50))
sns.boxenplot(x='Method', y='Relative Error (%)', data=df_err)
plt.xticks(rotation=65, horizontalalignment='right')
ax.yaxis.set_major_formatter(ticker.PercentFormatter())
plt.tight_layout()
fig.savefig(f'{save_folder}/boxplot_multifold_iter.png')

print('Saving all results...')

df_err = df_err.reset_index(drop=True)
df_err.to_csv(f'{save_folder}/df_err.csv')
times = pd.DataFrame([times]).T
times.to_csv(f'{save_folder}/times.csv')

print(f'Total time: {time.time() - total_time}\n')

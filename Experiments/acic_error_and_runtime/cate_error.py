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

from Experiments.helpers import get_acic_data
from other_methods import pymalts, bart, causalforest, prognostic
from src.linear_coef_matching_mf import LCM_MF
import pickle

warnings.filterwarnings("ignore")
np.random.seed(0)


acic_year = os.getenv('ACIC_YEAR').replace("'", '').replace('"', '')
acic_file = os.getenv('ACIC_FILE').replace("'", '').replace('"', '')
k_est = int(os.getenv('K_EST'))
save_folder = os.getenv('SAVE_FOLDER').replace("'", '').replace('"', '')
n_splits = int(os.getenv('N_SPLITS'))
n_samples_per_split = int(os.getenv('N_SAMPLES_PER_SPLIT'))
malts_max = int(os.getenv('MALTS_MAX'))

nn_retries = 5  # for some reason I keep getting a KNN internal error. until i found cause, try retries

config = {'n_splits': n_splits, 'k_est': k_est}

with open(f'{save_folder}/config.txt', 'w') as f:
    json.dump(config, f, indent=2)

df_err = pd.DataFrame(columns=['Method', 'True_CATE', 'Est_CATE', 'Relative Error (%)'])

print(f'Running {acic_year} file {acic_file}...')

total_time = time.time()

df_data, df_true, binary, categorical, dummy_cols = get_acic_data(year=acic_year, file=acic_file, n_train=0)
df_true.to_csv(f'{save_folder}/df_true.csv')

if acic_year == 'acic_2018':
    new_n_splits = df_data.shape[0] // n_samples_per_split
    n_splits = max(min(new_n_splits, 10), n_splits)

run_malts = True
if df_data.shape[0] > malts_max:
    print(f'**Not running malts. > {malts_max} samples.')
    run_malts = False


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
lcm = LCM_MF(outcome='Y', treatment='T', data=df_dummy_data, n_splits=n_splits, n_repeats=1)
lcm.fit(double_model=False)
attempt = 0
while attempt < nn_retries:
    try:
        lcm.MG(k=k_est)
        break
    except RuntimeError as e:
        print(f'{method_name} attempt {attempt+1}: {e}')
        attempt +=1
        if attempt == nn_retries:
            raise e
lcm.CATE(cate_methods=[['linear_pruned', False]])
times[method_name] = time.time() - start

cate_df = lcm.cate_df
cate_df = cate_df.rename(columns={'avg.CATE': 'Est_CATE'})
cate_df['True_CATE'] = df_true['TE'].to_numpy()
cate_df['Relative Error (%)'] = np.abs((cate_df['Est_CATE']-cate_df['True_CATE'])/np.abs(cate_df['True_CATE']).mean())
cate_df['Method'] = [method_name for i in range(cate_df.shape[0])]
df_err = df_err.append(cate_df[['Method', 'True_CATE', 'Est_CATE', 'Relative Error (%)']].copy(deep=True))
print(f'{method_name} method complete: {time.time() - start}')

split_strategy = lcm.gen_skf  # save split strategy to use for all other methods
with open(f'{save_folder}/split.pkl', 'wb') as f:
    pickle.dump(split_strategy, f)


# method_name = 'DoubleML'
# start = time.time()
# cate_est_doubleml = doubleml.doubleml('Y', 'T', df_dummy_data, gen_skf=split_strategy)
# times[method_name] = time.time() - start

# df_err_bart = pd.DataFrame()
# df_err_bart['Method'] = [method_name for i in range(cate_est_bart.shape[0])]
# df_err_bart['Relative Error (%)'] = np.abs(
#     (cate_est_bart['avg.CATE'].to_numpy() - df_true['TE'].to_numpy()) / np.abs(df_true['TE']).mean())
# df_err_bart['True_CATE'] = df_true['TE'].to_numpy()
# df_err_bart['Est_CATE'] = cate_est_bart['avg.CATE'].to_numpy()
# df_err = df_err.append(df_err_bart[['Method', 'True_CATE', 'Est_CATE', 'Relative Error (%)']])
# print(f'{method_name} complete: {time.time() - start}')


if run_malts:
    method_name = 'MALTS Matching'
    attempt = 0
    while attempt < nn_retries:
        try:
            start = time.time()
            # lcm2 = LCM_MF(outcome='Y', treatment='T', data=df_data, n_splits=n_splits, n_repeats=1)
            # lcm2.gen_skf = split_strategy
            # lcm2.fit(double_model=False)
            m = pymalts.malts_mf('Y', 'T', data=df_dummy_data, discrete=dummy_cols, k_tr=15, k_est=k_est,
                                 n_splits=n_splits, estimator='linear', smooth_cate=False,
                                 gen_skf=split_strategy, M_init=lcm.M_list)
            times[method_name] = time.time() - start
            break
        except RuntimeError as e:
            print(f'{method_name} attempt {attempt+1}: {e}')
            attempt += 1
            if attempt == nn_retries:
                raise e
    cate_df = m.CATE_df
    cate_df = cate_df.rename(columns={'avg.CATE': 'Est_CATE'})
    cate_df['True_CATE'] = df_true['TE'].to_numpy()
    cate_df['Relative Error (%)'] = np.abs(
        (cate_df['Est_CATE'] - cate_df['True_CATE']) / np.abs(cate_df['True_CATE']).mean())
    cate_df['Method'] = [method_name for i in range(cate_df.shape[0])]
    df_err = df_err.append(cate_df[['Method', 'True_CATE', 'Est_CATE', 'Relative Error (%)']].copy(deep=True))
    print(f'{method_name} complete: {time.time() - start}')

method_name = 'Prognostic Score Matching'
attempt = 0
while attempt < nn_retries:
    try:
        start = time.time()
        cate_est_prog, _, _ = prognostic.prognostic_cv('Y', 'T', df_dummy_data,
                                                       k_est=k_est, gen_skf=split_strategy)
        times[method_name] = time.time() - start
        break
    except RuntimeError as e:
        print(f'{method_name} attempt {attempt + 1}: {e}')
        attempt += 1
        if attempt == nn_retries:
            raise e
df_err_prog = pd.DataFrame()
df_err_prog['Method'] = [method_name for i in range(cate_est_prog.shape[0])]
df_err_prog['Relative Error (%)'] = np.abs((cate_est_prog['avg.CATE'].to_numpy() - df_true['TE'].to_numpy())/np.abs(df_true['TE']).mean())
df_err_prog['True_CATE'] = df_true['TE'].to_numpy()
df_err_prog['Est_CATE'] = cate_est_prog['avg.CATE'].to_numpy()
df_err = df_err.append(df_err_prog[['Method', 'True_CATE', 'Est_CATE', 'Relative Error (%)']])
print(f'{method_name} complete: {time.time() - start}')

method_name = 'BART'
start = time.time()
cate_est_bart = bart.bart('Y', 'T', df_dummy_data, gen_skf=split_strategy)
times[method_name] = time.time() - start

df_err_bart = pd.DataFrame()
df_err_bart['Method'] = [method_name for i in range(cate_est_bart.shape[0])]
df_err_bart['Relative Error (%)'] = np.abs(
    (cate_est_bart['avg.CATE'].to_numpy() - df_true['TE'].to_numpy()) / np.abs(df_true['TE']).mean())
df_err_bart['True_CATE'] = df_true['TE'].to_numpy()
df_err_bart['Est_CATE'] = cate_est_bart['avg.CATE'].to_numpy()
df_err = df_err.append(df_err_bart[['Method', 'True_CATE', 'Est_CATE', 'Relative Error (%)']])
print(f'{method_name} complete: {time.time() - start}')

method_name = 'Causal Forest'
start = time.time()
cate_est_cf = causalforest.causalforest('Y', 'T', df_dummy_data, gen_skf=split_strategy)
times[method_name] = time.time() - start

df_err_cf = pd.DataFrame()
df_err_cf['Method'] = [method_name for i in range(cate_est_cf.shape[0])]
df_err_cf['Relative Error (%)'] = np.abs((cate_est_cf['avg.CATE'].to_numpy() - df_true['TE'].to_numpy())/np.abs(df_true['TE']).mean())
df_err_cf['True_CATE'] = df_true['TE'].to_numpy()
df_err_cf['Est_CATE'] = cate_est_cf['avg.CATE'].to_numpy()
df_err = df_err.append(df_err_cf[['Method', 'True_CATE', 'Est_CATE', 'Relative Error (%)']])
print(f'{method_name} complete: {time.time() - start}')

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

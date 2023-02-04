# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 01:31:42 2020
@author: Harsh
"""

import json
import numpy as np
import os
import pandas as pd
import pickle
import time
import warnings

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

from Experiments.helpers import get_acic_data, summarize_warnings, get_errors
from other_methods import bart, causalforest, prognostic, doubleml, \
    drlearner, causalforest_dml, pymalts
from src.linear_coef_matching_mf import LCM_MF

np.random.seed(1)
random_state = 1

acic_year = os.getenv('ACIC_YEAR').replace("'", '').replace('"', '')
acic_file = os.getenv('ACIC_FILE').replace("'", '').replace('"', '')
k_est_per_500 = int(os.getenv('K_EST_PER_500'))
k_est_max = int(os.getenv('K_EST_MAX'))
save_folder = os.getenv('SAVE_FOLDER').replace("'", '').replace('"', '')
min_n_splits = int(os.getenv('MIN_N_SPLITS'))
max_n_splits = int(os.getenv('MAX_N_SPLITS'))
n_samples_per_split = int(os.getenv('N_SAMPLES_PER_SPLIT'))
n_repeats = int(os.getenv('N_REPEATS'))
malts_max = int(os.getenv('MALTS_MAX'))

df_err = pd.DataFrame(columns=['Method', 'True_CATE', 'Est_CATE', 'Relative Error (%)'])

print(f'Running {acic_year} file {acic_file}...')

total_time = time.time()

df_data, df_true, binary, categorical, dummy_cols, categorical_to_dummy = get_acic_data(year=acic_year, file=acic_file, n_train=0)
# df_true.to_csv(f'{save_folder}/df_true.csv')

new_n_splits = df_data.shape[0] // n_samples_per_split
n_splits = max(min(new_n_splits, max_n_splits), min_n_splits)
k_est = min(k_est_max, int(k_est_per_500 * (((df_data.shape[0] // n_splits) * (n_splits - 1)) / 500)))

run_malts = True
if df_data.shape[0] > malts_max:
    print(f'**Not running malts. > {malts_max} samples.')
    run_malts = False

run_bart = True
if acic_year == 'acic_2018' and acic_file == 'd09f96200455407db569ae33fe06b0d3':
    print('**Not running bart. BART fails to create predictions due to small '
          'size of treated group.')
    run_bart = False

config = {'n_splits': n_splits, 'k_est': k_est, 'run_bart': run_bart,
          'run_malts': run_malts}

with open(f'{save_folder}/config.txt', 'w') as f:
    json.dump(config, f, indent=2)

df_dummy_data = df_data.copy(deep=True)
if dummy_cols is not None:
    df_data = df_data.drop(columns=dummy_cols)
    df_dummy_data = df_dummy_data.drop(columns=categorical)
    # with open(f'{save_folder}/dummy_cols.txt', 'w') as f:
    #     f.write(str(dummy_cols))
    # df_dummy_data.to_csv(f'{save_folder}/df_dummy_data.csv')
# df_data.to_csv(f'{save_folder}/df_data.csv')
# with open(f'{save_folder}/binary_cols.txt', 'w') as f:
#     f.write(str(binary))
# with open(f'{save_folder}/categorical_cols.txt', 'w') as f:
#     f.write(str(categorical))


times = {}

method_name = 'LASSO Coefficient Matching'
start = time.time()
with warnings.catch_warnings(record=True) as warning_list:
    lcm = LCM_MF(outcome='Y', treatment='T', data=df_dummy_data,
                 n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    lcm.fit(model='linear')
    lcm.MG(k=k_est)
    lcm.CATE(cate_methods=['mean'])
times[method_name] = time.time() - start
df_err = pd.concat([df_err,
                    get_errors(lcm.cate_df[['avg.CATE_mean']],
                               df_true[['TE']],
                               method_name=method_name)
                    ])
print(f'\n{method_name} method complete: {time.time() - start}')
summarize_warnings(warning_list, method_name)
print()

split_strategy = lcm.gen_skf  # save split strategy to use for all other methods
with open(f'{save_folder}/split.pkl', 'wb') as f:
    pickle.dump(split_strategy, f)

# method_name = 'Tree Feature Importance Matching'
# start = time.time()
# with warnings.catch_warnings(record=True) as warning_list:
#     lcm = LCM_MF(outcome='Y', treatment='T', data=df_dummy_data,
#                  n_splits=n_splits, n_repeats=1,
#                  random_state=random_state)
#     lcm.gen_skf = split_strategy
#     lcm.fit(model='tree')
#     lcm.MG(k=k_est)
#     lcm.CATE(cate_methods=['mean'])
# times[method_name] = time.time() - start
# df_err = pd.concat([df_err,
#                     get_errors(lcm.cate_df[['avg.CATE_mean']],
#                                df_true[['TE']],
#                                method_name=method_name)
#                     ])
# print(f'\n{method_name} method complete: {time.time() - start}')
# summarize_warnings(warning_list, method_name)
# print()
#
# method_name = 'GBR Feature Importance Matching'
# start = time.time()
# with warnings.catch_warnings(record=True) as warning_list:
#     lcm = LCM_MF(outcome='Y', treatment='T', data=df_dummy_data,
#                  n_splits=n_splits, n_repeats=1,
#                  random_state=random_state)
#     lcm.gen_skf = split_strategy
#     lcm.fit(model='ensemble')
#     lcm.MG(k=k_est)
#     lcm.CATE(cate_methods=['mean'])
# times[method_name] = time.time() - start
# df_err = pd.concat([df_err,
#                     get_errors(lcm.cate_df[['avg.CATE_mean']],
#                                df_true[['TE']],
#                                method_name=method_name)
#                     ])
# print(f'\n{method_name} method complete: {time.time() - start}')
# summarize_warnings(warning_list, method_name)
# print()

method_name = 'GBR Feature Importance Matching'
start = time.time()
with warnings.catch_warnings(record=True) as warning_list:
    lcm = LCM_MF(outcome='Y', treatment='T', data=df_dummy_data,
                 n_splits=n_splits, n_repeats=1,
                 random_state=random_state)
    lcm.gen_skf = split_strategy
    lcm.fit(model='ensemble', params={'max_depth': 1},
            separate_treatments=True)
    lcm.MG(k=k_est)
    lcm.CATE(cate_methods=['mean'])
times[method_name] = time.time() - start
df_err = pd.concat([df_err,
                    get_errors(lcm.cate_df[['avg.CATE_mean']],
                               df_true[['TE']],
                               method_name=method_name)
                    ])
print(f'\n{method_name} method complete: {time.time() - start}')
summarize_warnings(warning_list, method_name)
print()



# method_name = 'Equal Weighted LASSO Matching'
# start = time.time()
# with warnings.catch_warnings(record=True) as warning_list:
#     lcm = LCM_MF(outcome='Y', treatment='T', data=df_dummy_data, n_splits=n_splits, n_repeats=1,
#                  random_state=random_state)
#     lcm.gen_skf = split_strategy
#     lcm.fit(model='linear', equal_weights=True)
#     lcm.MG(k=k_est)
#     lcm.CATE(cate_methods=['mean'])
# times[method_name] = time.time() - start
# df_err = pd.concat([df_err,
#                     get_errors(lcm.cate_df[['avg.CATE_mean']],
#                                df_true[['TE']],
#                                method_name=method_name)
#                     ])
# print(f'\n{method_name} method complete: {time.time() - start}')
# summarize_warnings(warning_list, method_name)
# print()
#
# if run_malts:
#     method_name = 'MALTS Matching'
#     start = time.time()
#     with warnings.catch_warnings(record=True) as warning_list:
#         m = pymalts.malts_mf('Y', 'T', data=df_data, discrete=binary + categorical,
#                              categorical=categorical,
#                              k_est=k_est,
#                              n_splits=n_splits, estimator='mean',
#                              smooth_cate=False,
#                              gen_skf=split_strategy, random_state=random_state)
#     times[method_name] = time.time() - start
#     df_err = pd.concat([df_err,
#                         get_errors(m.CATE_df[['avg.CATE']],
#                                    df_true[['TE']],
#                                    method_name=method_name)
#                         ])
#     print(f'\n{method_name} method complete: {time.time() - start}')
#     print(f'{method_name} complete: {time.time() - start}')
#     summarize_warnings(warning_list, method_name)
#     print()
#
#
method_name = 'Linear Prognostic Score Matching'
start = time.time()
with warnings.catch_warnings(record=True) as warning_list:
    cate_est_prog, _, _ = prognostic.prognostic_cv('Y', 'T', df_dummy_data,
                                                   method='linear',
                                                   double=True,
                                                   k_est=k_est,
                                                   est_method='mean',
                                                   gen_skf=split_strategy,
                                                   random_state=random_state)
times[method_name] = time.time() - start
df_err = pd.concat([df_err,
                    get_errors(cate_est_prog[['avg.CATE']],
                               df_true[['TE']],
                               method_name=method_name)
                    ])
print(f'\n{method_name} method complete: {time.time() - start}')
summarize_warnings(warning_list, method_name)
print()

method_name = 'Ensemble Prognostic Score Matching'
start = time.time()
with warnings.catch_warnings(record=True) as warning_list:
    cate_est_prog, c_mg, t_mg = prognostic.prognostic_cv('Y', 'T', df_dummy_data,
                                                   method='ensemble',
                                                   double=True,
                                                   k_est=k_est,
                                                   est_method='mean',
                                                   gen_skf=split_strategy,
                                                   random_state=random_state)
times[method_name] = time.time() - start
df_err = pd.concat([df_err,
                    get_errors(cate_est_prog[['avg.CATE']],
                               df_true[['TE']],
                               method_name=method_name)
                    ])
print(f'\n{method_name} method complete: {time.time() - start}')
summarize_warnings(warning_list, method_name)
print()


method_name = 'DoubleML'
start = time.time()
with warnings.catch_warnings(record=True) as warning_list:
    cate_est_doubleml = doubleml.doubleml('Y', 'T', df_dummy_data, gen_skf=split_strategy, random_state=random_state)
times[method_name] = time.time() - start
df_err = pd.concat([df_err,
                    get_errors(cate_est_doubleml[['avg.CATE']],
                               df_true[['TE']],
                               method_name=method_name)
                    ])
print(f'\n{method_name} method complete: {time.time() - start}')
summarize_warnings(warning_list, method_name)
print()


method_name = 'DRLearner'
start = time.time()
with warnings.catch_warnings(record=True) as warning_list:
    cate_est_drlearner = drlearner.drlearner('Y', 'T', df_dummy_data, gen_skf=split_strategy, random_state=random_state)
times[method_name] = time.time() - start
df_err = pd.concat([df_err,
                    get_errors(cate_est_drlearner[['avg.CATE']],
                               df_true[['TE']],
                               method_name=method_name)
                    ])
print(f'\n{method_name} method complete: {time.time() - start}')
summarize_warnings(warning_list, method_name)
print()

if run_bart:
    method_name = 'BART'
    start = time.time()
    with warnings.catch_warnings(record=True) as warning_list:
        cate_est_bart = bart.bart('Y', 'T', df_dummy_data, gen_skf=split_strategy, random_state=random_state)
    times[method_name] = time.time() - start
    df_err = pd.concat([df_err,
                        get_errors(cate_est_bart[['avg.CATE']],
                                   df_true[['TE']],
                                   method_name=method_name)
                        ])
    print(f'\n{method_name} method complete: {time.time() - start}')
    summarize_warnings(warning_list, method_name)
    print()

method_name = 'Causal Forest'
start = time.time()
with warnings.catch_warnings(record=True) as warning_list:
    cate_est_cf = causalforest.causalforest('Y', 'T', df_dummy_data, gen_skf=split_strategy, random_state=random_state)
times[method_name] = time.time() - start
df_err = pd.concat([df_err,
                    get_errors(cate_est_cf[['avg.CATE']],
                               df_true[['TE']],
                               method_name=method_name)
                    ])
print(f'\n{method_name} method complete: {time.time() - start}')
summarize_warnings(warning_list, method_name)
print()

method_name = 'Causal Forest DML'
start = time.time()
with warnings.catch_warnings(record=True) as warning_list:
    cate_est_cf = causalforest_dml.causalforest_dml('Y', 'T', df_dummy_data, gen_skf=split_strategy,
                                                    random_state=random_state)
times[method_name] = time.time() - start
df_err = pd.concat([df_err,
                    get_errors(cate_est_cf[['avg.CATE']],
                               df_true[['TE']],
                               method_name=method_name)
                    ])
print(f'\n{method_name} method complete: {time.time() - start}')
summarize_warnings(warning_list, method_name)
print()

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

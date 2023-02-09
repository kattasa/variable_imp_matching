import copy

import numpy as np
import pandas as pd
import time

import seaborn as sns
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from datagen.dgp import data_generation_dense_mixed_endo
from Experiments.helpers import get_errors
from src.linear_coef_matching_mf import LCM_MF

random_state = 0

k_est = 10
est_method = 'mean'

n_samples = 500
n_splits = 5
x_imp = 3
x_unimp = 1
t_imp = 3

preset_weights = [
    [0, {'control': 20, 'treated': 0}],
    [1, {'control': 0, 'treated': 20}],
    [2, {'control': 10, 'treated': 10}]
]

df_orig, df_true, binary = data_generation_dense_mixed_endo(num_samples=n_samples,
                                                       num_cont_imp=x_imp,
                                                       num_disc_imp=0,
                                                       num_cont_unimp=x_unimp,
                                                       num_disc_unimp=0,
                                                       weights=preset_weights)
df = df_orig.copy(deep=True)
x_cols = [c for c in df.columns if 'X' in c]
df[x_cols] = StandardScaler().fit_transform(df[x_cols])


df_err = pd.DataFrame(columns=['Method', 'True_CATE', 'Est_CATE', 'Relative Error (%)'])
df_weights = pd.DataFrame(columns=[f'X{i}' for i in range(x_imp+x_unimp)] + ['Method'])
df_mgs = pd.DataFrame(columns=list(range(k_est)) + ['Method', 'T'])

start = time.time()
lcm = LCM_MF(outcome='Y', treatment='T', data=df, n_splits=n_splits,
             n_repeats=1, random_state=random_state)
lcm.fit(metalearner=False)
these_weights = pd.DataFrame(lcm.M_list)
these_weights.columns = [f'X{i}' for i in range(x_imp+x_unimp)]
these_weights = these_weights.div(these_weights.sum(axis=1), axis=0)
these_weights['Method'] = 'LCM'
df_weights = pd.concat([df_weights, these_weights.copy(deep=True)])
lcm.MG(k=k_est)

lcm_diffs = {}
lcm_metalearner_diffs = {}

idxs = np.concatenate([lcm.gen_skf[i][0] for i in range(n_splits)]).reshape(-1)

for c in x_cols:
    values = df_orig[c].to_numpy()[idxs].reshape(-1, 1)
    mg_values = df_orig[c].to_numpy()[pd.concat([pd.concat([lcm.get_MGs()[i][0] for i in range(n_splits)]), pd.concat([lcm.get_MGs()[i][1] for i in range(n_splits)])], axis=1).to_numpy()]
    lcm_diffs[c] = copy.copy(np.mean(np.abs(mg_values - values), axis=1))

lcm.CATE(cate_methods=[est_method])
df_err = pd.concat([df_err, get_errors(lcm.cate_df[['avg.CATE_mean']],
                                       df_true[['TE']], method_name='LCM',
                                       scale=np.abs(df_true['TE']).mean())])
print(f'LCM complete: {time.time() - start}')

start = time.time()
lcm.fit(metalearner=True)
these_weights = pd.DataFrame([v[0] for v in lcm.M_list])
these_weights.columns = [f'X{i}' for i in range(x_imp+x_unimp)]
these_weights = these_weights.div(these_weights.sum(axis=1), axis=0)
these_weights['Method'] = 'Metalearner\nLCM M_C'
df_weights = pd.concat([df_weights, these_weights.copy(deep=True)])

these_weights = pd.DataFrame([v[1] for v in lcm.M_list])
these_weights.columns = [f'X{i}' for i in range(x_imp+x_unimp)]
these_weights = these_weights.div(these_weights.sum(axis=1), axis=0)
these_weights['Method'] = 'Metalearner\nLCM M_T'
df_weights = pd.concat([df_weights, these_weights.copy(deep=True)])

lcm.MG(k=k_est)

for c in x_cols:
    values = df_orig[c].to_numpy()[idxs].reshape(-1, 1)
    mg_values = df_orig[c].to_numpy()[pd.concat([pd.concat([lcm.get_MGs()[i][0] for i in range(n_splits)]), pd.concat([lcm.get_MGs()[i][1] for i in range(n_splits)])], axis=1).to_numpy()]
    lcm_diffs[c] = copy.copy(np.mean(np.abs(mg_values - values), axis=1))

lcm.CATE(cate_methods=[est_method])
df_err = pd.concat([df_err, get_errors(lcm.cate_df[['avg.CATE_mean']],
                                       df_true[['TE']], method_name='Metalearner\nLCM',
                                       scale=np.abs(df_true['TE']).mean())])
print(f'Metalearner LCM complete: {time.time() - start}')

df_true.to_csv('Results/df_true.csv')
df_err.to_csv('Results/df_err.csv')
df_weights.to_csv('Results/df_weights.csv')
df_mgs.to_csv('Results/df_mgs.csv')

df_err.loc[:, 'Relative Error (%)'] = df_err.loc[:, 'Relative Error (%)'] * 100
sns.set_context("paper")
sns.set_style("darkgrid")
sns.set(font_scale=2)
fig, ax = plt.subplots(figsize=(10, 12))
sns.boxenplot(x='Method', y='Relative Error (%)', data=df_err)
ax.yaxis.set_major_formatter(ticker.PercentFormatter())
plt.tight_layout()
fig.savefig(f'Results/boxplot_err.png')

x_imp += 1  # to include unimportant covariate in plotting
df_weights = df_weights[['Method'] + [f'X{i}' for i in range(x_imp)]].melt(id_vars=['Method'])
df_weights = df_weights.rename(columns={'variable': 'Covariate', 'value': 'Relative Weight (%)'})
sns.set_context("paper")
sns.set_style("darkgrid")
sns.set(font_scale=1)
fig, ax = plt.subplots(figsize=(10, 12))
sns.catplot(data=df_weights, x="Covariate", y="Relative Weight (%)", hue="Method", kind="bar", legend=False)
ax.yaxis.set_major_formatter(ticker.PercentFormatter())
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig('Results/barplot_weights.png')

lcm_std = {f'X{i}': {0: np.array([]), 1: np.array([])} for i in range(x_imp)}
meta_lcm_std = {f'X{i}': {0: np.array([]), 1: np.array([])} for i in range(x_imp)}
this_iter = 0
for est_idx, _ in lcm.gen_skf:
    for i in range(x_imp):
        cov = f'X{i}'
        cov_values = df.loc[est_idx, cov].to_numpy()
        lcm_std[cov][0] = np.concatenate([lcm_std[cov][0],
                                          np.mean(np.abs(cov_values[lcm_c_mgs[this_iter].to_numpy()] -
                                                         cov_values.reshape(-1, 1)),
                                                  axis=1)])
        lcm_std[cov][1] = np.concatenate([lcm_std[cov][1],
                                          np.mean(np.abs(cov_values[lcm_t_mgs[this_iter].to_numpy()] -
                                                         cov_values.reshape(-1, 1)),
                                                  axis=1)])
        meta_lcm_std[cov][0] = np.concatenate([meta_lcm_std[cov][0],
                                               np.mean(np.abs(cov_values[meta_lcm_c_mgs[this_iter].to_numpy()] -
                                                              cov_values.reshape(-1, 1)),
                                                       axis=1)])
        meta_lcm_std[cov][1] = np.concatenate([meta_lcm_std[cov][1],
                                               np.mean(np.abs(cov_values[meta_lcm_t_mgs[this_iter].to_numpy()] -
                                                              cov_values.reshape(-1, 1)),
                                                       axis=1)])
    this_iter += 1

lcm_std_df = []
meta_lcm_std_df = []
for i in range(x_imp):
    cov = f'X{i}'
    this_df = pd.DataFrame(lcm_std[cov]).T.reset_index().rename(columns={'index': 'T'})
    this_df['Covariate'] = cov
    lcm_std_df.append(this_df.copy(deep=True))
    this_df = pd.DataFrame(meta_lcm_std[cov]).T.reset_index().rename(columns={'index': 'T'})
    this_df['Covariate'] = cov
    meta_lcm_std_df.append(this_df.copy(deep=True))

lcm_std_df = pd.concat(lcm_std_df)
meta_lcm_std_df = pd.concat(meta_lcm_std_df)
lcm_std_df['Method'] = 'LCM'
meta_lcm_std_df['Method'] = 'LCM Metalearner'
mg_std = pd.concat([lcm_std_df, meta_lcm_std_df])
mg_std = pd.melt(mg_std, id_vars=['Method', 'T', 'Covariate'])
mg_std = mg_std.rename(columns={'value': 'MG Average Difference'})
mg_hue = mg_std.apply(lambda x: f"{x['Method']} {'Treatment' if x['T'] == 1 else 'Control'} Matches", axis=1)
mg_hue.name = 'Method, T'


sns.set_context("paper")
sns.set_style("darkgrid")
sns.set(font_scale=2)
fig, ax = plt.subplots(figsize=(10, 12))
sns.boxplot(data=mg_std, x='Covariate', y='MG Average Difference', hue=mg_hue)
plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig('Results/barplot_mg_avg_diff.png')


print('done')

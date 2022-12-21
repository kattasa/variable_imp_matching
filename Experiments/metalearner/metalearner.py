import copy

import numpy as np
import pandas as pd
import time
import warnings

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from datagen.dgp_df import dgp_dense_mixed_endo_df
from src.linear_coef_matching_mf import LCM_MF

warnings.filterwarnings("ignore")

k_est = 15
est_method = ['linear_pruned', False]

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

_, df, df_true, x_cols, binary = dgp_dense_mixed_endo_df(n=n_samples, nci=x_imp, ndi=0, ncu=x_unimp, ndu=0, std=1.5,
                                                         t_imp=t_imp, overlap=1000, n_train=0, weights=preset_weights)

sample_idx = df[(df['X0'].mean() + (2*df['X0'].std())) < df['X0']].index[0]

df_err = pd.DataFrame(columns=['Method', 'True_CATE', 'Est_CATE', 'Relative Error (%)'])
df_weights = pd.DataFrame(columns=[f'X{i}' for i in range(x_imp+x_unimp)] + ['Method'])
df_mgs = pd.DataFrame(columns=list(range(k_est)) + ['Method', 'T'])

start = time.time()
lcm = LCM_MF(outcome='Y', treatment='T', data=df, n_splits=n_splits,  n_repeats=1)
sample_mg_splits = []
i = 0
for est_idx, _ in lcm.gen_skf:
    if sample_idx in est_idx:
        sample_mg_splits.append(i)
    i += 1
lcm.fit(double_model=False)
these_weights = pd.DataFrame(lcm.M_list)
these_weights.columns = [f'X{i}' for i in range(x_imp+x_unimp)]
these_weights = these_weights.div(these_weights.sum(axis=1), axis=0)
these_weights['Method'] = 'LCM'
df_weights = df_weights.append(these_weights.copy(deep=True))
lcm.MG(k=k_est)
for s in sample_mg_splits:
    this_control_mg = pd.DataFrame(lcm.gen_skf[s][0][lcm.C_MG_list[s].iloc[
        np.where(lcm.gen_skf[s][0] == sample_idx)[0]].to_numpy()])
    this_treatment_mg = pd.DataFrame(lcm.gen_skf[s][0][lcm.T_MG_list[s].iloc[
        np.where(lcm.gen_skf[s][0] == sample_idx)[0]].to_numpy()])
    this_control_mg['Method'] = 'LCM'
    this_control_mg['T'] = 0
    this_treatment_mg['Method'] = 'LCM'
    this_treatment_mg['T'] = 1
    this_control_mg.index = [sample_idx]
    this_treatment_mg.index = [sample_idx]
    df_mgs = df_mgs.append(this_control_mg.copy(deep=True))
    df_mgs = df_mgs.append(this_treatment_mg.copy(deep=True))
lcm_c_mgs = copy.deepcopy(lcm.C_MG_list)
lcm_t_mgs = copy.deepcopy(lcm.T_MG_list)
lcm.CATE(cate_methods=[est_method])
cate_df = lcm.cate_df.sort_index()
cate_df = cate_df.rename(columns={'avg.CATE': 'Est_CATE'})
cate_df['True_CATE'] = df_true['TE'].to_numpy()
cate_df['Relative Error (%)'] = np.abs((cate_df['Est_CATE'] - cate_df['True_CATE']) / np.abs(cate_df['True_CATE']).mean())
cate_df['Method'] = ['LCM' for i in range(cate_df.shape[0])]
df_err = df_err.append(cate_df[['Method', 'True_CATE', 'Est_CATE', 'Relative Error (%)']].copy(deep=True))
print(f'LCM complete: {time.time() - start}')

start = time.time()
lcm.fit(double_model=True)
these_weights = pd.DataFrame(lcm.M_C_list)
these_weights.columns = [f'X{i}' for i in range(x_imp+x_unimp)]
these_weights = these_weights.div(these_weights.sum(axis=1), axis=0)
these_weights['Method'] = 'LCM Metalearner M_C'
df_weights = df_weights.append(these_weights.copy(deep=True))
these_weights = pd.DataFrame(lcm.M_T_list)
these_weights.columns = [f'X{i}' for i in range(x_imp+x_unimp)]
these_weights = these_weights.div(these_weights.sum(axis=1), axis=0)
these_weights['Method'] = 'LCM Metalearner M_T'
df_weights = df_weights.append(these_weights.copy(deep=True))
lcm.MG(k=k_est)
for s in sample_mg_splits:
    this_control_mg = pd.DataFrame(lcm.gen_skf[s][0][lcm.C_MG_list[s].iloc[
        np.where(lcm.gen_skf[s][0] == sample_idx)[0]].to_numpy()])
    this_treatment_mg = pd.DataFrame(lcm.gen_skf[s][0][lcm.T_MG_list[s].iloc[
        np.where(lcm.gen_skf[s][0] == sample_idx)[0]].to_numpy()])
    this_control_mg['Method'] = 'LCM Metalearner'
    this_control_mg['T'] = 0
    this_treatment_mg['Method'] = 'LCM Metalearner'
    this_treatment_mg['T'] = 1
    this_control_mg.index = [sample_idx]
    this_treatment_mg.index = [sample_idx]
    df_mgs = df_mgs.append(this_control_mg.copy(deep=True))
    df_mgs = df_mgs.append(this_treatment_mg.copy(deep=True))
meta_lcm_c_mgs = copy.deepcopy(lcm.C_MG_list)
meta_lcm_t_mgs = copy.deepcopy(lcm.T_MG_list)
lcm.CATE(cate_methods=[est_method])
cate_df = lcm.cate_df.sort_index()
cate_df = cate_df.rename(columns={'avg.CATE': 'Est_CATE'})
cate_df['True_CATE'] = df_true['TE'].to_numpy()
cate_df['Relative Error (%)'] = np.abs((cate_df['Est_CATE'] - cate_df['True_CATE']) / np.abs(cate_df['True_CATE']).mean())
cate_df['Method'] = ['LCM Metalearner' for i in range(cate_df.shape[0])]
df_err = df_err.append(cate_df[['Method', 'True_CATE', 'Est_CATE', 'Relative Error (%)']].copy(deep=True))
print(f'LCM Metalearner complete: {time.time() - start}')

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

df_weights = df_weights[['Method'] + [f'X{i}' for i in range(x_imp)]].melt(id_vars=['Method'])
df_weights = df_weights.rename(columns={'variable': 'Covariate', 'value': 'Relative Weight (%)'})
sns.set_context("paper")
sns.set_style("darkgrid")
sns.set(font_scale=1)
fig, ax = plt.subplots(figsize=(10, 12))
sns.catplot(data=df_weights, x="Covariate", y="Relative Weight (%)", hue="Method", kind="bar",)
ax.yaxis.set_major_formatter(ticker.PercentFormatter())
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig('Results/barplot_weights.png')

x_imp += 1
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
plt.xticks(rotation=65, horizontalalignment='right')
plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig('Results/barplot_mg_avg_diff.png')


print('done')

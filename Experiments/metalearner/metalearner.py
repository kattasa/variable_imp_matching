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

lcm_c_diffs = {}
lcm_t_diffs = {}
lcm_c_metalearner_diffs = {}
lcm_t_metalearner_diffs = {}

idxs = np.concatenate([lcm.gen_skf[i][0] for i in range(n_splits)]).reshape(-1)

for c in x_cols:
    values = df_orig[c].to_numpy()[idxs].reshape(-1, 1)
    mg_values = df_orig[c].to_numpy()[pd.concat([lcm.get_MGs()[i][0] for i in range(n_splits)]).to_numpy()]
    lcm_c_diffs[c] = copy.copy(np.mean(np.abs(mg_values - values), axis=1))

    mg_values = df_orig[c].to_numpy()[
        pd.concat([lcm.get_MGs()[i][1] for i in range(n_splits)]).to_numpy()]
    lcm_t_diffs[c] = copy.copy(np.mean(np.abs(mg_values - values), axis=1))

lcm.CATE(cate_methods=[est_method], diameter_prune=None)
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
    mg_values = df_orig[c].to_numpy()[pd.concat([lcm.get_MGs()[i][0] for i in range(n_splits)]).to_numpy()]
    lcm_c_metalearner_diffs[c] = copy.copy(np.mean(np.abs(mg_values - values), axis=1))

    mg_values = df_orig[c].to_numpy()[
        pd.concat([lcm.get_MGs()[i][1] for i in range(n_splits)]).to_numpy()]
    lcm_t_metalearner_diffs[c] = copy.copy(np.mean(np.abs(mg_values - values), axis=1))

lcm.CATE(cate_methods=[est_method], diameter_prune=None)
df_err = pd.concat([df_err, get_errors(lcm.cate_df[['avg.CATE_mean']],
                                       df_true[['TE']], method_name='Metalearner\nLCM',
                                       scale=np.abs(df_true['TE']).mean())])
print(f'Metalearner LCM complete: {time.time() - start}')

df_true.to_csv('Results/df_true.csv')
df_err.to_csv('Results/df_err.csv')
df_weights.to_csv('Results/df_weights.csv')

method_order = ['LCM', 'Linear PGM', 'Ensemble PGM', 'MALTS', '', '', 'GenMatch',
                'Metalearner\nLCM']
palette = {method_order[i]: sns.color_palette()[i] for i in
           range(len(method_order))}
order = [m for m in method_order if m in df_err['Method'].unique()]

df_err.loc[:, 'Relative Error (%)'] = df_err.loc[:, 'Relative Error (%)'] * 100
plt.figure()
sns.set_context("paper")
sns.set_style("darkgrid")
sns.set(font_scale=1.8)
ax = sns.boxplot(x='Method', y='Relative Error (%)',
                 data=df_err, showfliers=False,
                 order=order, palette=palette)
ax.yaxis.set_major_formatter(ticker.PercentFormatter())
ax.get_figure().savefig(f'Results/metalearner_boxplot_err.png', bbox_inches='tight')

palette = {'LCM': sns.color_palette()[0],
           'Metalearner\nLCM M_C': sns.color_palette()[7],
           'Metalearner\nLCM M_T': sns.color_palette()[8]}
order = ['LCM', 'Metalearner\nLCM M_C', 'Metalearner\nLCM M_T']

x_imp += 1  # to include unimportant covariate in plotting
df_weights = df_weights[['Method'] + [f'X{i}' for i in range(x_imp)]].melt(id_vars=['Method'])
df_weights = df_weights.rename(columns={'variable': 'Covariate', 'value': 'Relative Weight (%)'})
df_weights['Relative Weight (%)'] *= 100

plt.figure(figsize=(6, 8))
sns.set_context("paper")
sns.set_style("darkgrid")
sns.set(font_scale=2)
ax = sns.barplot(data=df_weights, x="Covariate", y="Relative Weight (%)",
                 hue="Method", hue_order=order, palette=palette)
ax.yaxis.set_major_formatter(ticker.PercentFormatter())
sns.move_legend(ax, "lower center", bbox_to_anchor=(.36, 1.02), ncol=3,
                title=None, handletextpad=0.4, columnspacing=0.5, fontsize=18)
plt.tight_layout()
ax.get_figure().savefig(f'Results/metalearner_barplot_weights.png')

lcm_c_diffs = pd.melt(pd.DataFrame.from_dict(lcm_c_diffs), var_name='Covariate', value_name='Mean Absolute Difference')
lcm_c_diffs['Method'] = 'LCM\nControl MGs'
lcm_t_diffs = pd.melt(pd.DataFrame.from_dict(lcm_t_diffs), var_name='Covariate', value_name='Mean Absolute Difference')
lcm_t_diffs['Method'] = 'LCM\nTreatment MGs'
lcm_c_metalearner_diffs = pd.melt(pd.DataFrame.from_dict(lcm_c_metalearner_diffs), var_name='Covariate', value_name='Mean Absolute Difference')
lcm_c_metalearner_diffs['Method'] = 'Metalearner LCM\nControl MGs'
lcm_t_metalearner_diffs = pd.melt(pd.DataFrame.from_dict(lcm_t_metalearner_diffs), var_name='Covariate', value_name='Mean Absolute Difference')
lcm_t_metalearner_diffs['Method'] = 'Metalearner LCM\nTreatment MGs'


mg_diffs = pd.concat([lcm_c_diffs, lcm_t_diffs, lcm_c_metalearner_diffs,
                      lcm_t_metalearner_diffs])
mg_diffs.to_csv('Results/mg_diffs.csv')

palette = {'LCM\nControl MGs': sns.color_palette()[0],
            'LCM\nTreatment MGs': sns.color_palette()[9],
           'Metalearner LCM\nControl MGs': sns.color_palette()[7],
           'Metalearner LCM\nTreatment MGs': sns.color_palette()[8]}
order = ['LCM\nControl MGs', 'LCM\nTreatment MGs',
         'Metalearner LCM\nControl MGs', 'Metalearner LCM\nTreatment MGs']

plt.figure(figsize=(6, 8))
sns.set_context("paper")
sns.set_style("darkgrid")
sns.set(font_scale=2)
ax = sns.boxplot(x='Covariate', y='Mean Absolute Difference', hue='Method',
                 data=mg_diffs, showfliers=False,
                 order=[f'X{i}' for i in range(x_imp)], palette=palette,
                 hue_order=order)
sns.move_legend(ax, "lower center", bbox_to_anchor=(.42, 1), ncol=2, title=None,
                handletextpad=0.4, columnspacing=0.5, fontsize=18)
plt.tight_layout()
ax.get_figure().savefig(f'Results/metalearner_barplot_mg_avg_diff.png')


print('done')

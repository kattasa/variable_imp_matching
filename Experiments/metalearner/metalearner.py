"""Runs metalearner analysis experiment."""

import copy

import numpy as np
import pandas as pd
import time

import seaborn as sns
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from datagen.dgp import dgp_sine
from Experiments.helpers import get_errors
from other_methods import prognostic, pymalts, tlearner, bart, matchit, causalforest
from src.variable_imp_matching import VIM_CF

random_state = 0

k_est = 10
est_method = 'mean'

n_samples = 500
n_splits = 5
x_imp = 2
x_unimp = 8

X, Y, T, Y0, Y1, TE, Y0_true, Y1_true = dgp_sine(n_samples, x_unimp)

df_orig = pd.DataFrame(np.concatenate([X, Y, T, Y0, Y1, TE, Y0_true, Y1_true], axis=1))
x_cols = [f'X{i}' for i in range(X.shape[1])]
df_orig.columns = [*x_cols, 'Y', 'T', 'Y0', 'Y1', 'TE', 'Y0_true', 'Y1_true']
df = df_orig.copy(deep=True)

df[x_cols] = StandardScaler().fit_transform(df[x_cols])
df['T'] = df['T'].astype(int)

df_true = df.copy(deep=True)
df = df.drop(columns=['Y0', 'Y1', 'TE', 'Y0_true', 'Y1_true'])

scaling_factor = np.abs(df_true['TE']).mean()

df_err = pd.DataFrame(columns=['Method', 'True_CATE', 'Est_CATE', 'Relative Error (%)'])
df_weights = pd.DataFrame(columns=[f'X{i}' for i in range(x_imp+x_unimp)] + ['Method'])

start = time.time()
lcm = VIM_CF(outcome='Y', treatment='T', data=df, n_splits=n_splits,
             n_repeats=1, random_state=random_state)
lcm.fit(metalearner=False)
these_weights = pd.DataFrame(lcm.M_list)
these_weights.columns = [f'X{i}' for i in range(x_imp+x_unimp)]
these_weights = these_weights.div(these_weights.sum(axis=1), axis=0)
these_weights['Method'] = 'LCM'
df_weights = pd.concat([df_weights, these_weights.copy(deep=True)])
lcm.create_mgs(k=k_est)

lcm_c_diffs = {}
lcm_t_diffs = {}
lcm_c_metalearner_diffs = {}
lcm_t_metalearner_diffs = {}

idxs = np.concatenate([lcm.split_strategy[i][0] for i in range(n_splits)]).reshape(-1)

for c in x_cols:
    values = df_orig[c].to_numpy()[idxs].reshape(-1, 1)
    mg_values = df_orig[c].to_numpy()[pd.concat([lcm.get_mgs()[i][0] for i in range(n_splits)]).to_numpy()]
    lcm_c_diffs[c] = copy.copy(np.mean(np.abs(mg_values - values), axis=1))

    mg_values = df_orig[c].to_numpy()[
        pd.concat([lcm.get_mgs()[i][1] for i in range(n_splits)]).to_numpy()]
    lcm_t_diffs[c] = copy.copy(np.mean(np.abs(mg_values - values), axis=1))

lcm.est_cate(cate_methods=[est_method], diameter_prune=None)
df_err = pd.concat([df_err, get_errors(lcm.cate_df[['avg.CATE_mean']],
                                       df_true[['TE']], method_name='LCM',
                                       scale=scaling_factor)])
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

lcm.create_mgs(k=k_est)

for c in x_cols:
    values = df_orig[c].to_numpy()[idxs].reshape(-1, 1)
    mg_values = df_orig[c].to_numpy()[pd.concat([lcm.get_mgs()[i][0] for i in range(n_splits)]).to_numpy()]
    lcm_c_metalearner_diffs[c] = copy.copy(np.mean(np.abs(mg_values - values), axis=1))

    mg_values = df_orig[c].to_numpy()[
        pd.concat([lcm.get_mgs()[i][1] for i in range(n_splits)]).to_numpy()]
    lcm_t_metalearner_diffs[c] = copy.copy(np.mean(np.abs(mg_values - values), axis=1))

lcm.est_cate(cate_methods=[est_method], diameter_prune=None)
df_err = pd.concat([df_err, get_errors(lcm.cate_df[['avg.CATE_mean']],
                                       df_true[['TE']], method_name='Metalearner\nLCM',
                                       scale=scaling_factor)])
print(f'Metalearner LCM complete: {time.time() - start}')

split_strategy = lcm.split_strategy

method_name = 'Linear\nPGM'
cate_est_lpgm, lpgm_c_mg, \
lpgm_t_mg, lpgm_fi   = prognostic.prognostic_cv('Y', 'T', df,
                                                method='linear', double=True,
                                                k_est=k_est, est_method='mean',
                                                diameter_prune=None,
                                                gen_skf=split_strategy,
                                                return_feature_imp=True,
                                                random_state=random_state)
df_err = pd.concat([df_err, get_errors(cate_est_lpgm[['avg.CATE']],
                                       df_true[['TE']],
                                       method_name=method_name,
                                       scale=scaling_factor)])
print(f'\n{method_name} method complete')

method_name = 'Nonparametric\nPGM'
cate_est_epgm, epgm_c_mg, \
epgm_t_mg, epgm_fi   = prognostic.prognostic_cv('Y', 'T', df,
                                                method='ensemble', double=True,
                                                k_est=k_est, est_method='mean',
                                                diameter_prune=None,
                                                gen_skf=split_strategy,
                                                return_feature_imp=True,
                                                random_state=random_state)
df_err = pd.concat([df_err, get_errors(cate_est_epgm[['avg.CATE']],
                                       df_true[['TE']],
                                       method_name=method_name,
                                       scale=scaling_factor)])
print(f'\n{method_name} method complete')

method_name = 'GenMatch'
ate, t_hat = matchit.matchit(outcome='Y', treatment='T', data=df,
                             method='genetic', replace=True)
df_err = pd.concat([df_err, get_errors(t_hat[['CATE']],
                                       df_true[['TE']],
                                       method_name=method_name,
                                       scale=scaling_factor)])
print(f'\n{method_name} method complete')

method_name = 'MALTS'
m = pymalts.malts_mf('Y', 'T', data=df,
                     discrete=[],
                     k_est=k_est,
                     n_splits=n_splits, estimator='mean',
                     smooth_cate=False,
                     split_strategy=split_strategy,
                     random_state=random_state)
df_err = pd.concat([df_err,
                    get_errors(m.CATE_df[['avg.CATE']],
                               df_true[['TE']],
                               method_name=method_name,
                               scale=scaling_factor)
                    ])
print(f'\n{method_name} method complete')

method_name = 'Causal Forest'
cate_est_cfor = causalforest.causalforest('Y', 'T', df,
                                          gen_skf=split_strategy,
                                          random_state=random_state)
df_err = pd.concat([df_err,
                    get_errors(cate_est_cfor[['avg.CATE']],
                               df_true[['TE']],
                               method_name=method_name,
                               scale=scaling_factor)
                    ])
print(f'\n{method_name} method complete')

method_name = 'BART'
cate_est_bart = bart.bart('Y', 'T', df,
                          gen_skf=split_strategy,
                          random_state=random_state)
df_err = pd.concat([df_err,
                    get_errors(cate_est_bart[['avg.CATE']],
                               df_true[['TE']],
                               method_name=method_name,
                               scale=scaling_factor)
                    ])
print(f'\n{method_name} method complete')

method_name = 'Linear\nTLearner'
cate_est_tlearner = tlearner.tlearner('Y', 'T', df, method='linear',
                                      gen_skf=split_strategy,
                                      random_state=random_state)
df_err = pd.concat([df_err,
                    get_errors(cate_est_tlearner[['avg.CATE']],
                               df_true[['TE']],
                               method_name=method_name,
                               scale=scaling_factor)
                    ])
print(f'\n{method_name} method complete')

method_name = 'Nonparametric\nTLearner'
cate_est_tlearner = tlearner.tlearner('Y', 'T', df, method='ensemble',
                                      gen_skf=split_strategy,
                                      random_state=random_state)
df_err = pd.concat([df_err,
                    get_errors(cate_est_tlearner[['avg.CATE']],
                               df_true[['TE']],
                               method_name=method_name,
                               scale=scaling_factor)
                    ])
print(f'\n{method_name} method complete')


df_true.to_csv('Results/df_true.csv')
df_err.to_csv('Results/df_err.csv')
df_weights.to_csv('Results/df_weights.csv')

color_order = ['LCM', 'Linear\nPGM', 'Nonparametric\nPGM', 'MALTS',
               'Metalearner\nLCM', 'BART', 'GenMatch', 'Linear\nTLearner',
               'Nonparametric\nTLearner']
palette = {color_order[i]: sns.color_palette()[i] for i in range(len(color_order))}

method_order = ['Metalearner\nLCM', 'LCM', 'MALTS', 'GenMatch', 'Linear\nPGM',
                'Nonparametric\nPGM',  'BART', 'Linear\nTLearner',
                'Nonparametric\nTLearner'
                ]

# method_order = ['Metalearner\nLCM', 'LCM', 'MALTS', 'Linear\nPGM',
#                 'Nonparametric\nPGM',  'BART', 'Causal Forest', 'Linear\nDML',
#                 'Causal Forest\nDML'
#                 ]
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
ax.get_figure().savefig(f'Results/metalearner_boxplot_err.png',
                        bbox_inches='tight')

palette = {'LCM': sns.color_palette()[0],
           'Metalearner\nLCM M_C': sns.color_palette()[7],
           'Metalearner\nLCM M_T': sns.color_palette()[8]}
order = ['LCM', 'Metalearner\nLCM M_C', 'Metalearner\nLCM M_T']

x_imp += 1  # to include unimportant covariate in plotting
df_weights = df_weights[['Method'] + [f'X{i}' for i in range(x_imp)]].melt(id_vars=['Method'])
df_weights = df_weights.rename(columns={'variable': 'Covariate', 'value': 'Relative Weight (%)'})
df_weights['Relative Weight (%)'] *= 100

plt.figure(figsize=(6, 6))
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

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib
import numpy as np
import os
from collections import Counter
import copy

from datagen.dgp_df import dgp_schools_df

from src.linear_coef_matching_mf import LCM_MF
from other_methods import prognostic

k_est = 10
random_state = 0
n_splits = 20
n_repeats = 50

df = dgp_schools_df()

lcm = LCM_MF(outcome='Y', treatment='T', data=df,
             n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)

lcm.fit(model='linear')
lcm.MG(k=k_est)
lcm.CATE(diameter_prune=None)
lcm_cates = lcm.cate_df['CATE_mean']
lcm_cates.to_csv('lcm_cates.csv')

lcm_ates = [lcm_cates.iloc[:, i*n_splits:(i+1)*n_splits].mean().mean() for i in range(n_repeats)]

lcm_ate = np.mean(lcm_ates)
lcm_std = np.std(lcm_ates)
lcm_ci = (lcm_ate - (1.96*lcm_std), lcm_ate + (1.96*lcm_std))
print('LCM Done')

with open('lcm_ate.txt', 'w') as f:
    f.write(f'{lcm_ate} ({lcm_ci[0]},{lcm_ci[1]})')

linear_prog_cates_full, linear_prog_c_mg, linear_prog_t_mg, linear_prog_fi = \
    prognostic.prognostic_cv(outcome='Y', treatment='T', data=df,
                             method='linear', double=True, k_est=k_est,
                             est_method='mean', gen_skf=lcm.gen_skf,
                             diameter_prune=None,
                             return_feature_imp=True,
                             random_state=random_state)
linear_prog_cates = linear_prog_cates_full['CATE']
linear_prog_cates.to_csv('linear_prog_cates.csv')
linear_prog_ates = [linear_prog_cates.iloc[:, i*n_splits:(i+1)*n_splits].mean().mean() for i in range(n_repeats)]
linear_prog_ate = np.mean(linear_prog_ates)
linear_prog_std = np.std(linear_prog_ates)
linear_prog_ci = (linear_prog_ate - (1.96*linear_prog_std), linear_prog_ate + (1.96*linear_prog_std))
print('Linear Prog Done')

with open('linear_prog_ate.txt', 'w') as f:
    f.write(f'{linear_prog_ate} ({linear_prog_ci[0]},{linear_prog_ci[1]})')

ensemble_prog_cates_full, ensemble_prog_c_mg, ensemble_prog_t_mg, ensemble_prog_fi = \
    prognostic.prognostic_cv(outcome='Y', treatment='T', data=df,
                             method='ensemble', double=True, k_est=k_est,
                             est_method='mean',
                             diameter_prune=None, gen_skf=lcm.gen_skf,
                             return_feature_imp=True,
                             random_state=random_state)
ensemble_prog_cates = ensemble_prog_cates_full['CATE']
ensemble_prog_cates.to_csv('ensemble_prog_cates.csv')
ensemble_prog_ates = [ensemble_prog_cates.iloc[:, i*n_splits:(i+1)*n_splits].mean().mean() for i in range(n_repeats)]
ensemble_prog_ate = np.mean(ensemble_prog_ates)
ensemble_prog_std = np.std(ensemble_prog_ates)
ensemble_prog_ci = (ensemble_prog_ate - (1.96*ensemble_prog_std), ensemble_prog_ate + (1.96*ensemble_prog_std))
print('Ensemble Prog Done')

with open('ensemble_prog_ate.txt', 'w') as f:
    f.write(f'{ensemble_prog_ate} ({ensemble_prog_ci[0]},{ensemble_prog_ci[1]})')

df_orig = pd.read_csv(f'{os.getenv("SCHOOLS_FOLDER")}/df.csv')
lcm_cates = lcm.cate_df.join(df_orig.drop(columns=['Z', 'Y']))

lcm_top_10 = pd.DataFrame(lcm.M_list, columns=lcm.covariates).mean().sort_values(ascending=False).iloc[:10].index
linear_prog_top_10 = pd.DataFrame(linear_prog_fi, columns=lcm.covariates).mean().sort_values().sort_values(ascending=False).iloc[:10].index
ensemble_prog_top_10 = pd.DataFrame(ensemble_prog_fi, columns=lcm.covariates).mean().sort_values().sort_values(ascending=False).iloc[:10].index
imp_covs = [c for c in lcm_top_10 if c in linear_prog_top_10]
imp_covs = [c for c in imp_covs if c in ensemble_prog_top_10]

categorical = [c for c in imp_covs if '_' in c]
continuous = [c for c in imp_covs if '_' not in c]
if 'C2' in imp_covs:
    categorical = categorical + ['C2']
    continuous.remove('C2')

lcm_diffs = {}
linear_prog_diffs = {}
ensemble_prog_diffs = {}

idxs = np.concatenate([lcm.gen_skf[i][0] for i in range(n_splits*n_repeats)])

for cov in imp_covs:
    if '_' in cov:
        c, val = cov.split('_')
        good_idxs = df_orig.loc[(df_orig[c] == int(val))].index
        good_idxs_i = np.array([i for i, v in enumerate(idxs) if v in good_idxs]).reshape(-1)
        good_idxs = np.array([c for c in idxs if c in good_idxs])
    else:
        c = cov
        good_idxs_i = np.array([range(idxs.shape[0])]).reshape(-1)
        good_idxs = idxs
    values = df_orig[c].to_numpy()[good_idxs].reshape(-1, 1)
    mg_values = df_orig[c].to_numpy()[pd.concat([pd.concat([lcm.get_MGs()[i][0] for i in range(n_splits * n_repeats)]), pd.concat([lcm.get_MGs()[i][1] for i in range(n_splits * n_repeats)])], axis=1).to_numpy()[good_idxs_i]]
    if cov in continuous:
        lcm_diffs[cov] = copy.copy(np.mean(np.abs(mg_values - values), axis=1))
    else:
        lcm_diffs[cov] = copy.copy((np.sum((mg_values == values).astype(int), axis=1) / (k_est*2)))
    mg_values = df_orig[c].to_numpy()[pd.concat(
        [pd.concat([linear_prog_c_mg[i] for i in range(n_splits * n_repeats)]),
         pd.concat([linear_prog_t_mg[i] for i in range(n_splits * n_repeats)])], axis=1).to_numpy()[good_idxs_i]]
    if cov in continuous:
        linear_prog_diffs[cov] = copy.copy(np.mean(np.abs(mg_values - values), axis=1))
    else:
        linear_prog_diffs[cov] = copy.copy((np.sum((mg_values == values).astype(int), axis=1) / (k_est*2)))
    mg_values = df_orig[c].to_numpy()[pd.concat(
        [pd.concat([ensemble_prog_c_mg[i] for i in range(n_splits * n_repeats)]),
         pd.concat(
             [ensemble_prog_t_mg[i] for i in range(n_splits * n_repeats)])],
        axis=1).to_numpy()[good_idxs_i]]
    if cov in continuous:
        ensemble_prog_diffs[cov] = copy.copy(np.mean(np.abs(mg_values - values), axis=1))
    else:
        ensemble_prog_diffs[cov] = copy.copy((np.sum((mg_values == values).astype(int), axis=1) / (k_est*2)))


method_order = ['LCM', 'Linear\nPrognostic Score', 'Ensemble\nPrognostic Score']
palette = {method_order[i]: sns.color_palette()[i] for i in range(len(method_order))}


cat_diff_df = []
cont_diff_df = []
for c in categorical:
    this_df = pd.melt(pd.DataFrame([lcm_diffs[c],
                                    linear_prog_diffs[c],
                                    ensemble_prog_diffs[c]],
                                   index=method_order).T,
                      var_name='Method',
                      value_name='% Match')
    this_df['Covariate'] = c.replace('_', '=')
    cat_diff_df.append(this_df.copy())
for c in continuous:
    this_df = pd.melt(pd.DataFrame([lcm_diffs[c],
                                    linear_prog_diffs[c],
                                    ensemble_prog_diffs[c]],
                                   index=method_order).T,
                      var_name='Method',
                      value_name='Mean Absolute Difference')
    this_df['Covariate'] = c
    cont_diff_df.append(this_df.copy())

cat_diff_df = pd.concat(cat_diff_df)
cont_diff_df = pd.concat(cont_diff_df)
cat_diff_df.to_csv('categorical_diff.csv')
cont_diff_df.to_csv('continuous_diff.csv')

matplotlib.rcParams.update({'font.size': 40})
sns.set_context("paper")
sns.set_style("darkgrid")
sns.set(font_scale=4)
fig, axes = plt.subplots(1, 2, figsize=(26, 14))
sns.barplot(ax=axes[0], data=cat_diff_df, x='Covariate',
            y='% Match', hue='Method', errorbar=None,
            hue_order=[c.replace('_', '=') for c in categorical].sort())
sns.boxplot(ax=axes[1], data=cont_diff_df, x='Covariate',
            y='Mean Absolute Difference', hue='Method', showfliers=False)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.1),
           ncol=3, fontsize=40,
           columnspacing=0.5)
for ax in axes:
    ax.set(xlabel=None)
    ax.get_legend().remove()
axes[0].yaxis.set_major_formatter(ticker.PercentFormatter())
# axes[1].set_ylim([0, 40])
fig.tight_layout()
fig.savefig(f'all_mg.png', bbox_inches='tight')


imp_covs2 = [c.split('_')[0] for c in imp_covs]
for cov, v in Counter(imp_covs2).items():
    if v > 1:
        duplicate_covs = [c for c in imp_covs if cov in c]
        for d in duplicate_covs[1:]:
            imp_covs.remove(d)
print(f'# imp covs: {len(imp_covs)}')
print(imp_covs)

np.random.seed(0)
sample = df_orig.copy(deep=True)
sample = sample[sample['S3'] == 5]
for c in imp_covs:
    if '_' in c:
        cov, val = c.split('_')
        sample = sample[sample[cov] == int(val)]
print(f'Possible matches: {sample.shape[0]}')

imp_covs = [c.split('_')[0] for c in imp_covs]
sample = sample.sample(n=1).loc[:, imp_covs + ['Z']]
sample_num = sample.index[0]

while True:
    iter_number = np.random.randint(0, n_splits*n_repeats)
    if sample_num in linear_prog_c_mg[iter_number].index:
        break
print(f'Pulling example MG from iteration {iter_number}')

lcm_c_mg = df_orig.loc[lcm.get_MGs()[iter_number][0].loc[sample_num], imp_covs + ['Z']]
lcm_t_mg = df_orig.loc[lcm.get_MGs()[iter_number][1].loc[sample_num], imp_covs + ['Z']]
linear_prog_c_mg = df_orig.loc[linear_prog_c_mg[iter_number].loc[sample_num], imp_covs + ['Z']]
linear_prog_t_mg = df_orig.loc[linear_prog_t_mg[iter_number].loc[sample_num], imp_covs + ['Z']]
ensemble_prog_c_mg = df_orig.loc[ensemble_prog_c_mg[iter_number].loc[sample_num], imp_covs + ['Z']]
ensemble_prog_t_mg = df_orig.loc[ensemble_prog_t_mg[iter_number].loc[sample_num], imp_covs + ['Z']]

lcm_mg = pd.concat([lcm_c_mg, lcm_t_mg])
linear_prog_mg = pd.concat([linear_prog_c_mg, linear_prog_t_mg])
ensemble_prog_mg = pd.concat([ensemble_prog_c_mg, ensemble_prog_t_mg])

categorical = list(lcm_mg.dtypes[lcm_mg.dtypes == 'int'].index)
categorical.remove('Z')
categorical.remove('S3')
continuous = list(lcm_mg.dtypes[lcm_mg.dtypes == 'float'].index)
continuous = ['S3'] + continuous

lcm_comps = {}
linear_prog_comps = {}
ensemble_prog_comps = {}

for c in categorical:
    lcm_comps[c] = ((lcm_mg[c] == sample[c].values[0]).astype(int).sum() / (k_est*2))*100
    linear_prog_comps[c] = ((linear_prog_mg[c] == sample[c].values[0]).astype(int).sum() / (
                k_est * 2)) * 100
    ensemble_prog_comps[c] = ((ensemble_prog_mg[c] == sample[c].values[0]).astype(int).sum() / (
                k_est * 2)) * 100
for c in continuous:
    lcm_comps[c] = np.abs(lcm_mg[c] - sample[c].values[0]).mean()
    linear_prog_comps[c] = np.abs(linear_prog_mg[c] - sample[c].values[0]).mean()
    ensemble_prog_comps[c] = np.abs(ensemble_prog_mg[c] - sample[c].values[0]).mean()

lcm_comps['Z'] = np.nan
linear_prog_comps['Z'] = np.nan
ensemble_prog_comps['Z'] = np.nan

lcm_mg = pd.concat([sample, lcm_mg, pd.DataFrame.from_dict([lcm_comps])])
linear_prog_mg = pd.concat([sample, linear_prog_mg, pd.DataFrame.from_dict([linear_prog_comps])])
ensemble_prog_mg = pd.concat([sample, ensemble_prog_mg, pd.DataFrame.from_dict([ensemble_prog_comps])])

lcm_mg = lcm_mg.rename(columns={'Z': 'T'})
linear_prog_mg = linear_prog_mg.rename(columns={'Z': 'T'})
ensemble_prog_mg = ensemble_prog_mg.rename(columns={'Z': 'T'})

lcm_mg.to_latex('school_lcm_mg.tex')
linear_prog_mg.to_latex('school_linear_prog_mg.tex')
ensemble_prog_mg.to_latex('school_ensemble_prog_mg.tex')

cate_df = lcm.cate_df.join(df_orig.drop(columns=['Z', 'Y']))
cate_df = cate_df[['avg.CATE_mean', 'XC', 'S3']]
cate_df = cate_df.rename(columns={'avg.CATE_mean': 'LCM',
                                  'XC': 'Urbanicity (XC)',
                                  'S3': 'Exp Success (S3)'})
cate_df = cate_df.join(linear_prog_cates_full['avg.CATE'])
cate_df = cate_df.rename(columns={'avg.CATE': 'Linear\nPrognostic Score'})
cate_df = cate_df.join(ensemble_prog_cates_full['avg.CATE'])
cate_df = cate_df.rename(columns={'avg.CATE': 'Ensemble\nPrognostic Score'})

cate_df = pd.melt(cate_df, id_vars=['Urbanicity (XC)', 'Exp Success (S3)'],
                  var_name='Method', value_name='Est CATE')

fig = plt.figure()
sns.set_context("paper")
sns.set_style("darkgrid")
sns.set(font_scale=1)
ax = sns.boxplot(data=cate_df, x="Urbanicity (XC)", y="Est CATE", hue='Method', hue_order=method_order, palette=palette, showfliers=False)
sns.move_legend(ax, "lower center", bbox_to_anchor=(.5, 1), ncol=3, title=None)
ax.get_figure().savefig(f'cate_by_xc.png')

plt.figure()
sns.set_context("paper")
sns.set_style("darkgrid")
sns.set(font_scale=1)
ax = sns.boxplot(data=cate_df, x="Exp Success (S3)", y="Est CATE", hue='Method', hue_order=method_order, palette=palette, showfliers=False)
sns.move_legend(ax, "lower center", bbox_to_anchor=(.5, 1), ncol=3, title=None)
ax.get_figure().savefig(f'cate_by_s3.png')

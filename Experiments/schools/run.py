import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os
from collections import Counter
import copy
import pickle

from datagen.dgp_df import dgp_schools_df

from Experiments.helpers import get_mg_comp
from src.variable_imp_matching_mf import VIM_MF
from other_methods import prognostic

save_folder = os.getenv('SAVE_FOLDER')
k_est = 3
random_state = 0
n_splits = 5
n_repeats = 1

df = dgp_schools_df()

lcm = VIM_MF(outcome='Y', treatment='T', data=df,
             n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)

lcm.fit(model='linear')
lcm.create_mgs(k=k_est)
lcm.est_cate(diameter_prune=None)
lcm_cates = lcm.cate_df['CATE_mean']
lcm_cates.to_csv(f'{save_folder}/lcm_cates.csv')

lcm_ates = [lcm_cates.iloc[:, i*n_splits:(i+1)*n_splits].mean().mean() for i in range(n_repeats)]

lcm_ate = np.mean(lcm_ates)
lcm_std = np.std(lcm_ates)
lcm_ci = (lcm_ate - (1.96*lcm_std), lcm_ate + (1.96*lcm_std))
print('LCM Done')

with open(f'{save_folder}/lcm_ate.txt', 'w') as f:
    f.write(f'{lcm_ate} ({lcm_ci[0]},{lcm_ci[1]})')

linear_prog_cates_full, linear_prog_c_mg, linear_prog_t_mg, linear_prog_fi = \
    prognostic.prognostic_cv(outcome='Y', treatment='T', data=df,
                             method='linear', double=True, k_est=k_est,
                             est_method='mean', gen_skf=lcm.split_strategy,
                             diameter_prune=None,
                             return_feature_imp=True,
                             random_state=random_state)
linear_prog_cates = linear_prog_cates_full['CATE']
linear_prog_cates.to_csv(f'{save_folder}/linear_prog_cates.csv')
linear_prog_ates = [linear_prog_cates.iloc[:, i*n_splits:(i+1)*n_splits].mean().mean() for i in range(n_repeats)]
linear_prog_ate = np.mean(linear_prog_ates)
linear_prog_std = np.std(linear_prog_ates)
linear_prog_ci = (linear_prog_ate - (1.96*linear_prog_std), linear_prog_ate + (1.96*linear_prog_std))
print('Linear Prog Done')

with open(f'{save_folder}/linear_prog_ate.txt', 'w') as f:
    f.write(f'{linear_prog_ate} ({linear_prog_ci[0]},{linear_prog_ci[1]})')

ensemble_prog_cates_full, ensemble_prog_c_mg, ensemble_prog_t_mg, ensemble_prog_fi = \
    prognostic.prognostic_cv(outcome='Y', treatment='T', data=df,
                             method='ensemble', double=True, k_est=k_est,
                             est_method='mean',
                             diameter_prune=None, gen_skf=lcm.split_strategy,
                             return_feature_imp=True,
                             random_state=random_state)
ensemble_prog_cates = ensemble_prog_cates_full['CATE']
ensemble_prog_cates.to_csv(f'{save_folder}/ensemble_prog_cates.csv')
ensemble_prog_ates = [ensemble_prog_cates.iloc[:, i*n_splits:(i+1)*n_splits].mean().mean() for i in range(n_repeats)]
ensemble_prog_ate = np.mean(ensemble_prog_ates)
ensemble_prog_std = np.std(ensemble_prog_ates)
ensemble_prog_ci = (ensemble_prog_ate - (1.96*ensemble_prog_std), ensemble_prog_ate + (1.96*ensemble_prog_std))
print('Ensemble Prog Done')

with open(f'{save_folder}/ensemble_prog_ate.txt', 'w') as f:
    f.write(f'{ensemble_prog_ate} ({ensemble_prog_ci[0]},{ensemble_prog_ci[1]})')

df_orig = pd.read_csv(f'{os.getenv("SCHOOLS_FOLDER")}/df.csv')
lcm_cates = lcm.cate_df.join(df_orig.drop(columns=['Z', 'Y']))

lcm_fi_df = pd.DataFrame(lcm.M_list, columns=lcm.covariates)
lcm_fi_df.to_csv(f'{save_folder}/lcm_fi_df.csv')
lcm_top_10 = lcm_fi_df.mean().sort_values(ascending=False).iloc[:10].index
linear_prog_fi_df = pd.DataFrame(linear_prog_fi, columns=lcm.covariates)
linear_prog_fi_df.to_csv(f'{save_folder}/linear_prog_fi_df.csv')
linear_prog_top_10 = linear_prog_fi_df.mean().sort_values(ascending=False).iloc[:10].index
ensemble_prog_fi_df = pd.DataFrame(ensemble_prog_fi, columns=lcm.covariates)
ensemble_prog_fi_df.to_csv(f'{save_folder}/ensemble_prog_fi_df.csv')
ensemble_prog_top_10 = ensemble_prog_fi_df.mean().sort_values(ascending=False).iloc[:10].index
imp_covs = [c for c in lcm_top_10 if c in linear_prog_top_10]
imp_covs = [c for c in imp_covs if c in ensemble_prog_top_10]

imp_covs = ['S3', 'X1', 'X2', 'C1_1', 'C1_13', 'C1_14']

categorical = [c for c in imp_covs if '_' in c]
continuous = [c for c in imp_covs if '_' not in c]
if 'C2' in imp_covs:
    categorical = categorical + ['C2']
    continuous.remove('C2')

lcm_diffs = {}
linear_prog_diffs = {}
ensemble_prog_diffs = {}

idxs = np.concatenate([lcm.split_strategy[i][0] for i in range(n_splits * n_repeats)])

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
    mg_values = df_orig[c].to_numpy()[pd.concat([pd.concat([lcm.get_mgs()[i][0] for i in range(n_splits * n_repeats)]), pd.concat([lcm.get_mgs()[i][1] for i in range(n_splits * n_repeats)])], axis=1).to_numpy()[good_idxs_i]]
    if cov in continuous:
        lcm_diffs[cov] = copy.copy(np.mean(np.abs(mg_values - values), axis=1))
    else:
        lcm_diffs[cov] = copy.copy((np.sum((mg_values != values).astype(int), axis=1) / (k_est*2)))
    mg_values = df_orig[c].to_numpy()[pd.concat(
        [pd.concat([linear_prog_c_mg[i] for i in range(n_splits * n_repeats)]),
         pd.concat([linear_prog_t_mg[i] for i in range(n_splits * n_repeats)])], axis=1).to_numpy()[good_idxs_i]]
    if cov in continuous:
        linear_prog_diffs[cov] = copy.copy(np.mean(np.abs(mg_values - values), axis=1))
    else:
        linear_prog_diffs[cov] = copy.copy((np.sum((mg_values != values).astype(int), axis=1) / (k_est*2)))
    mg_values = df_orig[c].to_numpy()[pd.concat(
        [pd.concat([ensemble_prog_c_mg[i] for i in range(n_splits * n_repeats)]),
         pd.concat(
             [ensemble_prog_t_mg[i] for i in range(n_splits * n_repeats)])],
        axis=1).to_numpy()[good_idxs_i]]
    if cov in continuous:
        ensemble_prog_diffs[cov] = copy.copy(np.mean(np.abs(mg_values - values), axis=1))
    else:
        ensemble_prog_diffs[cov] = copy.copy((np.sum((mg_values != values).astype(int), axis=1) / (k_est*2)))


method_order = ['LCM', 'Linear\nPGM', 'Nonparametric\nPGM']
palette = {method_order[i]: sns.color_palette()[i] for i in range(len(method_order))}


cat_diff_df = []
cont_diff_df = []
for c in categorical:
    this_df = pd.melt(pd.DataFrame([lcm_diffs[c],
                                    linear_prog_diffs[c],
                                    ensemble_prog_diffs[c]],
                                   index=method_order).T,
                      var_name='Method',
                      value_name='% Mismatch')
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
cat_diff_df.to_csv(f'{save_folder}/categorical_diff.csv')
cont_diff_df.to_csv(f'{save_folder}/continuous_diff.csv')

cat_diff_df['% Mismatch'] *= 100

sns.set_context("paper")
sns.set_style("darkgrid")
sns.set(font_scale=3.3)
fig, axes = plt.subplots(1, 2, figsize=(18, 9))
sns.barplot(ax=axes[0], data=cat_diff_df, x='Covariate',
            y='% Mismatch', hue='Method', errorbar=None,
            hue_order=[c.replace('_', '=') for c in categorical].sort())
sns.boxplot(ax=axes[1], data=cont_diff_df, x='Covariate',
            y='Mean Absolute Difference', hue='Method', showfliers=False)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.55, 0.93),
           ncol=3, fontsize=40, handletextpad=0.4,
           columnspacing=0.5)
for ax in axes:
    ax.set(xlabel=None)
    ax.get_legend().remove()
axes[0].yaxis.set_major_formatter(ticker.PercentFormatter())
fig.tight_layout()
fig.savefig(f'{save_folder}/all_mg.png', bbox_inches='tight')

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
for c in imp_covs:
    if '_' in c:
        cov, val = c.split('_')
        sample = sample[sample[cov] == int(val)]

imp_covs = [c.split('_')[0] for c in imp_covs]
this_sample = sample.sample(n=1).loc[:, imp_covs + ['Z']]
sample_num = this_sample.index[0]

print(f'Possible matches: {sample[sample["S3"] == this_sample["S3"].values[0]].shape[0]}')

pickle.dump(lcm.get_mgs(), open(f"{save_folder}/lcm_mgs.pkl", "wb"))
pickle.dump(linear_prog_c_mg, open(f"{save_folder}/linear_prog_c_mg.pkl", "wb"))
pickle.dump(linear_prog_t_mg, open(f"{save_folder}/linear_prog_t_mg.pkl", "wb"))
pickle.dump(ensemble_prog_c_mg, open(f"{save_folder}/ensemble_prog_c_mg.pkl", "wb"))
pickle.dump(ensemble_prog_t_mg, open(f"{save_folder}/ensemble_prog_t_mg.pkl", "wb"))

lcm_mg, linear_prog_mg, ensemble_prog_mg = get_mg_comp(df_orig, sample_num, this_sample,
                                                       lcm.get_mgs(),
                                                       linear_prog_c_mg,
                                                       linear_prog_t_mg,
                                                       ensemble_prog_c_mg,
                                                       ensemble_prog_t_mg,
                                                       n_iters=n_splits*n_repeats,
                                                       treatment='Z',
                                                       ordinal=['S3'],
                                                       k_est=k_est,
                                                       imp_covs=imp_covs)

lcm_mg = lcm_mg.rename(columns={'Z': 'T'})
linear_prog_mg = linear_prog_mg.rename(columns={'Z': 'T'})
ensemble_prog_mg = ensemble_prog_mg.rename(columns={'Z': 'T'})

lcm_mg.to_latex(f'{save_folder}/school_lcm_mg.tex')
linear_prog_mg.to_latex(f'{save_folder}/school_linear_prog_mg.tex')
ensemble_prog_mg.to_latex(f'{save_folder}/school_ensemble_prog_mg.tex')

cate_df = lcm.cate_df.join(df_orig.drop(columns=['Z', 'Y']))
cate_df = cate_df[['avg.CATE_mean', 'XC', 'S3']]
cate_df = cate_df.rename(columns={'avg.CATE_mean': 'LCM',
                                  'XC': 'Urbanicity (XC)',
                                  'S3': 'Exp Success (S3)'})
cate_df = cate_df.join(linear_prog_cates_full['avg.CATE'])
cate_df = cate_df.rename(columns={'avg.CATE': 'Linear\nPGM'})
cate_df = cate_df.join(ensemble_prog_cates_full['avg.CATE'])
cate_df = cate_df.rename(columns={'avg.CATE': 'Nonparametric\nPGM'})

cate_df = pd.melt(cate_df, id_vars=['Urbanicity (XC)', 'Exp Success (S3)'],
                  var_name='Method', value_name='Estimated CATE')


plt.figure(figsize=(8, 6))
sns.set_context("paper")
sns.set_style("darkgrid")
sns.set(font_scale=2)
ax = sns.boxplot(data=cate_df, x="Urbanicity (XC)", y="Estimated CATE", hue='Method', hue_order=method_order, palette=palette, showfliers=False)
sns.move_legend(ax, "lower center", bbox_to_anchor=(.47, 1.02), ncol=3,
                title=None, handletextpad=0.4, columnspacing=0.5, fontsize=20)
plt.tight_layout()
ax.get_figure().savefig(f'{save_folder}/cate_by_xc.png')
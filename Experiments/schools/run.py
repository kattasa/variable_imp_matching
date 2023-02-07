import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
import os
from collections import Counter

from datagen.dgp_df import dgp_schools_df

from src.linear_coef_matching_mf import LCM_MF
from other_methods import prognostic

k_est = 10
random_state = 0
n_splits = 25
n_repeats = 5
# n_splits = 2
# n_repeats = 1

df = dgp_schools_df()

lcm = LCM_MF(outcome='Y', treatment='T', data=df,
             n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)

lcm.fit(model='linear')
lcm.MG(k=k_est)
lcm.CATE()
lcm_cates = lcm.cate_df['CATE_mean']
lcm_cates.to_csv('lcm_cates.csv')

lcm_ates = [lcm_cates.iloc[:, i*n_splits:(i+1)*n_splits].mean().mean() for i in range(n_repeats)]

lcm_ate = np.mean(lcm_ates)
lcm_ci = st.t.interval(confidence=0.95, df=len(lcm_ates)-1, loc=np.mean(lcm_ates), scale=st.sem(lcm_ates))
print('LCM Done')

with open('schools_results/lcm_ate.txt', 'w') as f:
    f.write(f'{lcm_ate} ({lcm_ci[0]},{lcm_ci[1]})')

linear_prog_cates, linear_prog_c_mg, linear_prog_t_mg, linear_prog_fi = \
    prognostic.prognostic_cv(outcome='Y', treatment='T', data=df,
                             method='linear', double=True, k_est=k_est,
                             est_method='mean', gen_skf=lcm.gen_skf,
                             return_feature_imp=True,
                             random_state=random_state)
linear_prog_cates = linear_prog_cates['CATE']
linear_prog_cates.to_csv('linear_prog_cates.csv')
linear_prog_ates = [linear_prog_cates.iloc[:, i*n_splits:(i+1)*n_splits].mean().mean() for i in range(n_repeats)]
linear_prog_ate = np.mean(linear_prog_ates)
linear_prog_ci = st.t.interval(confidence=0.95, df=len(linear_prog_ates)-1, loc=np.mean(linear_prog_ates), scale=st.sem(linear_prog_ates))
print('Linear Prog Done')

with open('schools_results/linear_prog_ate.txt', 'w') as f:
    f.write(f'{linear_prog_ate} ({linear_prog_ci[0]},{linear_prog_ci[1]})')

ensemble_prog_cates, ensemble_prog_c_mg, ensemble_prog_t_mg, ensemble_prog_fi = \
    prognostic.prognostic_cv(outcome='Y', treatment='T', data=df,
                             method='ensemble', double=True, k_est=k_est,
                             est_method='mean', gen_skf=lcm.gen_skf,
                             return_feature_imp=True,
                             random_state=random_state)
ensemble_prog_cates = ensemble_prog_cates['CATE']
ensemble_prog_cates.to_csv('ensemble_prog_cates.csv')
ensemble_prog_ates = [ensemble_prog_cates.iloc[:, i*n_splits:(i+1)*n_splits].mean().mean() for i in range(n_repeats)]
ensemble_prog_ate = np.mean(ensemble_prog_ates)
ensemble_prog_ci = st.t.interval(confidence=0.95, df=len(ensemble_prog_ates)-1, loc=np.mean(ensemble_prog_ates), scale=st.sem(ensemble_prog_ates))
print('Ensemble Prog Done')

with open('schools_results/ensemble_prog_ate.txt', 'w') as f:
    f.write(f'{ensemble_prog_ate} ({ensemble_prog_ci[0]},{ensemble_prog_ci[1]})')

df_orig = pd.read_csv(f'{os.getenv("SCHOOLS_FOLDER")}/df.csv')
lcm_cates = lcm.cate_df.join(df_orig.drop(columns=['Z', 'Y']))

lcm_top_10 = pd.DataFrame(lcm.M_list, columns=lcm.covariates).mean().sort_values(ascending=False).iloc[:8].index
linear_prog_top_10 = pd.DataFrame(linear_prog_fi, columns=lcm.covariates).mean().sort_values().sort_values(ascending=False).iloc[:8].index
ensemble_prog_top_10 = pd.DataFrame(ensemble_prog_fi, columns=lcm.covariates).mean().sort_values().sort_values(ascending=False).iloc[:8].index
imp_covs = [c for c in lcm_top_10 if c in linear_prog_top_10]
imp_covs = [c for c in imp_covs if c in ensemble_prog_top_10]
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
sample = sample[sample['S3'] == 3]
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
continuous = list(lcm_mg.dtypes[lcm_mg.dtypes == 'float'].index)

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
cate_df = cate_df.rename(columns={'avg.CATE_mean': 'Est CATE',
                                  'XC': 'Urbanicity (XC)',
                                  'S3': 'Exp Success (S3)'})

plt.figure()
sns.set_context("paper")
sns.set_style("darkgrid")
sns.set(font_scale=1)
sns.boxplot(data=cate_df, x="Urbanicity (XC)", y="Est CATE")
plt.tight_layout()
plt.savefig(f'cate_by_xc.png')

plt.figure()
sns.set_context("paper")
sns.set_style("darkgrid")
sns.set(font_scale=1)
sns.boxplot(data=cate_df, x="Exp Success (S3)", y="Est CATE")
plt.tight_layout()
plt.savefig(f'cate_by_s3.png')

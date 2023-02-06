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

with open('lcm_ate.txt', 'w') as f:
    f.write(f'{lcm_ate} ({lcm_ci[0]},{lcm_ci[1]})')

prog_cates, prog_c_mg, prog_t_mg, prog_fi = \
    prognostic.prognostic_cv(outcome='Y', treatment='T', data=df,
                             method='ensemble', double=True, k_est=k_est,
                             est_method='mean', gen_skf=lcm.gen_skf,
                             return_feature_imp=True,
                             random_state=random_state)
prog_cates = prog_cates['CATE']
prog_cates.to_csv('prog_cates.csv')
prog_ates = [prog_cates.iloc[:, i*n_splits:(i+1)*n_splits].mean().mean() for i in range(n_repeats)]
prog_ate = np.mean(prog_ates)
prog_ci = st.t.interval(confidence=0.95, df=len(prog_ates)-1, loc=np.mean(prog_ates), scale=st.sem(prog_ates))
print('Prog Done')

with open('prog_ate.txt', 'w') as f:
    f.write(f'{prog_ate} ({prog_ci[0]},{prog_ci[1]})')

df_orig = pd.read_csv(f'{os.getenv("SCHOOLS_FOLDER")}/df.csv')
lcm_cates = lcm.cate_df.join(df_orig.drop(columns=['Z', 'Y']))

lcm_top_10 = pd.DataFrame(lcm.M_list, columns=lcm.covariates).mean().sort_values(ascending=False).iloc[:8].index
prog_top_10 = pd.DataFrame(prog_fi, columns=lcm.covariates).mean().sort_values().sort_values(ascending=False).iloc[:8].index
imp_covs = [c for c in lcm_top_10 if c in prog_top_10]
imp_covs2 = [c.split('_')[0] for c in imp_covs]
for cov, v in Counter(imp_covs2).items():
    if v > 1:
        duplicate_covs = [c for c in lcm_top_10 if cov in c]
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
    if sample_num in prog_c_mg[iter_number].index:
        break
print(f'Pulling example MG from iteration {iter_number}')

lcm_c_mg = df_orig.loc[lcm.get_MGs()[iter_number][0].loc[sample_num], imp_covs + ['Z']]
lcm_t_mg = df_orig.loc[lcm.get_MGs()[iter_number][1].loc[sample_num], imp_covs + ['Z']]
prog_c_mg = df_orig.loc[prog_c_mg[iter_number].loc[sample_num], imp_covs + ['Z']]
prog_t_mg = df_orig.loc[prog_t_mg[iter_number].loc[sample_num], imp_covs + ['Z']]

lcm_mg = pd.concat([lcm_c_mg, lcm_t_mg])
prog_mg = pd.concat([prog_c_mg, prog_t_mg])

categorical = list(lcm_mg.dtypes[lcm_mg.dtypes == 'int'].index)
categorical.remove('Z')
continuous = list(lcm_mg.dtypes[lcm_mg.dtypes == 'float'].index)

lcm_comps = {}
prog_comps = {}

for c in categorical:
    lcm_comps[c] = ((lcm_mg[c] == sample[c].values[0]).astype(int).sum() / (k_est*2))*100
    prog_comps[c] = ((prog_mg[c] == sample[c].values[0]).astype(int).sum() / (
                k_est * 2)) * 100
for c in continuous:
    lcm_comps[c] = np.abs(lcm_mg[c] - sample[c].values[0]).mean()
    prog_comps[c] = np.abs(prog_mg[c] - sample[c].values[0]).mean()

lcm_comps['Z'] = np.nan
prog_comps['Z'] = np.nan

lcm_mg = pd.concat([sample, lcm_mg, pd.DataFrame.from_dict([lcm_comps])])
prog_mg = pd.concat([sample, prog_mg, pd.DataFrame.from_dict([prog_comps])])

lcm_mg = lcm_mg.rename(columns={'Z': 'T'})
prog_mg = prog_mg.rename(columns={'Z': 'T'})

lcm_mg.to_latex('school_lcm_mg.tex')
prog_mg.to_latex('school_prog_mg.tex')

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

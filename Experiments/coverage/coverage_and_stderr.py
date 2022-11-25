
import json
import numpy as np
import os
import pandas as pd
import time
import warnings

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from sklearn.model_selection import RepeatedStratifiedKFold

from datagen.dgp import data_generation_dense_mixed_endo
from other_methods import bart
from src.linear_coef_matching_mf import LCM_MF
import scipy.stats as st
import scipy.special as sp
from sklearn.preprocessing import StandardScaler


warnings.filterwarnings("ignore")
np.random.seed(0)

save_folder = os.getenv('SAVE_FOLDER').replace("'", '').replace('"', '')
n_samples = int(os.getenv('N_SAMPLES'))
n_splits = int(os.getenv('N_SPLITS'))
n_repeats = int(os.getenv('N_REPEATS'))
n_iters = int(os.getenv('N_ITERS'))
k_est = int(os.getenv('K_EST'))


def u(x):
    T = []
    for row in x:
        l = sp.expit(np.sum(row[:2]) - 2 + np.random.normal(0, 1))
        t = int(l > 0.5)
        T.append(t)
    return np.array(T)

df, df_true, binary = data_generation_dense_mixed_endo(num_samples=n_samples, num_cont_imp=5, num_disc_imp=0,
                                                       num_cont_unimp=10, num_disc_unimp=0, std=1.5, t_imp=2, overlap=1)

x_cols = [c for c in df.columns if 'X' in c]
original_x_cols = df[x_cols].iloc[:, :15].to_numpy()
df[x_cols] = StandardScaler().fit_transform(df[x_cols])
bart_cates = []
lcm_cates = []
augmented_lcm_cates = []

start = time.time()
for i in range(n_iters):
    new_T = u(original_x_cols)
    df['T'] = new_T
    df['Y'] = (new_T * df_true['Y1']) + ((1 - new_T) * df_true['Y0'])

    skf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=0)
    split_strategy = list(skf.split(df, df['T']))

    cate_est_bart, bart_control_preds, bart_treatment_preds = bart.bart('Y', 'T', df,
                                                                        gen_skf=split_strategy,
                                                                        n_splits=n_splits, result='full')
    bart_cates.append(cate_est_bart['CATE'])

    bart_control_preds = bart_control_preds.T
    bart_treatment_preds = bart_treatment_preds.T
    bart_control_preds = [bart_control_preds.iloc[i, :].dropna().tolist() for i in range(n_splits*n_repeats)]
    bart_treatment_preds = [bart_treatment_preds.iloc[i, :].dropna().tolist() for i in range(n_splits*n_repeats)]

    lcm = LCM_MF(outcome='Y', treatment='T', data=df, n_splits=n_splits, n_repeats=n_repeats)
    lcm.gen_skf = split_strategy
    lcm.fit(double_model=False)
    lcm.MG(k=k_est)
    lcm.CATE(cate_methods=[['double_linear_pruned', False], ['double_linear_pruned', True]],
             precomputed_control_preds=bart_control_preds,
             precomputed_treatment_preds=bart_treatment_preds)
    lcm_cates.append(lcm.cate_df['CATE_double_linear_pruned'])
    augmented_lcm_cates.append(lcm.cate_df['CATE_double_linear_pruned_augmented'])

    print(f'Iter {i+1}: {time.time() - start}')

ate_dof = (n_splits * n_repeats) - 1
cate_dof = ((n_splits - 1)*n_repeats) - 1
ate = df_true['TE'].mean()
cates = df_true['TE'].to_numpy()

bart_ate_df = pd.DataFrame([[*st.t.interval(alpha=0.95, df=ate_dof, loc=np.mean(l.mean(axis=0).to_numpy()),
                                            scale=st.sem(l.mean(axis=0).to_numpy())),
                             st.sem(l.mean(axis=0).to_numpy())] for l in bart_cates], columns=['lb', 'ub', 'stderr'])
bart_cate_dfs = [pd.DataFrame([*st.t.interval(alpha=0.95, df=cate_dof, loc=l.mean(axis=1).to_numpy(), scale=l.sem(axis=1).to_numpy()),
                             l.sem(axis=1).to_numpy()]).T.rename(columns={0: 'lb', 1: 'ub', 2: 'stderr'}) for l in bart_cates]

lcm_ate_df = pd.DataFrame([[*st.t.interval(alpha=0.95, df=ate_dof, loc=np.mean(l.mean(axis=0).to_numpy()),
                                            scale=st.sem(l.mean(axis=0).to_numpy())),
                             st.sem(l.mean(axis=0).to_numpy())] for l in lcm_cates], columns=['lb', 'ub', 'stderr'])
lcm_cate_dfs = [pd.DataFrame([*st.t.interval(alpha=0.95, df=cate_dof, loc=l.mean(axis=1).to_numpy(), scale=l.sem(axis=1).to_numpy()),
                             l.sem(axis=1).to_numpy()]).T.rename(columns={0: 'lb', 1: 'ub', 2: 'stderr'}) for l in lcm_cates]

aug_lcm_ate_df = pd.DataFrame([[*st.t.interval(alpha=0.95, df=ate_dof, loc=np.mean(l.mean(axis=0).to_numpy()),
                                            scale=st.sem(l.mean(axis=0).to_numpy())),
                             st.sem(l.mean(axis=0).to_numpy())] for l in augmented_lcm_cates], columns=['lb', 'ub', 'stderr'])
aug_lcm_cate_dfs = [pd.DataFrame([*st.t.interval(alpha=0.95, df=cate_dof, loc=l.mean(axis=1).to_numpy(), scale=l.sem(axis=1).to_numpy()),
                             l.sem(axis=1).to_numpy()]).T.rename(columns={0: 'lb', 1: 'ub', 2: 'stderr'}) for l in augmented_lcm_cates]

bart_ate_df['coverage'] = ((bart_ate_df['lb'] <= ate) & (bart_ate_df['ub'] >= ate)).astype(int)
lcm_ate_df['coverage'] = ((lcm_ate_df['lb'] <= ate) & (lcm_ate_df['ub'] >= ate)).astype(int)
aug_lcm_ate_df['coverage'] = ((aug_lcm_ate_df['lb'] <= ate) & (aug_lcm_ate_df['ub'] >= ate)).astype(int)

bart_ate_df['Method'] = 'BART'
lcm_ate_df['Method'] = 'LCM'
aug_lcm_ate_df['Method'] = 'Augmented LCM'
ate_df = pd.concat([bart_ate_df, lcm_ate_df, aug_lcm_ate_df])
ate_df.index = ate_df.index.rename('iter')
ate_df = ate_df.reset_index()

for i in range(n_iters):
    bart_cate_dfs[i]['coverage'] = ((bart_cate_dfs[i]['lb'] <= cates) & (bart_cate_dfs[i]['ub'] >= cates)).astype(int)
    lcm_cate_dfs[i]['coverage'] = ((lcm_cate_dfs[i]['lb'] <= cates) & (lcm_cate_dfs[i]['ub'] >= cates)).astype(int)
    aug_lcm_cate_dfs[i]['coverage'] = ((aug_lcm_cate_dfs[i]['lb'] <= cates) & (aug_lcm_cate_dfs[i]['ub'] >= cates)).astype(int)

    bart_cate_dfs[i]['iter'] = i
    lcm_cate_dfs[i]['iter'] = i
    aug_lcm_cate_dfs[i]['iter'] = i

bart_cate_df = pd.concat(bart_cate_dfs)
lcm_cate_df = pd.concat(lcm_cate_dfs)
aug_lcm_cate_df = pd.concat(aug_lcm_cate_dfs)

bart_cate_df['Method'] = 'BART'
lcm_cate_df['Method'] = 'LCM'
aug_lcm_cate_df['Method'] = 'Augmented LCM'
cate_df = pd.concat([bart_cate_df, lcm_cate_df, aug_lcm_cate_df])
cate_df.index = cate_df.index.rename('sample')
cate_df = cate_df.reset_index()

df_true.to_csv(f'{save_folder}/df_true.csv')
ate_df.to_csv(f'{save_folder}/ate_df.csv')
cate_df.to_csv(f'{save_folder}/cate_df.csv')

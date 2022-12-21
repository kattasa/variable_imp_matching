import copy

import numpy as np
import os
import pandas as pd
import rpy2
import time

from sklearn.model_selection import RepeatedStratifiedKFold

from datagen.dgp_df import dgp_dense_mixed_endo_df
from other_methods import bart
from src.linear_coef_matching_mf import LCM_MF
import scipy.stats as st


save_folder = os.getenv('SAVE_FOLDER').replace("'", '').replace('"', '')
n_samples = int(os.getenv('N_SAMPLES'))
n_splits = int(os.getenv('N_SPLITS'))
n_repeats = int(os.getenv('N_REPEATS'))
n_iters = int(os.getenv('N_ITERS'))
k_est = int(os.getenv('K_EST'))

lcm_cate_method = 'mean'

x_imp = 5
x_unimp = 10

_, df, df_true, x_cols, binary = dgp_dense_mixed_endo_df(n=n_samples, nci=x_imp, ndi=0, ncu=x_unimp, ndu=0, std=1.5,
                                                         t_imp=2, overlap=1000, n_train=0)

bart_ates = []
lcm_ates = []
augmented_lcm_ates = []

start = time.time()
for i in range(n_iters):
    new_T = np.random.binomial(1, 0.5, size=(n_samples,))  # randomize T
    df['T'] = new_T
    df['Y'] = (new_T * df_true['Y1']) + ((1 - new_T) * df_true['Y0'])

    these_bart_ates = []
    these_lcm_ates = []
    these_augmented_lcm_ates = []
    t = 0
    while t < n_repeats:
        random_state = np.random.randint(100000)
        this_df = df.sample(frac=1, replace=True, random_state=random_state).reset_index(drop=True)
        skf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=1, random_state=random_state)
        split_strategy = list(skf.split(df, df['T']))

        try:
            cate_est_bart, bart_control_preds, bart_treatment_preds = bart.bart('Y', 'T', this_df,
                                                                                gen_skf=split_strategy,
                                                                                n_splits=n_splits, result='full')
        except rpy2.rinterface_lib.embedded.RRuntimeError as e:
            print(f'Bart runtime error')
            continue
        these_bart_ates.append(cate_est_bart['CATE'].mean().mean())

        bart_control_preds = bart_control_preds.T
        bart_treatment_preds = bart_treatment_preds.T
        bart_control_preds = [bart_control_preds.iloc[i, :].dropna().tolist() for i in range(n_splits)]
        bart_treatment_preds = [bart_treatment_preds.iloc[i, :].dropna().tolist() for i in range(n_splits)]

        lcm = LCM_MF(outcome='Y', treatment='T', data=this_df, n_splits=n_splits, n_repeats=1)
        lcm.gen_skf = split_strategy
        lcm.fit(double_model=False)
        lcm.MG(k=k_est)
        lcm.CATE(cate_methods=[[lcm_cate_method, False], [lcm_cate_method, True]],
                 precomputed_control_preds=bart_control_preds,
                 precomputed_treatment_preds=bart_treatment_preds)
        these_lcm_ates.append(lcm.cate_df[f'CATE_{lcm_cate_method}'].mean().mean())
        these_augmented_lcm_ates.append(lcm.cate_df[f'CATE_{lcm_cate_method}_augmented'].mean().mean())
        t += 1

    bart_ates.append(copy.deepcopy(these_bart_ates))
    lcm_ates.append(copy.deepcopy(these_lcm_ates))
    augmented_lcm_ates.append(copy.deepcopy(these_augmented_lcm_ates))
    print(f'Iter {i+1}: {time.time() - start}')

ate_dof = n_repeats - 1
ate = df_true['TE'].mean()

bart_ate_df = pd.DataFrame([[*st.t.interval(confidence=0.95, df=ate_dof, loc=np.mean(l), scale=st.sem(l)),
                             st.sem(l)] for l in bart_ates], columns=['lb', 'ub', 'sem'])

lcm_ate_df = pd.DataFrame([[*st.t.interval(confidence=0.95, df=ate_dof, loc=np.mean(l), scale=st.sem(l)),
                            st.sem(l)] for l in lcm_ates], columns=['lb', 'ub', 'sem'])

aug_lcm_ate_df = pd.DataFrame([[*st.t.interval(confidence=0.95, df=ate_dof, loc=np.mean(l), scale=st.sem(l)),
                                st.sem(l)] for l in augmented_lcm_ates], columns=['lb', 'ub', 'sem'])

bart_ate_df['coverage'] = ((bart_ate_df['lb'] <= ate) & (bart_ate_df['ub'] >= ate)).astype(int)
lcm_ate_df['coverage'] = ((lcm_ate_df['lb'] <= ate) & (lcm_ate_df['ub'] >= ate)).astype(int)
aug_lcm_ate_df['coverage'] = ((aug_lcm_ate_df['lb'] <= ate) & (aug_lcm_ate_df['ub'] >= ate)).astype(int)

bart_ate_df['Method'] = 'BART'
lcm_ate_df['Method'] = 'LCM'
aug_lcm_ate_df['Method'] = 'Augmented LCM'
ate_df = pd.concat([bart_ate_df, lcm_ate_df, aug_lcm_ate_df])
ate_df.index = ate_df.index.rename('iter')
ate_df = ate_df.reset_index()

df_true.to_csv(f'{save_folder}/df_true.csv')
ate_df.to_csv(f'{save_folder}/ate_df.csv')

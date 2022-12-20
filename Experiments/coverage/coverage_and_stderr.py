import numpy as np
import os
import pandas as pd
import time
import warnings

from sklearn.model_selection import RepeatedStratifiedKFold

from datagen.dgp_df import dgp_dense_mixed_endo_df
from other_methods import bart
from src.linear_coef_matching_mf import LCM_MF
import scipy.stats as st


warnings.filterwarnings("ignore")
np.random.seed(0)

save_folder = os.getenv('SAVE_FOLDER').replace("'", '').replace('"', '')
n_samples = int(os.getenv('N_SAMPLES'))
n_splits = int(os.getenv('N_SPLITS'))
n_repeats = int(os.getenv('N_REPEATS'))
n_iters = int(os.getenv('N_ITERS'))
k_est = int(os.getenv('K_EST'))

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

    for t in range(n_repeats):
        skf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=1, random_state=0)
        split_strategy = list(skf.split(df, df['T']))

        cate_est_bart, bart_control_preds, bart_treatment_preds = bart.bart('Y', 'T', df,
                                                                            gen_skf=split_strategy,
                                                                            n_splits=n_splits, result='full')
        bart_ates.append([cate_est_bart.iloc[:, (n_splits)*i:(n_splits)*(i+1)].mean().mean() for i in range(n_repeats)])

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
        lcm_ates.append([lcm.cate_df['CATE_double_linear_pruned'].iloc[:, (n_splits)*i:(n_splits)*(i+1)].mean().mean() for i in range(n_repeats)])
        augmented_lcm_ates.append([lcm.cate_df['CATE_double_linear_pruned_augmented'].iloc[:, (n_splits)*i:(n_splits)*(i+1)].mean().mean() for i in range(n_repeats)])

    print(f'Iter {i+1}: {time.time() - start}')

ate_dof = n_repeats - 1
ate = df_true['TE'].mean()

bart_ate_df = pd.DataFrame([[*st.t.interval(alpha=0.95, df=ate_dof, loc=np.mean(l), scale=np.std(l)),
                             np.std(l)] for l in bart_ates], columns=['lb', 'ub', 'stdev'])

lcm_ate_df = pd.DataFrame([[*st.t.interval(alpha=0.95, df=ate_dof, loc=np.mean(l), scale=np.std(l)),
                            np.std(l)] for l in lcm_ates], columns=['lb', 'ub', 'stdev'])

aug_lcm_ate_df = pd.DataFrame([[*st.t.interval(alpha=0.95, df=ate_dof, loc=np.mean(l), scale=np.std(l)),
                                np.std(l)] for l in augmented_lcm_ates], columns=['lb', 'ub', 'stdev'])

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

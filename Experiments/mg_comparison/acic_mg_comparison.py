import numpy as np
import os
import shutil
import pandas as pd
import warnings

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

from other_methods import prognostic
from src.linear_coef_matching import LCM
import pickle

warnings.filterwarnings("ignore")
np.random.seed(1)
random_state = 1

n_imp_covs = 10
only_pos = False
binary_y_label = 'MG % Match'
cont_y_label = 'MG Average Difference'

acic_results_folder = f"{os.getenv('RESULTS_FOLDER')}/{os.getenv('ACIC_FOLDER')}"[:-1]
split_num = int(os.getenv('SPLIT_NUM'))
k_est = int(os.getenv('K_EST'))

with open(f'{acic_results_folder}/split.pkl', 'rb') as f:
    est_idx, train_idx = pickle.load(f)[split_num]


acic_name = acic_results_folder.split('_')[-2].split('-')[-1]
save_folder = f'Results/{acic_name}{"_only_pos" if only_pos else ""}'
if os.path.exists(save_folder):
    shutil.rmtree(save_folder)
os.makedirs(save_folder)

df_train = pd.read_csv(f'{acic_results_folder}/df_dummy_data.csv', index_col=0).loc[train_idx].reset_index(drop=True)
df_est = pd.read_csv(f'{acic_results_folder}/df_dummy_data.csv', index_col=0).loc[est_idx].reset_index(drop=True)
df_true = pd.read_csv(f'{acic_results_folder}/df_true.csv', index_col=0).loc[est_idx].reset_index(drop=True)

lcm = LCM(outcome='Y', treatment='T', data=df_train, random_state=random_state)
lcm.fit(method='linear')
lcm_c_mg, lcm_t_mg, _, _ = lcm.get_matched_groups(df_est, k=k_est)
print('LCM done')

ewl = LCM(outcome='Y', treatment='T', data=df_train, random_state=random_state)
ewl.fit(method='linear', equal_weights=True)
ewl_c_mg, ewl_t_mg, _, _ = ewl.get_matched_groups(df_est, k=k_est)
print('EWL done')

prog = prognostic.Prognostic('Y', 'T', df_train, binary=df_train['Y'].nunique() == 2, random_state=random_state)
_, prog_c_mg, prog_t_mg = prog.get_matched_group(df_est, k=k_est, binary=False)
print('Prog done')

if np.sum(lcm.M > 0) < n_imp_covs:
    raise Exception

lcm_rank = np.array(lcm.covariates)[np.argsort(-lcm.M)]
prog_rank = np.array(prog.cov)[np.argsort(-prog.hc.feature_importances_)]

# imp_covs = [c for c in lcm_imp_covs if c in prog_imp_covs]
imp_covs = [c for c in prog_rank[:n_imp_covs] if c in lcm_rank[:n_imp_covs]]
lcm_rank = [list(lcm_rank).index(t) for t in imp_covs]
lcm_score = lcm.M[np.argsort(-lcm.M)][lcm_rank]
lcm_score = lcm_score / np.sum(lcm.M)
prog_rank = [list(prog_rank).index(t) for t in imp_covs]
prog_score = prog.hc.feature_importances_[np.argsort(-prog.hc.feature_importances_)][prog_rank]
print(f'# Imp: {len(imp_covs)}')
print(imp_covs)
print(f'LCM Rankings: {lcm_rank}')
print(f'Prog Rankings: {prog_rank}')


int_types = np.array(imp_covs)[list(df_est[imp_covs].nunique() <= 2)]
float_types = np.array(imp_covs)[list(df_est[imp_covs].nunique() > 2)]
if len(float_types) > 0:
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer
    sample_ids = pd.read_csv(
        f'/Users/qlanners/projects/linear_coef_matching/datagen/acic_2018/{acic_name}.csv')
    df = pd.read_csv('/Users/qlanners/projects/linear_coef_matching/datagen/acic_2018/covariates/x_preprocessed.csv')
    df = df.join(pd.DataFrame(sample_ids).set_index('sample_id'), on='sample_id', how='inner').reset_index(drop=True)
    df = df.loc[est_idx].reset_index(drop=True)
    df_est[float_types] = df[float_types]

imp_counts = {}
for i in int_types:
    imp_counts[i] = {
        0: df_est[df_est['T'] == 0][i].value_counts().to_dict(),
        1: df_est[df_est['T'] == 1][i].value_counts().to_dict(),
                     }
    imp_counts[i] = {
        0: np.minimum(k_est, df_est.apply(lambda x: imp_counts[i][0][x[i]], axis=1).to_numpy()),
        1: np.minimum(k_est, df_est.apply(lambda x: imp_counts[i][1][x[i]], axis=1).to_numpy())
    }


lcm_diffs = {i: np.array([]) for i in int_types}
ewl_diffs = {i: np.array([]) for i in int_types}
prog_diffs = {i: np.array([]) for i in int_types}
for i in int_types:
     lcm_diffs[i] = np.concatenate([lcm_diffs[i], imp_counts[i][0] -
                                    np.sum(np.transpose(df_est[i].to_numpy()[lcm_c_mg.T.to_numpy()] ==
                                                                   df_est[i].to_numpy()), axis=1) ])
     lcm_diffs[i] = np.concatenate([lcm_diffs[i], imp_counts[i][1] -
                                    np.sum(np.transpose(df_est[i].to_numpy()[lcm_t_mg.T.to_numpy()] ==
                                                                   df_est[i].to_numpy()), axis=1) ])
     ewl_diffs[i] = np.concatenate([ewl_diffs[i], imp_counts[i][0] -
                                    np.sum(np.transpose(df_est[i].to_numpy()[ewl_c_mg.T.to_numpy()] ==
                                                                   df_est[i].to_numpy()), axis=1) ])
     ewl_diffs[i] = np.concatenate([ewl_diffs[i], imp_counts[i][1] -
                                    np.sum(np.transpose(df_est[i].to_numpy()[ewl_t_mg.T.to_numpy()] ==
                                                                   df_est[i].to_numpy()), axis=1) ])
     prog_diffs[i] = np.concatenate([prog_diffs[i], imp_counts[i][0] -
                                    np.sum(np.transpose(df_est[i].to_numpy()[prog_c_mg.T.to_numpy()] ==
                                                                   df_est[i].to_numpy()), axis=1) ])
     prog_diffs[i] = np.concatenate([prog_diffs[i], imp_counts[i][1] -
                                    np.sum(np.transpose(df_est[i].to_numpy()[prog_t_mg.T.to_numpy()] ==
                                                                   df_est[i].to_numpy()), axis=1) ])

lcm_diffs = pd.DataFrame(lcm_diffs)
ewl_diffs = pd.DataFrame(ewl_diffs)
prog_diffs = pd.DataFrame(prog_diffs)
if only_pos:
    for i in int_types:
        lcm_diffs[i] = np.where(pd.concat([df_est[i], df_est[i]]).to_numpy(),
                                lcm_diffs[i].to_numpy(), np.nan)
        ewl_diffs[i] = np.where(pd.concat([df_est[i], df_est[i]]).to_numpy(),
                                ewl_diffs[i].to_numpy(), np.nan)
        prog_diffs[i] = np.where(pd.concat([df_est[i], df_est[i]]).to_numpy(),
                                prog_diffs[i].to_numpy(), np.nan)
lcm_diffs['Method'] = 'Linear Coefficient\nMatching'
ewl_diffs['Method'] = 'Equal Weighted\nLASSO Matching'
prog_diffs['Method'] = 'Prognostic Score\nMatching'
binary_sims = pd.concat([lcm_diffs, ewl_diffs, prog_diffs])
binary_sims[int_types] = (k_est - binary_sims[int_types]) / k_est
binary_sims = pd.melt(binary_sims, id_vars=['Method']).dropna(subset=['value'])
binary_sims = binary_sims.rename(columns={'variable': 'Binary Covariate', 'value': binary_y_label})

if len(float_types) > 0:
    lcm_std = {i: np.array([]) for i in float_types}
    ewl_std = {i: np.array([]) for i in float_types}
    prog_std = {i: np.array([]) for i in float_types}
    for i in float_types:
        lcm_std[i] = np.concatenate([lcm_std[i],
                                     np.mean(np.abs(np.transpose(df_est[i].to_numpy()[lcm_c_mg.T.to_numpy()] -
                                                         df_est[i].to_numpy())), axis=1)
                                     ])
        lcm_std[i] = np.concatenate([lcm_std[i],
                                     np.mean(np.abs(np.transpose(df_est[i].to_numpy()[lcm_t_mg.T.to_numpy()] -
                                                         df_est[i].to_numpy())), axis=1)
                                     ])
        ewl_std[i] = np.concatenate([ewl_std[i],
                                     np.mean(np.abs(np.transpose(df_est[i].to_numpy()[ewl_c_mg.T.to_numpy()] -
                                                         df_est[i].to_numpy())), axis=1)
                                     ])
        ewl_std[i] = np.concatenate([ewl_std[i],
                                     np.mean(np.abs(np.transpose(df_est[i].to_numpy()[ewl_t_mg.T.to_numpy()] -
                                                         df_est[i].to_numpy())), axis=1)
                                     ])
        prog_std[i] = np.concatenate([prog_std[i],
                                     np.mean(np.abs(np.transpose(df_est[i].to_numpy()[prog_c_mg.T.to_numpy()] -
                                                         df_est[i].to_numpy())), axis=1)
                                     ])
        prog_std[i] = np.concatenate([prog_std[i],
                                     np.mean(np.abs(np.transpose(df_est[i].to_numpy()[prog_t_mg.T.to_numpy()] -
                                                         df_est[i].to_numpy())), axis=1)
                                     ])
    lcm_std = pd.DataFrame(lcm_std)
    ewl_std = pd.DataFrame(ewl_std)
    prog_std = pd.DataFrame(prog_std)
    lcm_std['Method'] = 'Linear Coefficient\nMatching'
    ewl_std['Method'] = 'Equal Weighted\nLASSO Matching'
    prog_std['Method'] = 'Prognostic Score\nMatching'
    cont_sims = pd.concat([lcm_std, ewl_std, prog_std])
    cont_sims = pd.melt(cont_sims, id_vars=['Method'])
    cont_sims = cont_sims.rename(columns={'variable': 'Continuous Covariate', 'value': cont_y_label})

if (len(int_types) > 0) & (len(float_types) > 0):
    fig, axes = plt.subplots(1, 2)
    sns.barplot(ax=axes[0], data=binary_sims, x='Binary Covariate', y=binary_y_label, hue='Method')
    sns.boxplot(ax=axes[1], data=cont_sims, x='Continuous Covariate', y=cont_y_label, hue='Method')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='right', bbox_to_anchor=(0.95, 1.05), ncol=3)
    for ax in axes:
        ax.get_legend().remove()
    fig.tight_layout()
    fig.savefig(f'{save_folder}/all_mg.png', bbox_inches='tight')
elif len(int_types) > 0:
    fig = plt.figure()
    sns.barplot(data=binary_sims, x='Binary Covariate', y=binary_y_label, hue='Method')
    plt.ylim(0.95, 1.002)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.tight_layout()
    fig.savefig(f'{save_folder}/all_mg.png', bbox_inches='tight')
elif len(float_types) > 0:
    fig = plt.figure()
    sns.boxplot(data=cont_sims, x='Continuous Covariate', y=cont_y_label, hue='Method')
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.tight_layout()
    fig.savefig(f'{save_folder}/all_mg.png', bbox_inches='tight')


df_est = df_est.join(df_true[['Y0_true', 'Y1_true']])
df_est.to_csv(f'{save_folder}/full_df.csv')

sample = np.random.choice(df_est[df_est['cig_rec_0'] == 1].index)

this_sample = df_est.loc[sample]
if this_sample['T'] == 0:
    prog_sample_mg = prog_t_mg.loc[sample][:5]
    lcm_sample_mg = lcm_t_mg.loc[sample][:5]
    ewl_sample_mg = ewl_t_mg.loc[sample][:5]
else:
    prog_sample_mg = prog_c_mg.loc[sample][:5]
    lcm_sample_mg = lcm_c_mg.loc[sample][:5]
    ewl_sample_mg = ewl_c_mg.loc[sample][:5]

lcm_unimp_covs = np.array(lcm.covariates)[lcm.M == 0]
prog_unimp_covs = np.array(prog.cov)[prog.hc.feature_importances_ == 0]
unimp_covs = [c for c in lcm_unimp_covs if c in prog_unimp_covs]
covs_by_nunique = list(df_est.nunique().sort_values(ascending=False).index)
unimp_covs = [c for c in covs_by_nunique if c in unimp_covs]
unimp_covs = unimp_covs[:len(imp_covs)]

focus_cols = imp_covs + unimp_covs + ['T', 'Y0_true', 'Y1_true', 'Y']

this_sample = df_est.loc[[sample], focus_cols]
lcm_sample_mg = df_est.loc[lcm_sample_mg, focus_cols]
ewl_sample_mg = df_est.loc[ewl_sample_mg, focus_cols]
prog_sample_mg = df_est.loc[prog_sample_mg, focus_cols]
this_sample['Method'] = 'Sample'
lcm_sample_mg['Method'] = 'Linear Coefficient Matching'
ewl_sample_mg['Method'] = 'Equal Weighted LASSO Matching'
prog_sample_mg['Method'] = 'Prognostic Score Matching'

sample_mg = pd.concat([this_sample, lcm_sample_mg, ewl_sample_mg, prog_sample_mg])
sample_mg = sample_mg[['Method'] + focus_cols]
sample_mg.to_csv(f'{save_folder}/sample_mg.csv')

cov_summary = pd.DataFrame(columns=['Info'] +imp_covs)
cov_summary.loc[0] = ['LCM Rank'] + lcm_rank
cov_summary.loc[1] = ['Prog Rank'] + prog_rank
cov_summary.loc[2] = ['LCM Score'] + list(lcm_score)
cov_summary.loc[3] = ['Prog Score'] + list(prog_score)
cov_summary.to_csv(f'{save_folder}/imp_cov_summary.csv')

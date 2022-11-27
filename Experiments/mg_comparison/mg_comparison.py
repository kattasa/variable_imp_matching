import json
import numpy as np
import os
import pandas as pd
import time
import warnings

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

from datagen.dgp_df import dgp_lalonde
from other_methods import pymalts, bart, causalforest, prognostic
from src.linear_coef_matching import LCM
from src.linear_coef_matching_mf import LCM_MF
import pickle

warnings.filterwarnings("ignore")
np.random.seed(0)


df_data, x_cols, discrete = dgp_lalonde()
n_splits = 5
n_repeats = 1
k_est = 25

ate = 886

lcm = LCM_MF(outcome='Y', treatment='T', data=df_data, n_splits=n_splits, n_repeats=n_repeats)
lcm.fit(double_model=False)
lcm.MG(k=k_est)
lcm.CATE(cate_methods=[['linear_pruned', False]])
print('.')

split_strategy = lcm.gen_skf

m = pymalts.malts_mf('Y', 'T', data=df_data, discrete=discrete, k_tr=15, k_est=k_est, n_splits=n_splits,
                     estimator='linear', smooth_cate=False, gen_skf=split_strategy)

cate_est_prog, _, _ = prognostic.prognostic_cv('Y', 'T', df_data, k_est=k_est, gen_skf=split_strategy)


print('hi')
# lcm_c_mg, lcm_t_mg, _, _ = lcm.get_matched_groups(df_est, k=k_est)
# print(np.array(lcm.covariates)[np.argsort(-lcm.M_list[0])][:8])

# prog = prognostic.Prognostic('Y', 'T', df_train, binary=binary)
# _, prog_c_mg, prog_t_mg = prog.get_matched_group(df_est, k=k_est, binary=False)
#
# print('hi')
#
# if np.sum(lcm.M > 0) < n_imp_covs:
#     raise Exception
#
# imp_covs = [c for c in np.array(lcm.covariates)[np.argsort(-lcm.M)][:5] if c in
#             np.array(prog.cov)[np.argsort(-prog.hc.feature_importances_)][:5]]
#
# n_imp_covs = len(imp_covs)
# print(n_imp_covs)
#
# lcm_imp_covs = df_est[imp_covs]
# prog_imp_covs = df_est[imp_covs]
# # lcm_imp_covs = df_est[np.array(lcm.covariates)[np.argsort(-lcm.M)][:n_imp_covs]]
# # prog_imp_covs = df_est[np.array(prog.cov)[np.argsort(-prog.hc.feature_importances_)][:n_imp_covs]]
#
# lcm_c_perc = []
# lcm_t_perc = []
# prog_c_perc = []
# prog_t_perc = []
#
# # for i in range(df_est.shape[0]):
# #     lcm_c_perc.append((lcm_imp_covs.iloc[lcm_c_mg.iloc[i].to_numpy()] == lcm_imp_covs.iloc[i]).sum() / k_est)
# #     lcm_t_perc.append((lcm_imp_covs.iloc[lcm_t_mg.iloc[i].to_numpy()] == lcm_imp_covs.iloc[i]).sum() / k_est)
# #     prog_c_perc.append((prog_imp_covs.iloc[prog_c_mg.iloc[i].to_numpy()] == prog_imp_covs.iloc[i]).sum() / k_est)
# #     prog_t_perc.append((prog_imp_covs.iloc[prog_t_mg.iloc[i].to_numpy()] == prog_imp_covs.iloc[i]).sum() / k_est)
# #     if i % 1000 == 0:
# #         print(i)
#
# for i in range(df_est.shape[0]):
#     lcm_c_perc.append(lcm_imp_covs.iloc[lcm_c_mg.iloc[i].to_numpy()].std())
#     lcm_t_perc.append(lcm_imp_covs.iloc[lcm_t_mg.iloc[i].to_numpy()].std())
#     prog_c_perc.append(prog_imp_covs.iloc[prog_c_mg.iloc[i].to_numpy()].std())
#     prog_t_perc.append(prog_imp_covs.iloc[prog_t_mg.iloc[i].to_numpy()].std())
#     if i % 1000 == 0:
#         print(i)
#
#
#
# lcm_c_perc = pd.concat(lcm_c_perc, axis=1).T
# lcm_c_perc['T'] = 0
# lcm_t_perc = pd.concat(lcm_t_perc, axis=1).T
# lcm_t_perc['T'] = 1
#
# lcm_pos_perc = []
# for c in lcm_imp_covs.columns:
#     lcm_pos_perc.append(lcm_c_perc[['T', c]].loc[lcm_imp_covs[c] == 1])
#     lcm_pos_perc.append(lcm_t_perc[['T', c]].loc[lcm_imp_covs[c] == 1])
#
# prog_c_perc = pd.concat(prog_c_perc, axis=1).T
# prog_c_perc['T'] = 0
# prog_t_perc = pd.concat(prog_t_perc, axis=1).T
# prog_t_perc['T'] = 1
#
# prog_pos_perc = []
# for c in prog_imp_covs.columns:
#     prog_pos_perc.append(prog_c_perc[['T', c]].loc[prog_imp_covs[c] == 1])
#     prog_pos_perc.append(prog_t_perc[['T', c]].loc[prog_imp_covs[c] == 1])
#
#
# lcm_c_perc = lcm_c_perc.melt(id_vars=['T'])
# lcm_t_perc = lcm_t_perc.melt(id_vars=['T'])
# lcm_perc = pd.concat([lcm_c_perc, lcm_t_perc])
# lcm_perc['Cov Rank'] = lcm_perc['variable'].map({lcm_imp_covs.columns[i-1]: i for i in range(1,n_imp_covs+1)})
# lcm_perc['Method'] = 'Linear Coefficient Matching'
#
# prog_c_perc = prog_c_perc.melt(id_vars=['T'])
# prog_t_perc = prog_t_perc.melt(id_vars=['T'])
# prog_perc = pd.concat([prog_c_perc, prog_t_perc])
# prog_perc['Cov Rank'] = prog_perc['variable'].map({prog_imp_covs.columns[i-1]: i for i in range(1,n_imp_covs+1)})
# prog_perc['Method'] = 'Prognostic Score Matching'
#
# all_perc = pd.concat([lcm_perc, prog_perc])
# all_perc = all_perc.rename(columns={'value': '% MG with same value'})
#
# # sns.boxplot(data=all_perc, x="Cov Rank", y="% MG with same value", hue="Method")
# # plt.show()
#
# lcm_pos_perc = pd.concat(lcm_pos_perc).melt(id_vars=['T']).dropna()
# lcm_pos_perc['Cov Rank'] = lcm_pos_perc['variable'].map({lcm_imp_covs.columns[i-1]: i for i in range(1,n_imp_covs+1)})
# lcm_pos_perc['Method'] = 'Linear Coefficient Matching'
# prog_pos_perc = pd.concat(prog_pos_perc).melt(id_vars=['T']).dropna()
# prog_pos_perc['Cov Rank'] = prog_pos_perc['variable'].map({prog_imp_covs.columns[i-1]: i for i in range(1,n_imp_covs+1)})
# prog_pos_perc['Method'] = 'Prognostic Score Matching'
#
# all_pos_perc = pd.concat([lcm_pos_perc, prog_pos_perc])
# all_pos_perc = all_pos_perc.rename(columns={'value': '% MG with same positive value'})
#
# print('hi')
#
# # sns.boxplot(data=all_pos_perc, x="Cov Rank", y="% MG with same positive value", hue="Method")
# # plt.show()
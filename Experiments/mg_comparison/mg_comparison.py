import json
import numpy as np
import os
import pandas as pd
import time
import warnings

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

from Experiments.helpers import get_data
from other_methods import pymalts, bart, causalforest, prognostic
from src.linear_coef_matching import LCM
import pickle

warnings.filterwarnings("ignore")
np.random.seed(0)

acic_folder = '/Users/qlanners/projects/linear_coef_matching/Experiments/acic_error_and_runtime/Results/acic_2019-3_000'
print_progress = True


with open(f'{acic_folder}/config.txt') as c:
    config = json.loads(c.read())
n_splits = config['n_splits']
k_est = config['k_est']
# k_est = 15
n_imp_covs = 3

df_dummy_data = pd.read_csv(f'{acic_folder}/df_dummy_data.csv', index_col=0)

train_size = df_dummy_data.shape[0] // n_splits
df_dummy_data = df_dummy_data.sample(frac=1)
df_train = df_dummy_data.iloc[:train_size].reset_index(drop=True)
df_est = df_dummy_data.iloc[train_size:].reset_index(drop=True)
binary = df_est['Y'].nunique() == 2

lcm = LCM(outcome='Y', treatment='T', data=df_train)
lcm.fit(double_model=False)
lcm_c_mg, lcm_t_mg, _, _ = lcm.get_matched_groups(df_est, k=k_est)
# print(np.array(lcm.covariates)[np.argsort(-lcm.M_list[0])][:8])

prog = prognostic.Prognostic('Y', 'T', df_train, binary=binary)
_, prog_c_mg, prog_t_mg = prog.get_matched_group(df_est, k=k_est, binary=False)

print('hi')

if np.sum(lcm.M > 0) < n_imp_covs:
    raise Exception

imp_covs = [c for c in np.array(lcm.covariates)[np.argsort(-lcm.M)][:5] if c in
            np.array(prog.cov)[np.argsort(-prog.hc.feature_importances_)][:5]]

n_imp_covs = len(imp_covs)
print(n_imp_covs)

lcm_imp_covs = df_est[imp_covs]
prog_imp_covs = df_est[imp_covs]
# lcm_imp_covs = df_est[np.array(lcm.covariates)[np.argsort(-lcm.M)][:n_imp_covs]]
# prog_imp_covs = df_est[np.array(prog.cov)[np.argsort(-prog.hc.feature_importances_)][:n_imp_covs]]

lcm_c_perc = []
lcm_t_perc = []
prog_c_perc = []
prog_t_perc = []

# for i in range(df_est.shape[0]):
#     lcm_c_perc.append((lcm_imp_covs.iloc[lcm_c_mg.iloc[i].to_numpy()] == lcm_imp_covs.iloc[i]).sum() / k_est)
#     lcm_t_perc.append((lcm_imp_covs.iloc[lcm_t_mg.iloc[i].to_numpy()] == lcm_imp_covs.iloc[i]).sum() / k_est)
#     prog_c_perc.append((prog_imp_covs.iloc[prog_c_mg.iloc[i].to_numpy()] == prog_imp_covs.iloc[i]).sum() / k_est)
#     prog_t_perc.append((prog_imp_covs.iloc[prog_t_mg.iloc[i].to_numpy()] == prog_imp_covs.iloc[i]).sum() / k_est)
#     if i % 1000 == 0:
#         print(i)

for i in range(df_est.shape[0]):
    lcm_c_perc.append(lcm_imp_covs.iloc[lcm_c_mg.iloc[i].to_numpy()].std())
    lcm_t_perc.append(lcm_imp_covs.iloc[lcm_t_mg.iloc[i].to_numpy()].std())
    prog_c_perc.append(prog_imp_covs.iloc[prog_c_mg.iloc[i].to_numpy()].std())
    prog_t_perc.append(prog_imp_covs.iloc[prog_t_mg.iloc[i].to_numpy()].std())
    if i % 1000 == 0:
        print(i)



lcm_c_perc = pd.concat(lcm_c_perc, axis=1).T
lcm_c_perc['T'] = 0
lcm_t_perc = pd.concat(lcm_t_perc, axis=1).T
lcm_t_perc['T'] = 1

lcm_pos_perc = []
for c in lcm_imp_covs.columns:
    lcm_pos_perc.append(lcm_c_perc[['T', c]].loc[lcm_imp_covs[c] == 1])
    lcm_pos_perc.append(lcm_t_perc[['T', c]].loc[lcm_imp_covs[c] == 1])

prog_c_perc = pd.concat(prog_c_perc, axis=1).T
prog_c_perc['T'] = 0
prog_t_perc = pd.concat(prog_t_perc, axis=1).T
prog_t_perc['T'] = 1

prog_pos_perc = []
for c in prog_imp_covs.columns:
    prog_pos_perc.append(prog_c_perc[['T', c]].loc[prog_imp_covs[c] == 1])
    prog_pos_perc.append(prog_t_perc[['T', c]].loc[prog_imp_covs[c] == 1])


lcm_c_perc = lcm_c_perc.melt(id_vars=['T'])
lcm_t_perc = lcm_t_perc.melt(id_vars=['T'])
lcm_perc = pd.concat([lcm_c_perc, lcm_t_perc])
lcm_perc['Cov Rank'] = lcm_perc['variable'].map({lcm_imp_covs.columns[i-1]: i for i in range(1,n_imp_covs+1)})
lcm_perc['Method'] = 'Linear Coefficient Matching'

prog_c_perc = prog_c_perc.melt(id_vars=['T'])
prog_t_perc = prog_t_perc.melt(id_vars=['T'])
prog_perc = pd.concat([prog_c_perc, prog_t_perc])
prog_perc['Cov Rank'] = prog_perc['variable'].map({prog_imp_covs.columns[i-1]: i for i in range(1,n_imp_covs+1)})
prog_perc['Method'] = 'Prognostic Score Matching'

all_perc = pd.concat([lcm_perc, prog_perc])
all_perc = all_perc.rename(columns={'value': '% MG with same value'})

# sns.boxplot(data=all_perc, x="Cov Rank", y="% MG with same value", hue="Method")
# plt.show()

lcm_pos_perc = pd.concat(lcm_pos_perc).melt(id_vars=['T']).dropna()
lcm_pos_perc['Cov Rank'] = lcm_pos_perc['variable'].map({lcm_imp_covs.columns[i-1]: i for i in range(1,n_imp_covs+1)})
lcm_pos_perc['Method'] = 'Linear Coefficient Matching'
prog_pos_perc = pd.concat(prog_pos_perc).melt(id_vars=['T']).dropna()
prog_pos_perc['Cov Rank'] = prog_pos_perc['variable'].map({prog_imp_covs.columns[i-1]: i for i in range(1,n_imp_covs+1)})
prog_pos_perc['Method'] = 'Prognostic Score Matching'

all_pos_perc = pd.concat([lcm_pos_perc, prog_pos_perc])
all_pos_perc = all_pos_perc.rename(columns={'value': '% MG with same positive value'})

print('hi')

# sns.boxplot(data=all_pos_perc, x="Cov Rank", y="% MG with same positive value", hue="Method")
# plt.show()
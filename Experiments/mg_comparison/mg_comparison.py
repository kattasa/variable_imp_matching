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
from src.linear_coef_matching_mf import LCM_MF
import pickle

warnings.filterwarnings("ignore")
np.random.seed(0)

acic_folder = '/Users/qlanners/projects/linear_coef_matching/Experiments/acic_error_and_runtime/Results/acic_2018-2e474ffd411b4fb085245e72de71c19a_000'
print_progress = True


with open(f'{acic_folder}/config.txt') as c:
    config = json.loads(c.read())
n_splits = config['n_splits']
k_est = config['k_est']

df_data =pd.read_csv(f'{acic_folder}/df_data.csv', index_col=0)
df_lcm_data =pd.read_csv(f'{acic_folder}/df_lcm_data.csv', index_col=0)

lcm = LCM_MF(outcome='Y', treatment='T', data=df_lcm_data, n_splits=n_splits, n_repeats=1)
lcm.fit(double_model=False)
print(f'Nonzero weights: {[np.sum(m > 0) for m in lcm.M_list]}')
print(np.array(lcm.covariates)[np.argsort(-lcm.M_list[0])][:8])
# lcm.MG(k=k_est)
# if print_progress:
#     print(f'M Nonzero weights: {np.sum(ad_m.M_C != 0)}')
#     print(f'MC >0.5 weights: {np.sum(ad_m.M_C > 0.5)}')

# py_m = pymalts.malts('Y', 'T', df_train, discrete)
# py_m.fit()
# if print_progress:
#     print(f'MALTS fit {time.time() - start}')

prog = prognostic.prognostic_cv('Y', 'T', df_data, k_est=k_est, gen_skf=lcm.gen_skf)

a_mg = ad_m.get_matched_groups(df_est, k=k_est)
m_mg = py_m.get_matched_groups(df_est, k=k_est)
_, p_mg = prog.get_matched_group(df_est=df_est, k=k_est)

ad_m.M_C = np.array([int((nci+ncu) / nci)] * nci + [0] * ncu)
ad_m.M_T = np.array([int((nci+ncu) / nci)] * nci + [0] * ncu)
o_mg = ad_m.get_matched_groups(df_est, k=k_est)

if print_progress:
    print(f'All MG found {time.time() - start}')

sample = np.random.randint(0, df_est.shape[0])
if print_progress:
    print(f'Exploring MG for sample #{sample}.')

a_mg = pd.concat([df_est.loc[a_mg[sample]['control']], df_est.loc[a_mg[sample]['treatment']]])
m_mg = m_mg.loc[sample][1:]
p_mg = pd.concat([df_est.loc[p_mg[sample]['control']], df_est.loc[p_mg[sample]['treatment']]])
o_mg = pd.concat([df_est.loc[o_mg[sample]['control']], df_est.loc[o_mg[sample]['treatment']]])

a_mg.to_csv(f'{save_folder}/admalts_mg.csv')
m_mg.to_csv(f'{save_folder}/malts_mg.csv')
p_mg.to_csv(f'{save_folder}/prognostic_mg.csv')
o_mg.to_csv(f'{save_folder}/psychic_mg.csv')

a_mg_std = a_mg.groupby('T').std().drop(columns=['Y']).T.reset_index().melt(id_vars=['index']).rename(columns={'index': 'Covariate', 'value': 'Standard Deviation'})
a_mg_std['Method'] = 'AdMALTS Lasso'
m_mg_std = m_mg.groupby('T').std().drop(columns=['distance', 'Y']).T.reset_index().melt(id_vars=['index']).rename(columns={'index': 'Covariate', 'value': 'Standard Deviation'})
m_mg_std['Method'] = 'MALTS'
p_mg_std = p_mg.groupby('T').std().drop(columns=['Y']).T.reset_index().melt(id_vars=['index']).rename(columns={'index': 'Covariate', 'value': 'Standard Deviation'})
p_mg_std['Method'] = 'Prognostic Score Lasso'
o_mg_std = o_mg.groupby('T').std().drop(columns=['Y']).T.reset_index().melt(id_vars=['index']).rename(columns={'index': 'Covariate', 'value': 'Standard Deviation'})
o_mg_std['Method'] = 'Psychic Matching'

all_mg_std = pd.concat([a_mg_std, m_mg_std, p_mg_std, o_mg_std]).reset_index(drop=True)

sns.set_context("paper")
sns.set_style("darkgrid")
sns.set(font_scale=1)
fig, axs = plt.subplots(2, 1, figsize=(10, 12))
sns.barplot(x='Covariate', y='Standard Deviation', hue='Method', data=all_mg_std[all_mg_std['T'] == 0], ax=axs[0])
axs[0].set_title('Standard Deviation for each Covariate inside Control Match Groups')
sns.barplot(x='Covariate', y='Standard Deviation', hue='Method', data=all_mg_std[all_mg_std['T'] == 1], ax=axs[1])
axs[1].set_title('Standard Deviation for each Covariate inside Treatment Match Groups')

plt.tight_layout()
fig.savefig(f'{save_folder}/st_dev_in_mgs.png')

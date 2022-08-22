import json
import numpy as np
import pandas as pd
import time

import pymalts

import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.append("..")
from helpers import get_data, create_folder
sys.path.append("..")
from MALTS.amect import Amect
from other_methods.prognostic import prognostic

data = 'dense_continuous'
num_samples = 20000
num_train = 2500
k_est = 50

nci = 45
ndi = 0
ncu = 300
ndu = 0

print_progress = True
config = {'k_est': k_est}
df_train, df_est, df_true, discrete, config = get_data(data, num_samples, config, imp_c=nci, imp_d=ndi, unimp_c=ncu,
                                                       unimp_d=ndu, n_train=num_train)

save_folder = create_folder('mg_comp', print_progress)
with open(f'{save_folder}/config.txt', 'w') as f:
    json.dump(config, f, indent=2)

start = time.time()
ad_m = Amect('Y', 'T', df_train)
ad_m.fit()
if print_progress:
    print(f'AdMALTS fit {time.time() - start}')
    print(f'MC Nonzero weights: {np.sum(ad_m.M_C != 0)}')
    print(f'MC >0.5 weights: {np.sum(ad_m.M_C > 0.5)}')
c_mg, t_mg, _, _ = ad_m.get_matched_groups(df_est, k=k_est)
print(f'MG: {time.time() - start}')
a_cate = ad_m.CATE(df_est, c_mg, t_mg, method='mean')
print(f'Mean CATE: {time.time() - start}')
a_cate = ad_m.CATE(df_est, c_mg, t_mg, method='linear_pruned')
print(f'Linear CATE: {time.time() - start}')

py_m = pymalts.malts('Y', 'T', df_train, discrete)
py_m.fit()
if print_progress:
    print(f'MALTS fit {time.time() - start}')

prog = prognostic('Y', 'T', df_train, 'lasso')
if print_progress:
    print(f'Prognostic fit {time.time() - start}')

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

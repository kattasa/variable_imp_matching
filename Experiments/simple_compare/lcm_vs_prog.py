import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LinearRegression

from src.linear_coef_matching import LCM
from src.linear_coef_matching_mf import LCM_MF
from other_methods.prognostic import Prognostic
from other_methods.pymalts import malts

from Experiments.helpers import get_data, get_acic_data
import warnings

warnings.filterwarnings("ignore")

# 5cc4cad434a74f20aa259898eb07af5d
# 7e4dbafff9bc4714bc4950f086bae4a0
random_state = 1
dataset = 'acic_2018'
acic_file = '5cc4cad434a74f20aa259898eb07af5d'
dataset_config = {
    'num_samples': 2500,
    'imp_c': 15,
    'imp_d': 0,
    'unimp_c': 250,
    'unimp_d': 0,
    'n_train': 1000,
    'alpha': 1
}
k = 10

if 'acic' in dataset:
    df_train, df_est, df_true, binary, categorical, dummy_cols, categorical_to_dummy = get_acic_data(
        dataset, acic_file, n_train=dataset_config['n_train'])
else:
    df_train, df_est, df_true, binary = get_data(data=dataset, config=dataset_config)
    categorical = []

if len(categorical) > 0:
    df_train = df_train.drop(columns=categorical)
    df_est = df_est.drop(columns=categorical)

lcm = LCM(outcome='Y', treatment='T', data=df_train, binary_outcome=False, random_state=random_state)
lcm.fit(model='linear')
match_groups, match_distances = lcm.get_matched_groups(df_estimation=df_est, k=k)
lcm_cates = lcm.CATE(df_estimation=df_est, match_groups=match_groups, match_distances=match_distances, method='mean')

# malts = malts(outcome='Y', treatment='T', data=df_train, discrete=binary)
# malts.fit()
# malts_mg = malts.get_matched_groups(df_estimation=df_est, k=k)
# malts_cates = malts.CATE(MG=malts_mg, outcome_discrete=False, model='mean')

double_lcm = LCM(outcome='Y', treatment='T', data=df_train, binary_outcome=False, random_state=random_state)
double_lcm.fit(model='ensemble', separate_treatments=False)
double_match_groups, double_match_distances = double_lcm.get_matched_groups(df_estimation=df_est, k=k)
double_lcm_cates = lcm.CATE(df_estimation=df_est, match_groups=double_match_groups, match_distances=double_match_distances, method='mean')

prog = Prognostic(Y='Y', T='T', df=df_train, method='ensemble', double=True, random_state=random_state)
prog_cates, prog_c_mg, prog_t_mg = prog.get_matched_group(df_est=df_est, k=k)

# cates = pd.concat([lcm_cates['CATE_mean'], prog_cates['CATE'], double_lcm_cates['CATE_mean'], malts_cates['CATE']], axis=1)
# methods = ['LASSO', 'Prog', 'GBR', 'MALTS']
cates = pd.concat([lcm_cates['CATE_mean'], prog_cates['CATE'], double_lcm_cates['CATE_mean']], axis=1)
methods = ['LASSO', 'Prog', 'GBR']
cates.columns = methods
cates['TE'] = df_true['TE']
for m in methods:
    cates[f'{m} Error'] = np.abs(cates[m] - cates['TE'])

sns.set_context("paper")
sns.set_style("darkgrid")
sns.set(font_scale=6)
fig, _ = plt.subplots(figsize=(40, 50))
sns.boxenplot(x='Method', y='Error',
              data=cates[[f'{m} Error' for m in methods]].melt(var_name='Method', value_name='Error'))
plt.xticks(rotation=65, horizontalalignment='right')
plt.tight_layout()
fig.savefig(f'err.png')
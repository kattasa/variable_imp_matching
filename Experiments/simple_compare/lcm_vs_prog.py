import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LinearRegression

from src.linear_coef_matching import LCM
from other_methods.prognostic import Prognostic
from other_methods.pymalts import malts

from Experiments.helpers import get_data, get_acic_data
import warnings

warnings.filterwarnings("ignore")

# 5cc4cad434a74f20aa259898eb07af5d
# 7e4dbafff9bc4714bc4950f086bae4a0
random_state = 1
dataset = 'friedman'
acic_file = '0a2adba672c7478faa7a47137a87a3ab'
dataset_config = {
    'num_samples': 2500,
    'imp_c': 15,
    'imp_d': 0,
    'unimp_c': 250,
    'unimp_d': 0,
    'n_train': 500,
    'alpha': 2
}
k = 20

if 'acic' in dataset:
    df_train, df_est, df_true, binary, categorical, dummy_cols, categorical_to_dummy = get_acic_data(
        dataset, acic_file, n_train=dataset_config['n_train'])
else:
    df_train, df_est, df_true, binary = get_data(data=dataset, config=dataset_config)
    categorical = []

if len(categorical) > 0:
    df_train = df_train.drop(columns=categorical)
    df_est = df_est.drop(columns=categorical)

print(f'# train: {df_train.shape[0]}')
print(f'# est: {df_est.shape[0]}')
print(f'# Covariates: {df_train.shape[1] - 2}')


def get_dists(df, covs, k):
    nn = NearestNeighbors(n_neighbors=k).fit(df[covs].to_numpy())
    return np.sum(np.abs(df['Y'].to_numpy()[nn.kneighbors(df[covs].to_numpy(), return_distance=False)[:, 1:]] -
                         df[['Y']].to_numpy()))


covs = [c for c in df_train.columns if c not in ['Y', 'T']]
lcm = LCM(outcome='Y', treatment='T', data=df_train, binary_outcome=False, random_state=random_state)
df_c = df_train[df_train['T'] == 0].reset_index(drop=True)
df_t = df_train[df_train['T'] == 1].reset_index(drop=True)

# start = time.time()
# all_diffs = {}
# for x in covs:
#     all_diffs[x] = (get_dists(df_c, [x], k=k), get_dists(df_t, [x], k=k))
# all_diffs = pd.DataFrame.from_dict(all_diffs, orient='index', columns=['C', 'T'])
# this_c_cov = all_diffs.sort_values(by='C').index[0]
# this_t_cov = all_diffs.sort_values(by='T').index[0]
# imp_c_covs = [this_c_cov]
# imp_t_covs = [this_t_cov]
# # prev_c_score = all_diffs.loc[starting_c_cov, 'C']
# # prev_t_score = all_diffs.loc[starting_t_cov, 'T']
#
# start = time.time()
# all_scores = {}
# for x in covs:
#     all_scores[x] = get_dists(df_train, [x], k=0.05)
# a = pd.DataFrame.from_dict(all_scores, orient='index', columns=['Dist'])
# imp_covs = list(a[a['Dist'] < (a.mean() - (std * a.std())).values[0]].index)
# if len(imp_covs) == 0:
#     imp_covs = [a.sort_values(by='Dist').iloc[0].name]
# #
# keep_searching = True
# threshold = 0.03
# while True:
#     keep_searching = False
#     pot_covs = [c for c in covs if c not in imp_covs]
#     np.random.shuffle(pot_covs)
#     all_scores = {}
#     for x in pot_covs:
#         all_scores[x] = get_dists2(df_train, imp_covs + [x], k=5)
#     a = pd.DataFrame.from_dict(all_scores, orient='index', columns=['Dist'])
#     new_covs = list(a[a['Dist'] < (a.mean() - (std * a.std())).values[0]].index)
#     if len(new_covs) == 0:
#         new_covs = [a.sort_values(by='Dist').iloc[0].name]
#         # imp_covs = imp_covs + new_covs
#         new_covs = [k for k, v in (a.loc[new_covs] < get_dists2(df_train, imp_covs, k=5)).to_dict()['Dist'].items() if v is True]
#         if len(new_covs) > 0:
#             imp_covs = imp_covs + new_covs
#         else:
#             break
#     else:
#         imp_covs = imp_covs + new_covs


# keep_searching = True
# threshold = 0.03
# while keep_searching:
#     keep_searching = False
#     pot_covs = [c for c in covs if c not in imp_c_covs]
#     np.random.shuffle(pot_covs)
#     all_c_scores = {}
#     for x in pot_covs:
#         all_c_scores[x] = get_dists(df_c, imp_c_covs + [x], k=k)
#     this_cov, this_c_score = [(k, v.values[0]) for k,v in pd.DataFrame.from_dict(all_c_scores, orient='index', columns=['Dist']).sort_values(by='Dist').iloc[[0]].iterrows()][0]
#     if (prev_c_score - this_c_score) / prev_c_score > threshold:
#         imp_c_covs.append(this_cov)
#         prev_c_score = this_c_score
#         keep_searching = True
#         threshold /= 2
    # for x in pot_covs:
    #     this_c_score = get_dists(df_t, imp_c_covs + [x], k=k)
    #     if (prev_c_score - this_c_score) / prev_c_score > 0.05:
    #         imp_c_covs.append(x)
    #         prev_c_score = this_c_score
    #         keep_searching = True
    #         break

# keep_searching = True
# threshold = 0.03
# while keep_searching:
#     keep_searching = False
#     pot_covs = [c for c in covs if c not in imp_t_covs]
#     np.random.shuffle(pot_covs)
#     all_t_scores = {}
#     for x in pot_covs:
#         all_t_scores[x] = get_dists(df_t, imp_t_covs + [x], k=k)
#     this_cov, this_t_score = [(k, v.values[0]) for k,v in pd.DataFrame.from_dict(all_t_scores, orient='index', columns=['Dist']).sort_values(by='Dist').iloc[[0]].iterrows()][0]
#     if (prev_t_score - this_t_score) / prev_t_score > threshold:
#         imp_t_covs.append(this_cov)
#         prev_t_score = this_t_score
#         keep_searching = True
#         threshold /= 2
    # for x in pot_covs:
    #     this_t_score = get_dists(df_t, imp_t_covs + [x], k=k)
    #     if (prev_t_score - this_t_score) / prev_t_score > 0.05:
    #         imp_t_covs.append(x)
    #         prev_t_score = this_t_score
    #         keep_searching = True
    #         break

# imp_covs = list(set(imp_c_covs + imp_t_covs))
# print(len(imp_covs))
# print(imp_covs)
imp_covs = ['X0', 'X1', 'X2', 'X3', 'X4']
m = np.zeros(df_train.shape[1] - 2)
m[[covs.index(x) for x in imp_covs]] += 1
lcm.M = m
# print(time.time() - start)

k = 10
c_mg, t_mg, c_dist, t_dist = lcm.get_matched_groups(df_estimation=df_est, k=k)
lcm_cates = lcm.CATE(df_estimation=df_est, control_match_groups=c_mg, treatment_match_groups=t_mg, method='mean',
                 augmented=False)


double_lcm = LCM(outcome='Y', treatment='T', data=df_train, binary_outcome=False, random_state=random_state)
double_lcm.fit(method='linear', equal_weights=False, double_model=False)
double_c_mg, double_t_mg, double_c_dist, double_t_dist = double_lcm.get_matched_groups(df_estimation=df_est, k=k)
double_lcm_cates = double_lcm.CATE(df_estimation=df_est, control_match_groups=double_c_mg, treatment_match_groups=double_t_mg, method='mean',
                     augmented=False)

mal = malts(outcome='Y', treatment='T', data=df_train,  k=k)
start = time.time()
mal.fit()
print(time.time() - start)
malts_mg = mal.get_matched_groups(df_estimation=df_est, k=k)
malts_cate = mal.CATE(MG=malts_mg, model='mean')

prog = Prognostic(Y='Y', T='T', df=df_train, method='ensemble', double_model=False, random_state=random_state)
prog_cates, prog_c_mg, prog_t_mg = prog.get_matched_group(df_est=df_est, k=k, est_method='mean')

double_prog = Prognostic(Y='Y', T='T', df=df_train, method='ensemble', double_model=True, random_state=random_state)
double_prog_cates, double_prog_c_mg, double_prog_t_mg = double_prog.get_matched_group(df_est=df_est, k=k, est_method='mean')

# cates = pd.concat([lcm_cates, prog_cates['CATE'], double_lcm_cates, double_prog_cates['CATE']], axis=1)
# methods = ['FS', 'Prog', 'LCM', 'Double Prog']
cates = pd.concat([lcm_cates, prog_cates['CATE'], double_lcm_cates, malts_cate['CATE'], double_prog_cates['CATE']], axis=1)
methods = ['FS', 'Prog', 'LCM', 'MALTS', 'Double Prog']
cates.columns = methods
cates['TE'] = df_true['TE']
for m in methods:
    cates[f'{m} Error'] = np.abs(cates[m] - cates['TE'])

lcm_y0_err = np.abs(np.mean(df_true['Y'].to_numpy()[c_mg.to_numpy()], axis=1) - df_true['Y0_true'].to_numpy())
prog_y0_err = np.abs(np.mean(df_true['Y'].to_numpy()[prog_c_mg.to_numpy()], axis=1) - df_true['Y0_true'].to_numpy())
double_lcm_y0_err = np.abs(np.mean(df_true['Y'].to_numpy()[double_c_mg.to_numpy()], axis=1) - df_true['Y0_true'].to_numpy())
double_prog_y0_err = np.abs(np.mean(df_true['Y'].to_numpy()[double_prog_c_mg.to_numpy()], axis=1) - df_true['Y0_true'].to_numpy())

lcm_y1_err = np.abs(np.mean(df_true['Y'].to_numpy()[t_mg.to_numpy()], axis=1) - df_true['Y1_true'].to_numpy())
prog_y1_err = np.abs(np.mean(df_true['Y'].to_numpy()[prog_t_mg.to_numpy()], axis=1) - df_true['Y1_true'].to_numpy())
double_lcm_y1_err = np.abs(np.mean(df_true['Y'].to_numpy()[double_t_mg.to_numpy()], axis=1) - df_true['Y1_true'].to_numpy())
double_prog_y1_err = np.abs(np.mean(df_true['Y'].to_numpy()[double_prog_t_mg.to_numpy()], axis=1) - df_true['Y1_true'].to_numpy())

fig, ax = plt.subplots(2, 1, figsize=(10, 12))
ax[0].hist(lcm_y0_err, label='lcm', bins=50)
ax[0].hist(prog_y0_err, label='prog', alpha=0.9, bins=50)
ax[0].hist(double_lcm_y0_err, label='double lcm', alpha=0.8, bins=50)
ax[0].hist(double_prog_y0_err, label='double prog', alpha=0.7, bins=50)
ax[1].hist(lcm_y1_err, label='lcm', bins=50)
ax[1].hist(prog_y1_err, label='prog', alpha=0.9, bins=50)
ax[1].hist(double_lcm_y1_err, label='double lcm', alpha=0.8, bins=50)
ax[1].hist(double_prog_y1_err, label='double prog', alpha=0.7, bins=50)
ax[0].set_title('Y0 Error')
ax[1].set_title('Y1 Error')
handles, labels = ax[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center')
fig.savefig(f'y_err.png')

sns.set_context("paper")
sns.set_style("darkgrid")
sns.set(font_scale=6)
fig, _ = plt.subplots(figsize=(40, 50))
sns.boxenplot(x='Method', y='Error',
              data=cates[[f'{m} Error' for m in methods]].melt(var_name='Method', value_name='Error'))
plt.xticks(rotation=65, horizontalalignment='right')
plt.tight_layout()
fig.savefig(f'err.png')

# print()
# print(f"LCM Better: {np.sum(cates['LCM/Prog Error'] < 1)}")
# print(f"Prog Better: {np.sum(cates['LCM/Prog Error'] > 1)}")
# print(f"Tie: {np.sum(cates['LCM/Prog Error'] == 1)}")
# print()
# print(f"LCM Better: {np.sum(cates['LCM/Double Prog Error'] < 1)}")
# print(f"Double Prog Better: {np.sum(cates['LCM/Double Prog Error'] > 1)}")
# print(f"Tie: {np.sum(cates['LCM/Double Prog Error'] == 1)}")

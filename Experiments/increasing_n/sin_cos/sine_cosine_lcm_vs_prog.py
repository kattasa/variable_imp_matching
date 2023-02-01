import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import time

from src.linear_coef_matching import LCM
from other_methods.prognostic import Prognostic

from datagen.dgp_df import dgp_df


def get_dists(df, covs, k):
    nn = NearestNeighbors(n_neighbors=k).fit(df[covs].to_numpy())
    return np.sum(np.abs(df['Y'].to_numpy()[nn.kneighbors(df[covs].to_numpy(), return_distance=False)[:, 1:]] -
                         df[['Y']].to_numpy()))


np.random.seed(99)

random_state = 1
n_train = 500
n_est = 5000
nci = 4
ncu = 16
k = 2
est_method = 'mean'

methods = ['LCM', 'MALTS', 'Linear Prog', 'Ensemble Prog']

all_cates = []
start = time.time()
for i in range(5):
    k = 2
    c = None
    full_medians = []
    df_train, full_df_est, full_df_true, _, _ = dgp_df('sine', n_samples=n_train+n_est, n_unimp=ncu,
                                                       n_train=n_train)

    lcm = LCM(outcome='Y', treatment='T', data=df_train, binary_outcome=False, random_state=random_state)
    df_c = df_train[df_train['T'] == 0].reset_index()
    df_t = df_train[df_train['T'] == 1].reset_index()

    all_diffs = {}
    for x in range(nci + ncu):
        all_diffs[f'X{x}'] = (get_dists(df_c, [f'X{x}'], k=k), get_dists(df_t, [f'X{x}'], k=k))
    all_diffs = pd.DataFrame.from_dict(all_diffs, orient='index', columns=['C', 'T'])
    starting_c_cov = all_diffs.sort_values(by='C').index[0]
    starting_t_cov = all_diffs.sort_values(by='T').index[0]
    imp_c_covs = [starting_c_cov]
    imp_t_covs = [starting_t_cov]
    prev_c_score = all_diffs.loc[starting_c_cov, 'C']
    prev_t_score = all_diffs.loc[starting_t_cov, 'T']

    keep_searching = True
    while keep_searching:
        keep_searching = False
        for x in [c for c in range(nci + ncu) if f'X{c}' not in imp_c_covs]:
            this_c_score = get_dists(df_c, imp_c_covs + [f'X{x}'], k=k)
            if this_c_score < prev_c_score:
                imp_c_covs.append(f'X{x}')
                prev_c_score = this_c_score
                keep_searching = True
                break

    keep_searching = True
    while keep_searching:
        keep_searching = False
        for x in [c for c in range(nci + ncu) if f'X{c}' not in imp_t_covs]:
            this_t_score = get_dists(df_t, imp_t_covs + [f'X{x}'], k=k)
            if this_t_score < prev_t_score:
                imp_t_covs.append(f'X{x}')
                prev_t_score = this_t_score
                keep_searching = True
                break

    imp_covs = list(set(imp_c_covs + imp_t_covs))
    print(imp_c_covs)
    print(imp_t_covs)
    print(imp_covs)
    m = np.zeros(nci + ncu)
    m[[int(c[1:]) for c in imp_covs]] += 1
    # m[list(range(nci))] += 1
    lcm.M = m

    # imp_covs = []
    # for j in range(nci):
    #     c = c*2 if c is not None else None
    #     all_diffs = []
    #     for x in [c for c in range(nci + ncu) if f'X{c}' not in imp_covs]:
    #         these_covs = imp_covs + [f'X{x}']
    #         diffs = []
    #         nn = NearestNeighbors(n_neighbors=k).fit(df_c[these_covs].to_numpy())
    #         if c is not None:
    #             diff_c = []
    #             for a in nn.radius_neighbors(df_c[these_covs].to_numpy(), radius=c, return_distance=False):
    #                 diff_c.append(np.abs(df_c.loc[a[1:], 'Y'] - df_c.loc[a[0], 'Y']).to_numpy())
    #             diff_c = np.concatenate(diff_c).reshape(-1, )
    #         else:
    #             diff_c = np.abs(
    #                 df_c['Y'].to_numpy()[nn.kneighbors(df_c[these_covs].to_numpy(), return_distance=False)[:, 1:]] - df_c[
    #                     ['Y']].to_numpy()).reshape(-1, )
    #         nn = NearestNeighbors(n_neighbors=k).fit(df_t[these_covs].to_numpy())
    #         if c is not None:
    #             diff_t = []
    #             for a in nn.radius_neighbors(df_t[these_covs].to_numpy(), radius=c, return_distance=False):
    #                 diff_t.append(np.abs(df_t.loc[a[1:], 'Y'] - df_t.loc[a[0], 'Y']).to_numpy())
    #             diff_t = np.concatenate(diff_t).reshape(-1, )
    #         else:
    #             diff_t = np.abs(
    #                 df_t['Y'].to_numpy()[nn.kneighbors(df_t[these_covs].to_numpy(), return_distance=False)[:, 1:]] - df_t[
    #                     ['Y']].to_numpy()).reshape(-1, )
    #         this_df = pd.DataFrame(pd.DataFrame([diff_c, diff_t], index=['C', 'T']).T)
    #         this_df['X'] = f'X{x}'
    #         all_diffs.append(this_df.copy(deep=True))
    #     all_diffs = pd.concat(all_diffs)
    #     all_diffs[['C', 'T']] /= all_diffs[['C', 'T']].mean(axis=0)
    #     all_diffs = all_diffs.melt(id_vars=['X'], value_name='Diff')
    #     all_diffs['Diff'] = all_diffs['Diff'].astype('float64')
    #     this_cov = all_diffs.groupby('X')['Diff'].mean().sort_values().index[0]
    #     imp_covs.append(this_cov)
    # print(imp_covs)
    # m = np.zeros(nci + ncu)
    # m[[int(c[1:]) for c in imp_covs]] += 1
    # # m[list(range(nci))] += 1
    # lcm.M = m

    ecm = LCM(outcome='Y', treatment='T', data=df_train, binary_outcome=False, random_state=random_state)
    ecm.fit(method='ensemble', equal_weights=False, double_model=False)

    prog = Prognostic(Y='Y', T='T', df=df_train, method='linear', double_model=False, random_state=random_state)
    prog2 = Prognostic(Y='Y', T='T', df=df_train, method='ensemble', double_model=False, random_state=random_state)

    k = 10
    for n in [50, 100, 250, 500, 1000, 2500, 5000]:
        df_est = full_df_est.iloc[:n]

        c_mg, t_mg, c_dist, t_dist = lcm.get_matched_groups(df_estimation=df_est, k=k)
        lcm_cates = lcm.CATE(df_estimation=df_est, control_match_groups=c_mg, treatment_match_groups=t_mg, method=est_method,
                         augmented=False)

        c_mg2, t_mg2, c_dist2, t_dist2 = ecm.get_matched_groups(df_estimation=df_est, k=k)
        ecm_cates = ecm.CATE(df_estimation=df_est, control_match_groups=c_mg2, treatment_match_groups=t_mg2, method=est_method,
                         augmented=False)

        prog_cates, prog_c_mg, prog_t_mg = prog.get_matched_group(df_est=df_est, k=k, est_method=est_method)

        prog_cates2, prog_c_mg2, prog_t_mg2 = prog2.get_matched_group(df_est=df_est, k=k, est_method='mean')

        cates = pd.concat([lcm_cates, ecm_cates, prog_cates['CATE'], prog_cates2['CATE']], axis=1)
        cates.columns = ['LCM', 'ECM', 'Linear Prog', 'Ensemble Prog']
        cates['TE'] = full_df_true.iloc[:n]['TE']
        cates['Iter'] = i
        cates['# Est Samples'] = n
        cates['LCM Error'] = np.abs(cates['LCM'] - cates['TE'])
        cates['ECM Error'] = np.abs(cates['ECM'] - cates['TE'])
        cates['Linear Prog Error'] = np.abs(cates['Linear Prog'] - cates['TE'])
        cates['Ensemble Prog Error'] = np.abs(cates['Ensemble Prog'] - cates['TE'])

        all_cates.append(cates[['Iter', '# Est Samples', 'LCM Error', 'ECM Error', 'Linear Prog Error',
                                'Ensemble Prog Error']])
    print(f'Iter {i+1} complete: {time.time() - start}')

all_cates = pd.concat(all_cates)
all_cates = all_cates.melt(id_vars=['Iter', '# Est Samples'], var_name='Method', value_name='Error')

all_cates_medians = all_cates.groupby(['Iter', '# Est Samples', 'Method'])['Error'].median().reset_index()

sns.lineplot(data=all_cates_medians, x="# Est Samples", y="Error", hue="Method", style="Method",
             hue_order=['Linear Prog Error', 'LCM Error', 'Ensemble Prog Error', 'ECM Error'],
             style_order=['Linear Prog Error', 'LCM Error', 'Ensemble Prog Error', 'ECM Error'])
plt.savefig('median_errors2.png')

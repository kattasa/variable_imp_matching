import numpy as np
import pandas as pd
import seaborn as sns
import time

from src.linear_coef_matching import LCM
from other_methods.prognostic import Prognostic
from other_methods.pymalts import malts
from sklearn.neighbors import NearestNeighbors

from datagen.dgp import data_generation_dense_mixed_endo
from datagen.dgp_df import create_dense_endo_df
import warnings

warnings.filterwarnings("ignore")

np.random.seed(99)

random_state = 1
acic_file = '8'
n_train = 1000
n_est = 5000
nci = 15
ncu = 125
ndi = 0
ndu = 0
k = 10

all_cates = []
start = time.time()

methods = ['LCM', 'MALTS', 'Linear Prog', 'Ensemble Prog']


# def get_dists(df_c, df_t, covs, k):
#     nn = NearestNeighbors(n_neighbors=k).fit(df_c[covs].to_numpy())
#     diff_c = np.abs(df_c['Y'].to_numpy()[nn.kneighbors(df_c[covs].to_numpy(), return_distance=False)[:, 1:]] -
#                     df_c[['Y']].to_numpy()).reshape(-1, )
#     nn = NearestNeighbors(n_neighbors=k).fit(df_t[covs].to_numpy())
#     diff_t = np.abs(df_t['Y'].to_numpy()[nn.kneighbors(df_t[covs].to_numpy(), return_distance=False)[:, 1:]] -
#                     df_t[['Y']].to_numpy()).reshape(-1, )
#     return np.sum(np.concatenate([diff_c, diff_t]))

def get_dists(df, covs, k):
    nn = NearestNeighbors(n_neighbors=k).fit(df[covs].to_numpy())
    return np.sum(np.abs(df['Y'].to_numpy()[nn.kneighbors(df[covs].to_numpy(), return_distance=False)[:, 1:]] -
                         df[['Y']].to_numpy()))


for i in range(1):
    full_medians = []
    orig_df, orig_df_true, binary, treatment_eff_sec = data_generation_dense_mixed_endo(num_samples=n_train + n_est,
                                                                              num_cont_imp=nci,
                                                                              num_disc_imp=ndi, num_cont_unimp=ncu,
                                                                              num_disc_unimp=ndu, std=1.5, t_imp=2,
                                                                              overlap=1, weights=None)
    for a in [5]:
        k = 10
        df_train, full_df_est, full_df_true, x_cols, binary = create_dense_endo_df(orig_df, orig_df_true, binary,
                                                                                   treatment_eff_sec,
                                                                                   n_train=n_train, alpha=a)
        lcm = LCM(outcome='Y', treatment='T', data=df_train, binary_outcome=False, random_state=random_state)
        df_c = df_train[df_train['T'] == 0].reset_index(drop=True)
        df_t = df_train[df_train['T'] == 1].reset_index(drop=True)

        start = time.time()
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
            pot_covs = [c for c in range(nci + ncu) if f'X{c}' not in imp_c_covs]
            np.random.shuffle(pot_covs)
            all_c_scores = {}
            for x in pot_covs:
                all_c_scores[f'X{x}'] = get_dists(df_c, imp_c_covs + [f'X{x}'], k=k)
            this_cov, this_c_score = [(k, v.values[0]) for k,v in pd.DataFrame.from_dict(all_c_scores, orient='index', columns=['Dist']).sort_values(by='Dist').iloc[[0]].iterrows()][0]
            if (prev_c_score - this_c_score) / prev_c_score > 0.001:
                imp_c_covs.append(this_cov)
                prev_c_score = this_c_score
                keep_searching = True

        keep_searching = True
        while keep_searching:
            keep_searching = False
            pot_covs = [c for c in range(nci + ncu) if f'X{c}' not in imp_t_covs]
            np.random.shuffle(pot_covs)
            all_t_scores = {}
            for x in pot_covs:
                all_t_scores[f'X{x}'] = get_dists(df_t, imp_t_covs + [f'X{x}'], k=k)
            this_cov, this_t_score = [(k, v.values[0]) for k,v in pd.DataFrame.from_dict(all_t_scores, orient='index', columns=['Dist']).sort_values(by='Dist').iloc[[0]].iterrows()][0]
            if (prev_t_score - this_t_score) / prev_t_score > 0.001:
                imp_t_covs.append(this_cov)
                prev_t_score = this_t_score
                keep_searching = True
            # for x in pot_covs:
            #     this_t_score = get_dists(df_t, imp_t_covs + [f'X{x}'], k=k)
            #     if (prev_t_score - this_t_score) / prev_t_score > 0.01:
            #         imp_t_covs.append(f'X{x}')
            #         prev_t_score = this_t_score
            #         keep_searching = True
            #         break

        imp_covs = list(set(imp_c_covs + imp_t_covs))
        # print(imp_c_covs)
        # print(imp_t_covs)
        # print(imp_covs)
        print(len(imp_covs))
        print(imp_covs)
        print(len([c for c in imp_covs if int(c[1:]) not in list(range(nci))]))
        m = np.zeros(nci + ncu)
        m[[int(c[1:]) for c in imp_covs]] += 1
        # m[list(range(nci))] += 1
        lcm.M = m
        print(time.time() - start)

        # all_diffs = {}
        # for x in range(nci + ncu):
        #     all_diffs[f'X{x}'] = get_dists(df_c, df_t, [f'X{x}'], k=k)
        # all_diffs = pd.DataFrame([all_diffs], index=['Diff']).T.sort_values(by='Diff')
        # prev_score = all_diffs.values[0]
        # imp_covs = [all_diffs.index[0]]
        #
        # keep_searching = True
        # while keep_searching:
        #     keep_searching = False
        #     for x in [c for c in range(nci + ncu) if f'X{c}' not in imp_covs]:
        #         this_score = get_dists(df_c, df_t, imp_covs + [f'X{x}'], k=k)
        #         if this_score < prev_score:
        #             imp_covs.append(f'X{x}')
        #             prev_score = this_score
        #             keep_searching = True
        #             break
        # print(imp_covs)
        # m = np.zeros(nci + ncu)
        # m[[int(c[1:]) for c in imp_covs]] += 1
        # # m[list(range(nci))] += 1
        # lcm.M = m

        if 'MALTS' in methods:
            mal = malts(outcome='Y', treatment='T', data=df_train,  k=k)
            start = time.time()
            mal.fit()
            print(time.time() - start)

        prog = Prognostic(Y='Y', T='T', df=df_train, method='linear', double_model=True, random_state=random_state)
        prog2 = Prognostic(Y='Y', T='T', df=df_train, method='ensemble', double_model=True, random_state=random_state)
        k = 10
        for n in [500, 1000, 2500]:
            df_est = full_df_est.iloc[:n]

            c_mg, t_mg, c_dist, t_dist = lcm.get_matched_groups(df_estimation=df_est, k=k)
            lcm_cates = lcm.CATE(df_estimation=df_est, control_match_groups=c_mg, treatment_match_groups=t_mg, method='mean',
                             augmented=False)

            if 'MALTS' in methods:
                malts_mg = mal.get_matched_groups(df_estimation=df_est, k=k)
                malts_cate = mal.CATE(MG=malts_mg, model='mean')

            prog_cates, prog_c_mg, prog_t_mg = prog.get_matched_group(df_est=df_est, k=k, est_method='mean')

            prog_cates2, prog_c_mg2, prog_t_mg2 = prog2.get_matched_group(df_est=df_est, k=k, est_method='mean')

            if 'MALTS' in methods:
                cates = pd.concat([lcm_cates, malts_cate['CATE'], prog_cates['CATE'], prog_cates2['CATE']], axis=1)
            else:
                cates = pd.concat([lcm_cates, prog_cates['CATE'], prog_cates2['CATE']], axis=1)
            cates.columns = methods
            cates['TE'] = full_df_true.iloc[:n]['TE']
            cates['Iter'] = i
            cates['Alpha'] = a
            cates['# Est Samples'] = n
            for m in methods:
                cates[f'{m} Error'] = np.abs(cates[m] - cates['TE'])

            all_cates.append(cates[['Iter', 'Alpha', '# Est Samples'] + [f'{m} Error' for m in methods]])
    print(f'Iter {i+1} complete: {time.time() - start}')

all_cates = pd.concat(all_cates)
all_cates = all_cates.melt(id_vars=['Iter', 'Alpha', '# Est Samples'], var_name='Method', value_name='Error')

all_cates_medians = all_cates.groupby(['Iter', 'Alpha', '# Est Samples', 'Method'])['Error'].median().reset_index()

g = sns.FacetGrid(all_cates_medians, row="Alpha", hue='Method', height=3, aspect=4,
                  hue_order=[f'{m} Error' for m in methods], sharey=False)
g.map(sns.lineplot, '# Est Samples', 'Error')
g.add_legend()
g.savefig('median_errors2.png')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.neighbors import NearestNeighbors

from src.linear_coef_matching import LCM

from datagen.dgp import data_generation_dense_mixed_endo
from datagen.dgp_df import create_dense_endo_df
import warnings

from sklearn.feature_selection import SelectKBest, SelectFromModel, r_regression, f_regression, mutual_info_regression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LassoCV

warnings.filterwarnings("ignore")


random_state = 1
acic_file = '0a2adba672c7478faa7a47137a87a3ab'
n_samples = 132
n_imp = 6
n_unimp = 195
k = 10
std = 3


def get_dists(df, covs, k):
    ys = df['Y'].to_numpy()
    nn = NearestNeighbors(radius=k, metric='manhattan').fit(
        df[covs].to_numpy())
    nn = nn.radius_neighbors(df[covs].to_numpy(), sort_results=True)[1]
    return np.mean(np.abs(
        np.concatenate([ys[nn[i]][1:] - ys[i] for i in range(len(nn))])))


def get_dists2(df, covs, k):
    nn = NearestNeighbors(n_neighbors=k + 1).fit(df[covs].to_numpy())
    return np.sum(np.abs(df['Y'].to_numpy()[nn.kneighbors(df[covs].to_numpy(),
                                                          return_distance=False)[
                                            :, 1:]] -
                         df[['Y']].to_numpy()))


def get_dists3(df, covs, k):
    nn = NearestNeighbors(n_neighbors=k + 1).fit(df[covs].to_numpy())
    dist, neigh = nn.kneighbors(df[covs].to_numpy(), return_distance=True)
    ys = df['Y'].to_numpy()[neigh[:, 1:]]
    dist = (1 / dist[:, 1:])
    dist /= np.repeat(dist.sum(axis=1).reshape(-1, 1), repeats=k, axis=1)
    ys = np.sum(ys * dist, axis=1)
    return np.sum(np.abs(ys - df['Y'].to_numpy()))


all_results = []
for w in range(50):
    np.random.seed(w)

    df, df_true, binary, treatment_eff_sec = \
        data_generation_dense_mixed_endo(num_samples=n_samples,
                                         num_cont_imp=n_imp,
                                         num_disc_imp=0,
                                         num_cont_unimp=n_unimp,
                                         num_disc_unimp=0)
    for alpha in [0, 2, 6, 8]:
        this_round = []
        df_train, _, _, _, _ = \
            create_dense_endo_df(df, df_true, binary, treatment_eff_sec,
                             perc_train=None, n_train=n_samples, alpha=alpha)

        covs = [c for c in df_train.columns if c not in ['Y', 'T']]
        lcm = LCM(outcome='Y', treatment='T', data=df_train,
                  binary_outcome=False, random_state=random_state)

        start = time.time()
        all_scores = {}
        for x in covs:
            # all_scores[x] = get_dists(df_train, [x], k=0.05)
            all_scores[x] = get_dists3(df_train, [x], k=3)
        a = pd.DataFrame.from_dict(all_scores, orient='index', columns=['Dist'])
        qr = np.percentile(a['Dist'], 75) - np.percentile(a['Dist'], 25)
        imp_covs = list(a[a['Dist'] < (a.mean() - (std * qr)).values[0]].index)
        if len(imp_covs) == 0:
            imp_covs = [a.sort_values(by='Dist').iloc[0].name]
        #
        keep_searching = True
        threshold = 0.03
        while len(imp_covs) < n_imp:
            keep_searching = False
            pot_covs = [c for c in covs if c not in imp_covs]
            np.random.shuffle(pot_covs)
            all_scores = {}
            for x in pot_covs:
                all_scores[x] = get_dists3(df_train, imp_covs + [x], k=3)
            a = pd.DataFrame.from_dict(all_scores, orient='index', columns=['Dist'])
            qr = np.percentile(a['Dist'], 75) - np.percentile(a['Dist'], 25)
            new_covs = list(a[a['Dist'] < (a.mean() - (std * qr)).values[0]].index)
            if len(new_covs) == 0:
                new_covs = [a.sort_values(by='Dist').iloc[0].name]
                imp_covs = imp_covs + new_covs
                # new_covs = [k for k, v in (a.loc[new_covs] < get_dists3(df_train, imp_covs, k=3)).to_dict()['Dist'].items() if v is True]
                # if len(new_covs) > 0:
                #     imp_covs = imp_covs + new_covs
                # else:
                #     break
            else:
                imp_covs = imp_covs + new_covs

        this_round.append(['KNN'] + [True if f'X{c}' in imp_covs else False for c in range(n_imp) ])

        a = SelectKBest(r_regression, k=n_imp).fit(df_train[covs], df_train['Y'])
        this_round.append(['R'] + [True if f'X{c}' in a.get_feature_names_out() else False for c in range(n_imp)])

        a = SelectKBest(f_regression, k=n_imp).fit(df_train[covs], df_train['Y'])
        this_round.append(['F'] + [True if f'X{c}' in a.get_feature_names_out() else False for c in range(n_imp)])

        a = SelectKBest(mutual_info_regression, k=n_imp).fit(df_train[covs], df_train['Y'])
        this_round.append(['Info'] + [True if f'X{c}' in a.get_feature_names_out() else False for c in range(n_imp)])

        a = SelectFromModel(GradientBoostingRegressor(), prefit=False, max_features=n_imp).fit(df_train[covs], df_train['Y'])
        this_round.append(['GBR'] + [True if f'X{c}' in a.get_feature_names_out() else False for c in range(n_imp)])

        a = SelectFromModel(LassoCV(), prefit=False, max_features=n_imp).fit(df_train[covs], df_train['Y'])
        this_round.append(['LASSO'] + [True if f'X{c}' in a.get_feature_names_out() else False for c in range(n_imp)])

        this_round = pd.DataFrame(this_round, columns=['Method'] + list(range(n_imp)))
        this_round['Iter'] = w
        this_round['Alpha'] = alpha
        all_results.append(this_round)

all_results = pd.concat(all_results)
all_results = all_results.melt(id_vars=['Iter', 'Alpha', 'Method'],
                               var_name='Covariate', value_name='Recovery')
all_results['Recovery'] = all_results['Recovery'].astype(int)
g = sns.FacetGrid(all_results, aspect=2, row='Covariate')
g.map_dataframe(sns.barplot, x='Alpha', y='Recovery', hue='Method',
                palette='hls',
                hue_order=['KNN', 'LASSO', 'GBR', 'F', 'R', 'Info'])
g.add_legend()
g.savefig('test3.png')

from itertools import combinations
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.neighbors import NearestNeighbors
import warnings


def get_match_groups(df_estimation, k, covariates, treatment, M,
                     return_original_idx=True, check_est_df=True):
    if check_est_df:
        check_df_estimation(df_cols=df_estimation.columns,
                            necessary_cols=covariates + [treatment])
    old_idx = np.array(df_estimation.index)
    df_estimation = df_estimation.reset_index(drop=True)
    X = df_estimation[covariates].to_numpy()
    T = df_estimation[treatment].to_numpy()
    match_groups = {}
    match_distances = {}
    for t in np.unique(T):
        if type(M) == dict:
            weights = M[t]
        else:
            weights = M
        this_X = weights[weights > 0] * X[:, weights > 0]
        this_dist, this_mg = get_nn(this_X, T, treatment=t, k=k)
        match_groups[t] = this_mg
        match_distances[t] = this_dist
    for t in np.unique(T):
        match_groups[t] = pd.DataFrame(np.array(df_estimation.loc[T == t].index)[match_groups[t]])
        match_distances[t] = pd.DataFrame(match_distances[t])
        if return_original_idx:
            match_groups[t] = convert_idx(match_groups[t], old_idx)
            match_distances[t].index = old_idx
    return match_groups, match_distances


def get_nn(X, T, treatment, k):
    nn = NearestNeighbors(n_neighbors=k, leaf_size=50, algorithm='auto',
                          metric='cityblock', n_jobs=10).fit(X[T == treatment])
    return nn.kneighbors(X, return_distance=True)


def convert_idx(mg, idx):
    return pd.DataFrame(idx[mg.to_numpy()], index=idx)


def get_CATES(df_estimation, match_groups, match_distances, outcome,
              covariates, M, method='mean', diameter_prune=None,
              cov_imp_prune=0.01, check_est_df=True):
    if check_est_df:
        check_df_estimation(df_cols=df_estimation.columns, necessary_cols=covariates + [outcome])
    df_estimation, old_idx = check_mg_indices(df_estimation, match_groups,
                                              match_distances)
    potential_outcomes = []
    for t, mgs in match_groups.items():
        if diameter_prune:
            diameters = match_distances[t].iloc[:, -1]
            good_mgs = diameters < (diameters.mean() +
                                    (diameter_prune*diameters.std()))
            these_mgs = mgs.loc[good_mgs].to_numpy()
            mgs_idx = old_idx[good_mgs]
        else:
            these_mgs = mgs.to_numpy()
            mgs_idx = old_idx
        if method == 'mean':
            y_pot = df_estimation[outcome].to_numpy()[these_mgs].mean(axis=1)
        elif 'linear' in method:
            if 'pruned' in method:
                if type(M) == dict:
                    weights = M[t]
                else:
                    weights = M
                imp_covs = prune_covariates(covariates, weights,
                                            prune_level=cov_imp_prune)
            else:
                imp_covs = covariates
            these_mgs = df_estimation[imp_covs + [outcome]].to_numpy()[these_mgs]
            these_samples = df_estimation[imp_covs].to_numpy()
            if diameter_prune:
                these_samples = these_samples[good_mgs]
            y_pot = [linear_cate(these_mgs[i], these_samples[i].reshape(1, -1))
                     for i in range(these_samples.shape[0])]
        else:
            raise Exception(f'CATE Method type {method} not supported. '
                            f'Supported methods are: mean, linear, and '
                            f'linear_pruned.')
        potential_outcomes.append(pd.DataFrame(y_pot, index=mgs_idx,
                                               columns=[f'Y{t}_{method}']))
    cates = pd.concat(potential_outcomes, axis=1).sort_index()
    if len([t for t in list(match_groups.keys()) if t not in [0, 1]]) == 0:
        cates[f'CATE_{method}'] = cates[f'Y1_{method}'] - cates[f'Y0_{method}']
    else:
        for t1, t2 in combinations(list(match_groups.keys()), r=2):
            cates[f'{t2}-{t1}_CATE_{method}'] = cates[f'Y{t2}_{method}'] - cates[f'Y{t1}_{method}']
    return cates


def linear_cate(mg, sample):
    return RidgeCV().fit(mg[:, :-1], mg[:, -1]).predict(sample)[0]


def check_df_estimation(df_cols, necessary_cols):
    missing_cols = [c for c in necessary_cols if c not in df_cols]
    if len(missing_cols) > 0:
        raise Exception(f'df_estimation missing necessary column(s) {missing_cols}')


def check_mg_indices(df_estimation, match_groups, match_distances):
    old_idx = df_estimation.index
    df_estimation = df_estimation.reset_index(drop=True)
    est_nrows = df_estimation.shape[0]
    if not np.all([len(mg) == est_nrows for mg in match_groups.values()]):
        raise Exception(
            f'Match group dataframe sizes do not match size of df_estimation')
    if match_distances is not None:
        if not np.all([len(mg) == est_nrows for mg in
                       match_distances.values()]):
            raise Exception(
                f'Match distances dataframe sizes do not match size of '
                f'df_estimation')
    return df_estimation, old_idx


def prune_covariates(covariates, M, prune_level=0.01):
    imp_covs = []
    while len(imp_covs) == 0:
        imp_covs = list(np.array(covariates)[M >= prune_level * M.shape[0]])
        prune_level *= 0.1
    return imp_covs


def get_model_weights(model, weight_attr, equal_weights, t_in_covs, t):
    if type(weight_attr) == str:
        weights = np.abs(getattr(model, weight_attr).reshape(-1,))
    else:
        weights = weight_attr(model)
    if t_in_covs:
        weights = weights[:-1]
    if np.all(weights == 0):
        warnings.warn(f'Model fit to treatment={t} had all zero weights.')
        return np.ones(len(weights))
    if equal_weights:
        weights = np.where(weights > 0, 1, 0)
    return (weights / np.sum(weights)) * len(weights)

import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.neighbors import NearestNeighbors

from utils import prune_covariates


def sample_match_group(df_estimation, sample_idx, k, covariates, treatment, M, combine_mg=True):
    X = M[M > 0] * df_estimation[covariates[M > 0]].to_numpy()
    T = df_estimation[treatment].to_numpy()
    this_sample = X[sample_idx, :].reshape(1, -1)
    control_nn = NearestNeighbors(n_neighbors=k, leaf_size=50, algorithm='auto',
                                  metric='cityblock', n_jobs=10).fit(X[T == 0])
    treatment_nn = NearestNeighbors(n_neighbors=k, leaf_size=50, algorithm='auto',
                                    metric='cityblock', n_jobs=10).fit(X[T == 1])
    if combine_mg:
        return pd.concat([df_estimation.loc[df_estimation['T'] == 0].iloc[
                              control_nn.kneighbors(this_sample, return_distance=False).reshape(-1)],
               df_estimation.loc[df_estimation['T'] == 1].iloc[
                   treatment_nn.kneighbors(this_sample, return_distance=False).reshape(-1)]])
    else:
        return df_estimation.loc[df_estimation['T'] == 0].iloc[
                   control_nn.kneighbors(this_sample, return_distance=False).reshape(-1)], \
               df_estimation.loc[df_estimation['T'] == 1].iloc[
                   treatment_nn.kneighbors(this_sample, return_distance=False).reshape(-1)]


def sample_linear_cate(mg, covariates, M, treatment, outcome, prune=True):
    if prune:
        imp_covs = prune_covariates(covariates, M)
    else:
        imp_covs = covariates
    return RidgeCV().fit(mg[imp_covs + [treatment]].to_numpy(), mg[outcome].to_numpy()).coef_[-1]


def sample_double_linear_cate(c_mg, t_mg, sample, covariates, M, outcome, prune=True):
    if prune:
        imp_covs = prune_covariates(covariates, M)
    else:
        imp_covs = covariates
    return RidgeCV().fit(t_mg[imp_covs].to_numpy(), t_mg[outcome].to_numpy()).predict(sample)[0] - \
           RidgeCV().fit(c_mg[imp_covs].to_numpy(), c_mg[outcome].to_numpy()).predict(sample)[0]

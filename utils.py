import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.linear_model import RidgeCV
from sklearn.neighbors import NearestNeighbors


def get_match_groups(df_estimation, k, covariates, treatment, M, M_C=None, M_T=None, return_original_idx=True,
                     check_est_df=True):
    if check_est_df:
        check_df_estimation(df_cols=df_estimation.columns, necessary_cols=covariates + [treatment])
    old_idx = np.array(df_estimation.index)
    df_estimation = df_estimation.reset_index(drop=True)
    X = df_estimation[covariates].to_numpy()
    T = df_estimation[treatment].to_numpy()
    control_dist, control_mg, treatment_dist, treatment_mg = get_mg_from_M(X, T, M, M_C, M_T, k)
    control_mg = pd.DataFrame(np.array(df_estimation.loc[df_estimation['T'] == 0].index)[control_mg])
    treatment_mg = pd.DataFrame(np.array(df_estimation.loc[df_estimation['T'] == 1].index)[treatment_mg])
    control_dist = pd.DataFrame(control_dist)
    treatment_dist = pd.DataFrame(treatment_dist)
    if return_original_idx:
        control_dist.index = old_idx
        treatment_dist.index = old_idx
        return convert_idx(control_mg, old_idx), convert_idx(treatment_mg, old_idx), control_dist, treatment_dist
    return control_mg, treatment_mg, control_dist, treatment_dist


def get_mg_from_M(X, T, M, M_C, M_T, k):
    if M is not None:
        X = M[M > 0] * X[:, M > 0]
        control_dist, control_mg = get_nn(X, T, treatment=0, k=k)
        treatment_dist, treatment_mg = get_nn(X, T, treatment=1, k=k)
    else:
        X_C = M_C[M_C > 0] * X[:, M_C > 0]
        X_T = M_T[M_T > 0] * X[:, M_T > 0]
        control_dist, control_mg = get_nn(X_C, T, treatment=0, k=k)
        treatment_dist, treatment_mg = get_nn(X_T, T, treatment=1, k=k)
    return control_dist, control_mg, treatment_dist, treatment_mg


def get_nn(X, T, treatment, k):
    nn = NearestNeighbors(n_neighbors=k, leaf_size=50, algorithm='auto', metric='cityblock', n_jobs=10).fit(X[T == treatment])
    return nn.kneighbors(X, return_distance=True)


def convert_idx(mg, idx):
    return pd.DataFrame(idx[mg.to_numpy()], index=idx)


def mg_to_training_set(df_estimation, control_mg, treatment_mg, covariates, treatment, outcome, augmented=False,
                       control_preds=None, treatment_preds=None):
    all_matches = np.concatenate([control_mg.to_numpy(), treatment_mg.to_numpy()], axis=1)
    if augmented:
        return np.concatenate([df_estimation[covariates + [treatment]].to_numpy()[all_matches],
                               np.expand_dims(df_estimation[outcome].to_numpy()[all_matches] -
                                              (((1 - df_estimation[treatment].to_numpy()[all_matches]) *
                                                control_preds[all_matches]) +
                                               (df_estimation[treatment].to_numpy()[all_matches] *
                                                treatment_preds[all_matches])), axis=2)], axis=2)
    else:
        return df_estimation[covariates + [treatment, outcome]].to_numpy()[all_matches]


def get_CATES(df_estimation, control_mg, treatment_mg, method, covariates, outcome, treatment, M,
              augmented=False, control_preds=None, treatment_preds=None, check_est_df=True, random_state=None):
    if check_est_df:
        check_df_estimation(df_cols=df_estimation.columns, necessary_cols=covariates + [outcome])
    df_estimation, old_idx = check_mg_indices(df_estimation, control_mg.shape[0], treatment_mg.shape[0])
    method_full_name = f'CATE_{method}{"_augmented" if augmented else ""}'
    if method == 'mean':
        cates = df_estimation[outcome].to_numpy()[treatment_mg.to_numpy()].mean(axis=1) - \
                df_estimation[outcome].to_numpy()[control_mg.to_numpy()].mean(axis=1)
    else:
        if 'pruned' in method:
            imp_covs = prune_covariates(covariates, M)
        else:
            imp_covs = covariates
        mg = mg_to_training_set(df_estimation, control_mg, treatment_mg, imp_covs, treatment, outcome,
                                augmented, control_preds, treatment_preds)
        mg_size = mg.shape[1] // 2
        method = method.replace('_pruned', '')
        if method == 'linear':
            if augmented:
                mg_cates = np.array([linear_augmented_cate(mg[i, :, :]) for i in range(mg.shape[0])])
            else:
                mg_cates = np.array([linear_cate(mg[i, :, :]) for i in range(mg.shape[0])])
        elif (method == 'double_linear') or (method == 'rf'):
            samples = df_estimation[imp_covs].to_numpy()[control_mg.index]
            control_mg = mg[:, :mg_size, :]
            treatment_mg = mg[:, mg_size:, :]
            if method == 'double_linear':
                mg_cates = np.array([dual_linear_cate(control_mg[i, :, :], treatment_mg[i, :, :], samples[i].reshape(1, -1)) for i in
                                     range(mg.shape[0])])
            elif method == 'rf':
                mg_cates = np.array([rf_cate(control_mg[i, :, :], treatment_mg[i, :, :], samples[i].reshape(1, -1), random_state=random_state) for i in
                                     range(mg.shape[0])])
        else:
            raise Exception(f'CATE Method type {method} not supported. Supported methods are: mean, linear, '
                            f'double_linear and rf.')
        if augmented:
            cates = treatment_preds - control_preds + mg_cates
        else:
            cates = mg_cates
    return pd.Series(cates, index=old_idx, name=method_full_name)


def linear_cate(mg):
    return RidgeCV().fit(mg[:, :-1], mg[:, -1]).coef_[-1]


def linear_augmented_cate(mg):
    return RidgeCV(fit_intercept=False).fit(mg[:, :-1], mg[:, -1]).coef_[-1]


def dual_linear_cate(c_mg, t_mg, this_sample):
    mc = RidgeCV().fit(c_mg[:, :-2], c_mg[:, -1])
    mt = RidgeCV().fit(t_mg[:, :-2], t_mg[:, -1])
    return mt.predict(this_sample)[0] - \
           mc.predict(this_sample)[0]


def rf_cate(c_mg, t_mg, this_sample, random_state=None):
    mc = RFR(random_state=random_state).fit(c_mg[:, :-2], c_mg[:, -1])
    mt = RFR(random_state=random_state).fit(t_mg[:, :-2], t_mg[:, -1])
    return mt.predict(this_sample)[0] - \
           mc.predict(this_sample)[0]


def check_df_estimation(df_cols, necessary_cols):
    missing_cols = [c for c in necessary_cols if c not in df_cols]
    if len(missing_cols) > 0:
        raise Exception(f'df_estimation missing necessary column(s) {missing_cols}')


def check_mg_indices(df_estimation, control_nrows, treatment_nrows):
    old_idx = df_estimation.index
    df_estimation = df_estimation.reset_index(drop=True)
    est_nrows = df_estimation.shape[0]
    if (est_nrows == control_nrows) and (est_nrows == treatment_nrows):
        return df_estimation, old_idx
    raise Exception(f'Control match group dataframe missing {len([c for c in df_idx if c not in control_idx])} and '
                    f'treatment match group dataframe missing {len([c for c in df_idx if c not in treatment_idx])} '
                    f'samples that are present in estimation dataframe.')


def prune_covariates(covariates, M):
    imp_covs = []
    prune_level = 0.01
    while len(imp_covs) == 0:
        imp_covs = list(np.array(covariates)[M >= prune_level * M.shape[0]])
        prune_level *= 0.1
    return imp_covs

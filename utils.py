import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.linear_model import RidgeCV, LogisticRegression
from sklearn.neighbors import NearestNeighbors


def get_match_groups(df_estimation, k, covariates, treatment, M_C, M_T, return_original_idx=True, check_est_df=True):
    if check_est_df:
        check_df_estimation(df_cols=df_estimation.columns, necessary_cols=covariates + [treatment])
    old_idx = np.array(df_estimation.index)
    df_estimation = df_estimation.reset_index(drop=True)
    X = df_estimation[covariates].to_numpy()
    T = df_estimation[treatment].to_numpy()
    X_C = X[:, M_C > 0]
    X_T = X[:, M_T > 0]
    M_C = M_C[M_C > 0]
    M_T = M_T[M_T > 0]
    control_nn = NearestNeighbors(n_neighbors=k, leaf_size=50, algorithm='kd_tree', n_jobs=10).fit(M_C * X_C[T == 0])
    treatment_nn = NearestNeighbors(n_neighbors=k, leaf_size=50, algorithm='kd_tree', n_jobs=10).fit(M_T * X_T[T == 1])
    control_dist, control_mg = control_nn.kneighbors(M_C * X_C, return_distance=True)
    treatment_dist, treatment_mg = treatment_nn.kneighbors(M_T * X_T, return_distance=True)
    control_mg = pd.DataFrame(np.array(df_estimation.loc[df_estimation['T'] == 0].index)[control_mg])
    treatment_mg = pd.DataFrame(np.array(df_estimation.loc[df_estimation['T'] == 1].index)[treatment_mg])
    control_dist = pd.DataFrame(control_dist)
    treatment_dist = pd.DataFrame(treatment_dist)
    if return_original_idx:
        control_dist.index = old_idx
        treatment_dist.index = old_idx
        return convert_idx(control_mg, old_idx), convert_idx(treatment_mg, old_idx), control_dist, treatment_dist
    return control_mg, treatment_mg, control_dist, treatment_dist


def convert_idx(mg, idx):
    return pd.DataFrame(idx[mg.to_numpy()], index=idx)


def get_CATES(df_estimation, control_mg, treatment_mg, method, covariates, outcome, treatment, model_C, model_T, MC, MT,
              augmented=False, propensity_model=None, control_preds=None, treatment_preds=None, check_est_df=True):
    if check_est_df:
        check_df_estimation(df_cols=df_estimation.columns, necessary_cols=covariates + [outcome])
    df_estimation, old_idx = check_mg_indices(df_estimation, control_mg.shape[0], treatment_mg.shape[0])
    if augmented:
        if control_preds is None:
            control_preds = model_C.predict(df_estimation[covariates])
        if treatment_preds is None:
            treatment_preds = model_T.predict(df_estimation[covariates])
    if method == 'model':
        model_cates = treatment_preds - control_preds
        mg_cates = np.zeros(model_cates.shape)
    elif method == 'mean':
        if augmented:
            model_cates = treatment_preds[treatment_mg.to_numpy()].mean(axis=1) - control_preds[
                control_mg.to_numpy()].mean(axis=1)

            if propensity_model is None:
                propensity_model = LogisticRegression().fit(df_estimation[covariates], df_estimation[treatment])
            prop_scores = propensity_model.predict_proba(df_estimation[covariates])[:, 1]
            treatment_prop_weights = (1 / prop_scores)[treatment_mg.to_numpy()] * (1 / (1 / prop_scores)[
                treatment_mg.to_numpy()].sum(axis=1))[:, None]
            control_prop_weights = (1 / (1 - prop_scores))[control_mg.to_numpy()] * (1 / (1 / (1 - prop_scores))[
                control_mg.to_numpy()].sum(axis=1))[:, None]

            mg_cates = ((df_estimation[outcome].to_numpy() - treatment_preds)[
                            treatment_mg.to_numpy()] * treatment_prop_weights).sum(axis=1) - \
                       ((df_estimation[outcome].to_numpy() - control_preds)[
                            control_mg.to_numpy()] * control_prop_weights).sum(axis=1)
        else:
            mg_cates = df_estimation[outcome].to_numpy()[treatment_mg.to_numpy()].mean(axis=1) - \
                    df_estimation[outcome].to_numpy()[control_mg.to_numpy()].mean(axis=1)
    else:
        if 'pruned' in method:
            control_covs = list(np.array(covariates)[MC > 0])
            treatment_covs = list(np.array(covariates)[MT > 0])
        else:
            control_covs = covariates
            treatment_covs = covariates
        if augmented:
            model_cates = treatment_preds - control_preds
            control_mg = np.concatenate([df_estimation[control_covs].to_numpy(),
                                         (df_estimation[outcome].to_numpy() - control_preds).reshape(-1, 1)],
                                        axis=1)[control_mg.reset_index().to_numpy()]
            treatment_mg = np.concatenate([df_estimation[treatment_covs].to_numpy(),
                                           (df_estimation[outcome].to_numpy() - treatment_preds).reshape(-1, 1)],
                                          axis=1)[treatment_mg.reset_index().to_numpy()]
        else:
            control_mg = df_estimation[control_covs + [outcome]].to_numpy()[control_mg.reset_index().to_numpy()]
            treatment_mg = df_estimation[treatment_covs + [outcome]].to_numpy()[treatment_mg.reset_index().to_numpy()]
        if 'linear' in method:
            model_type = 'linear'
        elif 'rf' in method:
            model_type = 'rf'
        else:
            model_type = 'linear'
        mg_cates = np.array([cate_model_pred(treatment_mg[i, :, :], model_type) for i in range(treatment_mg.shape[0])]) - \
                   np.array([cate_model_pred(control_mg[i, :, :], model_type) for i in range(control_mg.shape[0])])
    if augmented:
        cates = model_cates + mg_cates
    else:
        cates = mg_cates
    return pd.Series(cates, index=old_idx, name=f'CATE_{method}')


def cate_model_pred(a, model_type='linear'):
    if model_type == 'linear':
        m = RidgeCV().fit(a[1:, :-1], a[1:, -1])
    elif model_type == 'rf':
        m = RFR().fit(a[1:, :-1], a[1:, -1])
    return m.predict(a[0, :-1].reshape(1, -1))[0]


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

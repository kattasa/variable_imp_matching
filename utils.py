import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor as RFR, RandomForestClassifier as RFC
from sklearn.linear_model import RidgeCV, LogisticRegressionCV
from sklearn.neighbors import NearestNeighbors


def get_match_groups(df_estimation, k, covariates, treatment, M, return_original_idx=True, check_est_df=True):
    if check_est_df:
        check_df_estimation(df_cols=df_estimation.columns, necessary_cols=covariates + [treatment])
    old_idx = np.array(df_estimation.index)
    df_estimation = df_estimation.reset_index(drop=True)
    X = df_estimation[covariates].to_numpy()
    T = df_estimation[treatment].to_numpy()
    X = M[M > 0] * X[:, M > 0]
    control_nn = NearestNeighbors(n_neighbors=k, leaf_size=50, algorithm='kd_tree', n_jobs=10).fit(X[T == 0])
    treatment_nn = NearestNeighbors(n_neighbors=k, leaf_size=50, algorithm='kd_tree', n_jobs=10).fit(X[T == 1])
    control_dist, control_mg = control_nn.kneighbors(X, return_distance=True)
    treatment_dist, treatment_mg = treatment_nn.kneighbors(X, return_distance=True)
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


def get_CATES(df_estimation, control_mg, treatment_mg, method, covariates, outcome, treatment, model_C, model_T, M,
              augmented=False, control_preds=None, treatment_preds=None, check_est_df=True):
    if check_est_df:
        check_df_estimation(df_cols=df_estimation.columns, necessary_cols=covariates + [outcome])
    df_estimation, old_idx = check_mg_indices(df_estimation, control_mg.shape[0], treatment_mg.shape[0])
    if method == 'mean':
        mg_cates = df_estimation[outcome].to_numpy()[treatment_mg.to_numpy()].mean(axis=1) - \
                df_estimation[outcome].to_numpy()[control_mg.to_numpy()].mean(axis=1)
    else:
        if 'pruned' in method:
            imp_covs = list(np.array(covariates)[M >= 0.01*M.shape[0]])
        else:
            imp_covs = covariates
        if augmented:
            if control_preds is None:
                control_preds = model_C.predict(df_estimation[covariates])
            if treatment_preds is None:
                treatment_preds = model_T.predict(df_estimation[covariates])
            model_cates = treatment_preds - control_preds
            control_mg = np.concatenate([df_estimation[imp_covs].to_numpy()[control_mg.reset_index().to_numpy()],
                                      np.expand_dims(((df_estimation[treatment].to_numpy()[control_mg.reset_index().to_numpy()] * control_preds[control_mg.reset_index().to_numpy()]) +
                                                      ((1 - df_estimation[treatment].to_numpy()[control_mg.reset_index().to_numpy()])*treatment_preds[control_mg.reset_index().to_numpy()])), axis=2)],
                                        axis=2)
            treatment_mg = np.concatenate([df_estimation[imp_covs].to_numpy()[treatment_mg.reset_index().to_numpy()],
                                      np.expand_dims(((df_estimation[treatment].to_numpy()[treatment_mg.reset_index().to_numpy()] * control_preds[treatment_mg.reset_index().to_numpy()]) +
                                                      ((1 - df_estimation[treatment].to_numpy()[treatment_mg.reset_index().to_numpy()])*treatment_preds[treatment_mg.reset_index().to_numpy()])), axis=2)],
                                        axis=2)

        else:
            control_mg = df_estimation[imp_covs + [outcome]].to_numpy()[control_mg.reset_index().to_numpy()]
            treatment_mg = df_estimation[imp_covs + [outcome]].to_numpy()[treatment_mg.reset_index().to_numpy()]
        if 'linear' in method:
            model_type = 'linear'
        elif 'rf' in method:
            model_type = 'rf'
        else:
            raise Exception(f'CATE Method type {method} not supported. Supported methods are: mean, linear, and rf.')
        binary = df_estimation[outcome].nunique() == 2
        mg_cates = np.array([cate_model_pred(control_mg[i, :, :], treatment_mg[i, :, :],
                                             method=model_type, binary=binary) for i in range(control_mg.shape[0])])
    if augmented:
        cates = model_cates + mg_cates
    else:
        cates = mg_cates
    return pd.Series(cates, index=old_idx, name=f'CATE_{method}')


def cate_model_pred(c_mg, t_mg, method, binary=False):
    if binary:
        mc = CustomClassifier(method=method).fit(c_mg[1:, :-1], c_mg[1:, -1])
        mt = CustomClassifier(method=method).fit(t_mg[1:, :-1], t_mg[1:, -1])
        return mt.predict_proba(c_mg[0, :-1].reshape(1, -1)) - \
               mc.predict_proba(c_mg[0, :-1].reshape(1, -1))
    elif method == 'linear':
        mc = RidgeCV().fit(c_mg[1:, :-1], c_mg[1:, -1])
        mt = RidgeCV().fit(t_mg[1:, :-1], t_mg[1:, -1])
    elif method == 'rf':
        mc = RFR().fit(c_mg[1:, :-1], c_mg[1:, -1])
        mt = RFR().fit(t_mg[1:, :-1], t_mg[1:, -1])
    # print()
    # print(mc.score(c_mg[1:, :-1], c_mg[1:, -1]))
    # print(mt.score(t_mg[1:, :-1], t_mg[1:, -1]))
    return mt.predict(c_mg[0, :-1].reshape(1, -1))[0] - \
           mc.predict(c_mg[0, :-1].reshape(1, -1))[0]


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


class CustomClassifier:
    def __init__(self, method):
        self.method = method
        self.model = None
        self.label = None

    def fit(self, x, y):
        if np.unique(y).shape[0] >= 2:
            if self.method == 'linear':
                self.model = LogisticRegressionCV().fit(x, y)
            elif self.method == 'rf':
                self.model = RFC().fit(x, y)
        else:
            self.label = y[0]
        return self

    def predict_proba(self, x):
        if self.model is not None:
            return self.model.predict_proba(x)[0][1]
        return self.label

    def score(self, x, y):
        if self.model is not None:
            return self.model.score(x, y)
        return 1.0

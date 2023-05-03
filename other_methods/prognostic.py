"""Prognostic Score Matching CATE Estimator."""

import numpy as np
import pandas as pd
import sklearn.ensemble as ensemble
import sklearn.linear_model as linear
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors

from utils import prune_covariates, linear_cate, get_model_weights


class Prognostic:
    """Fits prognostic score model to df using specified method.

    Parameters
    ----------
    Y : str
        Column label corresponding to the outcome.
    T : str
        Column label corresponding to the treatment.
    df : pandas.DataFrame
        Data that must include Y and T columns specified above.
    method : str, default='ensemble'
        Prognostic score model to fit. Options are 'ensemble' and 'linear'.
    double: bool, default=False
        Whether to fit a prognostic score to both the treated and control space
        (True) or only use a prognostic score model fit on the control space
        (False).
    random_state : None or int, default=None
        Random state to run on.
    """
    def __init__(self, Y, T, df, method='ensemble', double=False,
                 random_state=None):
        self.Y = Y
        self.T = T
        self.cov = [c for c in df.columns if c not in [Y, T]]
        self.double = double
        self.binary_outcome = df[self.Y].nunique() == 2
        self.model = method

        df_c = df.loc[df[T] == 0]
        Xc, Yc = df_c[self.cov].to_numpy(), df_c[Y].to_numpy()
        if self.double:
            df_t = df.loc[df[T] == 1]
            Xt, Yt = df_t[self.cov].to_numpy(), df_t[Y].to_numpy()
        if method == 'ensemble':
            if self.binary_outcome:
                self.hc = ensemble.GradientBoostingClassifier(
                    random_state=random_state).fit(Xc, Yc)
            else:
                self.hc = ensemble.GradientBoostingRegressor(
                    random_state=random_state).fit(Xc, Yc)
            if self.double:
                if self.binary_outcome:
                    self.ht = ensemble.GradientBoostingClassifier(
                        random_state=random_state).fit(Xc, Yc)
                else:
                    self.ht = ensemble.GradientBoostingRegressor(
                        random_state=random_state).fit(Xt, Yt)
        elif method == 'linear':
            if self.binary_outcome:
                self.hc = linear.LogisticRegressionCV(
                    penalty='l1', solver='saga',
                    max_iter=500, random_state=random_state).fit(Xc, Yc)
            else:
                self.hc = linear.LassoCV(random_state=random_state).fit(Xc, Yc)
            if self.double:
                if self.binary_outcome:
                    self.ht = linear.LogisticRegressionCV(
                        penalty='l1', solver='saga',
                        max_iter=500, random_state=random_state).fit(Xt, Yt)
                else:
                    self.ht = linear.LassoCV(random_state=random_state).fit(Xt, Yt)

    def get_matched_group(self, df_est, k=10, method='mean', diameter_prune=3):
        """Gets match groups for df_est using fitted prognostic score model."""
        X_est, Y_est, T_est = df_est[self.cov].to_numpy(), \
                              df_est[self.Y].to_numpy(), \
                              df_est[self.T].to_numpy()
        if self.double:
            if self.binary_outcome:
                hat_Y = np.concatenate([self.hc.predict_proba(X_est)[:, [1]],
                                        self.ht.predict_proba(X_est)[:, [1]]],
                                       axis=1)
            else:
                hat_Y = np.concatenate([self.hc.predict(X_est).reshape(-1, 1),
                                        self.ht.predict(X_est).reshape(-1, 1)],
                                       axis=1)
        else:
            if self.binary_outcome:
                hat_Y = self.hc.predict_proba(X_est)[:, [1]]
            else:
                hat_Y = self.hc.predict(X_est).reshape(-1, 1)
        control_nn = NearestNeighbors(n_neighbors=k, leaf_size=50,
                                      algorithm='auto',
                                      n_jobs=10).fit(hat_Y[T_est == 0])
        c_dist, c_mg = control_nn.kneighbors(hat_Y)
        treatment_nn = NearestNeighbors(n_neighbors=k, leaf_size=50,
                                        algorithm='auto',
                                        n_jobs=10).fit(hat_Y[T_est == 1])
        t_dist, t_mg = treatment_nn.kneighbors(hat_Y)

        if method == 'mean':
            yc = df_est[T_est == 0][self.Y].to_numpy()[c_mg].mean(axis=1)
            yt = df_est[T_est == 1][self.Y].to_numpy()[t_mg].mean(axis=1)
        elif method == 'linear_pruned':
            if self.model == 'linear':
                model_weight_attr = 'coef_'
            else:
                model_weight_attr = 'feature_importances_'
            M = get_model_weights(self.hc, model_weight_attr, False, False, 0)
            imp_covs = prune_covariates(self.cov, M)
            these_mgs = df_est[T_est == 0][imp_covs + [self.Y]].to_numpy()[c_mg]
            these_samples = df_est[imp_covs].to_numpy()
            yc = [linear_cate(these_mgs[i], these_samples[i].reshape(1, -1))
                  for i in range(these_samples.shape[0])]
            if self.double:
                M = get_model_weights(self.ht, model_weight_attr, False, False,
                                      1)
                imp_covs = prune_covariates(self.cov, M)
                these_samples = df_est[imp_covs].to_numpy()
            these_mgs = df_est[T_est == 1][imp_covs + [self.Y]].to_numpy()[
                t_mg]
            yt = [linear_cate(these_mgs[i], these_samples[i].reshape(1, -1))
                  for i in range(these_samples.shape[0])]
        if diameter_prune:
            c_diam = c_dist[:, -1]
            yc = np.where(
                c_diam <= np.mean(c_diam) + (diameter_prune*np.std(c_diam)), yc,
                np.nan)
            t_diam = t_dist[:, -1]
            yt = np.where(
                t_diam <= np.mean(t_diam) + (diameter_prune * np.std(t_diam)), yt,
                np.nan)
        df_mg = pd.DataFrame([yc, yt, T_est]).T
        df_mg.columns = ['Yc', 'Yt', 'T']
        df_mg['CATE'] = df_mg['Yt'] - df_mg['Yc']

        df_mg.index = df_est.index
        df_mg = df_mg.loc[~df_mg['CATE'].isna()]
        c_mg = pd.DataFrame(np.array(df_est.loc[T_est == 0].index)[c_mg])
        c_mg.index = df_est.index
        t_mg = pd.DataFrame(np.array(df_est.loc[T_est == 1].index)[t_mg])
        t_mg.index = df_est.index
        return df_mg, c_mg, t_mg


def prognostic_cv(outcome, treatment, data, method='ensemble', double=False,
                  k_est=1, est_method='mean', n_splits=5, gen_skf=None,
                  diameter_prune=None,
                  return_feature_imp=False, random_state=None):
    """Generates CATE estimates using prognostic score matching."""
    if gen_skf is None:
        skf = StratifiedKFold(n_splits=n_splits)
        gen_skf = skf.split(data, data[treatment])
    cate_est = pd.DataFrame()
    control_mgs = []
    treatment_mgs = []
    feature_imps = []
    for est_idx, train_idx in gen_skf:
        df_train = data.iloc[train_idx]
        df_est = data.iloc[est_idx]
        prog = Prognostic(outcome, treatment, df_train, method=method,
                          double=double, random_state=random_state)
        if return_feature_imp:
            feature_imps.append(get_feature_imp(prog))
        prog_mg, c_mgs, t_mgs = prog.get_matched_group(df_est, k=k_est,
                                                       diameter_prune=diameter_prune,
                                                       method=est_method)
        control_mgs.append(c_mgs)
        treatment_mgs.append(t_mgs)
        cate_est_i = pd.DataFrame(prog_mg['CATE'])
        cate_est = pd.concat([cate_est, cate_est_i], join='outer', axis=1)
    cate_est = cate_est.sort_index()
    cate_est['avg.CATE'] = cate_est.mean(axis=1)
    cate_est['std.CATE'] = cate_est.std(axis=1)
    if return_feature_imp:
        return cate_est, control_mgs, treatment_mgs, feature_imps
    return cate_est, control_mgs, treatment_mgs


def get_feature_imp(prog):
    """Gets feature importance measures for fitted prognostic score model."""
    if prog.model == 'linear':
        weight_attr = 'coef_'
    elif prog.model == 'ensemble':
        weight_attr = 'feature_importances_'
    if prog.double:
        M = [get_model_weights(prog.hc, weight_attr, False, 0, 0),
             get_model_weights(prog.ht, weight_attr, False, 0, 0)]
        M = sum(M) / 2
    else:
        M = get_model_weights(prog.hc, weight_attr, False, 0, 0)
    return M
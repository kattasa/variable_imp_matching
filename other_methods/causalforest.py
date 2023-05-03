"""Causal Forest CATE Estimator implemented using R grf package and rpy2."""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from rpy2.robjects.packages import importr
import rpy2.robjects as ro
from rpy2.robjects.conversion import localconverter
import rpy2.robjects.numpy2ri
from rpy2.robjects import pandas2ri

rpy2.robjects.numpy2ri.activate()
pandas2ri.activate()

base = importr('base')
grf = importr('grf')


def causalforest(outcome, treatment, data, n_splits=2, result='brief',
                 gen_skf=None, random_state=0):
    """Generates CATE estimates with Causal Forest."""
    if gen_skf is None:
        skf = StratifiedKFold(n_splits=n_splits)
        gen_skf = skf.split(data, data[treatment])
    cate_est = pd.DataFrame()
    covariates = [c for c in data.columns if c not in [outcome, treatment]]
    for est_idx, train_idx in gen_skf:
        df_train = data.iloc[train_idx]
        df_est = data.iloc[est_idx]
        Ycrf = df_train[outcome]
        Tcrf = df_train[treatment]
        X = df_train[covariates]
        Xtest = df_est[covariates]

        crf = grf.causal_forest(X, Ycrf, Tcrf, seed=random_state)
        tauhat = grf.predict_causal_forest(crf, Xtest)
        # t_hat_crf = np.array(tauhat[0])
        with localconverter(ro.default_converter + pandas2ri.converter):
            tauhat = ro.conversion.rpy2py(tauhat)
        t_hat_crf = np.array(tauhat['predictions'])

        cate_est_i = pd.DataFrame(t_hat_crf, index=df_est.index, columns=['CATE'])
        cate_est = pd.concat([cate_est, cate_est_i], join='outer', axis=1)

    cate_est = cate_est.sort_index()
    cate_est['avg.CATE'] = cate_est.mean(axis=1)
    cate_est['std.CATE'] = cate_est.std(axis=1)
    if result == 'full':
        return cate_est, crf
    return cate_est

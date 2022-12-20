from econml.dml import CausalForestDML
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from econml.sklearn_extensions.linear_model import WeightedLassoCV
from sklearn.linear_model import LogisticRegressionCV


def causalforest_dml(outcome, treatment, data, n_splits=2, gen_skf=None, random_state=None):
    if gen_skf is None:
        skf = StratifiedKFold(n_splits=n_splits)
        gen_skf = skf.split(data, data[treatment])
    covariates = [c for c in data.columns if c not in [outcome, treatment]]
    cate_est = pd.DataFrame()
    for est_idx, train_idx in gen_skf:
        df_train = data.iloc[train_idx]
        df_est = data.iloc[est_idx]
        X = np.array(df_train.loc[:, covariates])
        Y = np.array(df_train.loc[:, outcome])
        T = np.array(df_train.loc[:, treatment])
        X_est = np.array(df_est.loc[:, covariates])
        est = CausalForestDML(model_y=WeightedLassoCV(max_iter=5000),
                              model_t=LogisticRegressionCV(solver='sag', max_iter=500),
                              featurizer=None, treatment_featurizer=None, random_state=random_state)
        est.fit(Y=Y, T=T, X=X, W=X)
        this_te_est = est.effect(X=X_est)
        this_index = df_est.index
        cate_est_i = pd.DataFrame(this_te_est, index=this_index, columns=['CATE'])
        cate_est = pd.concat([cate_est, cate_est_i], join='outer', axis=1)

    cate_est = cate_est.sort_index()
    cate_est['avg.CATE'] = cate_est.mean(axis=1)
    cate_est['std.CATE'] = cate_est.std(axis=1)
    return cate_est


def causalforest_dml_sample(outcome, treatment, df_train, sample, covariates, random_state=None):
    X = np.array(df_train.loc[:, covariates])
    Y = np.array(df_train.loc[:, outcome])
    T = np.array(df_train.loc[:, treatment])
    est = CausalForestDML(model_y=WeightedLassoCV(max_iter=5000),
                          model_t=LogisticRegressionCV(solver='sag', max_iter=500),
                          featurizer=None, treatment_featurizer=None, random_state=random_state)
    est.fit(Y=Y, T=T, X=X, W=X)
    return est.effect(sample)

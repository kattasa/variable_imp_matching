from econml.dml import SparseLinearDML
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import time


def doubleml(outcome, treatment, data, n_splits=2, gen_skf=None):
    if gen_skf is None:
        skf = StratifiedKFold(n_splits=n_splits)
        gen_skf = skf.split(data, data[treatment])
    cate_est = pd.DataFrame()
    for est_idx, train_idx in gen_skf:
        df_train = data.iloc[train_idx]
        df_est = data.iloc[est_idx]

        covariates = list(set(data.columns) - set([outcome, treatment]))

        X = np.array(df_train.loc[:, covariates])
        Y = np.array(df_train.loc[:, outcome])
        T = np.array(df_train.loc[:, treatment])

        X_est = np.array(df_est.loc[:, covariates])

        est = SparseLinearDML(discrete_treatment=True)
        print('1')
        start = time.time()
        est.fit(Y=Y, T=T, X=X)
        print('2')
        print(time.time() - start)
        start = time.time()
        print(est.effect(X=X_est))
        print('3')
        print(time.time() - start)

        print('hi')
    #     Xtest = df_est[covariates].to_numpy()
    #     bart_res_c = dbarts.bart(Xc, Yc, Xtest, keeptrees=True, verbose=False, ntree=500, ndpost=10000)
    #     if discrete_outcome:
    #         y_c_hat_bart = norm.cdf(bart_res_c[2]).mean(axis=0)
    #     else:
    #         y_c_hat_bart = np.array(bart_res_c[7])
    #     bart_res_t = dbarts.bart(Xt, Yt, Xtest, keeptrees=True, verbose=False, ntree=500, ndpost=10000)
    #     if discrete_outcome:
    #         y_t_hat_bart = norm.cdf(bart_res_t[2]).mean(axis=0)
    #     else:
    #         y_t_hat_bart = np.array(bart_res_t[7])
    #     t_hat_bart = np.array(y_t_hat_bart - y_c_hat_bart)
    #     this_index = df_est.index
    #     control_preds_i = pd.DataFrame(y_c_hat_bart, index=this_index, columns=['Y0'])
    #     treatment_preds_i = pd.DataFrame(y_t_hat_bart, index=this_index, columns=['Y1'])
    #     cate_est_i = pd.DataFrame(t_hat_bart, index=this_index, columns=['CATE'])
    #
    #     control_preds = pd.concat([control_preds, control_preds_i], join='outer', axis=1)
    #     treatment_preds = pd.concat([treatment_preds, treatment_preds_i], join='outer', axis=1)
    #     cate_est = pd.concat([cate_est, cate_est_i], join='outer', axis=1)
    # control_preds['avg.Y0'] = control_preds.mean(axis=1)
    # control_preds['std.Y0'] = control_preds.std(axis=1)
    # treatment_preds['avg.Y1'] = treatment_preds.mean(axis=1)
    # treatment_preds['std.Y1'] = treatment_preds.std(axis=1)
    # cate_est = cate_est.sort_index()
    # cate_est['avg.CATE'] = cate_est.mean(axis=1)
    # cate_est['std.CATE'] = cate_est.std(axis=1)
    # if result == 'full':
    #     return cate_est, control_preds, treatment_preds
    # return cate_est

def bart_sample(outcome, treatment, df_train, sample, covariates, binary=False):
    Xc = np.array(df_train.loc[df_train[treatment] == 0, covariates])
    Yc = np.array(df_train.loc[df_train[treatment] == 0, outcome])

    Xt = np.array(df_train.loc[df_train[treatment] == 1, covariates])
    Yt = np.array(df_train.loc[df_train[treatment] == 1, outcome])

    if binary:
        # for some reason bart can't do one sample inference with binary outcome. so we add a dummy sample
        sample = np.concatenate([sample, np.zeros(shape=sample.shape)], axis=0)
        return norm.cdf(dbarts.bart(Xt, Yt, sample, keeptrees=False, verbose=False)[2][:, 0]).mean() - \
               norm.cdf(dbarts.bart(Xc, Yc, sample, keeptrees=False, verbose=False)[2][:, 0]).mean()
    return dbarts.bart(Xt, Yt, sample, keeptrees=False, verbose=False)[7][0] - \
           dbarts.bart(Xc, Yc, sample, keeptrees=False, verbose=False)[7][0]
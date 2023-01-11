#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 15:01:21 2020
@author: harshparikh
"""

import numpy as np
import pandas as pd
import sklearn.ensemble as ensemble
import sklearn.linear_model as linear
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors

from utils import prune_covariates, linear_cate


class Prognostic:
    def __init__(self, Y, T, df, method='ensemble', random_state=None):
        self.Y = Y
        self.T = T
        self.df = df
        self.cov = [c for c in df.columns if c not in [Y, T]]
        self.df_c = df.loc[df[T] == 0]
        self.Xc, self.Yc = self.df_c[self.cov].to_numpy(), self.df_c[Y].to_numpy()
        if method == 'ensemble':
            self.hc = ensemble.GradientBoostingRegressor(random_state=random_state).fit(self.Xc, self.Yc)
        elif method == 'linear':
            self.hc = linear.LassoCV(max_iter=5000, random_state=random_state).fit(self.Xc, self.Yc)

    def get_sample_cate(self, df_est, sample_idx, k=10):
        X_est, Y_est, T_est = df_est[self.cov].to_numpy(), df_est[self.Y].to_numpy(), df_est[self.T].to_numpy()
        hat_Y = self.hc.predict(X_est)
        control_nn = NearestNeighbors(n_neighbors=k, leaf_size=50, algorithm='auto', n_jobs=10).fit(
            hat_Y[T_est == 0].reshape(-1, 1))
        treatment_nn = NearestNeighbors(n_neighbors=k, leaf_size=50, algorithm='auto', n_jobs=10).fit(
            hat_Y[T_est == 1].reshape(-1, 1))
        c_mg = control_nn.kneighbors(hat_Y[sample_idx].reshape(1, -1), return_distance=False).reshape(-1, )
        yc = df_est[T_est == 0][self.Y].to_numpy()[c_mg].mean()
        t_mg = treatment_nn.kneighbors(hat_Y[sample_idx].reshape(1, -1), return_distance=False).reshape(-1, )
        yt = df_est[T_est == 1][self.Y].to_numpy()[t_mg].mean()
        return yt - yc

    def get_matched_group(self, df_est, k=10, est_method='mean'):
        X_est, Y_est, T_est = df_est[self.cov].to_numpy(), df_est[self.Y].to_numpy(), df_est[self.T].to_numpy()
        hat_Y = self.hc.predict(X_est)
        control_nn = NearestNeighbors(n_neighbors=k, leaf_size=50, algorithm='auto', n_jobs=10).fit(
            hat_Y[T_est == 0].reshape(-1, 1))
        treatment_nn = NearestNeighbors(n_neighbors=k, leaf_size=50, algorithm='auto', n_jobs=10).fit(
            hat_Y[T_est == 1].reshape(-1, 1))
        _, c_mg = control_nn.kneighbors(hat_Y.reshape(-1, 1))
        _, t_mg = treatment_nn.kneighbors(hat_Y.reshape(-1, 1))

        if est_method == 'mean':
            yc = df_est[T_est == 0][self.Y].to_numpy()[c_mg].mean(axis=1)
            yt = df_est[T_est == 1][self.Y].to_numpy()[t_mg].mean(axis=1)
            df_mg = pd.DataFrame([yc, yt, T_est]).T
            df_mg.columns = ['Yc', 'Yt', 'T']
            df_mg['CATE'] = df_mg['Yt'] - df_mg['Yc']
        elif est_method == 'linear_pruned':
            M = np.abs(self.hc.coef_)
            M = M / np.sum(M) * len(self.cov) if not np.all(M == 0) else np.ones(len(self.cov))
            imp_covs = prune_covariates(self.cov, M)
            covs = np.concatenate([df_est[T_est == 0][imp_covs + [self.T]].to_numpy()[c_mg],
                                   df_est[T_est == 1][imp_covs + [self.T]].to_numpy()[t_mg]], axis=1)
            outcomes = np.concatenate(
                [df_est[T_est == 0][self.Y].to_numpy()[c_mg], df_est[T_est == 1][self.Y].to_numpy()[t_mg]], axis=1)
            mg = np.concatenate([covs, outcomes.reshape(outcomes.shape[0], outcomes.shape[1], 1)], axis=2)
            df_mg = pd.DataFrame([np.array([linear_cate(mg[i, :, :]) for i in range(mg.shape[0])])]).T
            df_mg.columns = ['CATE']

        df_mg.index = df_est.index
        c_mg = pd.DataFrame(np.array(df_est.loc[T_est == 0].index)[c_mg])
        t_mg = pd.DataFrame(np.array(df_est.loc[T_est == 1].index)[t_mg])
        return df_mg, c_mg, t_mg


def prognostic_cv(outcome, treatment, data, method='ensemble', k_est=1, est_method='mean', n_splits=5, gen_skf=None,
                  random_state=None):
    if gen_skf is None:
        skf = StratifiedKFold(n_splits=n_splits)
        gen_skf = skf.split(data, data[treatment])
    cate_est = pd.DataFrame()
    control_mgs = []
    treatment_mgs = []
    for est_idx, train_idx in gen_skf:
        df_train = data.iloc[train_idx]
        df_est = data.iloc[est_idx]
        prog = Prognostic(outcome, treatment, df_train, method=method, random_state=random_state)
        prog_mg, c_mgs, t_mgs = prog.get_matched_group(df_est, k=k_est, est_method=est_method)
        control_mgs.append(c_mgs)
        treatment_mgs.append(t_mgs)
        cate_est_i = pd.DataFrame(prog_mg['CATE'])
        cate_est = pd.concat([cate_est, cate_est_i], join='outer', axis=1)
    cate_est = cate_est.sort_index()
    cate_est['avg.CATE'] = cate_est.mean(axis=1)
    cate_est['std.CATE'] = cate_est.std(axis=1)
    return cate_est, control_mgs, treatment_mgs

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 15:01:21 2020
@author: harshparikh
"""

import numpy as np
import pandas as pd
import sklearn.ensemble as ensemble

from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors


class Prognostic:
    def __init__(self, Y, T, df, binary=False, random_state=None):
        self.Y = Y
        self.T = T
        self.df = df
        self.cov = [c for c in df.columns if c not in [Y, T]]
        self.df_c = df.loc[df[T] == 0]
        self.Xc, self.Yc = self.df_c[self.cov].to_numpy(), self.df_c[Y].to_numpy()
        if binary:
            self.hc = ensemble.GradientBoostingClassifier(random_state=random_state).fit(self.Xc, self.Yc)
        else:
            self.hc = ensemble.GradientBoostingRegressor(random_state=random_state).fit(self.Xc, self.Yc)

    def get_sample_cate(self, df_est, sample_idx, k=10, binary=False):
        X_est, Y_est, T_est = df_est[self.cov].to_numpy(), df_est[self.Y].to_numpy(), df_est[self.T].to_numpy()
        if binary:
            hat_Y = self.hc.predict_proba(X_est)[:, 1]
        else:
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

    def get_matched_group(self, df_est, k=10, binary=False):
        X_est, Y_est, T_est = df_est[self.cov].to_numpy(), df_est[self.Y].to_numpy(), df_est[self.T].to_numpy()
        if binary:
            hat_Y = self.hc.predict_proba(X_est)[:, 1]
        else:
            hat_Y = self.hc.predict(X_est)
        control_nn = NearestNeighbors(n_neighbors=k, leaf_size=50, algorithm='auto', n_jobs=10).fit(
            hat_Y[T_est == 0].reshape(-1, 1))
        treatment_nn = NearestNeighbors(n_neighbors=k, leaf_size=50, algorithm='auto', n_jobs=10).fit(
            hat_Y[T_est == 1].reshape(-1, 1))
        _, c_mg = control_nn.kneighbors(hat_Y.reshape(-1, 1))
        yc = df_est[T_est == 0][self.Y].to_numpy()[c_mg].mean(axis=1)
        c_mg = pd.DataFrame(np.array(df_est.loc[T_est == 0].index)[c_mg])
        _, t_mg = treatment_nn.kneighbors(hat_Y.reshape(-1, 1))
        yt = df_est[T_est == 1][self.Y].to_numpy()[t_mg].mean(axis=1)
        t_mg = pd.DataFrame(np.array(df_est.loc[T_est == 1].index)[t_mg])
        df_mg = pd.DataFrame([yc, yt, T_est]).T
        df_mg.columns = ['Yc', 'Yt', 'T']
        df_mg['CATE'] = df_mg['Yt'] - df_mg['Yc']
        df_mg.index = df_est.index

        return df_mg, c_mg, t_mg


def prognostic_cv(outcome, treatment, data, k_est=1, n_splits=5, gen_skf=None, random_state=None):
    if gen_skf is None:
        skf = StratifiedKFold(n_splits=n_splits)
        gen_skf = skf.split(data, data[treatment])
    cate_est = pd.DataFrame()
    control_mgs = []
    treatment_mgs = []
    binary = data[outcome].nunique() == 2
    for est_idx, train_idx in gen_skf:
        df_train = data.iloc[train_idx]
        df_est = data.iloc[est_idx]
        prog = Prognostic(outcome, treatment, df_train, binary=binary, random_state=random_state)
        prog_mg, c_mgs, t_mgs = prog.get_matched_group(df_est, k=k_est, binary=binary)
        control_mgs.append(c_mgs)
        treatment_mgs.append(t_mgs)
        cate_est_i = pd.DataFrame(prog_mg['CATE'])
        cate_est = pd.concat([cate_est, cate_est_i], join='outer', axis=1)
    cate_est = cate_est.sort_index()
    cate_est['avg.CATE'] = cate_est.mean(axis=1)
    cate_est['std.CATE'] = cate_est.std(axis=1)
    return cate_est, control_mgs, treatment_mgs

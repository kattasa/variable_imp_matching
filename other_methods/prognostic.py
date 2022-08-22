#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 15:01:21 2020
@author: harshparikh
"""

import numpy as np
import sklearn.ensemble as ensemble
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import Lasso, LassoCV


class prognostic:
    def __init__(self, Y, T, df, method):
        self.Y = Y
        self.T = T
        self.df = df
        self.cov = [c for c in df.columns if c not in [Y, T]]
        self.df_c = df.loc[df[T] == 0]
        self.df_t = df.loc[df[T] == 1]
        self.Xc, self.Yc = self.df_c[self.cov], self.df_c[Y]
        self.Xt, self.Yt = self.df_t[self.cov], self.df_t[Y]
        if method == 'rf':
            self.hc = ensemble.RandomForestRegressor(n_estimators=100).fit(self.Xc, self.Yc)
            self.ht = ensemble.RandomForestRegressor(n_estimators=100).fit(self.Xt, self.Yt)
        elif method == 'lasso':
            self.hc = LassoCV().fit(self.Xc, self.Yc)
            self.ht = LassoCV().fit(self.Xt, self.Yt)

    def get_matched_group(self, df_est, k=1, est_method='exact'):
        df_mg = pd.DataFrame(columns=['Yc', 'Yt', 'T', 'CATE'])
        df_e_c = df_est.loc[df_est[self.T] == 0]
        df_e_t = df_est.loc[df_est[self.T] == 1]
        Xec, Yec = df_e_c[self.cov].to_numpy(), df_e_c[self.Y].to_numpy()
        Xet, Yet = df_e_t[self.cov].to_numpy(), df_e_t[self.Y].to_numpy()
        hatYcc = self.hc.predict(Xec)
        hatYct = self.hc.predict(Xet)
        hatYtc = self.ht.predict(Xec)
        hatYtt = self.ht.predict(Xet)
        mgs = {}
        for i in range(0, len(hatYct)):
            mgs[df_e_t.index[i]] = {'control': [], 'treatment': []}
            ps = hatYct[i]
            dis = np.abs(hatYcc - ps)
            idx = np.argpartition(dis, k)
            df_temp = pd.DataFrame()
            yc = np.mean(Yec[idx[:k]])
            mgs[df_e_t.index[i]]['control'] = list(df_e_c.index[idx[:k]])

            ps = hatYtt[i]
            dis = np.abs(hatYtt - ps)
            idx = np.argpartition(dis, k)
            mgs[df_e_t.index[i]]['treatment'] = list(df_e_t.index[idx[:k]])
            if est_method == 'smooth':
                yt = np.mean(Yet[idx[:k]])
            elif est_method == 'exact':
                yt = Yet[i]
            elif est_method == 'check':
                yc = hatYct[i]
                yt = hatYtt[i]
            df_temp['Yc'] = [yc]
            df_temp['Yt'] = [yt]
            df_temp['T'] = [1]
            df_temp['CATE'] = yt - yc
            df_temp = df_temp.rename(index={0: df_e_t.index[i]})
            df_mg = df_mg.append(df_temp)
        for i in range(0, len(hatYtc)):
            mgs[df_e_c.index[i]] = {'control': [], 'treatment': []}
            df_temp = pd.DataFrame()
            ps = hatYcc[i]
            dis = np.abs(hatYcc - ps)
            idx = np.argpartition(dis, k)
            mgs[df_e_c.index[i]]['control'] = list(df_e_c.index[idx[:k]])
            if est_method == 'smooth':
                yc = np.mean(Yec[idx[:k]])
            elif est_method == 'exact':
                yc = Yec[i]
            ps = hatYtc[i]
            dis = np.abs(hatYtt - ps)
            idx = np.argpartition(dis, k)
            mgs[df_e_c.index[i]]['treatment'] = list(df_e_t.index[idx[:k]])
            yt = np.mean(Yet[idx[:k]])
            if est_method == 'check':
                yc = hatYcc[i]
                yt = hatYtc[i]
            df_temp['Yc'] = [yc]
            df_temp['Yt'] = [yt]
            df_temp['T'] = [0]
            df_temp['CATE'] = yt - yc
            df_temp = df_temp.rename(index={0: df_e_c.index[i]})
            df_mg = df_mg.append(df_temp)
        return df_mg, mgs


def prognostic_cv(outcome, treatment, data, method, k_est=1, est_method='exact', n_splits=5):
    np.random.seed(0)
    skf = StratifiedKFold(n_splits=n_splits)
    gen_skf = skf.split(data, data[treatment])
    cate_est = pd.DataFrame()
    all_mgs = []
    for est_idx, train_idx in gen_skf:
        df_train = data.iloc[train_idx]
        df_est = data.iloc[est_idx]
        prog = prognostic(outcome, treatment, df_train, method=method)
        prog_mg, mgs = prog.get_matched_group(df_est, k=k_est, est_method=est_method)
        all_mgs.append(mgs)
        cate_est_i = pd.DataFrame(prog_mg['CATE'])
        cate_est = pd.concat([cate_est, cate_est_i], join='outer', axis=1)
    cate_est['avg.CATE'] = cate_est.mean(axis=1)
    cate_est['std.CATE'] = cate_est.std(axis=1)
    return cate_est, all_mgs
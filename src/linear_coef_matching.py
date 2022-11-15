#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 14 2022

@author: quinn.lanners
"""
import numpy as np

import sklearn.linear_model as linear

from utils import get_match_groups, get_CATES


class LCM:
    """
    A class to calculate the conditional treatment effect on samples.

    Arguments
    ----------
    outcome (str) : column label corresponding to the outcome values in the DataFrame passed as the data argument
    treatment (str) : column label corresponding to the treatment values in the DataFrame passed as the data argument
    data (pandas.DataFrame) : data to train on. Must include 'outcome' and 'treatment' columns specified above
    discrete (list) : indicates which columns of data to treat as discrete covariates
    adaptive (boolean, DEFAULT=TRUE) : whether to learn adaptive stretch weights
    M_init (np.array, DEFAULT=None) : if passed, and adaptive=True, used as the initial M value from which to learn
        adaptive M weights. If None, this value will be determined using a subset of the training data when .fit()
        is called.
    k (int, DEFAULT=10) : number of neighbors to use for finding optimal M
    gpu (Boolean, DEFAULT=False) : whether to run code optimized for GPU or not.

    Methods
    -------
    calc_x_distances(data):
        Run during initialization. Computes the distance between each covariates for each sample. Stored and used
        for optimizing M.
    set_init_M():
        scales and reformats initial M value into appropriate form for training.
    get_delta(M=None, treatment_filter=None):
        get objective function loss for a given M. If M is None, use self.M. If treatment_filter is not None, only
        compute loss for control group (0) or treatment group (1)
    get_pod_args(pod_center, pod_size):
        collects the arguments needed to perform optimization for a given pod_center and pod_size.
    fit(...):
        fits adaptive Amect model (if self.adaptive == True). See method doc for further details.
    train_M_model(...):
        trains model to predict adaptive M weights. See method doc for further details.
    model_compare(...):
        compares models to use to predict adaptive M weights. See method doc for further details.
    get_X(treatment_filter=None):
        get X values.
    get_matched_groups(df_estimation, k, check_df=True, return_error=False):
        get match groups of size k for a passed df_estimation.
    CATE(self, df_estimation, match_groups=None, k=10, method='mean'):
        calculate CATE values for samples in df_estimation.
    check_df_estimation(df_estimation):
        checks df_estimation is properly formatted.
    """
    def __init__(self, outcome, treatment, data):
        self.n, self.p =data.shape
        self.p -= 2
        self.outcome = outcome
        self.treatment = treatment
        self.covariates = [c for c in data.columns if c not in [outcome, treatment]]
        self.col_order = [*self.covariates, self.treatment, self.outcome]
        data = data[self.col_order]
        self.X = data[self.col_order[:-1]].to_numpy()
        self.T = data[self.treatment].to_numpy()
        self.Y = data[self.outcome].to_numpy()
        self.M = None

    def fit(self, params=None, double_model=False, return_score=False, random_state=0):
        if params is None:
            params = {'alpha': 0.1}
        params['random_state'] = random_state
        if double_model:
            model_C = linear.Lasso(**params).fit(self.X[self.T == 0, :-1], self.Y[self.T == 0])
            model_T = linear.Lasso(**params).fit(self.X[self.T == 1, :-1], self.Y_T[self.T == 1])
        else:
            model = linear.Lasso(**params).fit(self.X, self.Y)
        if double_model:
            M_C_hat = np.abs(self.model_C.coef_).reshape(-1,)
            M_T_hat = np.abs(self.model_T.coef_).reshape(-1,)
            M_C_hat = M_C_hat / np.sum(M_C_hat) if not np.all(M_C_hat == 0) else np.zeros(self.p)
            M_T_hat = M_T_hat / np.sum(M_T_hat) if not np.all(M_T_hat == 0) else np.zeros(self.p)
            M_hat = (M_C_hat * (np.sum(self.T == 0) / self.T.shape[0])) + \
                    (M_T_hat * (np.sum(self.T == 0) / self.T.shape[0]))
        else:
            M_hat = np.abs(model.coef_[:-1]).reshape(-1,)
            M_hat = M_hat if not np.all(M_hat == 0) else np.ones(self.p)
        self.M = (M_hat / np.sum(M_hat)) * self.p
        if return_score:
            if double_model:
                return (model_C.score(self.X[self.T == 0, :-1], self.Y[self.T == 0])) * \
                       (np.sum(self.T == 0)/ self.T.shape[0]) + \
                       (model_T.score(self.X[self.T == 1, :-1], self.Y[self.T == 1])) * \
                       (np.sum(self.T == 1) / self.T.shape[0]) if double_model else model.score(self.X, self.Y)
            return model.score(self.X, self.Y)

    def get_matched_groups(self, df_estimation, k=10, return_original_idx=False, check_est_df=False):
        """Get the match groups for a given

        :param df_estimation:
        :param k:
        :param check_df:
        :return:
        """
        return get_match_groups(df_estimation, k, self.covariates, self.treatment, M=self.M,
                                return_original_idx=return_original_idx, check_est_df=check_est_df)

    def CATE(self, df_estimation, control_match_groups=None, treatment_match_groups=None, k=10, method='mean',
             augmented=True, check_est_df=False):
        if (control_match_groups is None) or (treatment_match_groups is None):
            control_match_groups, treatment_match_groups, _, _ = self.get_matched_groups(df_estimation=df_estimation,
                                                                                         k=k, return_original_idx=False,
                                                                                         check_est_df=check_est_df)
        return get_CATES(df_estimation, control_match_groups, treatment_match_groups, method,
                         self.covariates, self.outcome, self.treatment, self.model_C, self.model_T, self.M,
                         augmented=augmented, control_preds=None, treatment_preds=None, check_est_df=check_est_df)



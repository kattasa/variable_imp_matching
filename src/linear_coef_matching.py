#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 14 2022

@author: quinn.lanners
"""
import numpy as np
import pandas as pd

from sklearn.base import clone
import sklearn.ensemble as ensemble
import sklearn.linear_model as linear
import sklearn.tree as tree

from utils import get_match_groups, get_CATES, get_model_weights


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
    def __init__(self, outcome, treatment, data, binary_outcome=False, random_state=None):
        self.n, self.p =data.shape
        self.p -= 2
        self.outcome = outcome
        self.treatment = treatment
        self.covariates = [c for c in data.columns if c not in [outcome, treatment]]
        self.col_order = [*self.covariates, self.treatment, self.outcome]
        data = data[self.col_order]
        self.binary_outcome = binary_outcome
        self.X = data[self.covariates].to_numpy()
        self.T = data[self.treatment].to_numpy()
        self.Y = data[self.outcome].to_numpy()
        self.treatments_classes = np.unique(self.T)
        self.M = None
        self.random_state = random_state

    def fit(self, model='linear', params=None, model_weight_attr=None,
            separate_treatments=True, equal_weights=False, metalearner=False,
            return_scores=False):
        self.M = None
        scores = None
        if return_scores:
            scores = {}
        m, weight_attr = self.create_model(model=model, params=params,
                                           weight_attr=model_weight_attr)
        if metalearner or separate_treatments:
            M = []
            for t in self.treatments_classes:
                m.fit(self.X[self.T == t, :], self.Y[self.T == t])
                if return_scores:
                    scores[t] = m.score(self.X[self.T == t, :],
                                        self.Y[self.T == t])
                M.append(get_model_weights(m, weight_attr, equal_weights,
                                           0, t))
                m = clone(estimator=m)
            if metalearner:
                self.M = dict(zip(self.treatments_classes, M))
            else:
                self.M = sum(M) / len(self.treatments_classes)
        else:
            t_dummy = pd.get_dummies(self.T.reshape(-1, 1)).to_numpy()
            m.fit(np.concatenate([self.X, t_dummy], axis=1),
                  self.Y)
            if return_scores:
                scores['all'] = m.score(np.concatenate([self.X,
                                                        t_dummy],
                                                       axis=1),
                                        self.Y)
            self.M = get_model_weights(m, weight_attr, equal_weights,
                                       t_dummy.shape[0], 'all')
        if return_scores:
            print(scores)
            return scores

    def get_matched_groups(self, df_estimation, k=10,
                           return_original_idx=False, check_est_df=False):
        """Get the match groups for a given

        :param df_estimation:
        :param k:
        :param check_df:
        :return:
        """
        return get_match_groups(df_estimation, k, self.covariates,
                                self.treatment, M=self.M,
                                return_original_idx=return_original_idx,
                                check_est_df=check_est_df)

    def CATE(self, df_estimation, match_groups=None, match_distances=None,
             k=10, method='mean', diameter_prune=None, cov_imp_prune=0.01,
             check_est_df=False):
        if (match_groups is None) or (match_distances is None):
            match_groups, match_distances = self.get_matched_groups(
                df_estimation=df_estimation, k=k, return_original_idx=False,
                check_est_df=check_est_df)
        return get_CATES(df_estimation, match_groups, match_distances,
                         self.outcome, self.covariates, self.M,
                         method=method, diameter_prune=diameter_prune,
                         cov_imp_prune=cov_imp_prune,
                         check_est_df=check_est_df)

    def create_model(self, model='linear', params=None, weight_attr=None):
        if params is None:
            if model == 'linear':
                if self.binary_outcome:
                    params = {'penalty': 'l1', 'solver': 'saga',
                              'max_iter': 500}
                else:
                    params = {'max_iter': 5000}
            elif model == 'tree':
                params = {'max_depth': 4}
            else:
                params = {}
        params['random_state'] = self.random_state
        if model == 'linear':
            if weight_attr is None:
                weight_attr = 'coef_'
            if self.binary_outcome:
                m = linear.LogisticRegressionCV(**params)
            else:
                m = linear.LassoCV(**params)
        elif model == 'tree':
            if weight_attr is None:
                weight_attr = 'feature_importances_'
            if self.binary_outcome:
                m = tree.DecisionTreeClassifier(**params)
            else:
                m = tree.DecisionTreeRegressor(**params)
        elif model == 'ensemble':
            if weight_attr is None:
                weight_attr = 'feature_importances_'
            if self.binary_outcome:
                m = ensemble.GradientBoostingClassifier(**params)
            else:
                m = ensemble.GradientBoostingRegressor(**params)
        else:
            m = model.set_params(**params)
        return m, weight_attr

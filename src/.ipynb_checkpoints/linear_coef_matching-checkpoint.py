#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 14 2022

@author: quinn.lanners
"""
import numpy as np

from sklearn.base import clone as clone_est
import sklearn.ensemble as ensemble
import sklearn.linear_model as linear
import sklearn.tree as tree

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
    def __init__(self, outcome, treatment, data, binary_outcome=False, random_state=None):
        self.n, self.p =data.shape
        self.p -= 2
        self.outcome = outcome
        self.treatment = treatment
        self.covariates = [c for c in data.columns if c not in [outcome, treatment]]
        self.col_order = [*self.covariates, self.treatment, self.outcome]
        data = data[self.col_order]
        self.binary_outcome = binary_outcome
        self.X = data[self.col_order[:-1]].to_numpy()
        self.T = data[self.treatment].to_numpy()
        self.Y = data[self.outcome].to_numpy()
        self.M = None
        self.M_C = None
        self.M_T = None
        self.est_C = None
        self.est_T = None
        self.random_state = random_state

    def fit(self, method='linear', params=None, equal_weights=False, double_model=False, return_score=False):
        self.M = None
        self.M_C = None
        self.M_T = None
        if params is None:
            params = {'max_iter': 5000}
        params['random_state'] = self.random_state
        if double_model:
            if method == 'linear':
                model_C = linear.LassoCV(**params).fit(self.X[self.T == 0, :-1], self.Y[self.T == 0])
                model_T = linear.LassoCV(**params).fit(self.X[self.T == 1, :-1], self.Y[self.T == 1])
                M_C_hat = np.abs(model_C.coef_).reshape(-1,)
                M_T_hat = np.abs(model_T.coef_).reshape(-1,)
                print(f'Control Score: {model_C.score(self.X[self.T == 0, :-1], self.Y[self.T == 0])}')
                print(f'Treatment Score: {model_T.score(self.X[self.T == 1, :-1], self.Y[self.T == 1])}')
            elif method == 'tree':
                model_C = tree.DecisionTreeRegressor(**params).fit(self.X[self.T == 0, :-1], self.Y[self.T == 0])
                model_T = tree.DecisionTreeRegressor(**params).fit(self.X[self.T == 1, :-1], self.Y[self.T == 1])
                M_C_hat = model_C.feature_importances_[:-1].reshape(-1, )
                M_T_hat = model_T.feature_importances_[:-1].reshape(-1, )
            else:
                raise Exception(f'Fit method not supported.')
            if equal_weights:
                M_C_hat = np.where(M_C_hat > 0, 1, 0)
                M_T_hat = np.where(M_T_hat > 0, 1, 0)
            self.M_C = M_C_hat / np.sum(M_C_hat) * self.p if not np.all(M_C_hat == 0) else np.ones(self.p)
            self.M_T = M_T_hat / np.sum(M_T_hat) * self.p if not np.all(M_T_hat == 0) else np.ones(self.p)
        else:
            if method == 'linear':
                # model = linear.LassoCV(**params).fit(self.X[self.T == 0, :-1], self.Y[self.T == 0])
                # M_hat = np.abs(model.coef_).reshape(-1,)
                # print('Control Score:')
                # print(model.score(self.X[self.T == 0, :-1], self.Y[self.T == 0]))
                # print('Treatment Score:')
                # print(model.score(self.X[self.T == 1, :-1], self.Y[self.T == 1]))
                params = {'penalty': 'l1', 'solver': 'saga'}
                model_C = linear.LogisticRegressionCV(**params).fit(self.X[self.T == 0, :-1], self.Y[self.T == 0])
                model_T = linear.LogisticRegressionCV(**params).fit(self.X[self.T == 1, :-1], self.Y[self.T == 1])
                print(f'Control Score: {model_C.score(self.X[self.T == 0, :-1], self.Y[self.T == 0])}')
                print(f'Treatment Score: {model_T.score(self.X[self.T == 1, :-1], self.Y[self.T == 1])}')
                mc = (np.abs(model_C.coef_).reshape(-1,) / np.sum(np.abs(model_C.coef_))) if \
                    not np.all(model_C.coef_ == 0) else np.ones(self.p)/self.p
                mt = (np.abs(model_T.coef_).reshape(-1,) / np.sum(np.abs(model_T.coef_))) if \
                    not np.all(model_T.coef_ == 0) else np.ones(self.p)/self.p
                M_hat = mc + mt
                # model = linear.LassoCV(**params).fit(self.X, self.Y)
                # M_hat = np.abs(model.coef_[:-1]).reshape(-1, )
                # print('Score:')
                # print(model.score(self.X, self.Y))
            elif method == 'tree':
                model = tree.DecisionTreeRegressor(**params).fit(self.X, self.Y)
                M_hat = model.feature_importances_[:-1].reshape(-1,)
            elif method == 'ensemble':
                model_C = ensemble.GradientBoostingRegressor(random_state=self.random_state).fit(
                    self.X[self.T == 0, :-1], self.Y[self.T == 0])
                model_T = ensemble.GradientBoostingRegressor(random_state=self.random_state).fit(
                    self.X[self.T == 1, :-1], self.Y[self.T == 1])
                mc = (model_C.feature_importances_.reshape(-1,) / np.sum(model_C.feature_importances_)) if \
                    not np.all(model_C.feature_importances_ == 0) else np.ones(self.p)/self.p
                mt = (model_T.feature_importances_.reshape(-1,) / np.sum(model_T.feature_importances_)) if \
                    not np.all(model_T.feature_importances_ == 0) else np.ones(self.p)/self.p
                M_hat = mc + mt
            else:
                raise Exception(f'Fit method not supported.')
            # if equal_weights:
            #     M_hat = np.where(M_hat > 0, 1, 0)
            self.M = (M_hat / np.sum(M_hat)) * self.p if not np.all(M_hat == 0) else np.ones(self.p)
            if equal_weights:
                print(f'Pre trim: {np.sum(self.M > 0)}')
                self.M = np.where(self.M > 0.03*self.p, 1, 0)
                print(f'Post trim: {np.sum(self.M > 0)}')
        if return_score:
            if double_model:
                return model_C.score(self.X[self.T == 0, :-1], self.Y[self.T == 0]), \
                       model_T.score(self.X[self.T == 1, :-1], self.Y[self.T == 1])
            return model.score(self.X, self.Y)

    def get_matched_groups(self, df_estimation, k=10, return_original_idx=False, check_est_df=False):
        """Get the match groups for a given

        :param df_estimation:
        :param k:
        :param check_df:
        :return:
        """
        return get_match_groups(df_estimation, k, self.covariates, self.treatment, M=self.M, M_C=self.M_C, M_T=self.M_T,
                                return_original_idx=return_original_idx, check_est_df=check_est_df)

    def CATE(self, df_estimation, control_match_groups=None, treatment_match_groups=None, k=10, method='mean',
             augmented=True, control_preds=None, treatment_preds=None, check_est_df=False):
        if (control_match_groups is None) or (treatment_match_groups is None):
            control_match_groups, treatment_match_groups, _, _ = self.get_matched_groups(df_estimation=df_estimation,
                                                                                         k=k, return_original_idx=False,
                                                                                         check_est_df=check_est_df)
        if augmented and ((control_preds is None) or (treatment_preds is None)):
            if self.binary_outcome:
                control_preds = self.est_C.predict_proba(df_estimation[self.covariates].to_numpy())[:, 1]
                treatment_preds = self.est_T.predict_proba(df_estimation[self.covariates].to_numpy())[:, 1]
            else:
                control_preds = self.est_C.predict(df_estimation[self.covariates].to_numpy())
                treatment_preds = self.est_T.predict(df_estimation[self.covariates].to_numpy())
        this_M = self.M if self.M is not None else self.M_C + self.M_T
        return get_CATES(df_estimation, control_match_groups, treatment_match_groups, method,
                         self.outcome, self.treatment, covariates=self.covariates, M=this_M, augmented=augmented,
                         control_preds=control_preds, treatment_preds=treatment_preds, check_est_df=check_est_df,
                         random_state=self.random_state)

    def augment(self, estimator=None):
        if estimator is None and self.binary_outcome:
            estimator = ensemble.GradientBoostingClassifier(random_state=self.random_state)
        elif estimator is None:
            estimator = ensemble.GradientBoostingRegressor(random_state=self.random_state)
        self.est_C = estimator
        self.est_T = clone_est(self.est_C)
        self.est_C.fit(self.X[self.T == 0, :-1], self.Y[self.T == 0])
        self.est_T.fit(self.X[self.T == 1, :-1], self.Y[self.T == 1])
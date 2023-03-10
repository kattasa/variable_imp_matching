#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 14 2022

@author: quinn.lanners
"""
import numpy as np
import pandas as pd
import warnings

from sklearn.base import clone
import sklearn.ensemble as ensemble
import sklearn.linear_model as linear
import sklearn.tree as tree

from utils import get_match_groups, get_CATES, get_model_weights


class VIM:
    """
    A class to calculate the conditional treatment effect on samples.

    Parameters
    ----------
    outcome : str
        Column label corresponding to the outcome.
    treatment : str
        Column label corresponding to the treatment.
    data : pandas.DataFrame
        Training data that must include 'outcome' and 'treatment' columns
        specified above.
    binary_outcome : boolean, default=False
        Whether the treatment is binary.
    random_state : None or int, default=None
        Random state to run on.

    Attributes
    -------
    n : int
        Number of samples.
    p : int
        Number of covariates.
    outcome : str
    treatment : str
    col_order : list
        Stores the order of the columns.
    binary_outcome : bool
    X : numpy.array
        Matrix of all training covariates.
    T : numpy.array
        Vector of training treatments.
    Y : numpy.array
        Vector of training outcomes.
    treatment_classes : numpy.array
        Vector of unique treatments.
    M : None, list type, or dict
        Covariate weights to use for matching. Is dictionary containing weights
        for each treatment is class run as metalearner.
    random_state : None or int
    """
    def __init__(self, outcome, treatment, data, binary_outcome=False, random_state=None):
        self.n, self.p = data.shape
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
        self.treatment_classes = np.unique(self.T)
        self.M = None
        self.random_state = random_state

    def fit(self, model='linear', params=None, model_weight_attr=None,
            separate_treatments=True, equal_weights=False, metalearner=False,
            return_scores=False):
        """
        Calculates variable importances to use for distance metric and stores
        in self.M.

        Parameters
        ----------
        model : str or machine learning class, default='linear'
            Model to use for calculating feature importances. If
            model='linear', L1-regularized regression (i.e. LASSO). If
            model='tree', uses shallow decision tree. If model='ensemble',
            uses gradient boosting regressor/classifier. Otherwise, if a model
            class is passed, it uses that model. Note that is a model class is
            passed, you must specify an accompanying 'model_weight_attr' that
            is used to pull the variable importance values from the trained
            model.
        params : None or dict, default=None
            Parameters for model. If None but model='linear' or model='tree',
            then it uses the default params laid out in the create_model()
            method. Otherwise, uses the default params of the specified model.
        model_weight_attr : None or str, default=None
            Indicates the name of the model attribute that stores the variable
            importance values. If None but model='linear', model='tree', or
            model='ensemble' then uses the attribute laid out in the
            create_model() method. Otherwise, the user must provide the correct
            attribute name.
        separate_treatments : bool, default=False
            Whether the fit separate models to each treatment. If False, will
            fit one model to all samples and use the treatment indicator as a
            predictive feature. Note that is metalearner=True, this value is
            overriden.
        equal_weights : bool, default=False
            Whether to give equal weight to all features that have a nonzero
            variable importance.
        metalearner : bool, default=False
            Whether to run metalearner VIM.
        return_scores : bool, default=False
            Whether to return the 'score()' of each fitted model.

        Raises
        ------
        ValueError
            No weights learned.

        Returns
        ------
        scores
            Only returned it return_scores=True.
        """
        self.M = None  # overwrite M is class previously fit
        scores = None
        if return_scores:
            scores = {}
        m, weight_attr = self.create_model(model=model, params=params,
                                           weight_attr=model_weight_attr)
        if metalearner or separate_treatments:
            M = []
            for t in self.treatment_classes:
                try:
                    m.fit(self.X[self.T == t, :], self.Y[self.T == t])
                    if return_scores:
                        scores[t] = m.score(self.X[self.T == t, :],
                                            self.Y[self.T == t])
                except ValueError as err:
                    setattr(m, weight_attr, np.zeros(shape=self.p))
                    warnings.warn(f'Set all weights to zero: {str(err)}')
                    if return_scores:
                        scores[t] = 0
                M.append(get_model_weights(m, weight_attr, equal_weights,
                                           0, t))
                m = clone(estimator=m)
            if metalearner:
                self.M = dict(zip(self.treatment_classes, M))
            else:
                self.M = sum(M) / len(self.treatment_classes)
        else:
            t_dummy = pd.get_dummies(self.T, drop_first=True).to_numpy()
            try:
                m.fit(np.concatenate([self.X, t_dummy], axis=1),
                      self.Y)
                if return_scores:
                    scores['all'] = m.score(np.concatenate([self.X,
                                                            t_dummy],
                                                           axis=1),
                                            self.Y)
            except ValueError as err:
                setattr(m, weight_attr, np.zeros(
                    shape=self.p +t_dummy.shape[1]))
                warnings.warn(f'Set all weights to zero: {str(err)}')
                if return_scores:
                    scores['all'] = 0
            self.M = get_model_weights(m, weight_attr, equal_weights,
                                       t_dummy.shape[1], 'all')
        if return_scores:
            return scores

    def get_matched_groups(self, df_estimation, k=10,
                           return_original_idx=False):
        """Get the match groups for a given estimation set.

        Parameters
        ----------
        df_estimation : pandas.DataFrame
            Estimation set. Should include same columns as the training set.
        k : int, default=10
            Matched group size for each treatment. I.e. is there are two
            treatments, each sample is matched to 10 samples that recieved
            each treatment.
        return_original_idx : bool, default=False
            Whether to return the matched groups using the original index to
            label the samples. If False, the index is reset before computing
            matched groups (i.e. the first sample in the dataframe will be
            sample 0).

        Returns
        -------
        match_groups
            Matched groups for each sample.
        match_distances
            The distances between each sample and each of its matched samples.
            The position of each distance corresponds to the match in the
            returned match_groups.
        """
        return get_match_groups(df_estimation, self.covariates,
                                self.treatment, M=self.M, k=k,
                                return_original_idx=return_original_idx)

    def CATE(self, df_estimation, match_groups=None, match_distances=None,
             k=10, method='mean', diameter_prune=None, cov_imp_prune=0.01):
        """Get CATE estimates for each sample in an estimation set.

        Parameters
        ----------
        df_estimation : pandas.DataFrame
            Estimation set. Should include same columns as the training set.
        return_original_idx : bool, default=False
            Whether to return the matched groups using the original index to
            label the samples. If False, the index is reset before computing
            matched groups (i.e. the first sample in the dataframe will be
            sample 0).

        Returns
        -------
        match_groups
            Matched groups for each sample.
        match_distances
            The distances between each sample and each of its matched samples.
            The position of each distance corresponds to the match in the
            returned match_groups.
        """
        if (match_groups is None) or (match_distances is None):
            match_groups, match_distances = self.get_matched_groups(
                df_estimation=df_estimation, k=k, return_original_idx=False)
        return get_CATES(df_estimation, match_groups, match_distances,
                         self.outcome, self.covariates, self.M,
                         method=method, diameter_prune=diameter_prune,
                         cov_imp_prune=cov_imp_prune)

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

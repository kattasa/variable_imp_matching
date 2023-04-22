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
    Uses data to learn a distance metric to implement Variable Importance
    Matching for CATE estimation.

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
    covariates : list[str]
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
    def __init__(self, outcome, treatment, data, binary_outcome=False,
                 random_state=None):
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
        self.M = None  # overwrite M if class previously fit
        scores = None
        if return_scores:
            scores = {}
        m, weight_attr = self.config_model(model=model, params=params,
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

    def create_mgs(self, df_estimation, k=10, return_original_idx=False):
        """Get the match groups for a given estimation set.

        Parameters
        ----------
        df_estimation : pandas.DataFrame
            Estimation set. Should include same columns as the training set.
        k : int, default=10
            Matched group size for each treatment. I.e. if there are two
            treatments, each sample is matched to 10 samples for
            each treatment for a total of 20 matched units.
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

    def est_cate(self, df_estimation, match_groups=None, match_distances=None,
                 k=10, method='mean', diameter_prune=None, cov_imp_prune=0.01):
        """Get CATE estimates for each sample in an estimation set.

        Parameters
        ----------
        df_estimation : pandas.DataFrame
            Estimation set. Should include same columns as the training set.
        match_groups : None or dict[str,pd.DataFrame], default=None
            Either an object of the same structure as that returned from
            self.create_mgs() or None. If None, this function runs
            self.create_mgs() first.
        match_distances : None or dict[str,pd.DataFrame], default=None
            Either an object of the same structure as that returned from
            self.create_mgs() or None. If None, this function runs
            self.create_mgs() first.
        k : int, default=10
            If match_groups or match_distances is None, used as the k for
            self.create_mgs()
        method : str, default='mean'
            Method to use inside match groups to estimate CATEs. Currently
            accepts 'mean', 'linear', and 'linear_pruned'.
        diameter_prune : None or numeric, default=None
            If numeric, prune all MGs for which the diameter is greater than
            diameter_prune standard deviations from the mean match group
            diameter.
        cov_imp_prune : float, default=0.01
            Minimum relative feature importance to not prune covariate. Only
            used if method == 'linear_pruned'.

        Returns
        -------
        pd.DataFrame with CATE estimates for each sample in df_estimation.
        """
        if (match_groups is None) or (match_distances is None):
            match_groups, match_distances = self.create_mgs(
                df_estimation=df_estimation, k=k, return_original_idx=False)
        return get_CATES(df_estimation, match_groups, match_distances,
                         self.outcome, self.covariates, self.M,
                         method=method, diameter_prune=diameter_prune,
                         cov_imp_prune=cov_imp_prune)

    def config_model(self, model='linear', params=None, weight_attr=None):
        """Configurate the appropriate model to use given the passed args.

        Parameters
        ----------
        model : str or sklearn model class, default=False
            Indicates what type of model to use. If str must be either
            'linear', 'tree', or 'ensemble'. Otherwise, can pass any sklearn
            model class as long as the corresponding 'weight_attr' is set to
            be the appropriate model attribute to retrieve the feature
            importance weights from.
        params : None or dict, default=None
            If None, default sklearn params are used. Otherwise, dict
            with model parameters.
        weight_attr : None or str, default=None
            If None and model == 'linear' then set to coef_
            If None and model == 'tree' or 'ensemble then set to
                feature_importances_
            Otherwise, if model is sklearn model class, must be str specifying
            the appropriate model attribute to use to retrieve feature
            importance weights.

        Returns
        -------
        m
            Untrained sklearn model with specified configuration.
        weight_attr
            String specifying the attribute to retrieve from m to use as the
            feature importance weights.
        """
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

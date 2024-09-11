from itertools import combinations
import numpy as np
import pandas as pd
from sklearn.base import clone
import sklearn.ensemble as ensemble
import sklearn.linear_model as linear
from sklearn.neighbors import NearestNeighbors
import sklearn.tree as tree
import warnings
from pathlib import Path

def config_model(model='linear', params=None, weight_attr=None,
                 binary_outcome=False, random_state=None):
    """Configure the appropriate model to use given the passed args.

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
    binary_outcome : bool, default=False
        Whether the outcome is binary or not.
    random_state : None or int, default=None
        Random state to use.

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
            if binary_outcome:
                params = {'penalty': 'l1', 'solver': 'saga',
                          'max_iter': 500}
            else:
                params = {'max_iter': 5000}
        elif model == 'tree':
            params = {'max_depth': 4}
        else:
            params = {}
    params['random_state'] = random_state
    if model == 'linear':
        if weight_attr is None:
            weight_attr = 'coef_'
        if binary_outcome:
            m = linear.LogisticRegressionCV(**params)
        else:
            m = linear.LassoCV(**params)
    elif model == 'tree':
        if weight_attr is None:
            weight_attr = 'feature_importances_'
        if binary_outcome:
            m = tree.DecisionTreeClassifier(**params)
        else:
            m = tree.DecisionTreeRegressor(**params)
    elif model == 'ensemble':
        if weight_attr is None:
            weight_attr = 'feature_importances_'
        if binary_outcome:
            m = ensemble.GradientBoostingClassifier(**params)
        else:
            m = ensemble.GradientBoostingRegressor(**params)
    else:
        m = model.set_params(**params)
    return m, weight_attr


def calc_var_imp(x_train, t_train, y_train, m, weight_attr,
                 separate_treatments=True, equal_weights=False,
                 metalearner=False, calc_scores=True):
    """Calculate the variable importance measures using the passed data and
    model.

    Parameters
    ----------
    x_train : np.array
        Matrix with covariate values for training data.
    t_train : np.array
        Vector with treatment values for training data.
    y_train : np.array
        Vector with outcome values for training data.
    m : sklearn.model
        Instance of sklearn model class to use to calculate variable
        importance. Use the returned m from the config_model() function to
        calculate this easily.
    weight_attr : str
        Indicates the attribute to use from the trained m as the feature
        importance measure. Use the returned weight_attr from the
        config_model() function to calculate this easily.
    separate_treatments : bool, default=True
        Whether to train separate feature importance models for each treatment
        subpopulation.
    equal_weights : bool, default=False
        Whether to weight all nonzero weights equally.
    metalearner : bool, default=False
        Whether to train the metalearner variety of VIM.
    calc_scores : bool, default=True
        Whether to calculate the .score() of each fit model.

    Returns
    -------
    final_m
        Learned covariate weights.
    scores
        If calc_scores == True, then the .score() of each fit model. Else, an
        empty dictionary.
    """
    scores = {}
    treatment_classes = np.unique(t_train)
    if metalearner or separate_treatments:
        all_ms = []
        for t in treatment_classes:
            try:
                m.fit(x_train[t_train == t, :], y_train[t_train == t])
                if calc_scores:
                    scores[t] = m.score(x_train[t_train == t, :],
                                        y_train[t_train == t])
            except ValueError as err:
                setattr(m, weight_attr, np.zeros(shape=x_train.shape[1]))
                warnings.warn(f'Set all weights to zero: {str(err)}')
                if calc_scores:
                    scores[t] = 0
            all_ms.append(get_model_weights(m, weight_attr, equal_weights,
                                            0, t))
            m = clone(estimator=m)
        if metalearner:
            final_m = dict(zip(treatment_classes, all_ms))
        else:
            final_m = sum(all_ms) / len(treatment_classes)
    else:
        t_dummy = pd.get_dummies(t_train, drop_first=True).to_numpy()
        try:
            m.fit(np.concatenate([x_train, t_dummy], axis=1), y_train)
            if calc_scores:
                scores['all'] = m.score(
                    np.concatenate([x_train, t_dummy], axis=1), y_train)
        except ValueError as err:
            setattr(m, weight_attr, np.zeros(
                shape=x_train.shape[1] + t_dummy.shape[1]))
            warnings.warn(f'Set all weights to zero: {str(err)}')
            if calc_scores:
                scores['all'] = 0
        final_m = get_model_weights(m, weight_attr, equal_weights,
                                    t_dummy.shape[1], 'all')
    return final_m, scores


def get_match_groups(df_estimation, covariates, treatment, M, k=None,
                     return_original_idx=True,
                     check_est_df=True):
    """Calculate match groups for an estimation dataset.

    Parameters
    ----------
    df_estimation : pd.DataFrame
        Estimation dataset.
    covariates : list[str]
        Covariate names. Used to make sure that df_estimation has the
        appropriate covaraites and columns are in the same order as the
        training dataset.
    treatment : str
        Treatment column label.
    M : np.array or list[numeric]
        Covariate weights. Must be in same order as covariates argument.
    k : int
        Match group size for each treatment.
    return_original_idx : bool, default=True
        Whether to return the match groups with the values of the original
        df_estimation index. If False, the matches correspond to the row #
        of the sample in df_estimation (i.e. the index after .reset_index()
        is run).
    check_est_df : bool, default=True
        Whether to check the df_estimation for the appropriate columns
        before running.

    Returns
    -------
    match_groups
        Match groups for each sample in df_estimation.
    match_distances
        Distance between each match, corresponding the matches in match_groups.
    """
    if check_est_df:
        check_df_estimation(df_cols=df_estimation.columns,
                            necessary_cols=covariates + [treatment])
    old_idx = np.array(df_estimation.index)
    df_estimation = df_estimation.reset_index(drop=True)
    X = df_estimation[covariates].to_numpy()
    T = df_estimation[treatment].to_numpy()
    match_groups = {}
    match_distances = {}
    for t in np.unique(T):
        if type(M) == dict:
            weights = M[t]
        else:
            weights = M
        this_X = weights[weights > 0] * X[:, weights > 0]
        this_dist, this_mg = get_nn(this_X, T, treatment=t, k=k)
        match_groups[t] = this_mg
        match_distances[t] = this_dist
    for t in np.unique(T):
        match_groups[t] = pd.DataFrame(np.array(df_estimation.loc[T == t].index)[match_groups[t]])
        match_distances[t] = pd.DataFrame(match_distances[t])
        if return_original_idx:
            match_groups[t] = convert_idx(match_groups[t], old_idx)
            match_distances[t].index = old_idx
    return match_groups, match_distances


def get_nn(X, T, treatment, k=None):
    """Get the k nn of a particular treatment for each sample."""
    nn = NearestNeighbors(n_neighbors=k, leaf_size=50, algorithm='auto',
                          metric='cityblock', n_jobs=10).fit(X[T == treatment])
    return nn.kneighbors(X, return_distance=True)


def convert_idx(mg, idx):
    """Convert the index of the match groups back to the original idx."""
    return pd.DataFrame(idx[mg.to_numpy()], index=idx)


def get_CATES(df_estimation, match_groups, match_distances, outcome,
              covariates, M, method='mean', diameter_prune=None,
              cov_imp_prune=0.01, check_est_df=True):
    """Calculate match groups for an estimation dataset.

    Parameters
    ----------
    df_estimation : pd.DataFrame
        Estimation dataset.
    match_groups : pd.DataFrame
        match groups for this df_estimation returned from get_match_groups().
    match_distances : pd.DataFrame
        match distances for this df_estimation returned from
        get_match_groups().
    outcome : str
        Outcome column label.
    covariates : list[str]
        Covariate names. Used to make sure that df_estimation has the
        appropriate covaraites and columns are in the same order as the
        training dataset.
    method : str, default='mean'
        CATE estimation method. Accepted values are 'mean', 'linear', and
        'linear_pruned'.
    diameter_prune : None or numeric, default=None
        If numeric, prune all MGs for which the diameter is greater than
        diameter_prune standard deviations from the mean match group
        diameter.
    cov_imp_prune : float, default=0.01
        Minimum relative feature importance to not prune covariate. Only
        used if method == 'linear_pruned'.
    check_est_df : bool, default=True
        Whether to check the df_estimation for the appropriate columns
        before running.

    Returns
    -------
    cates
        Estimated CATE values for each sample in df_estimation.
    """
    if check_est_df:
        check_df_estimation(df_cols=df_estimation.columns, necessary_cols=covariates + [outcome])
    df_estimation, old_idx = check_mg_indices(df_estimation, match_groups,
                                              match_distances)
    potential_outcomes = []
    max_dists = []
    for t, mgs in match_groups.items():
        if diameter_prune:
            diameters = match_distances[t].iloc[:, -1]
            good_mgs = diameters < (diameters.mean() +
                                    (diameter_prune*diameters.std()))
            these_mgs = mgs.loc[good_mgs].to_numpy()
            mgs_idx = old_idx[good_mgs]
        else:
            these_mgs = mgs.to_numpy()
            mgs_idx = old_idx
        if method == 'mean':
            y_pot = df_estimation[outcome].to_numpy()[these_mgs].mean(axis=1)
        elif 'linear' in method:
            if 'pruned' in method:
                if type(M) == dict:
                    weights = M[t]
                else:
                    weights = M
                imp_covs = prune_covariates(covariates, weights,
                                            prune_level=cov_imp_prune)
            else:
                imp_covs = covariates
            these_mgs = df_estimation[imp_covs + [outcome]].to_numpy()[these_mgs]
            these_samples = df_estimation[imp_covs].to_numpy()
            if diameter_prune:
                these_samples = these_samples[good_mgs]
            y_pot = [linear_cate(these_mgs[i], these_samples[i].reshape(1, -1))
                     for i in range(these_samples.shape[0])]
        else:
            raise Exception(f'CATE Method type {method} not supported. '
                            f'Supported methods are: mean, linear, and '
                            f'linear_pruned.')
        potential_outcomes.append(pd.DataFrame(y_pot, index=mgs_idx,
                                               columns=[f'Y{t}_{method}']))
        max_dists_t = match_distances[t].max(axis = 1)
        max_dists.append(pd.DataFrame(max_dists_t, index = mgs_idx, columns = [f'dist_{t}']))
    cates = pd.concat(potential_outcomes, axis=1).sort_index()
    max_dists_df = pd.concat(max_dists, axis = 1).sort_index()
    if len([t for t in list(match_groups.keys()) if t not in [0, 1]]) == 0:
        cates[f'CATE_{method}'] = cates[f'Y1_{method}'] - cates[f'Y0_{method}']
    else:
        for t1, t2 in combinations(list(match_groups.keys()), r=2):
            cates[f'{t2}-{t1}_CATE_{method}'] = cates[f'Y{t2}_{method}'] - cates[f'Y{t1}_{method}']
    return cates.join(max_dists_df)


def linear_cate(mg, sample):
    """Calculate CATE using a linear estiamtor inside the match group."""
    return linear.RidgeCV().fit(mg[:, :-1], mg[:, -1]).predict(sample)[0]


def check_df_estimation(df_cols, necessary_cols):
    """Check that the appropriate columns are in a dataframe."""
    missing_cols = [c for c in necessary_cols if c not in df_cols]
    if len(missing_cols) > 0:
        raise Exception(f'df_estimation missing necessary column(s) {missing_cols}')


def check_mg_indices(df_estimation, match_groups, match_distances):
    """Check that all the samples in a match groups dataframe are in
    the estimation dataframe."""
    old_idx = df_estimation.index
    df_estimation = df_estimation.reset_index(drop=True)
    est_nrows = df_estimation.shape[0]
    if not np.all([len(mg) == est_nrows for mg in match_groups.values()]):
        raise Exception(
            f'Match group dataframe sizes do not match size of df_estimation')
    if match_distances is not None:
        if not np.all([len(mg) == est_nrows for mg in
                       match_distances.values()]):
            raise Exception(
                f'Match distances dataframe sizes do not match size of '
                f'df_estimation')
    return df_estimation, old_idx


def prune_covariates(covariates, M, prune_level=0.01):
    """Prune covariates below a certain importance level. Used for the
    linear_pruned estimator."""
    imp_covs = []
    while len(imp_covs) == 0:
        imp_covs = list(np.array(covariates)[M >= prune_level * M.shape[0]])
        prune_level *= 0.1
    return imp_covs


def get_model_weights(model, weight_attr, equal_weights, t_covs, t):
    """Get the feature importance values from a model."""
    if type(weight_attr) == str:
        weights = np.abs(getattr(model, weight_attr).reshape(-1,))
    else:
        weights = weight_attr(model)
    if t_covs > 0:
        weights = weights[:-t_covs]
    if np.all(weights == 0):
        warnings.warn(f'Model fit to treatment={t} had all zero weights.')
        return np.zeros(len(weights))
    if equal_weights:
        weights = np.where(weights > 0, 1, 0)
    return (weights / np.sum(weights)) * len(weights)


def save_df_to_csv(df, file_path):
    """
    Save a DataFrame to a CSV file, ensuring the directory exists.
    
    Parameters:
    - df: pd.DataFrame, the DataFrame to save
    - file_path: str or Path, the path to save the CSV file (including directories and filename)
    """
    file_path = Path(file_path)  # Convert to Path object if not already
    file_path.parent.mkdir(parents=True, exist_ok=True)  # Create directories if they do not exist
    df.to_csv(file_path, index=False)  # Save DataFrame to CSV

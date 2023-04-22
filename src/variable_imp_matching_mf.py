import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold

from src.variable_imp_matching import VIM as LCM

from utils import get_match_groups, get_CATES, convert_idx


class VIM_MF:
    """
    Uses cross-fitting to learn a distance metric for Variable Importance
    Matching and estimate CATEs for all samples in a dataset.

    Parameters
    ----------
    outcome : str
        Column label corresponding to the outcome.
    treatment : str
        Column label corresponding to the treatment.
    data : pandas.DataFrame
        Data that must include 'outcome' and 'treatment' columns
        specified above.
    n_splits : int, default=5
        Number of splits to use for cross-fitting. For each split, 1/n_splits
        of the samples are used for training and the remaining samples are
        used for estimation
    n_repeats : int, default=1
        Number of times to shuffle the data and repeat the splitting process.
    random_state : None or int, default=None
        Random state to run on.

    Attributes
    -------
    covariates : list[str]
    outcome : str
    treatment : str
    p : int
        Number of covariates.
    data : pd.DataFrame
        Copy of data with columns ordered properly.
    binary_outcome : bool
        Indicates whether the outcome is binary or not.
    split_strategy : list[(list[int],list[int])]
        Split strategy where each value in list is tuple indicating the indices
        of the (est_set, training_set)
    M_list : list[list[float]]
        Each list corresponds to the stretches for that split. Run
        self.fit() to populate.
    model_scores : list[dict[str,float]]
        Each dict contains the scores of each model fit on the training data.
        Scores are computed on the same training set using the sklearn
        .score() function associated with the model.
    MGs : list[dict[str,pd.DataFrame]]
        Each dictionary corresponds to the MGs generated for each split. Run
        self.MG()
    MG_distances : list[dict[str,pd.DataFrame]]
        Each value to the distance of each matched sample from
        the corresponding MGs dictionary.
    cate_df : pd.DataFrame
        CATE estimates for each sample. Run self.est_cate() to populate.
    est_C_list :
    random_state : None or int
    """
    def __init__(self, outcome, treatment, data, n_splits=5, n_repeats=1,
                 random_state=None):
        self.covariates = [c for c in data.columns if c not in
                           [outcome, treatment]]
        self.outcome = outcome
        self.treatment = treatment
        self.p = len(self.covariates)
        self.data = data[[*self.covariates,
                          self.treatment, self.outcome]].reset_index(drop=True)
        self.binary_outcome = self.data[self.outcome].nunique() == 2

        skf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                      random_state=random_state)
        self.split_strategy = list(skf.split(data, data[treatment]))
        self.M_list = []
        self.model_scores = []
        self.MGs = []
        self.MG_distances = []
        self.cate_df = pd.DataFrame()
        self.random_state = random_state

    def fit(self, model='linear', params=None, model_weight_attr=None,
            separate_treatments=True, equal_weights=False, metalearner=False):
        """
        Uses the base VIM.fit() method to calculate variable importances to
        use for the distance metric for each split. Overwrites and populates
        self.M_list and self.model_scores.

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

        Raises
        ------
        ValueError
            No weights learned.
        """
        self.M_list = []
        self.model_scores = []
        for _, train_idx in self.split_strategy:
            df_train = self.data.loc[train_idx]
            m = LCM(outcome=self.outcome, treatment=self.treatment,
                    data=df_train, binary_outcome=self.binary_outcome,
                    random_state=self.random_state)
            scores = m.fit(model=model, params=params,
                           model_weight_attr=model_weight_attr,
                           separate_treatments=separate_treatments,
                           equal_weights=equal_weights,
                           metalearner=metalearner, return_scores=True)
            self.M_list.append(m.M)
            self.model_scores.append(scores)

    def create_mgs(self, k=10):
        """Calculate match groups for each split. Overwrites and populates
        self.MGs and self.MG_distances.

        Parameters
        ----------
        k : int, default=10
            Matched group size for each treatment. I.e. if there are two
            treatments, each sample is matched to 10 samples for
            each treatment for a total of 20 matched units.
        """
        self.MGs = []
        self.MG_distances = []

        i = 0
        for est_idx, _ in self.split_strategy:
            df_estimation = self.data.loc[est_idx]
            mgs, mg_dists = get_match_groups(df_estimation, self.covariates,
                                             self.treatment, M=self.M_list[i],
                                             k=k,
                                             return_original_idx=False,
                                             check_est_df=False)
            self.MGs.append(mgs)
            self.MG_distances.append(mg_dists)
            i += 1

    def est_cate(self, cate_methods=None, diameter_prune=3,
                 cov_imp_prune=0.01):
        """Calculates CATE estimates for each split. Populates self.cate_df
        with each split estimates and the avg and std of each sample's CATE
        estimates across all the n_splits-1 estimates.

        Parameters
        ----------
        cate_methods : list[str], default=['mean']
            Methods to use inside match groups to estimate CATEs. Currently
            accepts 'mean', 'linear', and 'linear_pruned'.
        diameter_prune : None or numeric, default=None
            If numeric, prune all MGs for which the diameter is greater than
            diameter_prune standard deviations from the mean match group
            diameter.
        cov_imp_prune : float, default=0.01
            Minimum relative feature importance to not prune covariate. Only
            used if method == 'linear_pruned'.
        """
        if cate_methods is None:
            cate_methods = ['mean']
        cates_list = []
        i = 0
        for est_idx, _ in self.split_strategy:
            df_estimation = self.data.loc[est_idx]
            cates = []
            for method in cate_methods:
                cates.append(get_CATES(df_estimation, self.MGs[i],
                                       self.MG_distances[i], self.outcome,
                                       self.covariates, self.M_list[i],
                                       method, diameter_prune, cov_imp_prune,
                                       check_est_df=False)
                             )
            cates = pd.concat(cates, axis=1).sort_index()
            cates_list.append(cates.copy(deep=True))
            i += 1
        self.cate_df = pd.concat(cates_list, axis=1).sort_index()
        for col in [c for c in np.unique(self.cate_df.columns) if 'CATE' in c]:
                self.cate_df[f'avg.{col}'] = self.cate_df[col].mean(axis=1)
                self.cate_df[f'std.{col}'] = self.cate_df[col].std(axis=1)
        self.cate_df[self.treatment] = self.data[self.treatment]
        self.cate_df[self.outcome] = self.data[self.outcome]

    def get_mgs(self, return_distance=False):
        """Get all match groups."""
        mg_list = []
        i = 0
        for est_idx, train_idx in self.split_strategy:
            this_mg = {}
            these_mgs = self.MGs[i]
            for t, mg in these_mgs.items():
                this_mg[t] = convert_idx(mg, est_idx)
            mg_list.append(this_mg)
            i += 1
        if return_distance:
            return mg_list, self.MG_distances
        else:
            return mg_list

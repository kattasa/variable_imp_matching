import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold

from src.linear_coef_matching import LCM

from utils import get_match_groups, get_CATES, convert_idx


class LCM_MF:
    def __init__(self, outcome, treatment, data, n_splits=5, n_repeats=1, random_state=0):

        self.covariates = [c for c in data.columns if c not in [outcome, treatment]]
        self.outcome = outcome
        self.treatment = treatment
        self.p = len(self.covariates)

        self.data = data[[*self.covariates, self.treatment, self.outcome]].reset_index(drop=True)
        self.binary_outcome = self.data[self.outcome].nunique() == 2

        skf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
        self.gen_skf = list(skf.split(data, data[treatment]))
        self.M_list = []
        self.model_scores = []
        self.col_orders = []
        self.MGs = []
        self.MG_distances = []
        self.cate_df = None
        self.est_C_list = []
        self.est_T_list = []
        self.random_state = random_state

    def fit(self, model='linear', params=None, model_weight_attr=None,
            separate_treatments=True, equal_weights=False, metalearner=False):
        self.M_list = []
        self.model_scores = []
        self.col_orders = []
        for _, train_idx in self.gen_skf:
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
            self.col_orders.append(m.col_order)

    def MG(self, k=10, treatment=None):
        if treatment is None:
            treatment = self.treatment
        self.MGs = []
        self.MG_distances = []

        i = 0
        for est_idx, _ in self.gen_skf:
            df_estimation = self.data.loc[est_idx]
            mgs, mg_dists = get_match_groups(df_estimation, k, self.covariates,
                                             treatment, M=self.M_list[i],
                                             return_original_idx=False,
                                             check_est_df=False)
            self.MGs.append(mgs)
            self.MG_distances.append(mg_dists)
            i += 1

    def CATE(self, cate_methods=None, outcome=None, treatment=None,
             diameter_prune=3, cov_imp_prune=0.01):
        if cate_methods is None:
            cate_methods = ['mean']
        if outcome is None:
            outcome = self.outcome
        if treatment is None:
            treatment = self.treatment
        cates_list = []
        i = 0
        for est_idx, _ in self.gen_skf:
            df_estimation = self.data.loc[est_idx]
            cates = []
            for method in cate_methods:
                cates.append(get_CATES(df_estimation, self.MGs[i],
                                       self.MG_distances[i], outcome,
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
        self.cate_df[treatment] = self.data[treatment]
        self.cate_df[outcome] = self.data[outcome]

    def get_MGs(self, return_distance=False):
        mg_list = []
        i = 0
        for est_idx, train_idx in self.gen_skf:
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

import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold

from MALTS.amect import Amect

from utils import get_match_groups, get_CATES, convert_idx


class Amect_mf:
    def __init__(self, outcome, treatment, data, n_splits=5, n_repeats=1, random_state=0):

        self.covariates = [c for c in data.columns if c not in [outcome, treatment]]
        self.outcome = outcome
        self.treatment = treatment
        self.p = len(self.covariates)

        self.col_order = [*self.covariates, self.treatment, self.outcome]
        self.data = data[self.col_order].reset_index(drop=True)

        skf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
        self.gen_skf = list(skf.split(data, data[treatment]))
        self.model_C_list = []
        self.model_T_list = []
        self.M_C_list = []
        self.M_T_list = []
        self.model_prop_score_list = []
        self.col_orders = []
        self.C_MG_list = []
        self.T_MG_list = []
        self.C_MG_distance = []
        self.T_MG_distance = []
        self.cates_list = []

    def fit(self, params=None, prune=0.01):
        for est_idx, train_idx in self.gen_skf:
            df_train = self.data.iloc[train_idx]

            m = Amect(outcome=self.outcome, treatment=self.treatment, data=df_train)
            m.fit(params=params, prune=prune)
            self.model_C_list.append(m.model_C)
            self.model_T_list.append(m.model_T)
            self.M_C_list.append(m.M_C)
            self.M_T_list.append(m.M_T)
            self.model_prop_score_list.append(m.model_prop_score)
            self.col_orders.append(m.col_order)

    def CATE(self, k=80, cate_methods=['linear'], augmented=True, outcome=None, return_distance=False):
        if outcome is None:
            outcome = self.outcome
        self.C_MG_list = []
        self.T_MG_list = []
        self.C_MG_distance = []
        self.T_MG_distance = []
        self.cates_list = []
        i = 0
        for est_idx, train_idx in self.gen_skf:
            df_estimation = self.data.iloc[est_idx]
            orig_idx = df_estimation.index
            control_mg, treatment_mg, control_dist, treatment_dist = get_match_groups(df_estimation, k, self.covariates,
                                                                                      self.treatment,
                                                                                      M_C=self.M_C_list[i],
                                                                                      M_T=self.M_T_list[i],
                                                                                      return_original_idx=False,
                                                                                      check_est_df=False)

            cates = []
            for method in cate_methods:
                cates.append(get_CATES(df_estimation, control_mg, treatment_mg, control_dist, treatment_dist, method, self.covariates, outcome,
                                       self.model_C_list[i], self.model_T_list[i], self.M_C_list[i], self.M_T_list[i],
                                       self.model_prop_score_list[i], augmented=augmented, check_est_df=False)
                             )

            self.C_MG_list.append(convert_idx(control_mg, orig_idx))
            self.T_MG_list.append(convert_idx(treatment_mg, orig_idx))
            if return_distance:
                control_dist.index = orig_idx
                treatment_dist.index = orig_idx
                self.C_MG_distance.append(control_dist)
                self.T_MG_distance.append(treatment_dist)
            cates = pd.DataFrame(cates).T
            self.cates_list.append(cates.copy(deep=True))

            i += 1

        self.cate_df = pd.concat(self.cates_list, axis=1)
        self.cate_df['avg.CATE'] = self.cate_df.mean(axis=1)
        self.cate_df['std.CATE'] = self.cate_df.iloc[:, :-1].std(axis=1)
        for method in cate_methods:
            self.cate_df[f'avg.CATE_{method}'] = self.cate_df[f'CATE_{method}'].mean(axis=1)
            self.cate_df[f'std.CATE_{method}'] = self.cate_df[f'CATE_{method}'].std(axis=1)
        self.cate_df[self.outcome] = self.data[self.outcome]
        self.cate_df[self.treatment] = self.data[self.treatment]

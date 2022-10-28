import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold

from linear_coef_matching.MALTS.amect import Amect

from linear_coef_matching.utils import get_match_groups, get_CATES, convert_idx


class Amect_mf:
    def __init__(self, outcome, treatment, data, n_splits=5, n_repeats=1, random_state=0):

        self.covariates = [c for c in data.columns if c not in [outcome, treatment]]
        self.outcome = outcome
        self.treatment = treatment
        self.p = len(self.covariates)

        self.col_order = [*self.covariates, self.treatment, self.outcome]
        self.data = data[self.col_order].reset_index(drop=True)

        skf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
        self.propensity_model = None
        self.gen_skf = list(skf.split(data, data[treatment]))
        self.model_C_list = []
        self.model_T_list = []
        self.modl_score_list = []
        self.M_list = []
        self.model_prop_score_list = []
        self.col_orders = []
        self.MG_size = None
        self.C_MG_list = []
        self.T_MG_list = []
        self.C_MG_distance = []
        self.T_MG_distance = []
        self.cates_list = []

    def fit(self, params=None, double_model=False):
        self.model_C_list = []
        self.model_T_list = []
        self.M_list = []
        self.col_orders = []
        for est_idx, train_idx in self.gen_skf:
            df_train = self.data.iloc[train_idx]

            m = Amect(outcome=self.outcome, treatment=self.treatment, data=df_train)
            m.fit(params=params, double_model=double_model)
            self.model_C_list.append(m.model_C)
            self.model_T_list.append(m.model_T)
            self.modl_score_list.append(m.model_score)
            self.M_list.append(m.M)
            self.col_orders.append(m.col_order)

    def MG(self, k=80, return_distance=False, treatment=None):
        if treatment is None:
            treatment = self.treatment
        self.MG_size = k
        self.C_MG_list = []
        self.T_MG_list = []
        self.C_MG_distance = []
        self.T_MG_distance = []

        i = 0
        for est_idx, train_idx in self.gen_skf:
            df_estimation = self.data.iloc[est_idx]
            control_mg, treatment_mg, control_dist, treatment_dist = get_match_groups(df_estimation, k, self.covariates,
                                                                                      treatment,
                                                                                      M=self.M_list[i],
                                                                                      return_original_idx=False,
                                                                                      check_est_df=False)
            self.C_MG_list.append(control_mg)
            self.T_MG_list.append(treatment_mg)
            if return_distance:
                control_dist.index = est_idx
                treatment_dist.index = est_idx
                self.C_MG_distance.append(control_dist)
                self.T_MG_distance.append(treatment_dist)
            i += 1

    def CATE(self, cate_methods=['linear'], augmented=False, outcome=None, treatment=None):
        if outcome is None:
            outcome = self.outcome
        if treatment is None:
            treatment = self.treatment
        self.cates_list = []
        i = 0
        for est_idx, train_idx in self.gen_skf:
            df_estimation = self.data.iloc[est_idx]
            cates = []
            for method in cate_methods:
                cates.append(get_CATES(df_estimation, self.C_MG_list[i], self.T_MG_list[i], method, self.covariates,
                                       outcome, treatment, self.model_C_list[i], self.model_T_list[i], self.M_list[i],
                                       augmented=augmented, control_preds=None, treatment_preds=None,
                                       check_est_df=False)
                             )
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

    def get_MGs(self, return_distance=False):
        c_mg_list = []
        t_mg_list = []
        for est_idx, train_idx in self.gen_skf:
            c_mg_list.append(convert_idx(self.C_MG_list[i], est_idx))
            t_mg_list.append(convert_idx(self.T_MG_list[i], est_idx))
        if return_distance:
            return c_mg_list, t_mg_list, self.C_MG_distance, self.T_MG_distance
        else:
            return c_mg_list, t_mg_list
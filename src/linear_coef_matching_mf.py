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
        self.M_C_list = []
        self.M_T_list = []
        self.double_model = False
        self.col_orders = []
        self.C_MG_list = []
        self.T_MG_list = []
        self.C_MG_distance = []
        self.T_MG_distance = []
        self.cate_df = None
        self.est_C_list = []
        self.est_T_list = []
        self.random_state = random_state

    def fit(self, method='linear', equal_weights=False, params=None, double_model=False, augment=False,
            augmented_est=None):
        self.M_list = []
        self.M_C_list = []
        self.M_T_list = []
        self.double_model = double_model
        self.col_orders = []
        for est_idx, train_idx in self.gen_skf:
            df_train = self.data.loc[train_idx]

            m = LCM(outcome=self.outcome, treatment=self.treatment, data=df_train, binary_outcome=self.binary_outcome,
                    random_state=self.random_state)
            m.fit(method=method, params=params, equal_weights=equal_weights, double_model=double_model)
            self.M_C_list.append(m.M_C)
            self.M_T_list.append(m.M_T)
            self.M_list.append(m.M)
            self.col_orders.append(m.col_order)
            if augment:
                m.augment(augmented_est)
                self.est_C_list.append(m.est_C)
                self.est_T_list.append(m.est_T)

    def MG(self, k=80, treatment=None):
        if treatment is None:
            treatment = self.treatment
        self.C_MG_list = []
        self.T_MG_list = []
        self.C_MG_distance = []
        self.T_MG_distance = []

        i = 0
        for est_idx, train_idx in self.gen_skf:
            df_estimation = self.data.loc[est_idx]
            control_mg, treatment_mg, control_dist, treatment_dist = get_match_groups(df_estimation, k, self.covariates,
                                                                                      treatment,
                                                                                      M=self.M_list[i],
                                                                                      M_C=self.M_C_list[i],
                                                                                      M_T=self.M_T_list[i],
                                                                                      return_original_idx=False,
                                                                                      check_est_df=False)
            self.C_MG_list.append(control_mg)
            self.T_MG_list.append(treatment_mg)
            self.C_MG_distance.append(control_dist)
            self.T_MG_distance.append(treatment_dist)
            i += 1

    def CATE(self, cate_methods=None, outcome=None, treatment=None, precomputed_control_preds=None,
             precomputed_treatment_preds=None):
        if cate_methods is None:
            cate_methods = [['linear_pruned', False]]
        if outcome is None:
            outcome = self.outcome
        if treatment is None:
            treatment = self.treatment
        cates_list = []
        i = 0
        for est_idx, train_idx in self.gen_skf:
            df_estimation = self.data.loc[est_idx]
            cates = []
            for method, augmented in cate_methods:
                control_preds = None
                treatment_preds = None
                if augmented:
                    control_preds, treatment_preds = self.get_outcome_preds(df_estimation, i,
                                                                            precomputed_control_preds,
                                                                            precomputed_treatment_preds)
                this_M = self.M_list[i] if not self.double_model else self.M_C_list[i] + self.M_T_list[i]
                cates.append(get_CATES(df_estimation, self.C_MG_list[i], self.T_MG_list[i], method,
                                       outcome, treatment, covariates=self.covariates, M=this_M, augmented=augmented,
                                       control_preds=control_preds, treatment_preds=treatment_preds,
                                       check_est_df=False, random_state=self.random_state)
                             )
            cates = pd.DataFrame(cates).T
            cates_list.append(cates.copy(deep=True))
            i += 1
        self.cate_df = pd.concat(cates_list, axis=1).sort_index()
        self.cate_df['avg.CATE'] = self.cate_df.mean(axis=1)
        self.cate_df['std.CATE'] = self.cate_df.iloc[:, :-1].std(axis=1)
        for method, augmented in cate_methods:
            self.cate_df[f'avg.CATE_{method}{"_augmented" if augmented else ""}'] = \
                self.cate_df[f'CATE_{method}{"_augmented" if augmented else ""}'].mean(axis=1)
            self.cate_df[f'std.CATE_{method}{"_augmented" if augmented else ""}'] = \
                self.cate_df[f'CATE_{method}{"_augmented" if augmented else ""}'].std(axis=1)
        self.cate_df[self.treatment] = self.data[self.treatment]
        self.cate_df[self.outcome] = self.data[self.outcome]

    def get_outcome_preds(self, df_estimation, idx, precomputed_control_preds=None, precomputed_treatment_preds=None):
        if (precomputed_control_preds is not None) and (precomputed_treatment_preds is not None):
            control_preds = np.array(precomputed_control_preds[idx])
            treatment_preds = np.array(precomputed_treatment_preds[idx])
        elif self.binary_outcome:
            control_preds = self.est_C_list[idx].predict_proba(df_estimation[self.covariates].to_numpy())[:,
                            1]
            treatment_preds = self.est_T_list[idx].predict_proba(df_estimation[self.covariates].to_numpy())[:,
                              1]
        else:
            control_preds = self.est_C_list[idx].predict(df_estimation[self.covariates].to_numpy())
            treatment_preds = self.est_T_list[idx].predict(df_estimation[self.covariates].to_numpy())
        return control_preds, treatment_preds

    def get_MGs(self, return_distance=False):
        c_mg_list = []
        t_mg_list = []
        i = 0
        for est_idx, train_idx in self.gen_skf:
            c_mg_list.append(convert_idx(self.C_MG_list[i], est_idx))
            t_mg_list.append(convert_idx(self.T_MG_list[i], est_idx))
            i += 1
        if return_distance:
            return c_mg_list, t_mg_list, self.C_MG_distance, self.T_MG_distance
        else:
            return c_mg_list, t_mg_list

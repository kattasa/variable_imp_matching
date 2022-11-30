import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold

from src.linear_coef_matching import LCM

from utils import get_match_groups, get_CATES, convert_idx, compare_CATE_methods


class LCM_MF:
    def __init__(self, outcome, treatment, data, n_splits=5, n_repeats=1, random_state=0):

        self.covariates = [c for c in data.columns if c not in [outcome, treatment]]
        self.outcome = outcome
        self.treatment = treatment
        self.p = len(self.covariates)

        self.col_order = [*self.covariates, self.treatment, self.outcome]
        self.data = data[self.col_order].reset_index(drop=True)
        self.binary = self.data[self.outcome].nunique() == 2

        skf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
        self.gen_skf = list(skf.split(data, data[treatment]))
        self.M_list = []
        self.model_prop_score_list = []
        self.col_orders = []
        self.MG_size = None
        self.C_MG_list = []
        self.T_MG_list = []
        self.C_MG_distance = []
        self.T_MG_distance = []
        self.cates_list = []
        self.cate_df = None
        self.est_C_list = []
        self.est_T_list = []

    def fit(self, method='linear', params=None, double_model=False, augmented_est=None):
        self.M_list = []
        self.col_orders = []
        for est_idx, train_idx in self.gen_skf:
            df_train = self.data.loc[train_idx]

            m = LCM(outcome=self.outcome, treatment=self.treatment, data=df_train, binary=self.binary)
            m.fit(method=method, params=params, double_model=double_model)
            self.M_list.append(m.M)
            self.col_orders.append(m.col_order)
            m.augment(augmented_est)
            self.est_C_list.append(m.est_C)
            self.est_T_list.append(m.est_T)


    def MG(self, k=80, treatment=None):
        if treatment is None:
            treatment = self.treatment
        self.MG_size = k
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
        self.cates_list = []
        i = 0
        for est_idx, train_idx in self.gen_skf:
            df_estimation = self.data.loc[est_idx]
            cates = []
            if (precomputed_control_preds is not None)  and (precomputed_treatment_preds is not None):
                control_preds = np.array(precomputed_control_preds[i])
                treatment_preds = np.array(precomputed_treatment_preds[i])
            elif self.binary:
                control_preds = self.est_C_list[i].predict_proba(df_estimation[self.covariates])[:, 1]
                treatment_preds = self.est_T_list[i].predict_proba(df_estimation[self.covariates])[:, 1]
            else:
                control_preds = self.est_C_list[i].predict(df_estimation[self.covariates])
                treatment_preds = self.est_T_list[i].predict(df_estimation[self.covariates])
            for method, augmented in cate_methods:
                cates.append(get_CATES(df_estimation, self.C_MG_list[i], self.T_MG_list[i], method, self.covariates,
                                       outcome, treatment, self.M_list[i], augmented=augmented,
                                       control_preds=control_preds, treatment_preds=treatment_preds,
                                       check_est_df=False)
                             )
            cates = pd.DataFrame(cates).T
            self.cates_list.append(cates.copy(deep=True))
            i += 1

        self.cate_df = pd.concat(self.cates_list, axis=1).sort_index()
        self.cate_df['avg.CATE'] = self.cate_df.mean(axis=1)
        self.cate_df['std.CATE'] = self.cate_df.iloc[:, :-1].std(axis=1)
        for method, augmented in cate_methods:
            self.cate_df[f'avg.CATE_{method}{"_augmented" if augmented else ""}'] = self.cate_df[f'CATE_{method}{"_augmented" if augmented else ""}'].mean(axis=1)
            self.cate_df[f'std.CATE_{method}{"_augmented" if augmented else ""}'] = self.cate_df[f'CATE_{method}{"_augmented" if augmented else ""}'].std(axis=1)
        self.cate_df[self.outcome] = self.data[self.outcome]
        self.cate_df[self.treatment] = self.data[self.treatment]

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

    def compare_CATE_methods(self, cate_methods=['linear', 'double_linear'], prune=True, n_test=200):
        test_per_iter = n_test // len(self.gen_skf)
        i = 0
        all_results = {m: [] for m in cate_methods}
        for est_idx, train_idx in self.gen_skf:
            df_estimation = self.data.loc[est_idx].reset_index(drop=True)
            test_samples = np.random.choice(range(len(est_idx)), size=(test_per_iter,), replace=False)
            c_mg = self.C_MG_list[i].loc[test_samples]
            t_mg = self.T_MG_list[i].loc[test_samples]
            these_results = compare_CATE_methods(c_mg=c_mg, t_mg=t_mg, df_est=df_estimation, covariates=self.covariates,
                                                 prune=prune, M=self.M_list[i], treatment=self.treatment,
                                                 outcome=self.outcome, methods=cate_methods)
            for m in cate_methods:
                all_results[m].append(these_results[m])
            i += 1
        all_results = {k: np.concatenate(v) for k, v in all_results.items()}
        for k, v in all_results.items():
            print(f'{k}: {np.mean(v)}')
            # print(df_estimation)





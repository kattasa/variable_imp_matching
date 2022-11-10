from src.linear_coef_matching import LCM
from utils import sample_match_group

import time
import numpy as np

def runtime(df_data, split_strategy, k_est):
    times = {}
    times['LCM'] = {'init': [], 'fit': [], 'mg': [], 'cate': []}
    times['MALTS'] = {'init': [], 'fit': [], 'mg': [], 'cate': []}

    for est_idx, train_idx in split_strategy:
        df_train = df_data.loc[train_idx]
        df_est = df_data.loc[est_idx].reset_index(drop=True)

        start = time.time()
        lcm = LCM(outcome='Y', treatment='T', data=df_train)
        init_time = time.time() - start

        start = time.time()
        lcm.fit(double_model=False)
        fit_time = time.time() - start


        df_est = df_est[lcm.col_order]
        covariates = np.array(lcm.covariates)
        M = lcm.M

        c_mg, t_mg = sample_match_group(df_estimation=df_est, sample_idx=0, k=k_est, covariates=covariates,
                                        treatment='T', M=M)
        print(c_mg)


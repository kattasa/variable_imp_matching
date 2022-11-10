from src.linear_coef_matching import LCM
from utils import sample_match_group

import numpy as np
import pickle
import time


def lcm_fit_runtime(df_train, save_folder, split_idx):
    start = time.time()
    lcm = LCM(outcome='Y', treatment='T', data=df_train)
    lcm.fit(double_model=False)
    fit_time = time.time() - start
    with open(f'{save_folder}/lcm_split{split_idx}.pkl', 'wb') as f:
        pickle.dump(lcm, f)
    return fit_time


def lcm_cate_runtime(df_est, k_est, save_folder, split_idx):
    with open(f'{save_folder}/lcm_split{split_idx}.pkl', 'rb') as f:
        lcm = pickle.load(f)
    covariates = np.array(lcm.covariates)
    sample_idx = np.random.randint(0, df_est.shape[0])
    start = time.time()
    c_mg, t_mg = sample_match_group(df_estimation=df_est, sample_idx=sample_idx, k=k_est,
                                    covariates=covariates, treatment='T', M=lcm.M)


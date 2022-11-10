import numpy as np
import os
import pandas as pd
import pickle
import time

from src.linear_coef_matching import LCM
from utils import sample_match_group, sample_linear_cate

acic_results_folder = f"{os.getenv('RESULTS_FOLDER')}/{os.getenv('ACIC_FOLDER')}"[:-1]
split_num = int(os.getenv('SPLIT_NUM'))
k_est = int(os.getenv('K_EST'))

print(split_num)
print(k_est)

with open(f'{acic_results_folder}/split.pkl', 'rb') as f:
    est_idx, train_idx = pickle.load(f)[split_num]

df_train = pd.read_csv(f'{acic_results_folder}/df_lcm_data.csv', index_col=0).loc[train_idx].reset_index(drop=True)
df_est = pd.read_csv(f'{acic_results_folder}/df_lcm_data.csv', index_col=0).loc[est_idx].reset_index(drop=True)

print(df_est.shape)
print(df_train.shape)

start = time.time()
lcm = LCM(outcome='Y', treatment='T', data=df_train)
lcm.fit(double_model=False)
fit_time = time.time() - start

covariates = np.array(lcm.covariates)
sample_idx = np.random.randint(0, df_est.shape[0])
df_est = df_est[lcm.col_order]

start = time.time()
M = lcm.M
X = M[M > 0] * df_est[covariates[M > 0]].to_numpy()
print(X.shape)
T = df_estimation['T'].to_numpy()
print(X[T == 0].shape)
mg = sample_match_group(df_estimation=df_est, sample_idx=sample_idx, k=k_est,
                                covariates=covariates, treatment='T', M=lcm.M)

sample_linear_cate(mg, lcm.covariates, lcm.M, treatment='T', outcome='Y', prune=True)
total_time = time.time() - start + fit_time
print(total_time)

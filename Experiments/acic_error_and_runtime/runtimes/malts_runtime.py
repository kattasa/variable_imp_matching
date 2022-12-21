import os
import numpy as np
import pandas as pd
import pickle
import time

from other_methods.pymalts import malts
from utils import sample_match_group, sample_double_linear_cate

acic_results_folder = f"{os.getenv('RESULTS_FOLDER')}/{os.getenv('ACIC_FOLDER')}"[:-1]
split_num = int(os.getenv('SPLIT_NUM'))
k_est = int(os.getenv('K_EST'))
random_state = int(os.getenv('RANDOM_STATE'))

np.random.seed(random_state)

with open(f'{acic_results_folder}/split.pkl', 'rb') as f:
    est_idx, train_idx = pickle.load(f)[split_num]
with open(f'{acic_results_folder}/binary_cols.txt', 'r') as f:
    binary = f.read().replace('[', '').replace(']','').replace("'", '').replace(' ','').split(',')
with open(f'{acic_results_folder}/categorical_cols.txt', 'r') as f:
    categorical = f.read().replace('[','').replace(']','').replace("'", '').replace(' ','').split(',')

discrete = binary + categorical

df_train = pd.read_csv(f'{acic_results_folder}/df_data.csv', index_col=0).loc[train_idx].reset_index(drop=True)
df_est = pd.read_csv(f'{acic_results_folder}/df_data.csv', index_col=0).loc[est_idx].reset_index(drop=True)

start = time.time()
malts = malts(outcome='Y', treatment='T', data=df_train, discrete=discrete, C=1, k=15, reweight=False)
malts.fit()
fit_time = time.time() - start

covariates = np.array([c for c in df_est.columns if c not in ['Y', 'T']])
sample_idx = np.random.randint(0, df_est.shape[0])
df_est = df_est[list(covariates) + ['T', 'Y']]
M = malts.M_opt[covariates].to_numpy().reshape(-1,)
sample = df_est.loc[sample_idx, covariates].to_numpy().reshape(1, -1)

start = time.time()
c_mg, t_mg = sample_match_group(df_estimation=df_est, sample_idx=sample_idx, k=k_est,
                                covariates=covariates, treatment='T', M=M, combine_mg=False)
sample_double_linear_cate(c_mg, t_mg, sample, list(covariates), M, outcome='Y', prune=False)
total_time = time.time() - start + fit_time
print(total_time)

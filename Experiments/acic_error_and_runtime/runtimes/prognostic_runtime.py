import numpy as np
import os
import pandas as pd
import pickle
import time

from other_methods.prognostic import Prognostic

np.random.seed(0)

acic_results_folder = f"{os.getenv('RESULTS_FOLDER')}/{os.getenv('ACIC_FOLDER')}"[:-1]
split_num = int(os.getenv('SPLIT_NUM'))
k_est = int(os.getenv('K_EST'))

with open(f'{acic_results_folder}/split.pkl', 'rb') as f:
    est_idx, train_idx = pickle.load(f)[split_num]

df_train = pd.read_csv(f'{acic_results_folder}/df_dummy_data.csv', index_col=0).loc[train_idx].reset_index(drop=True)
df_est = pd.read_csv(f'{acic_results_folder}/df_dummy_data.csv', index_col=0).loc[est_idx].reset_index(drop=True)

binary = df_train['Y'].nunique() == 2
sample_idx = np.random.randint(0, df_est.shape[0])

start = time.time()
prog = Prognostic(Y='Y', T='T', df=df_train, binary=binary)
cate = prog.get_sample_cate(df_est=df_est, sample_idx=sample_idx, k=k_est, binary=binary)
total_time = time.time() - start
print(total_time)

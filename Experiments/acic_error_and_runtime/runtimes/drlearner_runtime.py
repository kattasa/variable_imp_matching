import numpy as np
import os
import pandas as pd
import pickle
import time

from other_methods.drlearner import drlearner_sample

acic_results_folder = f"{os.getenv('RESULTS_FOLDER')}/{os.getenv('ACIC_FOLDER')}"[:-1]
split_num = int(os.getenv('SPLIT_NUM'))
random_state = int(os.getenv('RANDOM_STATE'))

np.random.seed(random_state)

with open(f'{acic_results_folder}/split.pkl', 'rb') as f:
    est_idx, train_idx = pickle.load(f)[split_num]

df_train = pd.read_csv(f'{acic_results_folder}/df_dummy_data.csv', index_col=0).loc[train_idx].reset_index(drop=True)
df_est = pd.read_csv(f'{acic_results_folder}/df_dummy_data.csv', index_col=0).loc[est_idx].reset_index(drop=True)

binary = df_train['Y'].nunique() == 2
covariates = [c for c in df_train.columns if c not in ['Y', 'T']]
sample = df_est.loc[np.random.randint(0, df_est.shape[0]), covariates].to_numpy().reshape(1, -1)

start = time.time()
cate = drlearner_sample(outcome='Y', treatment='T', df_train=df_train, sample=sample, covariates=covariates,
                        random_state=random_state)
print(time.time() - start)

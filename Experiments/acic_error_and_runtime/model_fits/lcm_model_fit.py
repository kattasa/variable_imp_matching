import numpy as np
import pandas as pd
import pickle

from src.variable_imp_matching import LCM
import os

acic_results_folder = f"{os.getenv('RESULTS_FOLDER')}/{os.getenv('ACIC_FOLDER')}"[:-1]
split_num = int(os.getenv('SPLIT_NUM'))
random_state = int(os.getenv('RANDOM_STATE'))

np.random.seed(random_state)

with open(f'{acic_results_folder}/split.pkl', 'rb') as f:
    est_idx, train_idx = pickle.load(f)[split_num]

df_train = pd.read_csv(f'{acic_results_folder}/df_dummy_data.csv', index_col=0).loc[train_idx].reset_index(drop=True)

lcm = LCM(outcome='Y', treatment='T', data=df_train, random_state=random_state)
print(lcm.fit(double_model=False, return_score=True))

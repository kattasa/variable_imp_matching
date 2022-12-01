import numpy as np
import os
import pandas as pd
import pickle
import time

from src.linear_coef_matching import LCM
from utils import sample_match_group, sample_linear_cate
from sklearn.neighbors import NearestNeighbors
import os

np.random.seed(0)

# acic_results_folder = f"{os.getenv('RESULTS_FOLDER')}/{os.getenv('ACIC_FOLDER')}"[:-1]
# split_num = int(os.getenv('SPLIT_NUM'))

split_num = 1
for acic_results_folder in [x[0] for x in os.walk('/Users/qlanners/projects/linear_coef_matching/Experiments/acic_error_and_runtime/Results_Tree')][1:]:
    with open(f'{acic_results_folder}/split.pkl', 'rb') as f:
        est_idx, train_idx = pickle.load(f)[split_num]

    df_train = pd.read_csv(f'{acic_results_folder}/df_dummy_data.csv', index_col=0).loc[train_idx].reset_index(drop=True)

    start = time.time()
    lcm = LCM(outcome='Y', treatment='T', data=df_train)
    print(acic_results_folder.split('/')[-1])
    print(lcm.fit(double_model=False, return_score=True))

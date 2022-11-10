import os
import pickle
import pandas as pd

from Experiments.acic_error_and_runtime.runtimes.lcm_runtime import lcm_fit_runtime

acic_results_folder = f"{os.getenv('RESULTS_FOLDER')}/{os.getenv('ACIC_FOLDER')}"[:-1]
split_num = int(os.getenv('SPLIT_NUM'))

with open(f'{acic_results_folder}/split.pkl', 'rb') as f:
    train_idx = pickle.load(f)[split_num][1]

df_train = pd.read_csv(f'{acic_results_folder}/df_lcm_data.csv').loc[train_idx]

print(lcm_fit_runtime(df_train, acic_results_folder, split_num))



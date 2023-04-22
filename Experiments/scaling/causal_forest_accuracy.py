import os
import pandas as pd

from other_methods.causalforest import causalforest

save_folder = os.getenv('RESULTS_FOLDER')
n_samples = int(os.getenv('N_SAMPLES'))
n_repeats = int(os.getenv('N_REPEATS'))
random_state = int(os.getenv('RANDOM_STATE'))
n_covs = [8, 16, 32, 64, 128, 256, 512, 1024]

df = pd.read_csv(f'{save_folder}/df.csv', nrows=n_samples)

for n in n_covs:
    this_df = df.iloc[:, :2+n].copy()
    cate_est = causalforest(outcome='Y', treatment='T', data=this_df,
                            n_splits=2, random_state=random_state)
    cate_est.to_csv(f'{save_folder}/num_covs/{n}/causal_forest_cates.csv')

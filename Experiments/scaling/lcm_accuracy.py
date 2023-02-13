import os
import pandas as pd

from src.linear_coef_matching_mf import LCM_MF

save_folder = os.getenv('RESULTS_FOLDER')
n_samples = int(os.getenv('N_SAMPLES'))
n_repeats = int(os.getenv('N_REPEATS'))
random_state = int(os.getenv('RANDOM_STATE'))
n_covs = [8, 16, 32, 64, 128, 256, 512, 1024]

df = pd.read_csv(f'{save_folder}/df.csv', nrows=n_samples)

for n in n_covs:
    this_df = df.iloc[:, :2+n].copy()
    lcm = LCM_MF(outcome='Y', treatment='T', data=this_df, n_splits=2,
                 n_repeats=n_repeats, random_state=random_state)
    lcm.fit()
    lcm.MG(k=10)
    lcm.CATE(diameter_prune=None)
    lcm.cate_df.to_csv(f'{save_folder}/num_covs/{n}/lcm_cates.csv')

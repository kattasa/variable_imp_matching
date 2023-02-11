import os
import pandas as pd

from pymalts2 import malts_mf

save_folder = os.getenv('RESULTS_FOLDER')
n_samples = os.getenv('N_SAMPLES')
n_repeats = int(os.getenv('N_REPEATS'))
random_state = int(os.getenv('RANDOM_STATE'))
n_covs = [8, 16, 32, 64, 128, 256, 512, 1024]

df = pd.read_csv(f'{save_folder}/df.csv', nrows=n_samples)

for n in n_covs:
    this_df = df.iloc[:, 2+n].copy()
    malts = malts_mf(outcome='Y', treatment='T', data=this_df, k_est=10,
                     estimator='mean', smooth_cate=False, reweight=False,
                     n_splits=2, n_repeats=n_repeats)
    malts.CATE_df.to_csv(f'{save_folder}/num_covs/{n}/malts_cates.csv')

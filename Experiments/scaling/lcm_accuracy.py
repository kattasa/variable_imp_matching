import os
import pandas as pd

from src.variable_imp_matching import VIM_CF

save_folder = os.getenv('RESULTS_FOLDER')
n_samples = int(os.getenv('N_SAMPLES'))
n_repeats = int(os.getenv('N_REPEATS'))
random_state = int(os.getenv('RANDOM_STATE'))
n_covs = [8, 16, 32, 64, 128, 256, 512, 1024]

df = pd.read_csv(f'{save_folder}/df.csv', nrows=n_samples)

for n in n_covs:
    this_df = df.iloc[:, :2+n].copy()
    lcm = VIM_CF(outcome='Y', treatment='T', data=this_df, n_splits=2,
                 n_repeats=n_repeats, random_state=random_state)
    lcm.fit()
    lcm.create_mgs(k=40)
    lcm.est_cate(diameter_prune=None, cate_methods=['mean'])
    lcm.cate_df.to_csv(f'{save_folder}/num_covs/{n}/lcm_cates.csv')

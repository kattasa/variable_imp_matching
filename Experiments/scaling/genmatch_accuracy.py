import os
import pandas as pd

from other_methods.matchit import matchit

save_folder = os.getenv('RESULTS_FOLDER')
n_samples = int(os.getenv('N_SAMPLES'))
n_repeats = int(os.getenv('N_REPEATS'))
random_state = int(os.getenv('RANDOM_STATE'))
n_covs = [8, 16, 32, 64, 128, 256, 512, 1024]

df = pd.read_csv(f'{save_folder}/df.csv', nrows=n_samples)

for n in n_covs:
    this_df = df.iloc[:, :2+n].copy()
    ate, t_hat = matchit(outcome='Y', treatment='T', data=this_df,
                         method='genetic', replace=True)
    with open(f'{save_folder}/num_covs/{n}/genmatch_ate.txt', 'w') as f:
        f.write(str(ate))
    t_hat.to_csv(f'{save_folder}/num_covs/{n}/genmatch_cates.csv')

import os
import pandas as pd

from other_methods.matchit import matchit

save_folder = os.getenv('SAVE_FOLDER')
n_repeats = int(os.getenv('N_REPEATS'))
random_state = int(os.getenv('RANDOM_STATE'))

df = pd.read_csv(f'{save_folder}/df.csv')

ate, t_hat = matchit(outcome='Y', treatment='T', data=df,
                     method='genetic', replace=True)

with open(f'{save_folder}/genmatch_ate.txt', 'w') as f:
    f.write(str(ate))

t_hat.to_csv(f'{save_folder}/genmatch_cates.csv')

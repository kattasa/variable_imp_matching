import os
import pandas as pd

from pymalts2 import malts_mf

save_folder = os.getenv('SAVE_FOLDER')
n_repeats = int(os.getenv('N_REPEATS'))
random_state = int(os.getenv('RANDOM_STATE'))

df = pd.read_csv(f'{save_folder}/df.csv')

malts = malts_mf(outcome='Y', treatment='T', data=df, k_est=10,
                 estimator='mean', smooth_cate=False, reweight=False,
                 n_splits=2, n_repeats=n_repeats)

malts.CATE_df.to_csv(f'{save_folder}/malts_cates.csv')

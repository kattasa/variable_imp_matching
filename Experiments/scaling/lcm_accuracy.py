import os
import pandas as pd
import time

from src.linear_coef_matching_mf import LCM_MF

save_folder = os.getenv('SAVE_FOLDER')
n_repeats = int(os.getenv('N_REPEATS'))
random_state = int(os.getenv('RANDOM_STATE'))

df = pd.read_csv(f'{save_folder}/df.csv')

start = time.time()
lcm = LCM_MF(outcome='Y', treatment='T', data=df, n_splits=2,
             n_repeats=n_repeats, random_state=random_state)
lcm.fit()
lcm.MG(k=10)
lcm.CATE(diameter_prune=None)
lcm.cate_df.to_csv(f'{save_folder}/lcm_cates.csv')

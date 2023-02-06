import os
import pandas as pd
import time

from src.linear_coef_matching_mf import LCM_MF

save_folder = os.getenv('SAVE_FOLDER')

df = pd.read_csv(f'{save_folder}/df.csv')

start = time.time()
lcm = LCM_MF(outcome='Y', treatment='T', data=df, n_splits=2, n_repeats=1)
lcm.fit()
lcm.MG()
lcm.CATE()
fit_time = time.time() - start
print(fit_time)

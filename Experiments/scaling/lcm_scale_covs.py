import os
import pandas as pd
import time

from src.linear_coef_matching_mf import LCM_MF

data_folder = os.getenv('RESULTS_FOLDER')
n_covs = int(os.getenv('N_COVS'))

df = pd.read_csv(f'{data_folder}/df.csv', nrows=2048,
                 usecols=list(range(2+n_covs)))

start = time.time()
lcm = LCM_MF(outcome='Y', treatment='T', data=df, n_splits=2, n_repeats=1)
lcm.fit()
lcm.MG()
lcm.CATE()
fit_time = time.time() - start
print(fit_time)

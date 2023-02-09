import os
import pandas as pd
import time

from pymalts2 import malts_mf

data_folder = os.getenv('RESULTS_FOLDER')
n_covs = int(os.getenv('N_COVS'))

df = pd.read_csv(f'{data_folder}/df.csv', nrows=2048,
                 usecols=list(range(2+n_covs)))

start = time.time()
malts = malts_mf(outcome='Y', treatment='T', data=df, k_est=10,
                 estimator='mean', smooth_cate=False, reweight=False,
                 n_splits=2, n_repeats=1)
fit_time = time.time() - start
print(fit_time)

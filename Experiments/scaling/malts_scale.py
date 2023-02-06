import os
import pandas as pd
import time

from pymalts2 import malts_mf

save_folder = os.getenv('SAVE_FOLDER')

df = pd.read_csv(f'{save_folder}/df.csv')

start = time.time()
malts = malts_mf(outcome='Y', treatment='T', data=df, k_est=10,
                 estimator='mean', smooth_cate=False, reweight=False,
                 n_splits=2, n_repeats=1)
fit_time = time.time() - start
print(fit_time)

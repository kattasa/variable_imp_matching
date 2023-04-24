import os
import pandas as pd
import time

from src.variable_imp_matching import VIM_CF

data_folder = os.getenv('RESULTS_FOLDER')
n_samples = int(os.getenv('N_SAMPLES'))

df = pd.read_csv(f'{data_folder}/df.csv', nrows=n_samples,
                 usecols=list(range(66)))

start = time.time()
lcm = VIM_CF(outcome='Y', treatment='T', data=df, n_splits=2, n_repeats=1)
lcm.fit()
lcm.create_mgs()
lcm.est_cate()
fit_time = time.time() - start
print(fit_time)

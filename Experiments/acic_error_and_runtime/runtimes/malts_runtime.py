from other_methods.pymalts import malts

import numpy as np
import pickle
import time

def malts_fit_runtime(df_train, save_folder, split_idx):
    start = time.time()
    lcm = malts(outcome='Y', treatment='T', data=df_train, discrete=[], C=1, k=10, reweight=False))
    lcm.fit(double_model=False)
    fit_time = time.time() - start
    with open(f'{save_folder}/lcm_split{split_idx}.pkl', 'wb') as f:
        pickle.dump(lcm, f)
    return fit_time
import json
import pandas as pd
import time

import pymalts

import sys
sys.path.append("..")
from helpers import get_data, create_folder
sys.path.append("..")
from MALTS.amect import Amect

data = 'dense_continuous'
num_samples = 2500
num_train = 2500

nci = 8
ndi = 0
ncu = 16
ndu = 0

config = {}
print_progress = True
print_every = 10
num_iters = 100

df = pd.DataFrame(columns=[f'X{i}' for i in range(nci+ncu)] + ['T', 'Method'])

start = time.time()
for i in range(num_iters):
    df_train, df_est, df_true, discrete, config, admalts_params = get_data(data, num_samples, config, admalts_params=None,
                                                                           imp_c=nci, imp_d=ndi, unimp_c=ncu, unimp_d=ndu,
                                                                           n_train=num_train)
    if i == 0:
        save_folder = create_folder('weights_comp', print_progress)
        with open(f'{save_folder}/config.txt', 'w') as f:
            json.dump(config, f, indent=2)

    m = Amect('Y', 'T', df_train, discrete=discrete)
    m.fit(model_type='lasso')

    py_m = pymalts.malts('Y', 'T', df_train, discrete)
    py_m.fit()

    df.loc[df.shape[0]] = list(m.M_C) + [0, 'AdMALTS']
    df.loc[df.shape[0]] = list(m.M_T) + [1, 'AdMALTS']
    df.loc[df.shape[0]] = list(py_m.M_opt[[f'X{i}' for i in range(nci + ncu)]].values[0]) + [0, 'MALTS']

    if print_progress:
        if (i+1) % print_every == 0:
            print(f'{i+1} iterations complete: {time.time() - start}')

df.to_csv(f'{save_folder}/weights.csv')

import os
import numpy as np
from Experiments.helpers import get_data

random_state = int(os.getenv('RANDOM_STATE'))
save_folder = os.getenv('SAVE_FOLDER')
np.random.seed(random_state)

dataset_config = {
    'num_samples': int(os.getenv('NUM_SAMPLES')),
    'imp_c': int(os.getenv('IMP_C')),
    'unimp_c': int(os.getenv('UNIMP_C')),
    'imp_d': 0,
    'unimp_d': 0
}
dataset_config['n_train'] = dataset_config['num_samples']

df_train, df_data, df_true, binary = get_data(data='dense_continuous', config=dataset_config)

df_train.to_csv(f'{save_folder}/df_train.csv', index=False)

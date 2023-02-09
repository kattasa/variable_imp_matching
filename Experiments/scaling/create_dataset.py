import os
from Experiments.helpers import get_data

save_folder = os.getenv('SAVE_FOLDER')

dataset_config = {
    'num_samples': 8192,
    'imp_c': 8,
    'unimp_c': 1016,
    'imp_d': 0,
    'unimp_d': 0,
    'n_train': 0
}

df_data, df_true, binary = get_data(data='dense_continuous',
                                    config=dataset_config)

df_data.to_csv(f'{save_folder}/df.csv', index=False)
df_true.to_csv(f'{save_folder}/df_true.csv', index=False)
print(f'Saved df.csv and df_true.csv to {save_folder}.')

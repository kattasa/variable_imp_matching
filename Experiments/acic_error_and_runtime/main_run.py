import glob
import os
import pandas as pd

from Experiments.acic_error_and_runtime.cate_error import cate_error
from Experiments.helpers import create_folder

print_progress = True
k_est = 60

all_acic_2019_files = list(range(1, 9))
all_acic_2018_files = [f.replace('.csv', '') for f in set([c.split('/')[-1].replace('_cf', '') for c in
                                                           glob.glob(f"{os.getenv('ACIC_2018_FOLDER')}/*.csv")])]
n_samples_per_split = 5000

for acic_file in all_acic_2019_files:
    acic_year = 'acic_2019'
    n_splits = 3
    save_folder = create_folder(f'{acic_year}-{acic_file}', print_progress)
    cate_error(acic_year=acic_year, acic_file=acic_file, n_splits=n_splits, k_est=k_est, save_folder=save_folder,
               print_progress=print_progress)


for acic_file in all_acic_2018_files:
    acic_year = 'acic_2018'
    n_splits = pd.read_csv(f"{os.getenv('ACIC_2018_FOLDER')}/{acic_file}.csv").shape[0] // n_samples_per_split
    n_splits = max(min(n_splits, 10), 3)
    save_folder = create_folder(f'{acic_year}-{acic_file}', print_progress)
    cate_error(acic_year=acic_year, acic_file=acic_file, n_splits=n_splits, k_est=k_est, save_folder=save_folder,
               print_progress=print_progress)

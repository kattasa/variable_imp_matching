import glob
import os

from Experiments.acic_error_and_runtime.cate_error import cate_error
from Experiments.helpers import create_folder

print_progress = True
k_est = 60

all_acic_2019_files = list(range(1, 9))
all_acic_2018_files = [f.replace('.csv', '') for f in set([c.split('/')[-1].replace('_cf', '') for c in
                                                           glob.glob(f"{os.getenv('ACIC_2018_FOLDER')}/*.csv")])]


for acic_file in all_acic_2019_files:
    acic_year = 'acic_2019'
    save_folder = create_folder(f'{acic_year}-{acic_file}', print_progress)
    cate_error(acic_year=acic_year, acic_file=acic_file, n_splits=3, k_est=k_est, save_folder=save_folder,
               print_progress=print_progress)


for acic_file in all_acic_2018_files:
    acic_year = 'acic_2018'
    save_folder = create_folder(f'{acic_year}-{acic_file}', print_progress)
    cate_error(acic_year=acic_year, acic_file=acic_file, n_splits=3, k_est=k_est, save_folder=save_folder,
               print_progress=print_progress)
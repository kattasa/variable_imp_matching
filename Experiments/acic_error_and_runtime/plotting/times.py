from glob import glob
import  json
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import seaborn as sns

all_folders = glob(f"{os.getenv('RESULTS_FOLDER')}/*/", recursive=True)

n_repeats = 25
q = 0.5
methods = ['LASSO Coefficient Matching', 'MALTS Matching', 'Prognostic Score Matching', 'BART', 'Causal Forest']
all_times = pd.DataFrame([], index=methods)
failed_files = {}
name_to_label = {}
methods_dirs = {
    'BART': 'bart_fit_times',
    'Causal Forest': 'causalforest_fit_times',
    'LASSO Coefficient Matching': 'lcm_fit_times',
    'MALTS Matching': 'malts_fit_times',
    'Prognostic Score Matching': 'prognostic_fit_times'
}

acic_2018_file_no = 1
acic_2019_file_no = 1
for f in all_folders:
    with open(f'{f}config.txt') as c:
        n_splits = json.loads(c.read())['n_splits']
    times = {}
    for m, d in methods_dirs.items():
        if os.path.isdir(f'{f}{d}'):
            these_times = []
            for t in glob(f'{f}{d}/*.txt'):
                try:
                    with open(t) as this_f:
                        these_times.append(float(this_f.read().replace('\n', '')))
                except Exception:
                    pass
            if len(these_times) != (n_splits * n_repeats):
                if f.split('/')[-2] in failed_files:
                    failed_files[f.split('/')[-2]].append(m)
                else:
                    failed_files[f.split('/')[-2]] = [m]
            else:
                times[m] = np.percentile(these_times, q)
        else:
            if f.split('/')[-2] in failed_files:
                failed_files[f.split('/')[-2]].append(m)
            else:
                failed_files[f.split('/')[-2]] = [m]
    if len(times) > 0:
        if 'acic_2019' in f:
            label = f'ACIC 2019 {acic_2019_file_no}'
            acic_2019_file_no += 1
        elif 'acic_2018' in f:
            label = f'ACIC 2018 {acic_2018_file_no}'
            acic_2018_file_no += 1
        all_times = all_times.join(pd.DataFrame([times], index=[label]).T)

all_times = all_times.reset_index().melt(id_vars=['index'])
all_times.columns = ['Method', 'ACIC File', 'Single CATE Runtime (s)']
all_times[['acic_year', 'acic_file_no']] = all_times['ACIC File'].str.split(expand=True).iloc[:, 1:].astype(int)
all_times = all_times.sort_values(['acic_year', 'acic_file_no'])
all_times = all_times.drop(columns=['acic_year', 'acic_file_no'])

sns.set_context("paper")
sns.set_style("darkgrid")
sns.set(font_scale=1)
ax = sns.catplot(data=all_times, x="ACIC File", y="Single CATE Runtime (s)", hue="Method", kind="bar")
plt.xticks(rotation=65, horizontalalignment='right')
plt.tight_layout()
plt.show()

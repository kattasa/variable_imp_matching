from glob import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

num_samples_folder = glob(f"{os.getenv('RESULTS_FOLDER')}/num_samples/*", recursive=True)
num_covs_folder = glob(f"{os.getenv('RESULTS_FOLDER')}/num_covs/*", recursive=True)

n_repeats = 10
methods = [
    'LCM',
    'MALTS',
    # 'GenMatch'
]

q = 0.5
sample_times_cols = ['Method', 'Time (s)', '# Samples']
covs_times_cols = ['Method', 'Time (s)', '# Covariates']
samples_times = pd.DataFrame([], columns=sample_times_cols)
covs_times = pd.DataFrame([], columns=covs_times_cols)
failed_files = {}
name_to_label = {}
methods_dirs = {
    'LCM': 'lcm_fit_times',
    'MALTS': 'malts_fit_times',
    # 'GenMatch': 'genmatch_fit_times'
}

for f in num_samples_folder:
    times = {}
    for m, d in methods_dirs.items():
        if os.path.isdir(f'{f}/{d}'):
            these_times = []
            for t in glob(f'{f}/{d}/*.txt'):
                try:
                    with open(t) as this_f:
                        these_times.append(float(this_f.read().replace('\n', '')))
                except Exception:
                    pass
            if len(these_times) != n_repeats:
                if '-'.join(f.split('/')[-2:]) in failed_files:
                    failed_files['-'.join(f.split('/')[-2:])].append(m)
                else:
                    failed_files['-'.join(f.split('/')[-2:])] = [m]
            else:
                times[m] = these_times
        else:
            if '-'.join(f.split('/')[-2:]) in failed_files:
                failed_files['-'.join(f.split('/')[-2:])].append(m)
            else:
                failed_files['-'.join(f.split('/')[-2:])] = [m]
    if len(times) > 0:
        times = pd.melt(pd.DataFrame.from_dict(times), var_name='Method', value_name='Time (s)')
        times['# Samples'] = int(f.split('/')[-1])
        samples_times = pd.concat([samples_times, times.copy()])


for f in num_covs_folder:
    times = {}
    for m, d in methods_dirs.items():
        if os.path.isdir(f'{f}/{d}'):
            these_times = []
            for t in glob(f'{f}/{d}/*.txt'):
                try:
                    with open(t) as this_f:
                        these_times.append(float(this_f.read().replace('\n', '')))
                except Exception:
                    pass
            if len(these_times) != n_repeats:
                if '-'.join(f.split('/')[-2:]) in failed_files:
                    failed_files['-'.join(f.split('/')[-2:])].append(m)
                else:
                    failed_files[f.split('/')[-2]] = [m]
            else:
                times[m] = these_times
        else:
            if '-'.join(f.split('/')[-2:]) in failed_files:
                failed_files['-'.join(f.split('/')[-2:])].append(m)
            else:
                failed_files['-'.join(f.split('/')[-2:])] = [m]
    if len(times) > 0:
        times = pd.melt(pd.DataFrame.from_dict(times), var_name='Method', value_name='Time (s)')
        times['# Covariates'] = int(f.split('/')[-1]) + 8
        covs_times = pd.concat([covs_times, times.copy()])

sns.set_context("paper")
sns.set_style("darkgrid")
sns.set(font_scale=1)
fig, axes = plt.subplots(1, 2)
sns.pointplot(ax=axes[0], data=samples_times, x='# Samples', y='Time (s)', hue='Method')
sns.pointplot(ax=axes[1], data=covs_times, x='# Covariates', y='Time (s)', hue='Method')
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='center', bbox_to_anchor=(0.5, 1.05), ncol=2)
for ax in axes:
    ax.get_legend().remove()
    # ax.set_xscale('log', basex=2)
fig.tight_layout()
fig.savefig(f'scaling.png', bbox_inches='tight')

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
    'GenMatch'
]

q = 0.5
samples_times = pd.DataFrame([], index=methods)
covs_times = pd.DataFrame([], index=methods)
failed_files = {}
name_to_label = {}
methods_dirs = {
    'LCM': 'lcm_fit_times',
    'MALTS': 'malts_fit_times',
    'GenMatch': 'genmatch_fit_times'
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
                times[m] = np.percentile(these_times, q)
        else:
            if '-'.join(f.split('/')[-2:]) in failed_files:
                failed_files['-'.join(f.split('/')[-2:])].append(m)
            else:
                failed_files['-'.join(f.split('/')[-2:])] = [m]
    if len(times) > 0:
        label = f.split('/')[-1]
        samples_times = samples_times.join(pd.DataFrame([times], index=[label]).T)


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
                times[m] = np.percentile(these_times, q)
        else:
            if '-'.join(f.split('/')[-2:]) in failed_files:
                failed_files['-'.join(f.split('/')[-2:])].append(m)
            else:
                failed_files['-'.join(f.split('/')[-2:])] = [m]
    if len(times) > 0:
        label = f.split('/')[-1]
        covs_times = covs_times.join(pd.DataFrame([times], index=[label]).T)

samples_times = samples_times.reset_index().melt(id_vars=['index'])
samples_times.columns = ['Method', '# Samples', 'Time (s)']
samples_times.loc[:, '# Samples'] = samples_times['# Samples'].astype(int)
samples_times = samples_times.sort_values('# Samples')

covs_times = covs_times.reset_index().melt(id_vars=['index'])
covs_times.columns = ['Method', '# Covariates', 'Time (s)']
covs_times.loc[:, '# Covariates'] = covs_times['# Covariates'].astype(int) + 8
covs_times = covs_times.sort_values('# Covariates')

sns.set_context("paper")
sns.set_style("darkgrid")
sns.set(font_scale=1)
fig, axes = plt.subplots(1, 2)
sns.scatterplot(ax=axes[0], data=samples_times, x='# Samples', y='Time (s)', hue='Method')
sns.scatterplot(ax=axes[1], data=covs_times, x='# Covariates', y='Time (s)', hue='Method')
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='center', bbox_to_anchor=(0.5, 1.05), ncol=2)
for ax in axes:
    ax.get_legend().remove()
    ax.set_xscale('log', base=2)
fig.tight_layout()
fig.savefig(f'scaling.png', bbox_inches='tight')

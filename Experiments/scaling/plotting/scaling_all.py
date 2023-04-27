from glob import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

num_samples_folder = glob(f"{os.getenv('RESULTS_FOLDER')}/num_samples/*", recursive=True)
num_covs_folder = glob(f"{os.getenv('RESULTS_FOLDER')}/num_covs/*", recursive=True)

n_repeats = 20
methods = [
    'LCM',
    'MALTS',
    'GenMatch'
    'AHB'
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
    'GenMatch': 'genmatch_fit_times',
    'AHB': 'ahb_fit_times'
}

for f in num_samples_folder:
    times = {}
    for m, d in methods_dirs.items():
        if os.path.isdir(f'{f}/{d}'):
            these_times = []
            for t in glob(f'{f}/{d}/*.txt'):
                try:
                    with open(t) as this_f:
                        if m == 'AHB':
                            these_times.append(
                                float(this_f.readlines()[-1].replace('\n', '')))
                        else:
                            these_times.append(float(this_f.readline().replace('\n', '')))
                except Exception:
                    pass
            if len(these_times) != n_repeats:
                this_file = '-'.join(f.split('/')[-2:])
                if this_file in failed_files:
                    failed_files[this_file].append(m)
                else:
                    failed_files[this_file] = [m]
                print(f'Failed {m} for {this_file}')
                print(f'Missing {n_repeats - len(these_times)} times.\n')
            else:
                times[m] = these_times
        else:
            this_file = '-'.join(f.split('/')[-2:])
            if this_file in failed_files:
                failed_files[this_file].append(m)
            else:
                failed_files[this_file] = [m]
            print(f'Failed {m} for {this_file}')
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
                        if m == 'AHB':
                            these_times.append(
                                float(this_f.readlines()[-1].replace('\n', '')))
                        else:
                            these_times.append(float(this_f.readline().replace('\n', '')))
                except Exception:
                    pass
            if len(these_times) != n_repeats:
                this_file = '-'.join(f.split('/')[-2:])
                if this_file in failed_files:
                    failed_files[this_file].append(m)
                else:
                    failed_files[this_file] = [m]
                print(f'Failed {m} for {this_file}')
                print(f'Missing {n_repeats - len(these_times)} times.\n')
            else:
                times[m] = these_times
        else:
            this_file = '-'.join(f.split('/')[-2:])
            if this_file in failed_files:
                failed_files[this_file].append(m)
            else:
                failed_files[this_file] = [m]
            print(f'Failed {m} for {this_file}')
    if len(times) > 0:
        times = pd.melt(pd.DataFrame.from_dict(times), var_name='Method', value_name='Time (s)')
        times['# Covariates'] = int(f.split('/')[-1])
        covs_times = pd.concat([covs_times, times.copy()])

# for k in failed_files:
#     test, num = k.split('-')
#     if test == 'num_samples':
#         samples_times = samples_times.loc[samples_times['# Samples'] != int(num)]
#     elif test == 'num_covs':
#         covs_times = covs_times.loc[covs_times['# Covariates'] != int(num)+8]

order = ['LCM', 'Linear PGM', 'Ensemble PGM', 'MALTS', '', '',  'GenMatch', 'AHB']
palette = {order[i]: sns.color_palette()[i] for i in range(len(order))}
method_order = [c for c in order if c in samples_times['Method'].unique()]
markers = ['o', '^', 's', 'p']

samples_times['# Samples'] = samples_times['# Samples'].astype(int)
covs_times['# Covariates'] = covs_times['# Covariates'].astype(int)

# samples_times[r"# Samples $_{\left(\log_2\right)}$"] = np.log2(samples_times['# Samples'].astype(int))
# samples_times[r"# Samples $_{\left(\log_2\right)}$"] = samples_times[r"# Samples $_{\left(\log_2\right)}$"].astype(int)
#
# covs_times[r"# Covariates $_{\left(\log_2\right)}$"] = np.log2(covs_times['# Covariates'].astype(int))
# covs_times[r"# Covariates $_{\left(\log_2\right)}$"] = covs_times[r"# Covariates $_{\left(\log_2\right)}$"].astype(int)


sns.set_context("paper")
sns.set_style("darkgrid")
sns.set(font_scale=2, font="times")
fig, axes = plt.subplots(2, 1, figsize=(8, 9))
sns.pointplot(ax=axes[0], data=samples_times, x="# Samples", y='Time (s)',
              errorbar=None, hue='Method',
              hue_order=method_order, palette=palette, markers=markers,
              scale=2)
sns.pointplot(ax=axes[1], data=covs_times, x="# Covariates", y='Time (s)',
              errorbar=None, hue='Method',
              hue_order=method_order, palette=palette, markers=markers,
              scale=2)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.55, 0.96),
           ncol=4, fontsize=20, handletextpad=0.2,
           columnspacing=0.5, markerscale=1.5)
for ax in axes:
    ax.get_legend().remove()
    ax.set_yscale('log')
    # ax.set_xscale('log')
# axes[1].set_ylabel(ylabel=None)
fig.tight_layout()
fig.savefig(f'scaling_all.png', bbox_inches='tight')

# plt.figure(figsize=(9, 6))
# sns.set_context("paper")
# sns.set_style("darkgrid")
# sns.set(font_scale=2, font="times")
# ax = sns.pointplot(data=samples_times, x=r"# Samples $_{\left(\log_2\right)}$", y='Time (s)',
#               errorbar='sd', hue='Method',
#               hue_order=method_order, palette=palette, markers=markers, scale=1.25)
# sns.move_legend(ax, "lower center", bbox_to_anchor=(.47, 1.02), ncol=3,
#                 title=None, handletextpad=0.4, columnspacing=0.5, fontsize=20)
# plt.tight_layout()
# ax.get_figure().savefig(f'scaling_all.png')
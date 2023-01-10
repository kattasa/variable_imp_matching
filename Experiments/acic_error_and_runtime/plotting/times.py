from glob import glob
import  json
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import seaborn as sns

all_folders = glob(f"{os.getenv('RESULTS_FOLDER')}/*/", recursive=True)

n_repeats = 10
plot_name = os.getenv('PLOT_NAME')
methods = [
    'LASSO Coefficient Matching',
    # 'Tree Feature Importance Matching',
    'Equal Weighted LASSO Matching',
    'MALTS Matching',
    'Prognostic Score Matching',
    'BART',
    'DoubleML',
    'DRLearner',
    'Causal Forest',
    'Causal Forest DML'
]

rename_methods = {
    "BART": "T-Learner BART",
    "Causal Forest DML": "Causal Forest\nDML",
    'LASSO Coefficient Matching': 'LASSO Coefficient\nMatching',
    "Equal Weighted LASSO Matching": "Equal Weighted\nLASSO Matching",
    'Prognostic Score Matching': 'Prognostic Score\nMatching',
    "DoubleML": "Linear DoubleML",
    "DRLearner": "Linear DRLearner"
}

order = [
    'LASSO Coefficient\nMatching',
    "Equal Weighted\nLASSO Matching",
    # 'Tree Feature\nImportance Matching',
    'MALTS Matching',
    'Prognostic Score\nMatching',
    "T-Learner BART",
    'Causal Forest',
    "Causal Forest\nDML",
    'Linear DoubleML',
    'Linear DRLearner'
]

all_times = pd.DataFrame([], index=methods)
failed_files = {}
name_to_label = {}
methods_dirs = {
    'BART': 'bart_fit_times',
    'Causal Forest': 'causalforest_fit_times',
    'LASSO Coefficient Matching': 'lcm_fit_times',
    'Equal Weighted LASSO Matching': 'equal_lcm_fit_times',
    'MALTS Matching': 'malts_fit_times',
    'Prognostic Score Matching': 'prognostic_fit_times',
    'DoubleML': 'doubleml_fit_times',
    'DRLearner': 'drlearner_fit_times',
    'Causal Forest DML': 'causalforest_dml_fit_times'
}

acic_2018_file_no = 1
for f in all_folders:
    n_samples = pd.read_csv(f'{f}df_true.csv').shape[0]
    if 'acic_2019' in f:
        n_splits = 3
    else:
        if n_samples < 2000:
            n_splits = 2
        else:
            with open(f'{f}config.txt') as c:
                n_splits = json.loads(c.read())['n_splits']
    if n_samples <= 5000:
        malts = True
    else:
        malts = False
    if "acic_2018-d09f96200455407db569ae33fe06b0d3_000" in f:
        bart = False
    else:
        bart = True
    times = {}
    for m, d in methods_dirs.items():
        if (m == 'MALTS Matching') and malts is not True:
            print(f'MALTS not run: {f}')
            continue
        if (m == 'BART') and bart is not True:
            print(f'BART not run: {f}')
            continue
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
                times[m] = these_times
        else:
            if f.split('/')[-2] in failed_files:
                failed_files[f.split('/')[-2]].append(m)
            else:
                failed_files[f.split('/')[-2]] = [m]
    if len(times) > 0:
        if 'acic_2019' in f:
            label = f"ACIC 2019 {f.split('/')[-2].split('_')[1].split('-')[1]}"
        elif 'acic_2018' in f:
            label = f'ACIC 2018 {acic_2018_file_no}'
            acic_2018_file_no += 1
        all_times = all_times.join(pd.DataFrame.from_dict(times, orient='index', columns=[label]*(n_splits*n_repeats)))

print('Failed Files:')
print(failed_files)

all_times = all_times.reset_index().melt(id_vars=['index'])
all_times.columns = ['Method', 'ACIC File', 'Single CATE Runtime (s)']
all_times[['acic_year', 'acic_file_no']] = all_times['ACIC File'].str.split(expand=True).iloc[:, 1:].astype(int)
all_times = all_times.sort_values(['acic_year', 'acic_file_no'])
all_times['Method'] = all_times['Method'].apply(lambda x: rename_methods[x] if x in rename_methods.keys() else x)


mean_times = all_times.groupby(['ACIC File', 'Method'])['Single CATE Runtime (s)'].mean().reset_index()
mean_times[['acic_year', 'acic_file_no']] = mean_times['ACIC File'].str.split(expand=True).iloc[:, 1:].astype(int)

matplotlib.rcParams.update({'font.size': 50})
plt.style.use(['seaborn-darkgrid'])
fig, axes = plt.subplots(2, 1, figsize=(40, 30))
sns.set_context("paper")
sns.set_style("darkgrid")
sns.set(font_scale=6)
b1 = sns.barplot(data=mean_times[(mean_times['acic_year'] == 2018) & (mean_times['acic_file_no'] <= 15)],
                 x="ACIC File", y='Single CATE Runtime (s)', hue="Method", hue_order=order, ax=axes[0])
b2 = sns.barplot(data=mean_times[((mean_times['acic_year'] == 2018) & (mean_times['acic_file_no'] > 15)) | (mean_times['acic_year'] == 2019)],
                 x="ACIC File", y='Single CATE Runtime (s)', hue="Method", hue_order=order, ax=axes[1])

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='right', bbox_to_anchor=(0.95, 1.07), ncol=3)
for ax in axes:
    ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_yscale('log')
    ax.get_legend().remove()
b1.set(xlabel=None, ylabel=None)
b2.set(xlabel=None, ylabel=None)
fig.text(0.5, -0.01, 'ACIC File', ha='center')
fig.text(-0.01, 0.5, 'Single CATE Runtime (s)', va='center', rotation='vertical')
fig.tight_layout()
fig.savefig(f'plots/acic_cate_times{plot_name}.png', bbox_inches='tight')

rankings = mean_times.sort_values(['ACIC File', 'Single CATE Runtime (s)'], ascending=True)
n_methods = rankings['Method'].nunique()
rankings['Ranking'] = list(range(1, n_methods+1))*(rankings.shape[0] // n_methods)
rankings = rankings[~rankings['Single CATE Runtime (s)'].isna()]

plt.figure(figsize=(26, 23))
sns.set_context("paper")
sns.set_style("darkgrid")
sns.set(font_scale=6)
sns.boxplot(data=rankings, x="Ranking", y="Method", order=order)
plt.xticks(list(range(1, 1+len(order))))
plt.tight_layout()
plt.savefig(f'plots/acic_cate_times_ranking{plot_name}.png')

nan_counts = all_times['Single CATE Runtime (s)'].isnull().groupby(all_times['ACIC File']).sum()
no_nan_acic_files = list(nan_counts[nan_counts == 0].index)
all_times_no_nan = all_times[all_times['ACIC File'].isin(no_nan_acic_files)]

plt.figure()
sns.set_context("paper")
sns.set_style("darkgrid")
sns.set(font_scale=1.2)
sns.boxenplot(data=all_times_no_nan, y='Single CATE Runtime (s)', x="Method", order=order)
plt.xticks(rotation=65, horizontalalignment='right')
plt.yscale('log')
plt.tight_layout()
plt.savefig(f'plots/acic_cate_times_by_method{plot_name}.png')

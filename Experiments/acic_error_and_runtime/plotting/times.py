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
plot_name = '_results_main'
q = 0.5
methods = [
    'LASSO Coefficient Matching',
    # 'Tree Feature Importance Matching',
    # 'Manhattan with Feature Selection',
    # 'Equal Weighted LASSO Matching',
    'MALTS Matching',
    'Prognostic Score Matching',
    'BART',
    'DoubleML',
    'DRLearner',
    'Causal Forest',
    'Causal Forest 2'
    # 'Causal Forest DML'
]

rename_methods = {
    "BART": "T-Learner BART",
    "Causal Forest 2": "Causal Forest\nDML",
    # "Causal Forest DML": "Causal Forest\nDML",
    'LASSO Coefficient Matching': 'LASSO Coefficient\nMatching',
    # "Manhattan with Feature Selection": "Equal Weighted\nLASSO Matching",
    # "Equal Weighted LASSO Matching": "Equal Weighted\nLASSO Matching",
    'Prognostic Score Matching': 'Prognostic Score\nMatching',
    "DoubleML": "Linear DoubleML",
    "DRLearner": "Linear DRLearner"
}

order = [
    'LASSO Coefficient\nMatching',
    # "Equal Weighted\nLASSO Matching",
    # 'Tree Feature Importance Matching',
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
    'Manhattan with Feature Selection': 'equal_lcm_fit_times',
    # 'Equal Weighted LASSO Matching': 'lcm_fit_times',
    'MALTS Matching': 'malts_fit_times',
    'Prognostic Score Matching': 'prognostic_fit_times',
    'DoubleML': 'doubleml_fit_times',
    'DRLearner': 'drlearner_fit_times',
    'Causal Forest 2': 'causalforest_dml_fit_times'
    # 'Causal Forest DML': 'causalforest_dml_fit_times
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
all_times['Method'] = all_times['Method'].apply(lambda x: rename_methods[x] if x in rename_methods.keys() else x)

plt.figure()
sns.set_context("paper")
sns.set_style("darkgrid")
sns.set(font_scale=1)
ax = sns.catplot(data=all_times, x="ACIC File", y="Single CATE Runtime (s)", hue="Method", kind="bar", hue_order=order)
plt.xticks(rotation=65, horizontalalignment='right')
plt.tight_layout()
plt.legend(loc='upper right', prop={'size': 10})
plt.yscale('log')
plt.savefig(f'plots/acic_cate_runtimes{plot_name}.png')

plt.figure()
sns.set_context("paper")
sns.set_style("darkgrid")
sns.set(font_scale=1)
sns.boxplot(data=all_times[all_times['Method'] != 'MALTS Matching'], x="Single CATE Runtime (s)", y="Method",
            order=order)
plt.xticks(rotation=65, horizontalalignment='right')
plt.tight_layout()
plt.savefig(f'plots/acic_cate_runtimes_by_method{plot_name}.png')

rankings = all_times.sort_values(['ACIC File', 'Single CATE Runtime (s)'],ascending=True)
n_methods = rankings['Method'].nunique()
rankings['Ranking'] = list(range(1, n_methods+1))*(rankings.shape[0] // n_methods)
rankings = rankings[~rankings['Single CATE Runtime (s)'].isna()]

plt.figure()
sns.set_context("paper")
sns.set_style("darkgrid")
sns.set(font_scale=1)
sns.boxplot(data=rankings, x="Ranking", y="Method", order=order)
plt.xticks(rotation=65, horizontalalignment='right')
plt.tight_layout()
plt.savefig(f'plots/acic_cate_runtimes_ranking{plot_name}.png')

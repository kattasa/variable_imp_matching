from glob import glob
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

all_folders = glob(f"{os.getenv('RESULTS_FOLDER')}/*/", recursive=True)

plot_name = os.getenv('PLOT_NAME')
q = 0.5
methods = [
    'LASSO Coefficient Matching',
    # 'Tree Feature Importance Matching',
    # 'GBR Feature Importance Matching',
    # 'GBR Single Model Feature Importance Matching',
    # 'Equal Weighted LASSO Matching',
    'Linear Prognostic Score Matching',
    'Ensemble Prognostic Score Matching',
    'DoubleML',
    # 'DRLearner',
    'BART',
    'Causal Forest',
    'Causal Forest DML'
]

rename_methods = {
    "BART": "T-Learner BART",
    "Causal Forest DML": "Causal Forest\nDML",
    'Tree Feature Importance Matching': 'Tree Feature\nImportance Matching',
    'LASSO Coefficient Matching': 'LASSO Coefficient\nMatching',
    "Equal Weighted LASSO Matching": "Equal Weighted\nLASSO Matching",
    'Linear Prognostic Score Matching': 'Linear Prognostic\nScore Matching',
    'Ensemble Prognostic Score Matching': 'Ensemble Prognostic\nScore Matching',
    "DoubleML": "Linear DoubleML",
    "DRLearner": "Linear DRLearner",
    'GBR Feature Importance Matching': 'GBR Feature\nImportance Matching',
    'GBR Single Model Feature Importance Matching': 'GBR Single Model\nFeature Importance Matching'
}

order = [
    'LASSO Coefficient\nMatching',
    "Equal Weighted\nLASSO Matching",
    'Tree Feature\nImportance Matching',
    'GBR Feature\nImportance Matching',
    'GBR Single Model\nFeature Importance Matching',
    'Linear Prognostic\nScore Matching',
    'Ensemble Prognostic\nScore Matching',
    "T-Learner BART",
    'Causal Forest',
    "Causal Forest\nDML",
    'Linear DoubleML',
    'Linear DRLearner',
]

all_errors = pd.DataFrame([], index=methods)
failed_files = []
name_to_label = {}

acic_2018_file_no = 1
for f in all_folders:
    if os.path.isfile(f'{f}df_err.csv'):
        if 'acic_2019' in f:
            label = f"ACIC 2019 {f.split('/')[-2].split('_')[1].split('-')[1]}"
        elif 'acic_2018' in f:
            label = f'ACIC 2018 {acic_2018_file_no}'
            acic_2018_file_no += 1
        all_errors = all_errors.join(pd.read_csv(f'{f}df_err.csv').groupby('Method')['Relative Error (%)'].quantile(q).rename(label).to_frame())
        # all_errors = all_errors.join(
        #     pd.read_csv(f'{f}df_err.csv').groupby('Method')[
        #         'Relative Error (%)'].mean().rename(label).to_frame())
        name_to_label[f.split('/')[-2]] = label
    else:
        failed_files.append(f.split('/')[-2])
        print(f"Failed: {f.split('/')[-2]}")

with open(f'plots/acic_file_to_num{plot_name}.txt', 'w') as f:
    f.write(json.dumps({v: k for k, v in name_to_label.items()}))

all_errors_original = all_errors.loc[~(all_errors.isna().sum(axis=1) == all_errors.shape[1]).values, :]
rename_methods = {k: v for k, v in rename_methods.items() if k in all_errors_original.index}
all_errors = all_errors_original.reset_index().melt(id_vars=['index'])
all_errors.columns = ['Method', 'ACIC File', 'Median Relative Error (%)']
all_errors[['acic_year', 'acic_file_no']] = all_errors['ACIC File'].str.split(expand=True).iloc[:, 1:].astype(int)
all_errors = all_errors.sort_values(['acic_year', 'acic_file_no'])
all_errors['Method'] = all_errors['Method'].apply(lambda x: rename_methods[x] if x in rename_methods.keys() else x)
order = [c for c in order if c in all_errors['Method'].unique()]

matplotlib.rcParams.update({'font.size': 50})
plt.style.use(['seaborn-darkgrid'])
fig, axes = plt.subplots(2, 1, figsize=(40, 30))
sns.set_context("paper")
sns.set_style("darkgrid")
sns.set(font_scale=6)
b1 = sns.barplot(data=all_errors[(all_errors['acic_year'] == 2018) & (all_errors['acic_file_no'] <= 15)],
                 x="ACIC File", y="Median Relative Error (%)", hue="Method", hue_order=order, ax=axes[0])
b2 = sns.barplot(data=all_errors[((all_errors['acic_year'] == 2018) & (all_errors['acic_file_no'] > 15) | (all_errors['acic_year'] == 2019))],
                 x="ACIC File", y="Median Relative Error (%)", hue="Method", hue_order=order, ax=axes[1])

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='right', bbox_to_anchor=(0.95, 1.07), ncol=3)
# axes[0].set_xticks(axes[0].get_xticks(), axes[0].get_xticklabels(), rotation=45, ha='right')
# axes[0].set_yscale('log')
# axes[0].get_legend().remove()
for ax in axes:
    ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_yscale('log')
    ax.get_legend().remove()
    # ax.set_ylim([0, 2])
b1.set(xlabel=None, ylabel=None)
# b2.set(xlabel=None, ylabel=None)
fig.text(0.5, -0.01, 'ACIC File', ha='center')
fig.text(-0.01, 0.5, 'Median Relative Error (%)', va='center', rotation='vertical')
fig.tight_layout()
fig.savefig(f'plots/acic_cate_errors{plot_name}.png', bbox_inches='tight')

rankings = all_errors.sort_values(['ACIC File', 'Median Relative Error (%)'],ascending=True)
n_methods = rankings['Method'].nunique()
rankings['Ranking'] = list(range(1, n_methods+1))*(rankings.shape[0] // n_methods)
rankings = rankings[~rankings['Median Relative Error (%)'].isna()]

plt.figure(figsize=(26, 23))
sns.set_context("paper")
sns.set_style("darkgrid")
sns.set(font_scale=6)
sns.boxplot(data=rankings, x="Ranking", y="Method", order=order)
plt.xticks(list(range(1, 1+len(order))))
plt.tight_layout()
plt.savefig(f'plots/acic_cate_errors_ranking{plot_name}.png')

plt.figure()
sns.set_context("paper")
sns.set_style("darkgrid")
sns.set(font_scale=1)
sns.boxplot(data=all_errors, x="Median Relative Error (%)", y="Method", order=order, showfliers=False)
plt.xticks(rotation=65, horizontalalignment='right')
plt.tight_layout()
plt.savefig(f'plots/acic_cate_errors_by_method{plot_name}.png')

relative_errors = all_errors_original.divide(all_errors_original.min(), axis=1)
relative_errors = relative_errors.reset_index().melt(id_vars=['index'])
relative_errors.columns = ['Method', 'ACIC File', 'Relative Median Error to Top Method (%)']
relative_errors[['acic_year', 'acic_file_no']] = relative_errors['ACIC File'].str.split(expand=True).iloc[:, 1:].astype(int)
relative_errors = relative_errors.sort_values(['acic_year', 'acic_file_no'])
relative_errors['Method'] = relative_errors['Method'].apply(lambda x: rename_methods[x] if x in rename_methods.keys() else x)

plt.figure()
sns.set_context("paper")
sns.set_style("darkgrid")
sns.set(font_scale=1)
sns.boxplot(data=relative_errors, x="Relative Median Error to Top Method (%)", y="Method", order=order)
plt.xticks(rotation=65, horizontalalignment='right')
plt.tight_layout()
plt.savefig(f'plots/acic_cate_relative_errors_by_method{plot_name}.png')
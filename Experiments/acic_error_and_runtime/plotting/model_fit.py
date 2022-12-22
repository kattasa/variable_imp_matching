from glob import glob
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import seaborn as sns

all_folders = glob(f"{os.getenv('RESULTS_FOLDER')}/*/", recursive=True)

n_repeats = 25
plot_name = '_results_2'
q = 0.5
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
    # 'Tree Feature Importance Matching',
    'MALTS Matching',
    'Prognostic Score\nMatching',
    "T-Learner BART",
    'Causal Forest',
    "Causal Forest\nDML",
    'Linear DoubleML',
    'Linear DRLearner'
]

all_errors = pd.DataFrame([], index=methods)
model_fit_scores = {}
failed_error_files = []
failed_model_fit_files = []
name_to_label = {}

acic_2018_file_no = 1
acic_2019_file_no = 1
for f in all_folders:
    if os.path.isfile(f'{f}df_err.csv'):
        if 'acic_2019' in f:
            label = f'ACIC 2019 {acic_2019_file_no}'
            acic_2019_file_no += 1
        elif 'acic_2018' in f:
            label = f'ACIC 2018 {acic_2018_file_no}'
            acic_2018_file_no += 1
        errors = pd.read_csv(f'{f}df_err.csv').groupby('Method')['Relative Error (%)'].quantile(0.5)
        #  get model fit scores
        with open(f'{f}config.txt') as c:
            n_splits = json.loads(c.read())['n_splits']
        if os.path.isdir(f'{f}lcm_model_fit_scores'):
            these_model_fits = []
            for t in glob(f'{f}lcm_model_fit_scores/*.txt'):
                try:
                    with open(t) as this_f:
                        these_model_fits.append(float(this_f.read().replace('\n', '')))
                except Exception:
                    pass
            if len(these_model_fits) != (n_splits * n_repeats):
                print(f"Failed Model Fits: {f.split('/')[-2]}")
                failed_model_fit_files.append(f.split('/')[-2])
            else:
                model_fit_scores[label] = np.percentile(these_model_fits, q)*100
        else:
            print(f"Failed Model Fits: {f.split('/')[-2]}")
            failed_model_fit_files.append(f.split('/')[-2])
        all_errors = all_errors.join(errors.rename(label).to_frame())
        name_to_label[f.split('/')[-2]] = label
    else:
        print(f"Failed Error File: {f.split('/')[-2]}")
        failed_error_files.append(f.split('/')[-2])

all_errors = all_errors.reset_index().melt(id_vars=['index'])
all_errors.columns = ['Method', 'ACIC File', 'Median Relative Error (%)']
all_errors[['acic_year', 'acic_file_no']] = all_errors['ACIC File'].str.split(expand=True).iloc[:, 1:].astype(int)
all_errors = all_errors.sort_values(['acic_year', 'acic_file_no'])
all_errors['Method'] = all_errors['Method'].apply(lambda x: rename_methods[x] if x in rename_methods.keys() else x)

model_fit_scores = pd.DataFrame([model_fit_scores]).T
model_fit_scores.columns = ['LCM LASSO R2 Score']

all_errors = all_errors.join(model_fit_scores, on='ACIC File', how='left')

all_errors = all_errors[all_errors['acic_year'] == 2019]  # Limit to only ACIC 2019 files for this plot
all_errors = all_errors.sort_values('Median Relative Error (%)').reset_index().rename(columns={'index': 'Ranking'})
for file in all_errors['ACIC File'].unique():
    all_errors.loc[all_errors['ACIC File'] == file, 'Ranking'] = [i + 1 for i in
                                                                  list(all_errors[all_errors['ACIC File'] == file]
                                                                       .reset_index().index)]
all_errors = all_errors[all_errors['Method'] == 'LASSO Coefficient\nMatching']
all_errors = all_errors.drop(columns=['acic_year', 'acic_file_no'])
plt.figure()
sns.set_context("paper")
sns.set_style("darkgrid")
sns.set(font_scale=1)
ax = sns.regplot(data=all_errors, x="LCM LASSO R2 Score", y="Median Relative Error (%)")
ax.yaxis.set_major_formatter(ticker.PercentFormatter())
plt.tight_layout()
plt.savefig(f'plots/acic_2019_model_fit_scores{plot_name}.png')

plt.figure()
sns.set_context("paper")
sns.set_style("darkgrid")
sns.set(font_scale=1)
ax = sns.regplot(data=all_errors, x="LCM LASSO R2 Score", y="Ranking")
plt.tight_layout()
plt.savefig(f'plots/acic_2019_model_fit_rankings{plot_name}.png')

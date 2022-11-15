from glob import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

all_folders = glob(f"{os.getenv('RESULTS_FOLDER')}/*/", recursive=True)

q = 0.5
methods = ['LASSO Coefficient Matching', 'MALTS Matching', 'Prognostic Score Matching', 'BART', 'Causal Forest']
all_errors = pd.DataFrame([], index=methods)
failed_files = []
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
        all_errors = all_errors.join(pd.read_csv(f'{f}df_err.csv').groupby('Method')['Relative Error (%)'].quantile(0.5).rename(label).to_frame())
        name_to_label[f.split('/')[-2]] = label
    else:
        failed_files.append(f.split('/')[-2])

all_errors = all_errors.reset_index().melt(id_vars=['index'])
all_errors.columns = ['Method', 'ACIC File', 'Median Relative Error (%)']
all_errors[['acic_year', 'acic_file_no']] = all_errors['ACIC File'].str.split(expand=True).iloc[:, 1:].astype(int)
all_errors = all_errors.sort_values(['acic_year', 'acic_file_no'])
all_errors = all_errors.drop(columns=['acic_year', 'acic_file_no'])

sns.set_context("paper")
sns.set_style("darkgrid")
sns.set(font_scale=1)
ax = sns.catplot(data=all_errors, x="ACIC File", y="Median Relative Error (%)", hue="Method", kind="bar")
plt.xticks(rotation=65, horizontalalignment='right')
plt.tight_layout()
plt.yscale('log')
plt.show()

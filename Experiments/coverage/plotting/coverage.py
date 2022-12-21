from glob import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

all_folders = glob(f"{os.getenv('RESULTS_FOLDER')}/*/", recursive=True)

all_ate = []

for f in all_folders:
    this_ate = pd.melt(pd.read_csv(f'{f}ate_df.csv').groupby('Method')[['coverage', 'sem']].mean().reset_index(),
                       id_vars=['Method'])
    this_ate['# samples'] = int(f.split('/')[-2].split('_')[0])
    all_ate.append(this_ate.copy(deep=True))

all_ate = pd.concat(all_ate)

all_ate_coverage = all_ate[all_ate['variable'] == 'coverage']
all_ate_sem = all_ate[all_ate['variable'] == 'sem']


plt.figure()
sns.set_context("paper")
sns.set_style("darkgrid")
sns.set(font_scale=1)
sns.catplot(data=all_ate_coverage, x="# samples", y="value", hue="Method", kind="bar", legend=False,
            aspect=3/2)
plt.xticks(rotation=65, horizontalalignment='right')
plt.tight_layout()
plt.legend(loc='upper right', prop={'size': 10})
plt.savefig('plots/ate_coverage.png')


plt.figure()
sns.set_context("paper")
sns.set_style("darkgrid")
sns.set(font_scale=1)
sns.catplot(data=all_ate_sem, x="# samples", y="value", hue="Method", kind="bar", legend=False,
            aspect=3/2)
plt.xticks(rotation=65, horizontalalignment='right')
plt.tight_layout()
plt.legend(loc='upper right', prop={'size': 10})
plt.savefig('plots/ate_sem.png')

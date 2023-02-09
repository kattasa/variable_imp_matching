from glob import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import seaborn as sns

from Experiments.helpers import get_errors

num_covs_folder = glob(f"{os.getenv('RESULTS_FOLDER')}/num_covs/*", recursive=True)

methods = [
    'LCM',
    'MALTS',
    'GenMatch'
]

lcm_cates = []
malts_cates = []
genmatch_cates = []

full_errors = []

for file in num_covs_folder:
    try:
        lcm = pd.read_csv(f'{file}/lcm_cates.csv', index_col=0)[['avg.CATE_mean']]
        malts = pd.read_csv(f'{file}/malts_cates.csv', index_col=0)[['avg.CATE']]
        genmatch = pd.read_csv(f'{file}/genmatch_cates.csv', index_col=0)
        genmatch.index -= 1
        df_true = pd.read_csv(f'{file}/df_true.csv')
        df = df_true[['TE']].join(lcm).join(malts).join(genmatch)
        df = df.rename(columns={'avg.CATE_mean': 'LCM', 'avg.CATE': 'MALTS',
                                'CATE': 'GenMatch'})
        ate = np.abs(df['TE']).mean()
        all_errors = []
        for m in methods:
            all_errors.append(get_errors(df[[m]], df[['TE']],
                                        method_name=m, scale=ate))
        all_errors = pd.concat(all_errors)
        all_errors['# Covariates'] = int(file.split('/')[-1]) + 8
        full_errors.append(all_errors.copy())
    except:
        print('uh oh')

full_errors = pd.concat(full_errors)

order = ['LCM', 'Linear PGM', 'Ensemble PGM', 'MALTS', '', 'GenMatch']
palette = {order[i]: sns.color_palette()[i] for i in range(len(order))}
method_order = [c for c in order if c in full_errors['Method'].unique()]

plt.figure()
sns.set_context("paper")
sns.set_style("darkgrid")
sns.set(font_scale=1)
ax = sns.boxplot(data=full_errors, x="# Covariates", y="Relative Error (%)",
                 hue='Method', hue_order=method_order, palette=palette,
                 showfliers=False)
sns.move_legend(ax, "lower center", bbox_to_anchor=(.5, 1), ncol=3, title=None)
ax.yaxis.set_major_formatter(ticker.PercentFormatter())
ax.get_figure().savefig(f'accuracy.png')

"""Replots errors from run_quadratic_ml.py"""


import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

results_folder = os.getenv('RESULTS_FOLDER')
quadratic_folder = os.getenv('DENSE_CONTINUOUS_FOLDER')

df = pd.read_csv(f'{results_folder}/{quadratic_folder}/df_err.csv')

order = ['LCM\nMean', 'LCM\nLinear', 'Linear\nDML',
         'Causal\nForest DML', 'Causal\nForest', 'BART\nTLearner']
palette = {order[i]: sns.color_palette()[i] for i in range(len(order))}

plt.figure(figsize=(8, 4))
sns.set_context("paper")
sns.set_style("darkgrid")
sns.set(font_scale=1.5, font="times")
ax = sns.boxplot(x='Method', y='Relative Error (%)',
                 data=df, showfliers=False,
                 order=order, palette=palette)
ax.yaxis.set_major_formatter(ticker.PercentFormatter())
ax.get_figure().savefig(f'{results_folder}/lcm_vs_ml.png', bbox_inches='tight')

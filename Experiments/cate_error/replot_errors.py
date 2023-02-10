import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

exp = pd.read_csv('Results/exp_002/df_err.csv')
sine = pd.read_csv('Results/sine_002/df_err.csv')

order = ['LCM', 'Linear PGM']
palette = {order[i]: sns.color_palette()[i] for i in range(len(order))}

sns.set_context("paper")
sns.set_style("darkgrid")
sns.set(font_scale=3.3)
fig, axes = plt.subplots(1, 2, figsize=(15, 9))
sns.boxplot(ax=axes[0], data=sine,
            x='Method', y='Relative Error (%)',
            showfliers=False,
            order=order,
            palette=palette)
sns.boxplot(ax=axes[1], data=exp,
            x='Method', y='Relative Error (%)',
            showfliers=False,
            order=order,
            palette=palette)

axes[0].set_xlabel(xlabel='Sine', labelpad=10, fontdict={'weight': 'bold'})
axes[1].set_xlabel(xlabel='Exponential', labelpad=10, fontdict={'weight': 'bold'})
axes[0].yaxis.set_major_formatter(ticker.PercentFormatter())
axes[1].yaxis.set_major_formatter(ticker.PercentFormatter())
axes[1].set_ylabel(ylabel=None)
fig.tight_layout()
fig.savefig(f'lcm_vs_lin_pgm.png', bbox_inches='tight')

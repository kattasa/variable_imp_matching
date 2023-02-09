import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

exp = pd.read_csv('Results/exp_000/df_err.csv')
sine = pd.read_csv('Results/sine_000/df_err.csv')

order = ['LCM', 'Linear PGM']
palette = {order[i]: sns.color_palette()[i] for i in range(len(order))}

matplotlib.rcParams.update({'font.size': 40})
sns.set_context("paper")
sns.set_style("darkgrid")
sns.set(font_scale=4)
fig, axes = plt.subplots(1, 2, figsize=(26, 14))
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

# fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.1),
#            ncol=3, fontsize=40,
#            columnspacing=0.5)

axes[0].set_xlabel(xlabel='Sine', labelpad=10, fontdict={'weight': 'bold'})
axes[1].set_xlabel(xlabel='Exponential', labelpad=10, fontdict={'weight': 'bold'})
axes[0].yaxis.set_major_formatter(ticker.PercentFormatter())
fig.tight_layout()
fig.savefig(f'lcm_vs_lin_pgm.png', bbox_inches='tight')

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

exp = pd.read_csv('Results/exp_003/df_err.csv')
sine = pd.read_csv('Results/sine_003/df_err.csv')
quad = pd.read_csv('Results/dense_continuous_003/df_err.csv')

rename = {'LASSO FS': 'LASSO\nFS', 'Oracle FS': 'Oracle\nFS'}
exp['Method'] = exp['Method'].replace(rename)
sine['Method'] = sine['Method'].replace(rename)
quad['Method'] = quad['Method'].replace(rename)
order = ['LCM', 'LASSO\nFS', 'Oracle\nFS']
palette = {
    'LCM': sns.color_palette()[0],
    'LASSO\nFS': sns.color_palette()[8],
    'Oracle\nFS': sns.color_palette()[9],
           }

sns.set_context("paper")
sns.set_style("darkgrid")
sns.set(font_scale=2, font="times")
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
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
sns.boxplot(ax=axes[2], data=quad,
            x='Method', y='Relative Error (%)',
            showfliers=False,
            order=order,
            palette=palette)

axes[0].set_xlabel(xlabel='Sine', labelpad=10, fontdict={'weight': 'bold'})
axes[1].set_xlabel(xlabel='Exponential', labelpad=10, fontdict={'weight': 'bold'})
axes[2].set_xlabel(xlabel='Quadratic', labelpad=10, fontdict={'weight': 'bold'})
axes[0].yaxis.set_major_formatter(ticker.PercentFormatter())
axes[1].yaxis.set_major_formatter(ticker.PercentFormatter())
axes[2].yaxis.set_major_formatter(ticker.PercentFormatter())
axes[1].set_ylabel(ylabel=None)
axes[2].set_ylabel(ylabel=None)
fig.tight_layout()
fig.savefig(f'lcm_vs_fs.png', bbox_inches='tight')

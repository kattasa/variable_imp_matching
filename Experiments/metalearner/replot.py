import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

df_err = pd.read_csv('Results2/df_err.csv', index_col=[0])

color_order = ['LCM', 'Linear\nPGM', 'Nonparametric\nPGM', 'MALTS',
               'Metalearner\nLCM', 'BART', 'Linear\nTLearner',
               'Nonparametric\nTLearner']
palette = {color_order[i]: sns.color_palette()[i] for i in range(len(color_order))}

method_order = ['Metalearner\nLCM', 'LCM', 'MALTS', 'Linear\nPGM',
                'Nonparametric\nPGM',  'BART', 'Linear\nTLearner',
                'Nonparametric\nTLearner'
                ]

order = [m for m in method_order if m in df_err['Method'].unique()]

df_err.loc[:, 'Relative Error (%)'] = df_err.loc[:, 'Relative Error (%)'] * 100
plt.figure(figsize=(8, 8))
sns.set_context("paper")
sns.set_style("darkgrid")
sns.set(font_scale=2)
ax = sns.boxplot(x='Method', y='Relative Error (%)',
                 data=df_err, showfliers=False,
                 order=order, palette=palette)
ax.yaxis.set_major_formatter(ticker.PercentFormatter())
ax.get_figure().savefig(f'Results/metalearner_boxplot_err.png',
                        bbox_inches='tight')

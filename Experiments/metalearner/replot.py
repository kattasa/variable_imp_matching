import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

df_err = pd.read_csv('Results/df_err.csv', index_col=[0])
df_weights = pd.read_csv('Results/df_weights.csv', index_col=[0])
mg_diffs = pd.read_csv('Results/mg_diffs.csv', index_col=[0])

rename_methods = {
    'BART': 'BART\nTLearner',
    'Nonparametric\nPGM': 'Nonparam\nPGM',
    'Nonparametric\nTLearner': 'Nonparam\nTLearner',
    'Causal Forest': 'Causal\nForest'
}

df_err['Method'] = df_err['Method'].replace(rename_methods)

color_order = ['LCM', 'Linear\nPGM', 'Nonparam\nPGM', 'MALTS',
               'Metalearner\nLCM', 'BART\nTLearner', 'GenMatch', 'Linear\nTLearner',
               'Nonparam\nTLearner', 'Causal\nForest']
palette = {color_order[i]: sns.color_palette()[i] for i in range(len(color_order))}

method_order = ['Metalearner\nLCM', 'LCM', 'MALTS', 'GenMatch', 'Linear\nPGM',
                'Nonparam\nPGM', 'Linear\nTLearner',
                'Nonparam\nTLearner', 'BART\nTLearner', 'Causal\nForest'
                ]

order = [m for m in method_order if m in df_err['Method'].unique()]

df_err.loc[:, 'Relative Error (%)'] = df_err.loc[:, 'Relative Error (%)'] * 100
plt.figure(figsize=(8, 8))
sns.set_context("paper")
sns.set_style("darkgrid")
sns.set(font_scale=2, font="times")
ax = sns.boxplot(y='Method', x='Relative Error (%)',
                 data=df_err, showfliers=False,
                 order=order, palette=palette)
ax.xaxis.set_major_formatter(ticker.PercentFormatter())
ax.set_xlim([-10, 300])
ax.get_figure().savefig(f'Results/metalearner_boxplot_err.png',
                        bbox_inches='tight')


rename_methods = {
    'LCM': 'LCM\n'+r"$\mathcal{M}^*$",
    'Metalearner\nLCM M_C': "Metalearner\n"+ r"LCM $\mathcal{M}^{(0)*}$",
    'Metalearner\nLCM M_T': "Metalearner\n"+ r"LCM $\mathcal{M}^{(1)*}$"
}

df_weights.columns = [f'X{i}' for i in range(1, len(df_weights.columns))] + ['Method']
df_weights['Method'] = df_weights['Method'].replace(rename_methods)

palette = {'LCM\n'+r"$\mathcal{M}^*$": sns.color_palette()[0],
           "Metalearner\n"+ r"LCM $\mathcal{M}^{(0)*}$": sns.color_palette()[7],
           "Metalearner\n"+ r"LCM $\mathcal{M}^{(1)*}$": sns.color_palette()[8]}
order = ['LCM\n'+r"$\mathcal{M}^*$", "Metalearner\n"+ r"LCM $\mathcal{M}^{(0)*}$",
         "Metalearner\n"+ r"LCM $\mathcal{M}^{(1)*}$"]

x_imp = 3  # to include unimportant covariate in plotting
df_weights = df_weights[['Method'] + [f'X{i}' for i in range(1,x_imp+1)]].melt(id_vars=['Method'])
df_weights = df_weights.rename(columns={'variable': 'Covariate', 'value': 'Relative Weight (%)'})
df_weights['Relative Weight (%)'] *= 100

plt.figure(figsize=(6, 5))
sns.set_context("paper")
sns.set_style("darkgrid")
sns.set(font_scale=2, font="times")
ax = sns.barplot(data=df_weights, x="Covariate", y="Relative Weight (%)",
                 hue="Method", hue_order=order, palette=palette,
                 errorbar=None)
ax.yaxis.set_major_formatter(ticker.PercentFormatter())
sns.move_legend(ax, "lower center", bbox_to_anchor=(.36, 1.02), ncol=3,
                title=None, handletextpad=0.4, columnspacing=0.5, fontsize=18)
plt.tight_layout()
ax.get_figure().savefig(f'Results/metalearner_barplot_weights.png')

mg_diffs['Method'] = mg_diffs['Method'].str.replace('MGs', 'KNN')
mg_diffs['Covariate'] = mg_diffs['Covariate'].apply(lambda x: f'X{int(x[-1])+1}')

palette = {'LCM\nControl KNN': sns.color_palette()[0],
            'LCM\nTreatment KNN': sns.color_palette()[9],
           'Metalearner LCM\nControl KNN': sns.color_palette()[7],
           'Metalearner LCM\nTreatment KNN': sns.color_palette()[8]}
order = ['LCM\nControl KNN', 'LCM\nTreatment KNN',
         'Metalearner LCM\nControl KNN', 'Metalearner LCM\nTreatment KNN']

plt.figure(figsize=(6, 7))
sns.set_context("paper")
sns.set_style("darkgrid")
sns.set(font_scale=2, font="times")
ax = sns.boxplot(x='Covariate', y='Mean Absolute Difference', hue='Method',
                 data=mg_diffs, showfliers=False,
                 order=[f'X{i}' for i in range(1,x_imp+1)], palette=palette,
                 hue_order=order)
sns.move_legend(ax, "lower center", bbox_to_anchor=(.45, 1), ncol=2, title=None,
                handletextpad=0.4, columnspacing=0.5, fontsize=18)
plt.tight_layout()
ax.get_figure().savefig(f'Results/metalearner_barplot_mg_avg_diff.png')


print('done')
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

folder = 'Results/acic_2019'
iters = 8
methods_to_plot = ['Double Model Lasso Matching Mean',
       # 'Double Model Lasso Matching Linear Pruned',
       # 'Single Model Lasso Matching Mean',
       # 'Single Model Lasso Matching Linear Pruned',
       #             'MALTS Matching Mean',
       #             'Linear Prognostic Score Matching',
                   'BART']

df_err = []
for i in range(iters):
    this_df_err = pd.read_csv(f'{folder}/acic_2019_00{i}/df_err.csv')
    this_df_err['Iter'] = i
    df_err.append(this_df_err)

df_err = pd.concat(df_err).reset_index(drop=True)
df_err = df_err.rename(columns={'Iter': 'ACIC File #'})
df_err['ACIC File #'] += 1
df_err = df_err[df_err.Method.isin(methods_to_plot)]

sns.set_context("paper")
sns.set_style("darkgrid")
sns.set(font_scale=3)
fig, ax = plt.subplots(figsize=(40, 50))
pp = sns.pointplot(data=df_err.reset_index(drop=True), x='ACIC File #', y='Relative Error (%)', errorbar=("pi", 95),
                   hue='Method', dodge=True, scale=5)
plt.setp(pp.get_legend().get_texts(), fontsize='50')
plt.setp(pp.get_legend().get_title(), fontsize='60')
plt.legend(ncol=2)
ax.yaxis.set_major_formatter(ticker.PercentFormatter())
plt.tight_layout()
fig.savefig(f'{folder}/all_files.png')

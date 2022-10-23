import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

folder = 'Results/dense_continuous_000'
iters = 5

title = f'dense_continuous CATE Errors as number of irrelevant covariates increases\n(# relevant covariates is always 8)'
save_name = f'increasing_irrelevant_covariates_no_sd.png'

df_err = pd.read_csv(f'{folder}/df_err.csv')
df_err['Method'] = df_err['Method'].replace({'AdMALTS_euclidean': "Euclidean Matching"})
df_err['Iter'] = 2**(df_err['Iter']+3)
df_err = df_err.rename(columns={'Iter': '# Covariates'})
# df_err = df_err[~df_err['Method'].isin(['Propensity Score', 'Causal Forest'])]

sns.set_context("paper")
sns.set_style("darkgrid")
sns.set(font_scale=6)
fig, ax = plt.subplots(figsize=(40, 50))
pp = sns.pointplot(data=df_err.reset_index(drop=True), x='# Covariates', y='Relative Error (%)', ci=0, hue='Method',
                   dodge=True, scale=5)
plt.setp(pp.get_legend().get_texts(), fontsize='50')
plt.setp(pp.get_legend().get_title(), fontsize='60')
plt.legend(ncol=2)
for line, name in zip(list(ax.lines)[::iters + 1], df_err['Method'].unique().tolist()):
    y = line.get_ydata()[-1]
    x = line.get_xdata()[-1]
    if not np.isfinite(y):
        y = next(reversed(line.get_ydata()[~line.get_ydata().mask]), float("nan"))
    if not np.isfinite(y) or not np.isfinite(x):
        continue
    text = ax.annotate(name, xy=(x+0.25, y), xytext=(0, 0), color=line.get_color(),
                       xycoords=(ax.get_xaxis_transform(), ax.get_yaxis_transform()), textcoords="offset points",
                       fontsize='50')
    text_width = (text.get_window_extent(fig.canvas.get_renderer()).transformed(ax.transData.inverted()).width)
    if np.isfinite(text_width):
        ax.set_xlim(ax.get_xlim()[0], text.xy[0] + text_width * 1.05)
plt.title(title)
# plt.xticks(rotation=65, horizontalalignment='right')
ax.yaxis.set_major_formatter(ticker.PercentFormatter())
plt.tight_layout()
fig.savefig(f'{folder}/{save_name}')

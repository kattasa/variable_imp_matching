import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

from datagen.dgp_df import dgp_poly_basic_df
from src.linear_coef_matching_mf import LCM


random_state = 0
n_samples = 1000
n_imp = 1
n_unimp = 9
k_est = 10

df_train, df_est, df_true, x_cols = dgp_poly_basic_df(n_samples=1000,  n_imp=n_imp, n_unimp=n_unimp, n_train=500)

df_err = pd.DataFrame(columns=['Method', 'True_CATE', 'Est_CATE', 'Relative Error (%)'])

method_name = 'LCM'
lcm = LCM(outcome='Y', treatment='T', data=df_train, random_state=random_state)
print('LASSO R^2 score:')
print(lcm.fit(return_scores=True))
cate_df = lcm.CATE(df_estimation=df_est, k=k_est, diameter_prune=None)
cate_df = cate_df.rename(columns={'CATE_mean': 'Est_CATE'})
cate_df['True_CATE'] = df_true['TE'].to_numpy()
cate_df['Relative Error (%)'] = np.abs((cate_df['Est_CATE']-cate_df['True_CATE'])/np.abs(cate_df['True_CATE']).mean())
cate_df['Method'] = [method_name for i in range(cate_df.shape[0])]
df_err = pd.concat([df_err, cate_df[['Method', 'True_CATE', 'Est_CATE', 'Relative Error (%)']].copy(deep=True)])
print(f'{method_name} method complete')

method_name = 'Tree Feature\nImportance Matching'
tree = LCM(outcome='Y', treatment='T', data=df_train,
           random_state=random_state)
tree.fit(model='tree', params={'max_depth': 3})
cate_df2 = tree.CATE(df_estimation=df_est, k=k_est, diameter_prune=None)
cate_df2 = cate_df2.rename(columns={'CATE_mean': 'Est_CATE'})
cate_df2['True_CATE'] = df_true['TE'].to_numpy()
cate_df2['Relative Error (%)'] = np.abs((cate_df2['Est_CATE']-cate_df2['True_CATE'])/np.abs(cate_df2['True_CATE']).mean())
cate_df2['Method'] = [method_name for i in range(cate_df2.shape[0])]
df_err = pd.concat([df_err, cate_df2[['Method', 'True_CATE', 'Est_CATE', 'Relative Error (%)']].copy(deep=True)])
print(f'{method_name} method complete')

palette = {'LCM': sns.color_palette()[0],
           'Tree Feature\nImportance Matching': sns.color_palette()[9]}
order = ['LCM', 'Tree Feature\nImportance Matching']

df_err.to_csv('df_err.csv')
df_err.loc[:, 'Relative Error (%)'] = df_err.loc[:, 'Relative Error (%)'] * 100

plt.figure()
sns.set_context("paper")
sns.set_style("darkgrid")
sns.set(font_scale=1.8)
ax = sns.boxplot(x='Method', y='Relative Error (%)',
                 data=df_err, showfliers=False,
                 order=order, palette=palette)
ax.yaxis.set_major_formatter(ticker.PercentFormatter())
ax.get_figure().savefig(f'lcm_vs_tree.png', bbox_inches='tight')

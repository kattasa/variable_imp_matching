"""Replots the results created by run.py script."""

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

results_folder = os.getenv('RESULTS_FOLDER')
save_folder = os.getenv('PLOTS_FOLDER')

method_order = ['LCM', 'Linear\nPGM', 'NP\nPGM']
palette = {method_order[i]: sns.color_palette()[i] for i in range(len(method_order))}

rename_methods = {
    'Nonparametric\nPGM': 'NP\nPGM'
}

cont_diff_df = pd.read_csv(f'{results_folder}/continuous_diff.csv', index_col=[0])
cont_diff_df['Method'] = cont_diff_df['Method'].replace(rename_methods)

plt.figure(figsize=(5, 6))
sns.set_context("paper")
sns.set_style("darkgrid")
sns.set(font_scale=2, font="times")
ax = sns.boxplot(data=cont_diff_df, x='Covariate',
                 y='Mean Absolute Difference', hue='Method', showfliers=False)
sns.move_legend(ax, "lower center", bbox_to_anchor=(.43, 1), ncol=3, title=None,
                handletextpad=0.4, columnspacing=0.5, fontsize=18)
plt.tight_layout()
ax.get_figure().savefig(f'{save_folder}/mg_diff.png')


df_orig = pd.read_csv(f'{os.getenv("SCHOOLS_FOLDER")}/df.csv')
lcm_cates_full = pd.read_csv(f'{results_folder}/lcm_cates.csv', index_col=[0])
linear_prog_cates_full = pd.read_csv(f'{results_folder}/linear_prog_cates.csv',
                                     index_col=[0])
ensemble_prog_cates_full = pd.read_csv(f'{results_folder}/ensemble_prog_cates.csv',
                                       index_col=[0])

lcm_cates_full['avg.CATE_mean'] = lcm_cates_full.mean(axis=1)
linear_prog_cates_full['avg.CATE'] = linear_prog_cates_full.mean(axis=1)
ensemble_prog_cates_full['avg.CATE'] = ensemble_prog_cates_full.mean(axis=1)

cate_df = lcm_cates_full.join(df_orig.drop(columns=['Z', 'Y']))
cate_df = cate_df[['avg.CATE_mean', 'XC', 'S3']]
cate_df = cate_df.rename(columns={'avg.CATE_mean': 'LCM',
                                  'XC': 'Urbanicity (XC)',
                                  'S3': 'Exp Success (S3)'})
cate_df = cate_df.join(linear_prog_cates_full['avg.CATE'])
cate_df = cate_df.rename(columns={'avg.CATE': 'Linear\nPGM'})
cate_df = cate_df.join(ensemble_prog_cates_full['avg.CATE'])
cate_df = cate_df.rename(columns={'avg.CATE': 'Nonparametric\nPGM'})

cate_df = pd.melt(cate_df, id_vars=['Urbanicity (XC)', 'Exp Success (S3)'],
                  var_name='Method', value_name='Estimated CATE')

plt.figure(figsize=(8, 6))
sns.set_context("paper")
sns.set_style("darkgrid")
sns.set(font_scale=2, font="times")
ax = sns.boxplot(data=cate_df, x="Urbanicity (XC)", y="Estimated CATE", hue='Method', hue_order=method_order, palette=palette, showfliers=False)
sns.move_legend(ax, "lower center", bbox_to_anchor=(.47, 1.02), ncol=3,
                title=None, handletextpad=0.4, columnspacing=0.5, fontsize=20)
plt.tight_layout()
ax.get_figure().savefig(f'{save_folder}/cate_by_xc.png')

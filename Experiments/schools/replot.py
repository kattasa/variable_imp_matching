import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib

save_folder = 'Results_Tables'
method_order = ['LCM', 'Linear PGM', 'Nonparametric PGM']
palette = {method_order[i]: sns.color_palette()[i] for i in range(len(method_order))}

rename_methods = {
    'LCM': 'LCM',
    'Linear\nPrognostic Score': 'Linear PGM',
    'Ensemble\nPrognostic Score': 'Nonparametric PGM'
}

cat_diff_df = pd.read_csv(f'Results2/categorical_diff.csv', index_col=[0])
cont_diff_df = pd.read_csv(f'Results2/continuous_diff.csv', index_col=[0])
cat_diff_df['Method'] = cat_diff_df['Method'].map(rename_methods)
cont_diff_df['Method'] = cont_diff_df['Method'].map(rename_methods)

categorical = ['XC_1', 'C1_4', 'C1_5', 'C2']

cat_diff_df['% Match'] *= 100

sns.set_context("paper")
sns.set_style("darkgrid")
sns.set(font_scale=3.3)
fig, axes = plt.subplots(1, 2, figsize=(18, 9))
sns.barplot(ax=axes[0], data=cat_diff_df, x='Covariate',
            y='% Match', hue='Method', errorbar=None,
            hue_order=[c.replace('_', '=') for c in categorical].sort())
sns.boxplot(ax=axes[1], data=cont_diff_df, x='Covariate',
            y='Mean Absolute Difference', hue='Method', showfliers=False)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.55, 0.93),
           ncol=3, fontsize=40, handletextpad=0.4,
           columnspacing=0.5)
for ax in axes:
    ax.set(xlabel=None)
    ax.get_legend().remove()
axes[0].yaxis.set_major_formatter(ticker.PercentFormatter())
fig.tight_layout()
fig.savefig(f'{save_folder}/all_mg.png', bbox_inches='tight')


# df_orig = pd.read_csv(f'{os.getenv("SCHOOLS_FOLDER")}/df.csv')
# lcm_cates_full = pd.read_csv('Results/lcm_cates.csv', index_col=[0])
# linear_prog_cates_full = pd.read_csv('Results/linear_prog_cates.csv',
#                                      index_col=[0])
# ensemble_prog_cates_full = pd.read_csv('Results/ensemble_prog_cates.csv',
#                                        index_col=[0])

# lcm_cates_full['avg.CATE_mean'] = lcm_cates_full.mean(axis=1)
# linear_prog_cates_full['avg.CATE'] = linear_prog_cates_full.mean(axis=1)
# ensemble_prog_cates_full['avg.CATE'] = ensemble_prog_cates_full.mean(axis=1)
#
# cate_df = lcm_cates_full.join(df_orig.drop(columns=['Z', 'Y']))
# cate_df = cate_df[['avg.CATE_mean', 'XC', 'S3']]
# cate_df = cate_df.rename(columns={'avg.CATE_mean': 'LCM',
#                                   'XC': 'Urbanicity (XC)',
#                                   'S3': 'Exp Success (S3)'})
# cate_df = cate_df.join(linear_prog_cates_full['avg.CATE'])
# cate_df = cate_df.rename(columns={'avg.CATE': 'Linear\nPGM'})
# cate_df = cate_df.join(ensemble_prog_cates_full['avg.CATE'])
# cate_df = cate_df.rename(columns={'avg.CATE': 'Nonparametric\nPGM'})
#
# cate_df = pd.melt(cate_df, id_vars=['Urbanicity (XC)', 'Exp Success (S3)'],
#                   var_name='Method', value_name='Estimated CATE')
#
# plt.figure(figsize=(8, 6))
# sns.set_context("paper")
# sns.set_style("darkgrid")
# sns.set(font_scale=2)
# ax = sns.boxplot(data=cate_df, x="Urbanicity (XC)", y="Estimated CATE", hue='Method', hue_order=method_order, palette=palette, showfliers=False)
# sns.move_legend(ax, "lower center", bbox_to_anchor=(.47, 1.02), ncol=3,
#                 title=None, handletextpad=0.4, columnspacing=0.5, fontsize=20)
# plt.tight_layout()
# ax.get_figure().savefig(f'{save_folder}/cate_by_xc.png')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from datagen.dgp_df import dgp_poly_basic_df, dgp_df
from src.variable_imp_matching import VIM
from scipy.spatial.distance import pdist
import warnings
from utils import save_df_to_csv
from collections import namedtuple

def read_df_w_error(file):
    try:
        return pd.read_csv(file)
    except FileNotFoundError:
        print(f'oops. file not found: {file} ')

args_df = pd.read_csv('./Experiments/variance/args.csv')
list_df = [read_df_w_error(f'./Experiments/variance/output_files/dgp_{args.dgp}/n_train_{args.n_train}/n_imp_{args.n_imp}/n_unimp_{args.n_unimp}/k_{args.k}/seed_{args.seed}/{args.fit}.csv') for _, args in args_df.iterrows()]

overall_df = pd.concat(list_df)
gb_df = overall_df.groupby(['dgp', 'seed', 'fit']).mean().reset_index()
# labels = gb_df['fit'].unique().tolist()

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure()
# Create the FacetGrid
g = sns.FacetGrid(gb_df, col="dgp", col_wrap=3, height=4, aspect=1.2)
# Map the scatter plot to the grid
g.map_dataframe(sns.scatterplot, x="se", y="contains_true_cate", hue="fit", alpha=0.7)
# scatter = sns.scatterplot(data = gb_df, x = 'se', y = 'contains_true_cate', hue = 'fit')
# scatter = plt.scatter(gb_df['se'], gb_df['contains_true_cate'], c = gb_df['fit'])
plt.axhline(y = 0.95, xmin = gb_df['se'].min(), xmax=gb_df['se'].max())
plt.xlabel('Squared Error')
plt.ylabel('Coverage')
# plt.legend(handles=scatter.legend_elements()[0], labels=labels)
# plt.showlegend()
plt.savefig('./trash.png', dpi = 150)

plt.figure()
# Create the FacetGrid
g = sns.FacetGrid(gb_df, col="dgp", col_wrap=3, height=4, aspect=1.2)
# Map the scatter plot to the grid
g.map_dataframe(sns.scatterplot, x="CATE_error_bound", y="contains_true_cate", hue="fit", alpha=0.7)
# scatter = sns.scatterplot(data = gb_df, x = 'CATE_error_bound', y = 'contains_true_cate', hue = 'fit', legend='brief')
plt.axhline(y = 0.95, xmin = gb_df['CATE_error_bound'].min(), xmax=gb_df['CATE_error_bound'].max())
plt.xlabel('1/2 Width of confidence interval')
plt.ylabel('Coverage')
# plt.legend(handles=scatter.legend_elements()[0], labels=labels)
# plt.showlegend()
plt.savefig('./trash2.png', dpi = 150)

plt.figure()
# Create the FacetGrid
g = sns.FacetGrid(gb_df, col="dgp", col_wrap=3, height=4, aspect=1.2)
# Map the scatter plot to the grid
g.map_dataframe(sns.scatterplot, x="se", y="CATE_error_bound", hue="fit", alpha=0.7)
# scatter = sns.scatterplot(data = gb_df, x = 'se', y = 'CATE_error_bound', hue = 'fit')
plt.axhline(y = 0.95, xmin = gb_df['se'].min(), xmax=gb_df['se'].max())
plt.xlabel('Squared error')
plt.ylabel('Radius of confidence interval')
# plt.legend(handles=scatter.legend_elements()[0], labels=labels)
# plt.showlegend()
plt.savefig('./trash3.png', dpi = 150)


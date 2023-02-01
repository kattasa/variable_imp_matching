from sklearn.linear_model import LogisticRegressionCV, RidgeCV, LassoCV
from Experiments.helpers import get_acic_data

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import NearestNeighbors

df_data, df_true, binary, categorical, dummy_cols, categorical_to_dummy = get_acic_data('acic_2019_low', 4, 0)
if len(categorical) > 0:
    df_data = df_data.drop(columns=categorical)
df_c = df_data[df_data['T'] == 0].reset_index(drop=True)
df_t = df_data[df_data['T'] == 1].reset_index(drop=True)

print('hi')

covs = [c for c in df_data.columns if c not in ['Y', 'T']]

pre_diff = (df_t.loc[:, covs].mean() - df_c.loc[:, covs].mean()) / df_t.loc[:, covs].std()

# m = LassoCV().fit(df_data.loc[df_data['T'] == 0, covs], df_data.loc[df_data['T'] == 0, 'Y'])
m = LogisticRegressionCV(penalty='l1', solver='saga').fit(df_data[covs], df_data['T'])
m2 = LogisticRegressionCV().fit(df_data[covs], df_data['T'])

full_mg = pd.DataFrame(columns=['C Match'])
weights = np.abs(m.coef_)
print(weights)

while (full_mg.shape[0] < df_t.shape[0]) and (full_mg.shape[0] < df_c.shape[0]):
    available_controls = [c for c in df_c.index if c not in full_mg['C Match']]
    # nn = NearestNeighbors(n_neighbors=1).fit(
    #     m.predict(df_c.loc[df_c.index.isin(available_controls), covs]).reshape(-1, 1))
    # mg = nn.kneighbors(m.predict(df_t.loc[~df_t.index.isin(full_mg.index), covs]).reshape(-1, 1),
    #                      return_distance=True)
    nn = NearestNeighbors(n_neighbors=1).fit(df_c.loc[df_c.index.isin(available_controls), covs].to_numpy() * weights)
    mg = nn.kneighbors(df_t.loc[~df_t.index.isin(full_mg.index), covs].to_numpy() * weights, return_distance=True)

    mg = pd.DataFrame(np.concatenate([mg[0], np.array(available_controls)[mg[1].reshape(-1)].reshape(-1, 1)], axis=1)).sort_values(0)
    mg.index = df_t.index[~df_t.index.isin(full_mg.index)]
    mg = mg.drop_duplicates(subset=[1]).drop(columns=[0]).rename(columns={1: 'C Match'})

    full_mg = pd.concat([full_mg, mg])

full_mg = full_mg.sort_index()
df_c1 = df_c.loc[full_mg['C Match']]

full_mg2 = pd.DataFrame(columns=['C Match'])

while (full_mg2.shape[0] < df_t.shape[0]) and (full_mg2.shape[0] < df_c.shape[0]):
    available_controls = [c for c in df_c.index if c not in full_mg2['C Match']]
    nn2 = NearestNeighbors(n_neighbors=1).fit(m2.predict_proba(df_c.loc[df_c.index.isin(available_controls), covs])[:, 1].reshape(-1, 1))
    mg2 = nn2.kneighbors(m2.predict_proba(df_t.loc[~df_t.index.isin(full_mg2.index), covs])[:, 1].reshape(-1, 1), return_distance=True)

    mg2 = pd.DataFrame(np.concatenate([mg2[0], np.array(available_controls)[mg2[1].reshape(-1)].reshape(-1, 1)], axis=1)).sort_values(0)
    mg2.index = df_t.index[~df_t.index.isin(full_mg2.index)]
    mg2 = mg2.drop_duplicates(subset=[1]).drop(columns=[0]).rename(columns={1: 'C Match'})

    full_mg2 = pd.concat([full_mg2, mg2])
full_mg2 = full_mg2.sort_index()
df_c2 = df_c.loc[full_mg2['C Match']]

post_diff1 = (df_t.loc[:, covs].mean() - df_c1.loc[:, covs].mean()) / df_t.loc[:, covs].std()
post_diff2 = (df_t.loc[:, covs].mean() - df_c2.loc[:, covs].mean()) / df_t.loc[:, covs].std()

print(f'Pre: {np.sum(np.abs(pre_diff))}')
print(f'Post1: {np.sum(np.abs(post_diff1))}')
print(f'Post2: {np.sum(np.abs(post_diff2))}')

plt.hist(pre_diff, label='Pre', alpha=1)
plt.hist(post_diff2, label='Post2', alpha=0.9)
plt.hist(post_diff1, label='Post1', alpha=0.8)
plt.legend()
plt.show()
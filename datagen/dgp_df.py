import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.linear_model import LassoCV

from datagen.dgp import dgp_poly_no_interaction, dgp_poly_interaction, dgp_exp_log_interaction,\
    dgp_friedman, data_generation_dense_mixed_endo

IHDP_FOLDER = '/Users/qlanners/projects/AME-for-Continuous-Exposure/datagen/ihdp'

ACIC_FOLDER = '/Users/qlanners/projects/AME-for-Continuous-Exposure/datagen/acic'
ACIC_FILE = 'highDim_testdataset[IDX].csv'


def dgp_df(dgp, n_samples, n_imp=None, n_unimp=None, perc_train=None, n_train=None):
    if dgp == 'poly_no_interaction':
        X, Y, T, Y0, Y1, TE, Y0_true, Y1_true = dgp_poly_no_interaction(n_samples, n_imp, n_unimp)
        discrete = []
    if dgp == 'poly_interaction':
        X, Y, T, Y0, Y1, TE, Y0_true, Y1_true = dgp_poly_interaction(n_samples, n_imp, n_unimp)
        discrete = []
    if dgp == 'exp_log_interaction':
        X, Y, T, Y0, Y1, TE, Y0_true, Y1_true = dgp_exp_log_interaction(n_samples, n_imp, n_unimp)
        discrete = []
    elif dgp == 'friedman':
        X, Y, T, Y0, Y1, TE, Y0_true, Y1_true = dgp_friedman(n_samples)
        discrete = []
    if perc_train:
        train_idx = int(n_samples*perc_train)
    else:
        train_idx = n_train
    df = pd.DataFrame(np.concatenate([X, Y, T, Y0, Y1, TE, Y0_true, Y1_true], axis=1))
    x_cols = [f'X{i}' for i in range(X.shape[1])]
    df.columns = [*x_cols, 'Y', 'T', 'Y0', 'Y1', 'TE', 'Y0_true', 'Y1_true']
    df[x_cols] = StandardScaler().fit_transform(df[x_cols])

    skf = KFold(n_splits=2).split(df, df['T'])
    new_y0s = []
    new_y1s = []
    for e_idx, t_idx in skf:
        m = LassoCV().fit(df.iloc[t_idx][df.iloc[t_idx]['T'] == 0][x_cols], df.iloc[t_idx][df.iloc[t_idx]['T'] == 0]['Y'])
        new_y0s.append(df.iloc[e_idx][df.iloc[e_idx]['T'] == 0]['Y'] - m.predict(df.iloc[e_idx][df.iloc[e_idx]['T'] == 0][x_cols]))
        new_y1s.append(df.iloc[e_idx][df.iloc[e_idx]['T'] == 1]['Y'] - m.predict(df.iloc[e_idx][df.iloc[e_idx]['T'] == 1][x_cols]))

    df['Y_new'] = pd.concat(new_y0s + new_y1s).sort_index()
    df_train = df.copy(deep=True)[:train_idx]
    df_train = df_train.drop(columns=['Y0', 'Y1', 'TE', 'Y0_true', 'Y1_true'])
    df_true = df.copy(deep=True)[train_idx:]
    df_assess = df_true.copy(deep=True).drop(columns=['Y0', 'Y1', 'TE', 'Y0_true', 'Y1_true'])
    return df_train.reset_index(drop=True), df_assess.reset_index(drop=True), df_true.reset_index(drop=True), x_cols, discrete


def dgp_dense_mixed_endo_df(n, nci, ndi, ncu, ndu, perc_train=None, n_train=None):
    df, df_true, discrete = data_generation_dense_mixed_endo(num_samples=n, num_cont_imp=nci, num_disc_imp=ndi,
                                                             num_cont_unimp=ncu, num_disc_unimp=ndu)
    x_cols = [c for c in df.columns if 'X' in c]
    df[x_cols] = StandardScaler().fit_transform(df[x_cols])
    if perc_train:
        train_idx = int(df.shape[0]*perc_train)
    else:
        train_idx = n_train

    df = df.join(df_true)

    skf = KFold(n_splits=2).split(df, df['T'])
    new_y0s = []
    new_y1s = []
    for e_idx, t_idx in skf:
        m = LassoCV().fit(df.iloc[t_idx][df.iloc[t_idx]['T'] == 0][x_cols], df.iloc[t_idx][df.iloc[t_idx]['T'] == 0]['Y'])
        new_y0s.append(df.iloc[e_idx][df.iloc[e_idx]['T'] == 0]['Y'] - m.predict(df.iloc[e_idx][df.iloc[e_idx]['T'] == 0][x_cols]))
        new_y1s.append(df.iloc[e_idx][df.iloc[e_idx]['T'] == 1]['Y'] - m.predict(df.iloc[e_idx][df.iloc[e_idx]['T'] == 1][x_cols]))

    df['Y_new'] = pd.concat(new_y0s + new_y1s).sort_index()
    df_train = df.copy(deep=True)[:train_idx]
    df_train = df_train.drop(columns=['Y0', 'Y1', 'TE', 'Y0_true', 'Y1_true'])
    df_true = df.copy(deep=True)[train_idx:]
    df_assess = df_true.copy(deep=True).drop(columns=['Y0', 'Y1', 'TE', 'Y0_true', 'Y1_true'])
    return df_train.reset_index(drop=True), df_assess.reset_index(drop=True), df_true.reset_index(drop=True), x_cols, discrete


def dgp_ihdp_df(dataset, k=None, perc_train=None, n_train=672):
    x_cols = [f'X{i}' for i in range(25)]
    if perc_train:
        train_idx = int(df.shape[0]*perc_train)
    else:
        train_idx = n_train
    train_file = f'{IHDP_FOLDER}/ihdp_npci_1-{dataset}.train.npz'
    test_file = f'{IHDP_FOLDER}/ihdp_npci_1-{dataset}.test.npz'
    train = np.load(train_file)
    test = np.load(test_file)
    if k is None:
        k = random.randint(0, int(dataset))
    discrete = [f'X{i}' for i in range(6, 25)]

    this_x = np.vstack([train['x'][:, :, k], test['x'][:, :, k]])
    this_y = np.hstack([train['yf'][:, k], test['yf'][:, k]])
    this_t = np.hstack([train['t'][:, k], test['t'][:, k]])
    this_ycf = np.hstack([train['ycf'][:, k], test['ycf'][:, k]])
    this_y0 = np.select([this_t == 0, this_t == 1], [this_y, this_ycf])
    this_y1 = np.select([this_t == 0, this_t == 1], [this_ycf, this_y])
    this_df = pd.DataFrame(this_x, columns=x_cols)
    this_df[x_cols] = StandardScaler().fit_transform(this_df[x_cols])
    this_df['Y'] = this_y
    this_df['T'] = this_t
    this_df['Y0'] = this_y0
    this_df['Y1'] = this_y1
    this_df['TE'] = this_y1 - this_y0
    df = this_df

    skf = KFold(n_splits=2).split(df, df['T'])
    new_y0s = []
    new_y1s = []
    for e_idx, t_idx in skf:
        m = LassoCV().fit(df.iloc[t_idx][df.iloc[t_idx]['T'] == 0][x_cols], df.iloc[t_idx][df.iloc[t_idx]['T'] == 0]['Y'])
        new_y0s.append(df.iloc[e_idx][df.iloc[e_idx]['T'] == 0]['Y'] - m.predict(df.iloc[e_idx][df.iloc[e_idx]['T'] == 0][x_cols]))
        new_y1s.append(df.iloc[e_idx][df.iloc[e_idx]['T'] == 1]['Y'] - m.predict(df.iloc[e_idx][df.iloc[e_idx]['T'] == 1][x_cols]))

    df['Y_new'] = pd.concat(new_y0s + new_y1s).sort_index()
    df_train = df.copy(deep=True)[:train_idx]
    df_train = df_train.drop(columns=['Y0', 'Y1', 'TE'])
    df_true = df.copy(deep=True)[train_idx:]
    df_assess = df_true.copy(deep=True).drop(columns=['Y0', 'Y1', 'TE'])
    return df_train.reset_index(drop=True), df_assess.reset_index(drop=True), df_true.reset_index(drop=True), x_cols, discrete


def dgp_acic_df(dataset_idx, perc_train=None, n_train=None):
    df = pd.read_csv(f'{ACIC_FOLDER}/{ACIC_FILE.replace("[IDX]", str(dataset_idx))}')
    df_cf = pd.read_csv(f'{ACIC_FOLDER}/{ACIC_FILE.replace("[IDX]", str(dataset_idx) + "_cf")}')
    x_cols = []
    rename_cols = {'A': 'T'}
    for i in range(df.shape[1]-2):
        rename_cols[f'V{i+1}'] = f'X{i}'
        x_cols.append(f'X{i}')
    df = df.rename(columns=rename_cols)
    discrete = []
    for i in range(len(x_cols)):
        if df.iloc[:, 2 + i].unique().shape[0] <= 2:
            discrete.append(f'X{i}')
    if perc_train:
        train_idx = int(df.shape[0]*perc_train)
    else:
        train_idx = n_train
    df_cf = df_cf.rename(columns={'ATE': 'TE', 'EY1': 'Y1_true', 'EY0': 'Y0_true'})
    df = pd.concat([df, df_cf], axis=1)
    df[x_cols] = StandardScaler().fit_transform(df[x_cols])

    skf = KFold(n_splits=2).split(df, df['T'])
    new_y0s = []
    new_y1s = []
    for e_idx, t_idx in skf:
        m = LassoCV().fit(df.iloc[t_idx][df.iloc[t_idx]['T'] == 0][x_cols], df.iloc[t_idx][df.iloc[t_idx]['T'] == 0]['Y'])
        new_y0s.append(df.iloc[e_idx][df.iloc[e_idx]['T'] == 0]['Y'] - m.predict(df.iloc[e_idx][df.iloc[e_idx]['T'] == 0][x_cols]))
        new_y1s.append(df.iloc[e_idx][df.iloc[e_idx]['T'] == 1]['Y'] - m.predict(df.iloc[e_idx][df.iloc[e_idx]['T'] == 1][x_cols]))

    df['Y_new'] = pd.concat(new_y0s + new_y1s).sort_index()
    df_train = df.copy(deep=True)[:train_idx]
    df_train = df_train.drop(columns=['TE', 'Y0_true', 'Y1_true'])
    df_true = df.copy(deep=True)[train_idx:]
    df_assess = df_true.copy(deep=True).drop(columns=['TE', 'Y0_true', 'Y1_true'])
    return df_train.reset_index(drop=True), df_assess.reset_index(drop=True), df_true.reset_index(drop=True), x_cols, discrete

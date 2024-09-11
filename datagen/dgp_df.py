"""
Functions to create pandas dataframes from data generation processes in
dgp.py.
"""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from datagen.dgp import dgp_poly_basic, dgp_sine, dgp_exp, \
    data_generation_dense_mixed_endo, dgp_nonlinear_mml, dgp_piecewise_mml

np.random.seed(0)


def dgp_poly_basic_df(n_samples, n_imp, n_unimp, powers=[2], perc_train=None,
                      n_train=None):
    """Create Basic Quadratic DGP dataframe."""
    if perc_train:
        train_idx = int(n_samples*perc_train)
    else:
        train_idx = n_train
    X, Y, T, Y0, Y1, TE, Y0_true, Y1_true = dgp_poly_basic(
        n_samples, n_imp, n_unimp, powers=powers)
    df = pd.DataFrame(np.concatenate([X, Y, T, Y0, Y1, TE, Y0_true, Y1_true],
                                     axis=1))
    x_cols = [f'X{i}' for i in range(X.shape[1])]
    df.columns = [*x_cols, 'Y', 'T', 'Y0', 'Y1', 'TE', 'Y0_true', 'Y1_true']
    df[x_cols] = StandardScaler().fit_transform(df[x_cols])
    df['T'] = df['T'].astype(int)
    df_train = df.copy(deep=True)[:train_idx]
    df_train = df_train.drop(columns=['Y0', 'Y1', 'TE', 'Y0_true', 'Y1_true'])
    df_true = df.copy(deep=True)[train_idx:]
    df_assess = df_true.copy(deep=True).drop(
        columns=['Y0', 'Y1', 'TE', 'Y0_true', 'Y1_true'])
    return df_train.reset_index(drop=True), df_assess.reset_index(drop=True), \
           df_true.reset_index(drop=True), x_cols


def dgp_df(dgp, n_samples, n_unimp=None, perc_train=None, n_train=None):
    """Create sine or exponential dataframe."""
    if dgp == 'sine':
        X, Y, T, Y0, Y1, TE, Y0_true, Y1_true = dgp_sine(n_samples, n_unimp)
        discrete = []
    if dgp == 'exp':
        X, Y, T, Y0, Y1, TE, Y0_true, Y1_true = dgp_exp(n_samples, n_unimp)
        discrete = []
    if dgp == 'nonlinear_mml':
        X, Y, T, Y0, Y1, TE, Y0_true, Y1_true = dgp_nonlinear_mml(n_samples, n_unimp)
        discrete = []
    if dgp == 'piecewise_mml':
        X, Y, T, Y0, Y1, TE, Y0_true, Y1_true = dgp_piecewise_mml(n_samples, n_unimp)
        discrete = []
    if perc_train:
        train_idx = int(n_samples*perc_train)
    else:
        train_idx = n_train
    df = pd.DataFrame(np.concatenate([X, Y, T, Y0, Y1, TE, Y0_true, Y1_true],
                                     axis=1))
    x_cols = [f'X{i}' for i in range(X.shape[1])]
    df.columns = [*x_cols, 'Y', 'T', 'Y0', 'Y1', 'TE', 'Y0_true', 'Y1_true']

    df[x_cols] = StandardScaler().fit_transform(df[x_cols])
    df['T'] = df['T'].astype(int)

    df_train = df.copy(deep=True)[:train_idx]
    df_train = df_train.drop(columns=['Y0', 'Y1', 'TE', 'Y0_true', 'Y1_true'])
    df_true = df.copy(deep=True)[train_idx:]
    df_assess = df_true.copy(deep=True).drop(
        columns=['Y0', 'Y1', 'TE', 'Y0_true', 'Y1_true'])
    return df_train.reset_index(drop=True), df_assess.reset_index(drop=True), \
           df_true.reset_index(drop=True), x_cols, discrete


def dgp_dense_mixed_endo_df(n, nci, ndi, ncu, ndu, std=1.5, t_imp=2, overlap=1,
                            perc_train=None, n_train=None, weights=None):
    """Create Quadratic DGP dataframe."""
    df, df_true, binary = \
        data_generation_dense_mixed_endo(num_samples=n, num_cont_imp=nci,
                                         num_disc_imp=ndi, num_cont_unimp=ncu,
                                         num_disc_unimp=ndu, std=std,
                                         t_imp=t_imp, overlap=overlap,
                                         weights=weights)
    x_cols = [c for c in df.columns if 'X' in c]
    df[x_cols] = StandardScaler().fit_transform(df[x_cols])
    if perc_train:
        train_idx = int(df.shape[0]*perc_train)
    else:
        train_idx = n_train
    df = df.join(df_true)

    df_train = df.copy(deep=True)[:train_idx]
    df_train = df_train.drop(columns=['Y0', 'Y1', 'TE', 'Y0_true', 'Y1_true'])
    df_true = df.copy(deep=True)[train_idx:]
    df_assess = df_true.copy(deep=True).drop(
        columns=['Y0', 'Y1', 'TE', 'Y0_true', 'Y1_true'])
    df_train = df_train[['Y', 'T'] + x_cols]
    df_assess = df_assess[['Y', 'T'] + x_cols]
    return df_train.reset_index(drop=True), df_assess.reset_index(drop=True), \
           df_true.reset_index(drop=True), x_cols, binary


def dgp_schools_df():
    """Create schools dataframe."""
    df = pd.read_csv(f'{os.getenv("SCHOOLS_FOLDER")}/df.csv')
    categorical = ['schoolid', 'C1', 'C2', 'C3', 'XC']
    df = df.rename(columns={'Z': 'T'})
    continuous = [c for c in df.columns if c not in categorical + ['T', 'Y']]
    df[continuous] = StandardScaler().fit_transform(df[continuous])
    categorical.remove('C2')
    categorical.remove('C3')
    df['C2'] = df['C2'].map({1: 0, 2: 1})
    return pd.get_dummies(df, columns=categorical)

"""
Functions to create pandas dataframes from data generation processes in
dgp.py.
"""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, minmax_scale

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

def linear_outcome(X, T):
    n_samples = X.shape[0]
    coef = np.array(list(range(2)) + [0] * (X.shape[1] - 2))
    Y0_true = X.dot(coef).reshape(-1,1)
    Y1_true = Y0_true + 1 + X.dot(coef).reshape(-1,1)
    error = np.random.uniform(-1, 1, size = (n_samples, 1))
    Y1 = Y1_true + error
    Y0 = Y0_true + error
    Y = (Y1 * T) + Y0 * (1 - T)
    TE = Y1_true - Y0_true
    return X, Y, T, Y0, Y1, TE, Y0_true, Y1_true

def dgp_linear(n_unimp, n_samples):
    n_imp = 2
    xmin = 0
    xmax = 5
    X = np.random.uniform(xmin, xmax, size = [n_samples, n_imp + n_unimp])
    T = np.random.binomial(n = 1, p = minmax_scale(X[:,0] + 3 * X[:, 1], feature_range=(0.25, 0.75))).reshape(-1,1)
    coef = np.array(list(range(n_imp)) + [0] * n_unimp)
    Y0_true = X.dot(coef).reshape(-1,1)
    Y1_true = Y0_true + 1 + X.dot(coef).reshape(-1,1)
    error = np.random.uniform(-1, 1, size = (n_samples, 1))
    Y1 = Y1_true + error
    Y0 = Y0_true + error
    Y = (Y1 * T) + Y0 * (1 - T)
    TE = Y1_true - Y0_true
    ymin = np.array([xmin] * (n_imp + n_unimp)).dot(coef)
    ymax = np.array([xmax] * (n_imp + n_unimp)).dot(coef)
    return X, Y, T, Y0, Y1, TE, Y0_true, Y1_true, ymin, ymax


def dgp_lihua(n_unimp, n_samples, corr, heteroskedasticity):
    n_imp = 2
    mean = np.array([0] * (n_imp + n_unimp))
    cov = np.ones(shape = [n_imp + n_unimp, n_imp + n_unimp]) * corr
    np.fill_diagonal(cov, np.repeat(a = 1, repeats=n_imp + n_unimp))
    X = np.random.multivariate_normal(mean, cov, shape = [n_samples, n_imp + n_unimp])
    xmax = 5
    xmin = -5
    X[X > xmax] = xmax
    X[X < -xmin] = -xmin
    import scipy.stats as stats
    T = np.random.binomial(n = 1, p = stats.beta.cdf(x = X[:, 0], a = 2, b = 4))
    def f(X): # bounded by [0,2]
        return 2/(1 + np.exp(-12 * (X - 0.5)))
    Y0_true = np.array([0] * n_samples)
    Y1_true = f(X[:, 0]) * f(X[:, 1])
    error = np.random.uniform(-1, 1, size = n_samples) # bounded by [-1,1]
    if heteroskedasticity:
        error = error * np.log(-X[:, 0])
    Y1 = Y1_true + error
    Y0 = Y0_true + error
    Y = (Y1 * T) + Y0 * (1 - T)
    TE = Y1_true - Y0_true
    return X, Y, T, Y0, Y1, TE, Y0_true, Y1_true


def dgp_df(dgp, n_samples, n_imp = None, n_unimp=None, perc_train=None, n_train=None, X = None, data_seed = 42, coef_seed = 42):
    """Create sine or exponential dataframe."""
    if dgp == 'sine':
        X, Y, T, Y0, Y1, TE, Y0_true, Y1_true = dgp_sine(n_samples, n_unimp)
        discrete = []
    if dgp == 'exp':
        X, Y, T, Y0, Y1, TE, Y0_true, Y1_true = dgp_exp(n_samples, n_unimp)
        discrete = []
    if dgp == 'nonlinear_mml':
        X, Y, T, Y0, Y1, TE, Y0_true, Y1_true = dgp_nonlinear_mml(n_samples, n_unimp, data_seed = data_seed, coef_seed = coef_seed)
        discrete = []
    if dgp == 'piecewise_mml':
        X, Y, T, Y0, Y1, TE, Y0_true, Y1_true = dgp_piecewise_mml(n_samples, n_unimp, data_seed = data_seed, coef_seed = coef_seed)
        discrete = []
    if dgp == 'poly':
        if n_imp is None:
            n_imp = 20
        return *dgp_poly_basic_df(n_samples, n_imp = n_imp, n_unimp = n_unimp, powers=[2], perc_train=perc_train,
                      n_train=n_train), []
    if dgp == 'linear':
        if n_imp is None:
            n_imp = 20
        X, Y, T, Y0, Y1, TE, Y0_true, Y1_true, ymin, ymax = dgp_linear(n_samples = n_samples, n_unimp = n_unimp)
        discrete = []
    if dgp == 'lihua_uncorr_homo':
        X, Y, T, Y0, Y1, TE, Y0_true, Y1_true = dgp_lihua(n_unimp = n_unimp, n_samples = n_samples, corr = 0, heteroskedasticity = False)
        discrete = []
    if dgp == 'lihua_corr_homo':
        X, Y, T, Y0, Y1, TE, Y0_true, Y1_true = dgp_lihua(n_unimp = n_unimp, n_samples = n_samples, corr = 0.9, heteroskedasticity = False)
        discrete = []
    if dgp == 'lihua_uncorr_hetero':
        X, Y, T, Y0, Y1, TE, Y0_true, Y1_true = dgp_lihua(n_unimp = n_unimp, n_samples = n_samples, corr = 0, heteroskedasticity = True)
        discrete = []
    if dgp == 'lihua_corr_hetero':
        X, Y, T, Y0, Y1, TE, Y0_true, Y1_true = dgp_lihua(n_unimp = n_unimp, n_samples = n_samples, corr = 0.9, heteroskedasticity = True)
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

    df_train = df.copy(deep=True).loc[:train_idx, ]
    df_train = df_train.drop(columns=['Y0', 'Y1', 'TE', 'Y0_true', 'Y1_true'])
    df_true = df.copy(deep=True).loc[train_idx:, ]
    df_assess = df_true.copy(deep=True).drop(
        columns=['Y0', 'Y1', 'TE', 'Y0_true', 'Y1_true'])
    return df_train.reset_index(drop=True), df_assess.reset_index(drop=True), \
           df_true.reset_index(drop=True), x_cols, discrete, ymin, ymax


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

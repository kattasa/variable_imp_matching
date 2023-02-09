import os
import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from datagen.dgp import dgp_poly_basic, dgp_friedman, \
    data_generation_dense_mixed_endo, dgp_sine, dgp_non_linear_mixed, \
    dgp_polynomials, dgp_test, dgp_combo, dgp_exp

np.random.seed(0)

IHDP_FOLDER = os.getenv('IHDP_FOLDER')

ACIC_2018_FOLDER = os.getenv('ACIC_2018_FOLDER')
ACIC_2019_FOLDER = os.getenv('ACIC_2019_FOLDER')
ACIC_2022_FOLDER = os.getenv('ACIC_2022_FOLDER')
NEWS_FOLDER = os.getenv('NEWS_FOLDER')


def dgp_poly_basic_df(n_samples, n_imp, n_unimp, powers=[2], perc_train=None, n_train=None):
    if perc_train:
        train_idx = int(n_samples*perc_train)
    else:
        train_idx = n_train
    X, Y, T, Y0, Y1, TE, Y0_true, Y1_true = dgp_poly_basic(n_samples, n_imp, n_unimp, powers=powers)
    df = pd.DataFrame(np.concatenate([X, Y, T, Y0, Y1, TE, Y0_true, Y1_true], axis=1))
    x_cols = [f'X{i}' for i in range(X.shape[1])]
    df.columns = [*x_cols, 'Y', 'T', 'Y0', 'Y1', 'TE', 'Y0_true', 'Y1_true']
    df[x_cols] = StandardScaler().fit_transform(df[x_cols])
    df['T'] = df['T'].astype(int)
    df_train = df.copy(deep=True)[:train_idx]
    df_train = df_train.drop(columns=['Y0', 'Y1', 'TE', 'Y0_true', 'Y1_true'])
    df_true = df.copy(deep=True)[train_idx:]
    df_assess = df_true.copy(deep=True).drop(columns=['Y0', 'Y1', 'TE', 'Y0_true', 'Y1_true'])
    return df_train.reset_index(drop=True), df_assess.reset_index(drop=True), df_true.reset_index(drop=True), x_cols


def dgp_df(dgp, n_samples, n_imp=None, n_unimp=None, perc_train=None, n_train=None):
    if dgp == 'polynomials':
        X, Y, T, Y0, Y1, TE, Y0_true, Y1_true = dgp_polynomials(n_samples, n_imp, n_unimp)
        discrete = []
    if dgp == 'sine':
        X, Y, T, Y0, Y1, TE, Y0_true, Y1_true = dgp_sine(n_samples, n_unimp)
        discrete = []
    if dgp == 'exp':
        X, Y, T, Y0, Y1, TE, Y0_true, Y1_true = dgp_exp(n_samples, n_unimp)
        discrete = []
    if dgp == 'non_linear_mixed':
        X, Y, T, Y0, Y1, TE, Y0_true, Y1_true = dgp_non_linear_mixed(n_samples, n_imp, n_unimp)
        discrete = []
    if dgp == 'test':
        X, Y, T, Y0, Y1, TE, Y0_true, Y1_true = dgp_test(n_samples, n_imp, n_unimp)
        discrete = []
    if dgp == 'combo':
        X, Y, T, Y0, Y1, TE, Y0_true, Y1_true = dgp_combo(n_samples, n_unimp)
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
    df['T'] = df['T'].astype(int)

    df_train = df.copy(deep=True)[:train_idx]
    df_train = df_train.drop(columns=['Y0', 'Y1', 'TE', 'Y0_true', 'Y1_true'])
    df_true = df.copy(deep=True)[train_idx:]
    df_assess = df_true.copy(deep=True).drop(columns=['Y0', 'Y1', 'TE', 'Y0_true', 'Y1_true'])
    return df_train.reset_index(drop=True), df_assess.reset_index(drop=True), df_true.reset_index(drop=True), x_cols, discrete


def dgp_dense_mixed_endo_df(n, nci, ndi, ncu, ndu, std=1.5, t_imp=2, overlap=1, perc_train=None, n_train=None,
                            weights=None):
    df, df_true, binary = data_generation_dense_mixed_endo(num_samples=n, num_cont_imp=nci, num_disc_imp=ndi,
                                                           num_cont_unimp=ncu, num_disc_unimp=ndu, std=std,
                                                           t_imp=t_imp, overlap=overlap, weights=weights)
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
    df_assess = df_true.copy(deep=True).drop(columns=['Y0', 'Y1', 'TE', 'Y0_true', 'Y1_true'])
    df_train = df_train[['Y', 'T'] + x_cols]
    df_assess = df_assess[['Y', 'T'] + x_cols]
    return df_train.reset_index(drop=True), df_assess.reset_index(drop=True), df_true.reset_index(drop=True), x_cols, binary


def dgp_schools_df():
    df = pd.read_csv(f'{os.getenv("SCHOOLS_FOLDER")}/df.csv')
    categorical = ['schoolid', 'C1', 'C2', 'C3', 'XC']
    df = df.rename(columns={'Z': 'T'})
    continuous = [c for c in df.columns if c not in categorical + ['T', 'Y']]
    df[continuous] = StandardScaler().fit_transform(df[continuous])
    categorical.remove('C2')
    df['C2'] = df['C2'].map({1: 0, 2: 1})
    return pd.get_dummies(df, columns=categorical)


def dgp_acic_2019_df(dataset_idx, perc_train=None, n_train=None, dummy_cutoff=10):
    df = pd.read_csv(f'{ACIC_2019_FOLDER}/{"highDim_testdataset[IDX].csv".replace("[IDX]", str(dataset_idx))}')
    df_cf = pd.read_csv(f'{ACIC_2019_FOLDER}/{"highDim_testdataset[IDX].csv".replace("[IDX]", str(dataset_idx) + "_cf")}')
    x_cols = []
    rename_cols = {'A': 'T'}
    for i in range(df.shape[1]-2):
        rename_cols[f'V{i+1}'] = f'X{i}'
        x_cols.append(f'X{i}')
    df = df.rename(columns=rename_cols)
    binary = []
    categorical = []
    dummy_cols = []
    categorical_to_dummy = {}
    n_x_cols = len(x_cols)
    for i in range(len(x_cols)):
        if df.iloc[:, 2 + i].unique().shape[0] <= 2:
            binary.append(f'X{i}')
        elif df.iloc[:, 2 + i].unique().shape[0] <= dummy_cutoff and df.iloc[:, 2 + i].dtype == int:
            categorical.append(f'X{i}')
            these_dummies = pd.get_dummies(df.iloc[:, 2+i])
            these_dummies.columns = [f'X{c}' for c in list(range(n_x_cols, n_x_cols + these_dummies.shape[1]))]
            n_x_cols += these_dummies.shape[1]
            categorical_to_dummy[f'X{i}'] = list(these_dummies.columns)
            dummy_cols.append(these_dummies)
    dummy_cols = pd.concat(dummy_cols, axis=1)
    df = pd.concat([df, dummy_cols], axis=1)
    # drop correlated columns
    corr = df.corr()
    cols = corr.columns
    corr_drop_cols = []
    n = corr.shape[0]
    for i in range(n):
        if cols[i] in corr_drop_cols:
            continue
        corr_drop_cols += list(corr.iloc[list(range(0, i)) +
                                         list(range(i + 1, n))].loc[
                                   corr.iloc[list(range(0, i)) +
                                             list(range(i + 1, n)), i] == 1]
                               .index)
    df = df.drop(columns=corr_drop_cols)
    x_cols = [c for c in x_cols if c not in corr_drop_cols]
    binary = [c for c in binary if c not in corr_drop_cols]
    categorical = [c for c in categorical if c not in corr_drop_cols]
    dummy_cols = [c for c in dummy_cols.columns if c not in corr_drop_cols]
    if perc_train:
        train_idx = int(df.shape[0]*perc_train)
    else:
        train_idx = n_train
    df_cf = df_cf.drop(columns=['ATE'])
    df_cf = df_cf.rename(columns={'EY1': 'Y1_true', 'EY0': 'Y0_true'})
    df_cf['TE'] = df_cf['Y1_true'] - df_cf['Y0_true']
    df = pd.concat([df, df_cf], axis=1)
    continuous = [x for x in x_cols if x not in binary + categorical + dummy_cols]
    df[continuous] = StandardScaler().fit_transform(df[continuous])

    df_train = df.copy(deep=True)[:train_idx]
    df_train = df_train.drop(columns=['TE', 'Y0_true', 'Y1_true'])
    df_true = df.copy(deep=True)[train_idx:]
    df_assess = df_true.copy(deep=True).drop(columns=['TE', 'Y0_true', 'Y1_true'])
    return df_train.reset_index(drop=True), df_assess.reset_index(drop=True), df_true.reset_index(drop=True), x_cols,\
           binary, categorical, dummy_cols, categorical_to_dummy


def dgp_acic_2018_df(acic_file, perc_train=None, n_train=None):
    if os.path.isfile(f'{ACIC_2018_FOLDER}/covariates/x_preprocessed.csv'):
        df = pd.read_csv(f'{ACIC_2018_FOLDER}/covariates/x_preprocessed.csv').set_index('sample_id')
        with open(f'{ACIC_2018_FOLDER}/covariates/x_binary.csv') as d:
            binary = d.read().replace('\n', '').split(',')
        with open(f'{ACIC_2018_FOLDER}/covariates/x_categorical.csv') as d:
            categorical = d.read().replace('\n', '').split(',')
        with open(f'{ACIC_2018_FOLDER}/covariates/x_dummy.csv') as d:
            dummy_cols = d.read().replace('\n', '').split(',')
    else:
        df = pd.read_csv(f'{ACIC_2018_FOLDER}/covariates/x.csv').set_index('sample_id')
        df, binary, categorical, dummy_cols = clean_2018_covariates(df)
    categorical_to_dummy = {}
    for c in categorical:
        categorical_to_dummy[c] = [k for k in dummy_cols if '_'.join(k.split('_')[:-1]) == c]
    df_results = pd.read_csv(f'{ACIC_2018_FOLDER}/{acic_file}.csv')
    df_cf = pd.read_csv(f'{ACIC_2018_FOLDER}/{acic_file}_cf.csv')
    df_cf = df_cf[['sample_id', 'y0', 'y1']]
    x_cols = [c for c in df.columns if c != 'sample_id']
    continuous = [x for x in x_cols if x not in binary + categorical + dummy_cols]
    df = df.join(df_results.set_index('sample_id'), how='inner')
    df = df.join(df_cf.set_index('sample_id'), how='inner')
    df = df.rename(columns={'z': 'T', 'y': 'Y', 'y0': 'Y0_true', 'y1': 'Y1_true'})
    df['TE'] = df['Y1_true'] - df['Y0_true']
    # drop correlated columns
    corr = df[x_cols].corr()
    cols = corr.columns
    corr_drop_cols = []
    n = corr.shape[0]
    for i in range(n):
        if cols[i] in corr_drop_cols:
            continue
        corr_drop_cols += list(corr.iloc[list(range(0, i)) +
                                         list(range(i + 1, n))].loc[
                                   corr.iloc[list(range(0, i)) +
                                             list(range(i + 1, n)), i] == 1]
                               .index)
    df = df.drop(columns=corr_drop_cols)
    x_cols = [c for c in x_cols if c not in corr_drop_cols]
    continuous = [c for c in continuous if c not in corr_drop_cols]
    binary = [c for c in binary if c not in corr_drop_cols]
    categorical = [c for c in categorical if c not in corr_drop_cols]
    dummy_cols = [c for c in dummy_cols if c not in corr_drop_cols]
    df[continuous] = StandardScaler().fit_transform(df[continuous])
    if perc_train:
        train_idx = int(df.shape[0]*perc_train)
    else:
        train_idx = n_train
    df_train = df.copy(deep=True)[:train_idx]
    df_train = df_train.drop(columns=['TE', 'Y0_true', 'Y1_true'])
    df_true = df.copy(deep=True)[train_idx:]
    df_assess = df_true.copy(deep=True).drop(columns=['TE', 'Y0_true', 'Y1_true'])
    return df_train.reset_index(drop=True), df_assess.reset_index(drop=True), df_true.reset_index(drop=True), x_cols, \
           binary, categorical, dummy_cols, categorical_to_dummy


def clean_2018_covariates(df):
    ordinal_unknowns = {'meduc': 9, 'fagecomb': 99, 'ufagecomb': 99, 'precare': 99, 'precare_rec': 5, 'uprevis': 99, 'previs_rec': 12,
                        'wtgain': 99, 'wtgain_rec': 9, 'cig_1': 99, 'cig_2': 99, 'cig_3': 99, 'rf_ncesar': 99, 'apgar5': 99, 'apgar5r': 5,
                        'estgest': 99, 'combgest': 99, 'gestrec10': 99, 'bwtr14': 14}
    int_type = ['mager41', 'mager14', 'mager9', 'meduc', 'fagecomb', 'ufagecomb', 'precare', 'precare_rec', 'uprevis', 'previs_rec',
               'wtgain', 'wtgain_rec', 'cig_1', 'cig_2', 'cig_3', 'rf_ncesar', 'apgar5', 'apgar5r', 'dplural', 'estgest', 'combgest',
               'gestrec10', 'dbwt', 'bwtr14']
    float_type = ['recwt']

    edited_cols = ['sample_id']

    ordinal_unknowns_cols = [list(df.columns).index(i) for i in ordinal_unknowns.keys()]
    binary = []
    for k, v in ordinal_unknowns.items():
        new_col = f'{k}_missing'
        binary.append(new_col)
        df.loc[:, new_col] = 0
        df.loc[df[k] == v, new_col] = 1
        df.loc[df[k] == v, k] = np.nan
    df.iloc[:, ordinal_unknowns_cols] = IterativeImputer().fit_transform(df[df.columns.difference(binary)])[:, ordinal_unknowns_cols]
    edited_cols += list(ordinal_unknowns.keys())

    for c in df.columns:
        if len(df[c].unique()) == 2:
            df[c] = df[c].map(dict(zip(set(df[c].unique()), [0, 1])))
            binary.append(c)
    df[int_type] = df[int_type].astype('int64')
    df[float_type] = df[float_type].astype('float32')
    edited_cols += int_type + float_type + binary

    categorical = [c for c in df.columns if c not in edited_cols]
    new_df = pd.concat([df, pd.get_dummies(df[categorical], columns=categorical)], axis=1)
    dummy_cols = [c for c in new_df.columns if c not in edited_cols+categorical]

    return new_df, binary, categorical, dummy_cols



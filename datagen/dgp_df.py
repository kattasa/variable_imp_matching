import glob
import os
import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.linear_model import LassoCV

from datagen.dgp import dgp_poly_no_interaction, dgp_poly_interaction,\
    dgp_friedman, data_generation_dense_mixed_endo, dgp_sine, dgp_non_linear_mixed, dgp_polynomials, dgp_test

IHDP_FOLDER = '/Users/qlanners/projects/linear_coef_matching/datagen/ihdp'

ACIC_FOLDER = '/Users/qlanners/projects/linear_coef_matching/datagen/acic'
ACIC_FILE = 'highDim_testdataset[IDX].csv'

ACIC_2018_FOLDER = '/Users/qlanners/projects/linear_coef_matching/datagen/acic_2018'
ACIC_2022_FOLDER = '/Users/qlanners/projects/linear_coef_matching/datagen/acic_2022'


def dgp_df(dgp, n_samples, n_imp=None, n_unimp=None, perc_train=None, n_train=None):
    if dgp == 'polynomials':
        X, Y, T, Y0, Y1, TE, Y0_true, Y1_true = dgp_polynomials(n_samples, n_imp, n_unimp)
        discrete = []
    if dgp == 'sine':
        X, Y, T, Y0, Y1, TE, Y0_true, Y1_true = dgp_sine(n_samples, n_imp, n_unimp)
        discrete = []
    if dgp == 'non_linear_mixed':
        X, Y, T, Y0, Y1, TE, Y0_true, Y1_true = dgp_non_linear_mixed(n_samples, n_imp, n_unimp)
        discrete = []
    if dgp == 'test':
        X, Y, T, Y0, Y1, TE, Y0_true, Y1_true = dgp_test(n_samples, n_imp, n_unimp)
        discrete = []
    if dgp == 'poly_no_interaction':
        X, Y, T, Y0, Y1, TE, Y0_true, Y1_true = dgp_poly_no_interaction(n_samples, n_imp, n_unimp)
        discrete = []
    if dgp == 'poly_interaction':
        X, Y, T, Y0, Y1, TE, Y0_true, Y1_true = dgp_poly_interaction(n_samples, n_imp, n_unimp)
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

    df_train = df.copy(deep=True)[:train_idx]
    df_train = df_train.drop(columns=['Y0', 'Y1', 'TE'])
    df_true = df.copy(deep=True)[train_idx:]
    df_assess = df_true.copy(deep=True).drop(columns=['Y0', 'Y1', 'TE'])
    return df_train.reset_index(drop=True), df_assess.reset_index(drop=True), df_true.reset_index(drop=True), x_cols, discrete


def dgp_acic_2019_df(dataset_idx, perc_train=None, n_train=None, dummy_cutoff=10):
    df = pd.read_csv(f'{ACIC_FOLDER}/{ACIC_FILE.replace("[IDX]", str(dataset_idx))}')
    df_cf = pd.read_csv(f'{ACIC_FOLDER}/{ACIC_FILE.replace("[IDX]", str(dataset_idx) + "_cf")}')
    x_cols = []
    rename_cols = {'A': 'T'}
    for i in range(df.shape[1]-2):
        rename_cols[f'V{i+1}'] = f'X{i}'
        x_cols.append(f'X{i}')
    df = df.rename(columns=rename_cols)
    discrete = []
    dummy_cols = []
    for i in range(len(x_cols)):
        if df.iloc[:, 2 + i].unique().shape[0] <= 2:
            discrete.append(f'X{i}')
        elif df.iloc[:, 2 + i].unique().shape[0] <= dummy_cutoff and df.iloc[:, 2 + i].dtype == int:
            discrete.append(f'X{i}')
            dummy_cols.append(pd.get_dummies(df.iloc[:, 2+i]))
    dummy_cols = pd.concat(dummy_cols, axis=1)
    dummy_cols.columns = [f'X{i}' for i in range(len(x_cols), len(x_cols)+dummy_cols.shape[1])]
    df = pd.concat([df, dummy_cols], axis=1)
    if perc_train:
        train_idx = int(df.shape[0]*perc_train)
    else:
        train_idx = n_train
    df_cf = df_cf.drop(columns=['ATE'])
    df_cf = df_cf.rename(columns={'EY1': 'Y1_true', 'EY0': 'Y0_true'})
    df_cf['TE'] = df_cf['Y1_true'] - df_cf['Y0_true']
    df = pd.concat([df, df_cf], axis=1)
    df[x_cols] = StandardScaler().fit_transform(df[x_cols])

    df_train = df.copy(deep=True)[:train_idx]
    df_train = df_train.drop(columns=['TE', 'Y0_true', 'Y1_true'])
    df_true = df.copy(deep=True)[train_idx:]
    df_assess = df_true.copy(deep=True).drop(columns=['TE', 'Y0_true', 'Y1_true'])
    return df_train.reset_index(drop=True), df_assess.reset_index(drop=True), df_true.reset_index(drop=True), x_cols,\
           discrete, list(dummy_cols.columns)


def dgp_acic_2018_df(perc_train=None, n_train=None):
    df = pd.read_csv(f'{ACIC_2018_FOLDER}/x.csv')
    df_results = pd.read_csv(f'{ACIC_2018_FOLDER}/f2e5cac9902246fba6e5a5c3b11d1605.csv')
    df_cf = pd.read_csv(f'{ACIC_2018_FOLDER}/f2e5cac9902246fba6e5a5c3b11d1605_cf.csv')
    df_cf = df_cf[['sample_id', 'y0', 'y1']]
    x_cols = [c for c in df.columns if c != 'sample_id']
    discrete = []  # will need to change df structure for MALTS
    df = df.join(df_results.set_index('sample_id'), on='sample_id', how='inner')
    df = df.join(df_cf.set_index('sample_id'), on='sample_id', how='inner')
    df = df.rename(columns={'z': 'T', 'y': 'Y', 'y0': 'Y0_true', 'y1': 'Y1_true'})
    df['TE'] = df['Y1_true'] - df['Y0_true']
    df[x_cols] = StandardScaler().fit_transform(df[x_cols])

    if perc_train:
        train_idx = int(df.shape[0]*perc_train)
    else:
        train_idx = n_train
    df_train = df.copy(deep=True)[:train_idx]
    df_train = df_train.drop(columns=['TE', 'Y0_true', 'Y1_true'])
    df_true = df.copy(deep=True)[train_idx:]
    df_assess = df_true.copy(deep=True).drop(columns=['TE', 'Y0_true', 'Y1_true'])
    return df_train.reset_index(drop=True), df_assess.reset_index(drop=True), df_true.reset_index(drop=True), x_cols, discrete


def dgp_acic_2022_df(track=2):
    practice_files = glob.glob(os.path.join(ACIC_2022_FOLDER, 'practice/*.csv'))
    practice_year_files = glob.glob(os.path.join(ACIC_2022_FOLDER, 'practice_year/*.csv'))

    practice_df = []
    practice_year_df = []

    for f in practice_files:
        practice_df.append(pd.read_csv(f))
    practice_df = pd.concat(practice_df, ignore_index=True)

    for f in practice_year_files:
        practice_year_df.append(pd.read_csv(f))
    practice_year_df = pd.concat(practice_year_df, ignore_index=True)

    return practice_df, practice_year_df


def dgp_lalonde():
    df_assess = pd.read_stata('http://www.nber.org/~rdehejia/data/nsw.dta')
    df_assess = df_assess.drop(columns=['data_id'])
    df_assess = df_assess.rename(columns={'treat': 'T', 're78': 'Y'})
    x_cols = [c for c in df_assess.columns if c not in ['T', 'Y']]
    discrete = ['black', 'hispanic', 'married', 'nodegree']
    ate = df_assess['Y'].mean()
    return None, df_assess.reset_index(drop=True), ate, x_cols, discrete

    # psid_control = pd.read_stata(‘http: // www.nber.org / ~rdehejia / data / psid_controls.dta’)
    # psid_control2 = pd.read_stata(‘http://www.nber.org/~rdehejia/data/psid_controls2.dta’)
    # psid_control3 = pd.read_stata(‘http://www.nber.org/~rdehejia/data/psid_controls3.dta’)


def clean_2018_covariates(df):
    float_type = ['recwt']
    int_type = ['mager41', 'mager14', 'mager9', 'meduc', 'fagecomb', 'ufagecomb', 'precare', 'precare_rec', 'uprevis', 'previs_rec',
               'wtgain', 'wtgain_rec', 'cig_1', 'cig_2', 'cig_3', 'rf_ncesar', 'apgar5', 'apgar5r', 'dplural', 'estgest', 'combgest',
               'gestrec10', 'dbwt', 'bwtr14']
    ordinal_unknowns = {'meduc': 9, 'fagecomb': 99, 'ufagecomb': 99, 'precare': 99, 'precare_rec': 5, 'uprevis': 99, 'previs_rec': 12,
                        'wtgain': 99, 'wtgain_rec': 9, 'cig_1': 99, 'cig_2': 99, 'cig_3': 99, 'rf_ncesar': 99, 'apgar5': 99, 'apgar5r': 5,
                        'estgest': 99, 'combgest': 99, 'gestrec10': 99, 'bwtr14': 14}
    edited_cols = ['sample_id']
    for c in df.columns:
        if len(df[c].unique()) == 2:
            df[c] = df[c].map(dict(zip(set(df[c].unique()), [0,1])))
            edited_cols.append(c)
    df[int_type] = df[int_type].astype('int64')
    df[float_type] = df[float_type].astype('float32')
    edited_cols += int_type + float_type
    for k, v in ordinal_unknowns.items():
        df.loc[df[k] == v, k] = df.loc[df[k] != v, k].mean()

    return pd.get_dummies(df, columns=[c for c in df.columns if c not in edited_cols])



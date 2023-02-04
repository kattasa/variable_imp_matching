import copy

import numpy as np
import os
import pandas as pd

from datagen.dgp_df import dgp_dense_mixed_endo_df, dgp_df, dgp_acic_2019_df, \
    dgp_acic_2018_df


def create_folder(data, print_progress=True):
    folders = [int(c.replace(f'{data}_', '')) for c in
               [d for d in os.listdir(os.getenv('RESULTS_FOLDER')) if data in d]]
    iter = max(folders) + 1 if len(folders) > 0 else 0
    save_folder = f'{os.getenv("RESULTS_FOLDER")}/{data}_{iter:03d}'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    if print_progress:
        print(f'Saving results to {save_folder}')
    return save_folder


def get_acic_data(year, file, n_train):
    if year == 'acic_2019':
        df_train, df_data, df_true, x_cols, binary, categorical, dummy_cols, categorical_to_dummy = dgp_acic_2019_df(
            file, n_train=n_train)
    elif year == 'acic_2018':
        df_train, df_data, df_true, x_cols, binary, categorical, dummy_cols, categorical_to_dummy = dgp_acic_2018_df(
            file, n_train=n_train)
    if n_train > 0:
        return df_train, df_data, df_true, binary, categorical, dummy_cols, categorical_to_dummy
    return df_data, df_true, binary, categorical, dummy_cols, categorical_to_dummy


def get_data(data, config):
    if 'dense' in data:
        df_train, df_data, df_true, x_cols, binary = dgp_dense_mixed_endo_df(config['num_samples'], config['imp_c'],
                                                                               config['imp_d'], config['unimp_c'],
                                                                               config['unimp_d'],
                                                                               n_train=config['n_train'])
    else:
        df_train, df_data, df_true, x_cols, binary = dgp_df(dgp=data, n_samples=config['num_samples'],
                                                              n_imp=config['imp_c'], n_unimp=config['unimp_c'],
                                                              n_train=config['n_train'])
    if config['n_train'] > 0:
        return df_train, df_data, df_true, binary
    return df_data, df_true, binary


def summarize_warnings(warning_list, method_name=None, print_warnings=True, return_warnings=False):
    method_warnings = np.array(np.unique([f'{w.filename}{w.lineno}' for w in warning_list], return_index=True,
                                         return_counts=True))[[1, 2]]
    method_warnings = {warning_list[int(method_warnings[0][i])].message: method_warnings[1][i] for i in
                       range(len(method_warnings[0]))}
    if print_warnings and len(method_warnings) > 0:
        print(f'{method_name} warnings:')
        for k, v in method_warnings.items():
            print(f'\t{v}: {k}')
    if return_warnings:
        return method_warnings


def lcm_to_malts_weights(lcm, malts_covs, categorical_to_dummy):
    matching_cols = [c for c in malts_covs if c in lcm.covariates]
    lcm_mismatches = [c for c in lcm.covariates if c not in matching_cols]
    malts_mismatches = [c for c in malts_covs if c not in matching_cols]
    for k, v in categorical_to_dummy.items():
        lcm_mismatches = [c for c in lcm_mismatches if c not in v]
        malts_mismatches = [c for c in malts_mismatches if c != k]
    if len(malts_mismatches) > 0:
        print(f'ERROR: {len(malts_mismatches)} malts covariate(s) not mapped to lcm covariate.')
    if len(lcm_mismatches) > 0:
        print(f'ERROR: {len(lcm_mismatches)} lcm covariate(s) not mapped to malts covariate.')

    malts_weights = []
    for i in range(len(lcm.gen_skf)):
        M = pd.DataFrame([lcm.M_list[i]], columns=lcm.covariates)
        these_malts_weights = pd.DataFrame([np.zeros(shape=len(malts_covs))], columns=malts_covs)
        these_malts_weights[matching_cols] = M[matching_cols]
        for k, v in categorical_to_dummy.items():
            these_malts_weights[k] = M[v].sum(axis=1).values[0]
        malts_weights.append(copy.deepcopy(these_malts_weights.to_numpy().reshape(-1,)))

    return np.array(malts_weights)


def weights_to_feature_selection(malts_weights, malts_covs):
    malts_features = []
    for w in malts_weights:
        malts_features.append(list(np.array(malts_covs)[w > 0]))
    return malts_features


def get_errors(est_cates, true_cates, method_name, scale=None):
    if scale is None:
        scale = np.abs(true_cates).mean()[0]
    cates = est_cates.join(true_cates, how='inner')
    cates.columns = ['Est_CATE', 'True_CATE']
    cates['Relative Error (%)'] = \
        np.abs(cates['Est_CATE'] - cates['True_CATE']) / scale
    cates['Method'] = method_name
    print(f'{method_name} ommitted {round(((true_cates.shape[0] - cates.shape[0]) / true_cates.shape[0])*100, 4)}% of samples.')
    return cates[['Method', 'True_CATE',
                  'Est_CATE', 'Relative Error (%)']].copy(deep=True)

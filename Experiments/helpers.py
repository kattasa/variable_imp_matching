"""Helper functions to run experiments."""
import numpy as np
import os
import pandas as pd

from datagen.dgp_df import dgp_dense_mixed_endo_df, dgp_df


def create_folder(data, print_progress=True):
    """Creates a folder to save results in."""
    folders = [int(c.replace(f'{data}_', '')) for c in
               [d for d in os.listdir(os.getenv('RESULTS_FOLDER')) if data in d]]
    iter = max(folders) + 1 if len(folders) > 0 else 0
    save_folder = f'{os.getenv("RESULTS_FOLDER")}/{data}_{iter:03d}'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    if print_progress:
        print(f'Saving results to {save_folder}')
    return save_folder


def get_data(data, config):
    """Get data to use for experiments from a config dictionary."""
    if 'dense' in data:
        std = config['std'] if 'std' in config else 1.5
        df_train, df_data, df_true, x_cols, binary = \
            dgp_dense_mixed_endo_df(config['num_samples'], config['imp_c'],
                                    config['imp_d'], config['unimp_c'],
                                    config['unimp_d'],
                                    n_train=config['n_train'], std=std)
    else:
        df_train, df_data, df_true, x_cols, binary = \
            dgp_df(dgp=data, n_samples=config['num_samples'],
                   n_unimp=config['unimp_c'],
                   n_train=config['n_train'])
    if config['n_train'] > 0:
        return df_train, df_data, df_true, binary
    return df_data, df_true, binary


def summarize_warnings(warning_list, method_name=None, print_warnings=True,
                       return_warnings=False):
    """Summarize warnings raised during script."""
    method_warnings = np.array(
        np.unique([f'{w.filename}{w.lineno}' for w in warning_list],
                  return_index=True, return_counts=True))[[1, 2]]
    method_warnings = {
        warning_list[int(method_warnings[0][i])].message: method_warnings[1][i]
        for i in range(len(method_warnings[0]))
    }
    if print_warnings and len(method_warnings) > 0:
        print(f'{method_name} warnings:')
        for k, v in method_warnings.items():
            print(f'\t{v}: {k}')
    if return_warnings:
        return method_warnings


def get_errors(est_cates, true_cates, method_name, scale=None, iter=None):
    """Calculate relative % error."""
    if scale is None:
        scale = np.abs(true_cates).mean()[0]
    cates = est_cates.join(true_cates, how='inner')
    cates.columns = ['Est_CATE', 'True_CATE']
    cates['Relative Error (%)'] = \
        np.abs(cates['Est_CATE'] - cates['True_CATE']) / scale
    cates['Method'] = method_name
    print(f'{method_name} ommitted {round(((true_cates.shape[0] - cates.shape[0]) / true_cates.shape[0])*100, 4)}% of samples.')
    cates = cates[['Method', 'True_CATE', 'Est_CATE', 'Relative Error (%)']]
    if iter is not None:
        cates['Iter'] = iter
    return cates


def get_mg_comp(df_orig, sample_num, sample, lcm_mgs, linear_prog_c_mg,
                linear_prog_t_mg, ensemble_prog_c_mg, ensemble_prog_t_mg,
                n_iters, treatment, ordinal, k_est, imp_covs):
    """Compare match groups between lcm, linear prog, and ensemble prog."""
    while True:
        iter_number = np.random.randint(0, n_iters)
        if sample_num in linear_prog_c_mg[iter_number].index:
            break
    print(f'Pulling example MG from iteration {iter_number}')

    lcm_c_mg = df_orig.loc[lcm_mgs[iter_number][0].loc[sample_num], imp_covs + [treatment]]
    lcm_t_mg = df_orig.loc[lcm_mgs[iter_number][1].loc[sample_num], imp_covs + [treatment]]
    linear_prog_c_mg = df_orig.loc[linear_prog_c_mg[iter_number].loc[sample_num], imp_covs + [treatment]]
    linear_prog_t_mg = df_orig.loc[linear_prog_t_mg[iter_number].loc[sample_num], imp_covs + [treatment]]
    ensemble_prog_c_mg = df_orig.loc[ensemble_prog_c_mg[iter_number].loc[sample_num], imp_covs + [treatment]]
    ensemble_prog_t_mg = df_orig.loc[ensemble_prog_t_mg[iter_number].loc[sample_num], imp_covs + [treatment]]

    lcm_mg = pd.concat([lcm_c_mg, lcm_t_mg])
    linear_prog_mg = pd.concat([linear_prog_c_mg, linear_prog_t_mg])
    ensemble_prog_mg = pd.concat([ensemble_prog_c_mg, ensemble_prog_t_mg])

    categorical = list(lcm_mg.dtypes[lcm_mg.dtypes == 'int'].index)
    categorical.remove(treatment)
    for o in ordinal:
        categorical.remove(o)
    continuous = list(lcm_mg.dtypes[lcm_mg.dtypes == 'float'].index)
    continuous = ordinal + continuous

    lcm_comps = {}
    linear_prog_comps = {}
    ensemble_prog_comps = {}

    for c in categorical:
        lcm_comps[c] = ((lcm_mg[c] == sample[c].values[0]).astype(int).sum() / (k_est*2))*100
        linear_prog_comps[c] = ((linear_prog_mg[c] == sample[c].values[0]).astype(int).sum() / (
                    k_est * 2)) * 100
        ensemble_prog_comps[c] = ((ensemble_prog_mg[c] == sample[c].values[0]).astype(int).sum() / (
                    k_est * 2)) * 100
    for c in continuous:
        lcm_comps[c] = np.abs(lcm_mg[c] - sample[c].values[0]).mean()
        linear_prog_comps[c] = np.abs(linear_prog_mg[c] - sample[c].values[0]).mean()
        ensemble_prog_comps[c] = np.abs(ensemble_prog_mg[c] - sample[c].values[0]).mean()

    lcm_comps[treatment] = np.nan
    linear_prog_comps[treatment] = np.nan
    ensemble_prog_comps[treatment] = np.nan

    lcm_mg = pd.concat([sample, lcm_mg, pd.DataFrame.from_dict([lcm_comps])])
    linear_prog_mg = pd.concat([sample, linear_prog_mg, pd.DataFrame.from_dict([linear_prog_comps])])
    ensemble_prog_mg = pd.concat([sample, ensemble_prog_mg, pd.DataFrame.from_dict([ensemble_prog_comps])])

    return lcm_mg, linear_prog_mg, ensemble_prog_mg
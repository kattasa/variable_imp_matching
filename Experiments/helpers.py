import os
import sys
sys.path.append("..")
from datagen.dgp_df import dgp_dense_mixed_endo_df, dgp_df, dgp_acic_df, dgp_ihdp_df


def create_folder(data, print_progress=True):
    folders = [int(c.replace(f'{data}_', '')) for c in
               [d for d in os.listdir('Results/') if data in d]]
    iter = max(folders) + 1 if len(folders) > 0 else 0
    save_folder = f'Results/{data}_{iter:03d}'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    if print_progress:
        print(f'Saving results to {save_folder}')
    return save_folder


def get_data(data, num_samples, config, imp_c=None, imp_d=None, unimp_c=None, unimp_d=None, n_train=0):
    if data == 'dense_continuous':
        if imp_c is None:
            imp_c = 5
            imp_d = 0
            unimp_c = 10
            unimp_d = 0
        df_train, df_data, df_true, x_cols, discrete = dgp_dense_mixed_endo_df(num_samples, imp_c, imp_d, unimp_c,
                                                                               unimp_d, n_train=n_train)
    elif data == 'dense_discrete':
        if imp_c is None:
            imp_c = 0
            imp_d = 15
            unimp_c = 0
            unimp_d = 10
        df_train, df_data, df_true, x_cols, discrete = dgp_dense_mixed_endo_df(num_samples, imp_c, imp_d, unimp_c,
                                                                               unimp_d, n_train=n_train)
    elif data == 'dense_mixed':
        if imp_c is None:
            imp_c = 5
            imp_d = 15
            unimp_c = 10
            unimp_d = 10
        df_train, df_data, df_true, x_cols, discrete = dgp_dense_mixed_endo_df(num_samples, imp_c, imp_d, unimp_c,
                                                                               unimp_d, n_train=n_train)

    elif data == 'poly_no_interaction':
        if imp_c is None:
            imp_c = 5
            unimp_c = 10
        df_train, df_data, df_true, x_cols, discrete = dgp_df(dgp='poly_no_interaction', n_samples=num_samples,
                                                              n_imp=imp_c, n_unimp=unimp_c, n_train=n_train)

    elif data == 'poly_interaction':
        if imp_c is None:
            imp_c = 5
            unimp_c = 10
        df_train, df_data, df_true, x_cols, discrete = dgp_df(dgp='poly_interaction', n_samples=num_samples,
                                                              n_imp=imp_c, n_unimp=unimp_c, n_train=n_train)

    elif data == 'exp_log_interaction':
        if imp_c is None:
            imp_c = 5
            unimp_c = 10
        df_train, df_data, df_true, x_cols, discrete = dgp_df(dgp='exp_log_interaction', n_samples=num_samples,
                                                              n_imp=imp_c, n_unimp=unimp_c, n_train=n_train)

    elif data == 'friedman':
        df_train, df_data, df_true, x_cols, discrete = dgp_df(dgp='friedman', n_samples=num_samples, n_train=n_train)

    elif data == 'acic':
        acic_file = 8
        df_train, df_data, df_true, x_cols, discrete = dgp_acic_df(acic_file, n_train=n_train)
        df_true = df_true.rename(columns={'ATE': 'TE'})
        config['acic_file'] = acic_file

    elif data == 'ihdp':
        ihdp_file = 100
        df_train, df_data, df_true, x_cols, discrete = dgp_ihdp_df(ihdp_file, n_train=n_train)
        config['ihdp_file'] = ihdp_file

    config['num_samples'] = df_train.shape[0] + df_data.shape[0]
    config['n_train'] = n_train
    config['imp_c'] = imp_c
    config['imp_d'] = imp_d
    config['unimp_c'] = unimp_c
    config['unimp_d'] = unimp_d

    if n_train > 0:
        return df_train, df_data, df_true, discrete, config

    return df_data, df_true, discrete, config
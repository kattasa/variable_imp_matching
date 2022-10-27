import os
import sys
sys.path.append("..")
from datagen.dgp_df import dgp_dense_mixed_endo_df, dgp_df, dgp_acic_2019_df, dgp_ihdp_df, dgp_lalonde, dgp_acic_2018_df


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


def get_data(data, num_samples, config, imp_c=None, imp_d=None, unimp_c=None, unimp_d=None, n_train=0, acic_file=8):
    dummy_cols = None  # col names for any cols that were transformed into several cols from a single categorical col
    if 'dense' in data:
        df_train, df_data, df_true, x_cols, discrete = dgp_dense_mixed_endo_df(num_samples, imp_c, imp_d, unimp_c,
                                                                               unimp_d, n_train=n_train)
    elif data == 'acic_2019':
        df_train, df_data, df_true, x_cols, discrete, dummy_cols = dgp_acic_2019_df(acic_file, n_train=n_train)
        config['acic_file'] = acic_file
    elif data == 'acic_2018':
        df_train, df_data, df_true, x_cols, discrete = dgp_acic_2018_df(n_train=n_train)
        config['acic_file'] = None

    elif data == 'ihdp':
        ihdp_file = 100
        df_train, df_data, df_true, x_cols, discrete = dgp_ihdp_df(ihdp_file, n_train=n_train)
        config['ihdp_file'] = ihdp_file
    elif data == 'lalonde':
        df_train, df_data, df_true, x_cols, discrete = dgp_lalonde()
    else:
        df_train, df_data, df_true, x_cols, discrete = dgp_df(dgp=data, n_samples=num_samples,
                                                              n_imp=imp_c, n_unimp=unimp_c, n_train=n_train)

    if n_train > 0:
        return df_train, df_data, df_true, discrete, config, dummy_cols

    return df_data, df_true, discrete, config, dummy_cols
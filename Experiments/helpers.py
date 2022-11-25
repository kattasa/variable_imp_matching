import os
from datagen.dgp_df import dgp_dense_mixed_endo_df, dgp_df, dgp_acic_2019_df, \
    dgp_ihdp_df, dgp_news, dgp_acic_2018_df


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
        df_train, df_data, df_true, x_cols, binary, categorical, dummy_cols = dgp_acic_2019_df(file, n_train=n_train)
    elif year == 'acic_2018':
        df_train, df_data, df_true, x_cols, binary, categorical, dummy_cols = dgp_acic_2018_df(file, n_train=n_train)
    if n_train > 0:
        return df_train, df_data, df_true, binary, categorical, dummy_cols
    return df_data, df_true, binary, categorical, dummy_cols


def get_data(data, config):
    if 'dense' in data:
        df_train, df_data, df_true, x_cols, binary = dgp_dense_mixed_endo_df(config['num_samples'], config['imp_c'],
                                                                               config['imp_d'], config['unimp_c'],
                                                                               config['unimp_d'],
                                                                               n_train=config['n_train'])
    elif data == 'ihdp':
        df_train, df_data, df_true, x_cols, discrete = dgp_ihdp_df(config['ihdp_file'], n_train=config['n_train'])
    elif data == 'news':
        df_train, df_data, df_true, x_cols, discrete = dgp_news(config['news_file'], n_train=config['n_train'])
    else:
        df_train, df_data, df_true, x_cols, discrete = dgp_df(dgp=data, n_samples=config['num_samples'],
                                                              n_imp=config['imp_c'], n_unimp=config['unimp_c'],
                                                              n_train=config['n_train'])
    if config['n_train'] > 0:
        return df_train, df_data, df_true, binary
    return df_data, df_true, binary

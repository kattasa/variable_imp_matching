import numpy as np
import pandas as pd
from sklearn.preprocessing import minmax_scale
from utils import save_df_to_csv
import warnings
from argparse import ArgumentParser
from collections import namedtuple
from datagen.dgp import dgp_linear, dgp_lihua

def gen_original_data(args):
    print(args)
    np.random.seed(42069)
    folder = f'./Experiments/variance/randomization_files/dgp_{args.dgp}/n_train_{args.n_train}/n_est_{args.n_est}/n_imp_{args.n_imp}/n_unimp_{args.n_unimp}'
    try:
        pd.read_csv(folder + '/X_train.csv')
        pd.read_csv(folder + '/train_df.csv')
        pd.read_csv(folder + '/train_true_df.csv')
        pd.read_csv(folder + '/X_est.csv')
        pd.read_csv(folder + '/est_df.csv')
        pd.read_csv(folder + '/est_true_df.csv')
        pd.read_csv(folder + '/X_query.csv')
        pd.read_csv(folder + '/query_df.csv')
        pd.read_csv(folder + '/query_true_df.csv')
    except FileNotFoundError:
        if args.dgp == 'linear_homoskedastic':
            gen_dgp = dgp_linear(xmin = -5, xmax = 5, n_imp = args.n_imp, n_unimp=args.n_unimp, heteroskedastic=False)
        elif args.dgp == 'linear_heteroskedastic':
            gen_dgp = dgp_linear(xmin = -5, xmax = 5, n_imp = args.n_imp, n_unimp=args.n_unimp, heteroskedastic=True)
        elif args.dgp == 'linear_heteroskedastic':
            gen_dgp = dgp_linear(xmin = -5, xmax = 5, n_imp = args.n_imp, n_unimp=args.n_unimp, heteroskedastic=True)
        elif args.dgp == 'lihua_uncorr_homoskedastic':
            gen_dgp = dgp_lihua(xmin = -5, xmax = 5, n_imp = args.n_imp, n_unimp=args.n_unimp, heteroskedastic=False, corr = False)
        elif args.dgp == 'lihua_uncorr_heteroskedastic':
            gen_dgp = dgp_lihua(xmin = -5, xmax = 5, n_imp = args.n_imp, n_unimp=args.n_unimp, heteroskedastic=True, corr = False)
        elif args.dgp == 'lihua_corr_homoskedastic':
            gen_dgp = dgp_lihua(xmin = -5, xmax = 5, n_imp = args.n_imp, n_unimp=args.n_unimp, heteroskedastic=False, corr = True)
        elif args.dgp == 'lihua_corr_heteroskedastic':
            gen_dgp = dgp_lihua(xmin = -5, xmax = 5, n_imp = args.n_imp, n_unimp=args.n_unimp, heteroskedastic=True, corr = True)
            
        X_train, train_df, train_true_df, X_est, est_df, est_true_df, X_query, query_df, query_true_df = gen_dgp.gen_train_est_query(n_train = args.n_train, n_est = args.n_est, n_query = args.n_query)
        
        save_df_to_csv(X_train, folder + '/X_train.csv')
        save_df_to_csv(train_df, folder + '/train_df.csv')
        save_df_to_csv(train_true_df, folder + '/train_true_df.csv')
        save_df_to_csv(X_est, folder + '/X_est.csv')
        save_df_to_csv(est_df, folder + '/est_df.csv')
        save_df_to_csv(est_true_df, folder + '/est_true_df.csv')
        save_df_to_csv(X_query, folder + '/X_query.csv')
        save_df_to_csv(query_df, folder + '/query_df.csv')
        save_df_to_csv(query_true_df, folder + '/query_true_df.csv')

def gen_resampled_data(args):
    np.random.seed(args.seed)
    folder = f'./Experiments/variance/randomization_files/dgp_{args.dgp}/n_train_{args.n_train}/n_est_{args.n_est}/n_imp_{args.n_imp}/n_unimp_{args.n_unimp}'
    try:
        pd.read_csv(folder + f'/seed_{args.seed}/train_df.csv')
        pd.read_csv(folder + f'/seed_{args.seed}/train_true_df.csv')
        pd.read_csv(folder + f'/seed_{args.seed}/est_df.csv')
        pd.read_csv(folder + f'/seed_{args.seed}/est_iter_df.csv')
    except FileNotFoundError:
        print('file not found. creating...')
        X_train_raw = pd.read_csv(folder + '/X_train.csv')
        X_est_raw = pd.read_csv(folder + '/X_est.csv')
        if args.dgp == 'linear_homoskedastic':
            gen_dgp = dgp_linear(xmin = -5, xmax = 5, n_imp = args.n_imp, n_unimp=args.n_unimp, heteroskedastic=False)
        elif args.dgp == 'linear_heteroskedastic':
            gen_dgp = dgp_linear(xmin = -5, xmax = 5, n_imp = args.n_imp, n_unimp=args.n_unimp, heteroskedastic=True)
        elif args.dgp == 'lihua_uncorr_homoskedastic':
            gen_dgp = dgp_lihua(xmin = -5, xmax = 5, n_imp = args.n_imp, n_unimp=args.n_unimp, heteroskedastic=False, corr = False)
        elif args.dgp == 'lihua_uncorr_heteroskedastic':
            gen_dgp = dgp_lihua(xmin = -5, xmax = 5, n_imp = args.n_imp, n_unimp=args.n_unimp, heteroskedastic=True, corr = False)
        elif args.dgp == 'lihua_corr_homoskedastic':
            gen_dgp = dgp_lihua(xmin = -5, xmax = 5, n_imp = args.n_imp, n_unimp=args.n_unimp, heteroskedastic=False, corr = True)
        elif args.dgp == 'lihua_corr_heteroskedastic':
            gen_dgp = dgp_lihua(xmin = -5, xmax = 5, n_imp = args.n_imp, n_unimp=args.n_unimp, heteroskedastic=True, corr = True)
        else:
            raise ValueError(f'Error. {args.dgp} not found...')
            
        train_iter_df, train_true_iter_df = gen_dgp.gen_resampled_dataset(X_train_raw)
        est_iter_df, est_true_iter_df = gen_dgp.gen_resampled_dataset(X_est_raw)

        save_df_to_csv(train_iter_df, folder + f'/seed_{args.seed}/train_df.csv')
        save_df_to_csv(train_true_iter_df, folder + f'/seed_{args.seed}/train_true_df.csv')
        save_df_to_csv(est_iter_df, folder + f'/seed_{args.seed}/est_df.csv')
        save_df_to_csv(est_true_iter_df, folder + f'/seed_{args.seed}/est_iter_df.csv')

def main():

    parser = ArgumentParser()
    parser.add_argument('--n_train', type = int)
    parser.add_argument('--n_est', type = int)
    parser.add_argument('--n_query', type = int)
    parser.add_argument('--n_imp', type = int)
    parser.add_argument('--n_unimp', type = int)
    parser.add_argument('--n_iter', type = int)
    parser.add_argument('--dgp', type = str)


    parser_args = parser.parse_args()

    ## generate original data
    gen_original_data(parser_args)

    ## generate resampled datasets
    for iter in range(parser_args.n_iter):
        parser_args.seed = 42 * iter
        gen_resampled_data(parser_args)

if __name__ == '__main__':
    main()

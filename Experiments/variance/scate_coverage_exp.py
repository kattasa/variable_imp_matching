
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import numpy as np
from variance_playground import *
from src.variable_imp_matching import VIM
from confseq.betting import betting_ci
from utils import save_df_to_csv
from sklearn.ensemble import RandomForestRegressor
from datagen.dgp_df import dgp_df
from datagen.dgp import add_noise

from argparse import ArgumentParser


def calib_bias(df, treatment, outcome, sklearn_model, T = 1, C = 0, args = {}):
    model_T = sklearn_model(**args)
    X_T = df.loc[df[treatment] == T].drop([outcome, treatment], axis = 1)
    Y_T = df.loc[df[treatment] == T][outcome]
    model_T.fit(X_T, Y_T)

    model_C = sklearn_model(**args)
    X_C = df.loc[df[treatment] == C].drop([outcome, treatment], axis = 1)
    Y_C = df.loc[df[treatment] == C][outcome]
    model_C.fit(X_C, Y_C)

    return {T : model_T, C : model_C}

def normalize_samples(samples, min, max):
    return (samples - min)/(max - min)

def unnormalize_samples(samples, min, max):
    return samples * (max - min) + min

def make_betting_ci(samples, alpha, min, max):
    y_min = samples.min()
    y_max = samples.max() 
    
    samples_normalized = normalize_samples(samples, min, max) # (samples - y_min)/(y_max - y_min)

    # print('normalized min, max:', samples_normalized.min(), samples_normalized.max())

    betting_cs_result = betting_ci(samples_normalized, alpha = alpha)
    betting_lb = betting_cs_result[0]
    betting_ub = betting_cs_result[1]

    betting_lb = unnormalize_samples(betting_lb, min, max) # betting_lb * (y_max - y_min) + y_min
    betting_ub = unnormalize_samples(betting_ub, min, max) # betting_ub * (y_max - y_min) + y_min
    
    return betting_lb, betting_ub

def get_betting_bounds(df_est, Y, tx, mgs, T, C, alpha, ymin, ymax):
    # def make_betting_ci(samples, alpha, y_min, y_max):
        
    #     samples_normalized = (samples - y_min)/(y_max - y_min)

    #     betting_cs_result = betting_ci(samples_normalized, alpha = alpha)
    #     betting_lb = betting_cs_result[0]
    #     betting_ub = betting_cs_result[1]

    #     betting_lb = betting_lb * (y_max - y_min) + y_min
    #     betting_ub = betting_ub * (y_max - y_min) + y_min
        
    #     return betting_lb, betting_ub

    # df_est[Y] = (df_est[Y].values  - df_est[Y].min())/(df_est[Y].max() - df_est[Y].min())
    df_est = df_est.copy()
    df_est[Y + '_normalized'] = df_est[Y].values * (df_est[tx].values) - df_est[Y].values * (1 - df_est[tx].values)
    # y_max = df_est[Y + '_normalized'].max()
    # y_min = df_est[Y + '_normalized'].min()
    # df_est[Y + '_normalized'] = (df_est[Y + '_normalized'].values  - y_min)/(y_max - y_min)
    Y_T = df_est[Y + '_normalized'].values[mgs[T]]
    Y_C = df_est[Y + '_normalized'].values[mgs[C]] 
    Y_stack = np.hstack([Y_T, Y_C])
    # get confidence intervals from hedging by betting sequences...
    # lb = np.apply_along_axis(arr = Y_stack, func1d = lambda x: hedged_cs(x, alpha=alpha)[0][-1], axis = 1)
    # ub = np.apply_along_axis(arr = Y_stack, func1d = lambda x: hedged_cs(x, alpha=alpha)[1][-1], axis = 1)

    y_min = -np.max(np.abs([ymin, ymax]))
    y_max = np.max(np.abs([ymin, ymax]))

    print('y_min', y_min)
    print('y_max', y_max)

    lb = np.apply_along_axis(arr = Y_stack, func1d = lambda x: make_betting_ci(x, alpha=alpha, min = y_min, max = y_max)[0], axis = 1)
    ub = np.apply_along_axis(arr = Y_stack, func1d = lambda x: make_betting_ci(x, alpha=alpha, min = y_min, max = y_max)[1], axis = 1)

    ## un-normalize the bounds
    # lb = lb * (y_max - y_min) + y_min
    # ub = ub * (y_max - y_min) + y_min
    return lb, ub
    
def get_CATE_bias_betting_bound(X_NN_T, X_NN_C, query_x, k, model_dict, T, C, df_est, Y, tx, mgs, ymin, ymax, alpha = 0.05):
    T_uq = []
    C_uq = []
    for i in range(k):
        T_uq.append(model_dict[T].predict(X_NN_T[i], T = 1))
        C_uq.append(model_dict[C].predict(X_NN_C[i], T = 0))

    T_uq = np.array(T_uq).mean(axis = 0)
    T_query = model_dict[T].predict(query_x, T = 1)
    # T_uq = np.abs(T_query - T_uq)
    print('T_uq shape:', T_uq.shape)

    C_uq = np.array(C_uq).mean(axis = 0)
    C_query = model_dict[C].predict(query_x, T = 0)
    # C_uq = np.abs(C_query - C_uq)
    print('C_uq shape:', C_uq.shape)

    bias = np.abs(T_uq - T_query - C_uq + T_query)

    lb, ub = get_betting_bounds(df_est = df_est, Y = Y, tx = tx, mgs = mgs, T = T, C = C, alpha = alpha, ymin = ymin, ymax = ymax)
    lb, ub = lb - bias, ub + bias
    
    return lb, ub

def bias_corr_betting_scate_ci(dgp, lcm, df_calib, df_est, k, ymin, ymax, query_x, cate_true):
    '''
    given a distance metric, calibration set, and estimation set, estimate confidence intervals for a query set
    
    inputs
    ----
    dgp : name of dgp
    lcm : trained VIM object
    df_calib : pandas dataframe used for training outcome regressions
    df_est : pandas dataframe used for constructing MGs
    k : number of nearest neighbors to be used for matching
    ymin : float. lower bound of outcome
    ymax : float. upper bound of outcome
    query_x : pandas dataframe for query points at which we want to construct conf ints
    cate_true : vector of true CATEs


    '''

    # estimate CATEs
    mgs, mgs_dists = lcm.create_mgs(df_estimation=df_est, k=k, return_original_idx=False, query_x = query_x)
    cate = lcm.est_cate(df_est, match_groups = mgs, match_distances = mgs_dists, k = k, diameter_prune = False, query_x = query_x)
    cate['CATE_true'] = cate_true

    # estimate confidence intervals for query set
    
    class sklearn_model:
        def __init__(self, model, dgp):
            self.dgp = dgp
            self.model = model()
        def fit(self, X_train, y_train):
            self.model.fit(X_train, y_train)
        def predict(self, X_est, T):
            return self.model.predict(X_est)

    model_dict = calib_bias(df = df_calib, treatment = 'T', outcome = 'Y', sklearn_model = sklearn_model, T = 1, C = 0, args = {'dgp' : dgp, 'model' : RandomForestRegressor})

    X_NN_T = {j : df_est.loc[mgs[1][j].values].drop(['T', 'Y'], axis = 1) for j in range(k)}
    X_NN_C = {j : df_est.loc[mgs[0][j].values].drop(['T', 'Y'], axis = 1) for j in range(k)}

    lb, ub = get_CATE_bias_betting_bound(X_NN_T = X_NN_T, X_NN_C = X_NN_C, query_x = query_x.drop(['T', 'Y'], axis = 1), k = k, model_dict = model_dict, T = 1, C = 0, df_est = df_est, Y = 'Y', tx = 'T', mgs = mgs, alpha = 0.05, ymin = ymin, ymax = ymax)
    return cate, lb, ub


def make_data(dgp, n_train, n_est, n_query, n_imp, n_unimp, n_iter = 100, query_seed = 100, sample_seed = 42069):
    np.random.seed(query_seed)
    df_train, query_x, df_true, x_cols, discrete, ymin, ymax = dgp_df(dgp = dgp, n_samples = n_query, n_imp = n_imp, n_unimp=n_unimp, perc_train=None, n_train=0)
    cate_true = df_true[['TE']]

    # save query df
    save_df_to_csv(query_x, f'./Experiments/variance/output_files/dgp_{dgp}/n_imp_{n_imp}/n_unimp_{n_unimp}/query_seed_{query_seed}/query_x.csv')
    save_df_to_csv(cate_true, f'./Experiments/variance/output_files/dgp_{dgp}/n_imp_{n_imp}/n_unimp_{n_unimp}/query_seed_{query_seed}/cate_true.csv')

    # make sample dataset
    np.random.seed(sample_seed)
    df_train, df_est, df_true, x_cols, discrete, ymin, ymax = dgp_df(dgp = dgp, n_samples = n_train + n_est, n_imp = n_imp, n_unimp=n_unimp, perc_train=None, n_train=n_train)

    ## split into train, calibrate, and estimation
    df_train_sub, df_calib = train_test_split(df_train, test_size = 0.5, stratify = df_train['T'].values, random_state = 42)

    # save training and calibration sets to re-use across runs
    save_df_to_csv(df_train_sub, f'./Experiments/variance/output_files/dgp_{dgp}/n_imp_{n_imp}/n_unimp_{n_unimp}/n_train_{n_train}/n_est_{n_est}/sample_seed_{sample_seed}/df_train_sub.csv')
    save_df_to_csv(df_calib, f'./Experiments/variance/output_files/dgp_{dgp}/n_imp_{n_imp}/n_unimp_{n_unimp}/n_train_{n_train}/n_est_{n_est}/sample_seed_{sample_seed}/df_calib.csv')
    save_df_to_csv(df_true, f'./Experiments/variance/output_files/dgp_{dgp}/n_imp_{n_imp}/n_unimp_{n_unimp}/n_train_{n_train}/n_est_{n_est}/sample_seed_{sample_seed}/df_true.csv')
    save_df_to_csv(df_est, f'./Experiments/variance/output_files/dgp_{dgp}/n_imp_{n_imp}/n_unimp_{n_unimp}/n_train_{n_train}/n_est_{n_est}/sample_seed_{sample_seed}/df_est.csv')
    
    # save ymin, ymax
    save_df_to_csv(pd.DataFrame({'ymin' : [ymin], 'ymax' : [ymax]}), f'./Experiments/variance/output_files/dgp_{dgp}/n_imp_{n_imp}/n_unimp_{n_unimp}/summaries.csv')

    # make estimation dataset to use for each run
    for seed in range(n_iter):
        seed *= 42
        np.random.seed(seed)

        # add noise and get the right Y
        df_est['Y'] = df_true['Y1_true'] * df_true['T'] + df_true['Y0_true'] * (1 - df_true['T']) + add_noise(df_true[x_cols], 'hetero' in dgp).flatten()
        save_df_to_csv(df_est, f'./Experiments/variance/output_files/dgp_{dgp}/n_imp_{n_imp}/n_unimp_{n_unimp}/n_train_{n_train}/n_est_{n_est}/sample_seed_{sample_seed}/final_seed_{seed}/df_est.csv')




def bias_corr_coverage(dgp, n_train, n_est, n_imp, n_unimp, k, fit, query_seed, sample_seed, seed):
    # read in training, calibration, true, and estimation sets
    df_train_sub = pd.read_csv(f'./Experiments/variance/output_files/dgp_{dgp}/n_imp_{n_imp}/n_unimp_{n_unimp}/n_train_{n_train}/n_est_{n_est}/sample_seed_{sample_seed}/df_train_sub.csv')
    df_calib = pd.read_csv(f'./Experiments/variance/output_files/dgp_{dgp}/n_imp_{n_imp}/n_unimp_{n_unimp}/n_train_{n_train}/n_est_{n_est}/sample_seed_{sample_seed}/df_calib.csv')
    df_true = pd.read_csv(f'./Experiments/variance/output_files/dgp_{dgp}/n_imp_{n_imp}/n_unimp_{n_unimp}/n_train_{n_train}/n_est_{n_est}/sample_seed_{sample_seed}/df_true.csv')
    df_est = pd.read_csv(f'./Experiments/variance/output_files/dgp_{dgp}/n_imp_{n_imp}/n_unimp_{n_unimp}/n_train_{n_train}/n_est_{n_est}/sample_seed_{sample_seed}/final_seed_{seed}/df_est.csv')

    query_x = pd.read_csv(f'./Experiments/variance/output_files/dgp_{dgp}/n_imp_{n_imp}/n_unimp_{n_unimp}/query_seed_{query_seed}/query_x.csv')
    cate_true = pd.read_csv(f'./Experiments/variance/output_files/dgp_{dgp}/n_imp_{n_imp}/n_unimp_{n_unimp}/query_seed_{query_seed}/cate_true.csv')[['TE']].values

    # get min/max of outcomes
    df_summaries = pd.read_csv(f'./Experiments/variance/output_files/dgp_{dgp}/n_imp_{n_imp}/n_unimp_{n_unimp}/summaries.csv')
    ymin = df_summaries.ymin.values[0]
    ymax = df_summaries.ymax.values[0]

    # fit LCM to find distance metric
    lcm = VIM(outcome = 'Y', treatment = 'T', data = df_train_sub, binary_outcome=False, random_state=None)
    lcm.fit(return_scores = False, model = 'ensemble')

    # save new estimation set
    if fit == 'bias_corr_betting':
        cate, lb, ub = bias_corr_betting_scate_ci(dgp, lcm, df_calib, df_est, k, ymin, ymax, query_x, cate_true)
    else:
        raise ValueError(f'fit unknown: `{fit}`')
    cate['CATE_lb'] = lb
    cate['CATE_ub'] = ub
    cate['CATE_error_bound'] = np.abs(cate['CATE_ub'] - cate['CATE_lb'])/2
    cate['contains_true_cate'] = (cate['CATE_lb'] <= cate['CATE_true']) * (cate['CATE_ub'] >= cate['CATE_true'])
    cate['seed'] = seed
    cate['id'] = cate.index.values
    cate['fit'] = fit
    
    # save CATE df
    save_df_to_csv(cate, f'./Experiments/variance/output_files/dgp_{dgp}/n_imp_{n_imp}/n_unimp_{n_unimp}/n_train_{n_train}/n_est_{n_est}/sample_seed_{sample_seed}/final_seed_{seed}/bias_corr_betting.csv')


def main():
    parser = ArgumentParser()
    parser.add_argument('--task_id', type = int)
    parser.add_argument('--fit', type = str)

    parser_args = parser.parse_args()
    args_df = pd.read_csv('./Experiments/variance/scate_args.csv')
    if parser_args.task_id >= args_df.shape[0]:
        print('task id is out of bounds for args_df size:', parser_args.task_id, args_df.shape[0])
    args = args_df.loc[parser_args.task_id - 1, ] # subtract 1 because arg generator gets first array task 

    if parser_args.fit == 'bias_corr_betting':
        args['fit'] = 'bias_corr_betting'
        bias_corr_coverage(**args)


if __name__ == '__main__':
    main()
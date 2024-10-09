import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from datagen.dgp_df import dgp_poly_basic_df, dgp_df
from src.variable_imp_matching import VIM
from scipy.spatial.distance import pdist
import warnings
from utils import save_df_to_csv
from collections import namedtuple
from argparse import ArgumentParser
from econml.dml import CausalForestDML, KernelDML
from econml.dr import DRLearner
from econml.metalearners import XLearner, TLearner, SLearner
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, HistGradientBoostingRegressor

def estimate_calibration_distance(df, treatment_col, outcome_col, M, metric = 'cityblock'):
    # scale df by M
    X = df.drop([treatment_col, outcome_col], axis = 1)
    X = X * M
    y = df[[outcome_col]]
    
    # for each observation in df with T=t,
    model_dict = {}
    unique_txs = np.unique(df[treatment_col].values)
    for tx in unique_txs:
        # create matrix of distances btw scaled covs
        X_tx = X.loc[df[treatment_col] == tx]
        y_tx = y.loc[df[treatment_col] == tx]
        print('X_tx size calib: ', tx, X_tx.shape)

        # create vector of distances btw outcomes
        X_distance_tx = pdist(X_tx, metric = metric)
        y_distance_tx = pdist(y_tx, metric = 'cityblock')
        
        print('X_distance_tx size calib: ', tx, X_distance_tx.shape)

        # regress outcome distances on cov distances using flexible, non-parametric method (e.g., deep tree)
        # with warnings.catch_warnings():
        #     warnings.filterwarnings("ignore", message="X has feature names, but KNeighborsRegressor was fitted without feature names")
        model = KNeighborsRegressor(n_neighbors=int(np.sqrt(y_tx.shape[0])))
        # model = DecisionTreeRegressor()
        model.fit(X_distance_tx.reshape(-1,1), y_distance_tx)

        # save distance function for this treatment arm
        model_dict[tx] = model
    return model_dict

# def get_CATE_error_bound(max_dist_T, max_dist_C, k, model_dict, T, C, max_y, alpha = 0.05):
    
#     T_uq = model_dict[T].predict(max_dist_T)
#     C_uq = model_dict[C].predict(max_dist_C)

#     delta_a = np.maximum(T_uq, C_uq)
    
#     return delta_a + 2 * np.sqrt( -2 * max_y / k * np.log((1 - alpha)/4) )

# def get_CATE_error_bound(max_dist_T, max_dist_C, k, model_dict, T, C, max_y, min_y = None, alpha = 0.05):
    
#     T_uq = model_dict[T].predict(max_dist_T[[k - 1]])
#     C_uq = model_dict[C].predict(max_dist_C[[k - 1]])

#     delta_a = np.maximum(T_uq, C_uq)
    
#     # return delta_a + 2 * max_y * np.sqrt( np.log(2/(alpha)) / (4 * k) )
#     return delta_a + 2 * max_y * np.sqrt( np.log(2/(alpha)) / (2 * k) )

def get_CATE_error_bound(dist_vec_T, dist_vec_C, k, model_dict, T, C, min_y, max_y, alpha = 0.05):
    T_uq = []
    C_uq = []
    for i in range(k):
        T_uq.append(model_dict[T].predict(dist_vec_T[[i]]))
        C_uq.append(model_dict[C].predict(dist_vec_C[[i]]))

    T_uq = np.array(T_uq).mean(axis = 0)
    C_uq = np.array(C_uq).mean(axis = 0)

    # delta_a = np.maximum(T_uq, C_uq)
    
    b_min_a = 2 * max(np.abs(max_y), np.abs(min_y))

    return b_min_a/2 * np.sqrt( np.log( 2 / (alpha) ) / k ) + T_uq + C_uq

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

def get_CATE_bias_bound(X_NN_T, X_NN_C, query_x, k, model_dict, T, C, min_y, max_y, alpha = 0.05):
    T_uq = []
    C_uq = []
    for i in range(k):
        T_uq.append(model_dict[T].predict(X_NN_T[i]))
        C_uq.append(model_dict[C].predict(X_NN_C[i]))

    T_uq = np.array(T_uq).mean(axis = 0)
    T_query = model_dict[T].predict(query_x)
    # T_uq = np.abs(T_query - T_uq)
    print('T_uq shape:', T_uq.shape)

    C_uq = np.array(C_uq).mean(axis = 0)
    C_query = model_dict[C].predict(query_x)
    # C_uq = np.abs(C_query - C_uq)
    print('C_uq shape:', C_uq.shape)

    bias = np.abs(T_query - T_uq + C_uq - C_query)

    # delta_a = np.maximum(T_uq, C_uq)
    
    b_min_a = 2 * max(np.abs(max_y), np.abs(min_y))

    return b_min_a/2 * np.sqrt( np.log( 2 / (alpha) ) / k ) + bias

from confseq.betting import hedged_cs, betting_ci
def get_betting_bounds(df_est, Y, tx, mgs, T, C, alpha):
    def make_betting_ci(samples, alpha):
        y_min = samples.min()
        y_max = samples.max() 
        
        samples_normalized = (samples - y_min)/(y_max - y_min)

        betting_cs_result = betting_ci(samples_normalized, alpha = alpha)
        betting_lb = betting_cs_result[0]
        betting_ub = betting_cs_result[1]

        betting_lb = betting_lb * (y_max - y_min) + y_min
        betting_ub = betting_ub * (y_max - y_min) + y_min
        
        return betting_lb, betting_ub

    # df_est[Y] = (df_est[Y].values  - df_est[Y].min())/(df_est[Y].max() - df_est[Y].min())
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

    lb = np.apply_along_axis(arr = Y_stack, func1d = lambda x: make_betting_ci(x, alpha=alpha)[0], axis = 1)
    ub = np.apply_along_axis(arr = Y_stack, func1d = lambda x: make_betting_ci(x, alpha=alpha)[1], axis = 1)

    ## un-normalize the bounds
    # lb = lb * (y_max - y_min) + y_min
    # ub = ub * (y_max - y_min) + y_min
    return lb, ub
    
def get_CATE_bias_betting_bound(X_NN_T, X_NN_C, query_x, k, model_dict, T, C, df_est, Y, tx, mgs, alpha = 0.05):
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

    lb, ub = get_betting_bounds(df_est = df_est, Y = Y, tx = tx, mgs = mgs, T = T, C = C, alpha = alpha)
    lb, ub = lb - bias, ub + bias
    
    return lb, ub

    


def get_mml_CATE_error_bound(df_est, treatment_col, outcome_col, mgs, alpha, T = 1, C = 0):
    n = df_est.shape[1]
    d = df_est.drop([treatment_col, outcome_col], axis = 1).shape[1] # number of covs
    y = df_est[outcome_col].values
    avg_treated = y[mgs[T]].mean(axis = 1) # get sample variance of treated nns
    avg_control = y[mgs[C]].mean(axis = 1) # get sample variance of treated nns
    cate = avg_treated - avg_control
    var_treated = y[mgs[T]].var(axis = 1, ddof = 1) # get sample variance of treated nns
    var_control = y[mgs[C]].var(axis = 1, ddof = 1) # get sample variance of control nns
    K_treated = mgs[T].shape[1]
    K_control = mgs[C].shape[0]
    r = 1/(2 + d)
    from scipy.special import ndtri
    z_score = ndtri(1 - alpha/2)
    var = var_treated/K_treated + var_control/K_control
    std_err = np.sqrt(var)/(n**r)
    error_bound = z_score * std_err
    return error_bound

def get_causal_forest_CATE_error_bound(df_train, df_est, treatment_col, outcome_col, alpha):
    X_train = df_train.drop([treatment_col, outcome_col], axis = 1)
    t_train = df_train[treatment_col]
    y_train = df_train[outcome_col]

    X_est = df_est.drop([treatment_col, outcome_col], axis = 1)
    t_est = df_est[treatment_col]
    y_est = df_est[outcome_col]

    # Initialize and fit the causal forest model
    cf = CausalForestDML(n_estimators=1000, min_samples_leaf=10, max_depth=None, 
                        verbose=0, random_state=123)
    
    cf.fit(Y=y_train, T=t_train, X=X_train)

    # Estimate CATE for the test set
    cate_pred = cf.effect(X_est)

    # Get confidence intervals
    cate_intervals = cf.effect_interval(X_est, alpha=alpha)  # 95% confidence interval
    error_bound = cate_intervals[1] - cate_pred

    return cate_pred, error_bound

def get_method_CATE_error_bound(df_train, df_est, treatment_col, outcome_col, alpha, estimator, fit_params):
    X_train = df_train.drop([treatment_col, outcome_col], axis = 1)
    t_train = df_train[treatment_col]
    y_train = df_train[outcome_col]

    X_est = df_est.drop([treatment_col, outcome_col], axis = 1)
    t_est = df_est[treatment_col]
    y_est = df_est[outcome_col]

    # Initialize and fit the estimator
    estimator.fit(Y=y_train, T=t_train, X=X_train, **fit_params)

    # Estimate CATE for the test set
    cate_pred = estimator.effect(X_est)

    # Get confidence intervals
    
    cate_intervals = estimator.effect_interval(X_est, alpha=alpha)  # 95% confidence interval
    # error_bound = cate_intervals[1] - cate_pred

    return cate_pred, cate_intervals[0], cate_intervals[1]


# n_samples = 10000
# n_imp = 20 # does not matter for all DGPs
# n_unimp = 0
# k = 50
def get_est_set(dgp, n_imp, n_unimp):
    try:
        est_df = pd.read_csv(f'./Experiments/variance/output_files/dgp_{dgp}/est.csv')
    except FileNotFoundError:
        np.random.seed(42069)
        df_train, est_df, df_true, x_cols, discrete = dgp_df(dgp = dgp, n_samples = 100, n_imp = n_imp, n_unimp=n_unimp, perc_train=None, n_train=0)
        est_df['CATE_true'] = df_true['TE']
        save_df_to_csv(est_df, f'./Experiments/variance/output_files/dgp_{dgp}/est.csv')
    return est_df


def get_data(dgp, n_train : int, n_est : int, n_imp : int, n_unimp : int, k : int, seed : int):
    try:
        print('Trying to find data.')
        df_train_sub = pd.read_csv(f'./Experiments/variance/output_files/dgp_{dgp}/n_train_{n_train}/n_est_{n_est}/n_imp_{n_imp}/n_unimp_{n_unimp}/k_{k}/seed_{seed}/df_train.csv').dropna(axis = 0)
        df_calib = pd.read_csv(f'./Experiments/variance/output_files/dgp_{dgp}/n_train_{n_train}/n_est_{n_est}/n_imp_{n_imp}/n_unimp_{n_unimp}/k_{k}/seed_{seed}/df_calib.csv').dropna(axis = 0)
        df_est = pd.read_csv(f'./Experiments/variance/output_files/dgp_{dgp}/n_train_{n_train}/n_est_{n_est}/n_imp_{n_imp}/n_unimp_{n_unimp}/k_{k}/seed_{seed}/df_est.csv').dropna(axis = 0)
        x_cols = pd.read_csv(f'./Experiments/variance/output_files/dgp_{dgp}/n_train_{n_train}/n_est_{n_est}/n_imp_{n_imp}/n_unimp_{n_unimp}/k_{k}/seed_{seed}/x_cols.csv')['x_cols'].tolist()
        print('Found data. Read in...')
    except FileNotFoundError:
        print('Files not found. Recreating...')
        np.random.seed(seed)
        df_train, df_est, df_true, x_cols, discrete = dgp_df(dgp = dgp, n_samples = n_train + n_est, n_imp = n_imp, n_unimp=n_unimp, perc_train=None, n_train=n_train)

        ## split into train, calibrate, and estimation
        df_train_sub, df_calib = train_test_split(df_train, test_size = 0.4, stratify = df_train['T'].values, random_state = 42)
        print('Calib treatment:', df_calib['T'].unique())

        save_df_to_csv(df_train_sub, f'./Experiments/variance/output_files/dgp_{dgp}/n_train_{n_train}/n_est_{n_est}/n_imp_{n_imp}/n_unimp_{n_unimp}/k_{k}/seed_{seed}/df_train.csv')
        save_df_to_csv(df_calib, f'./Experiments/variance/output_files/dgp_{dgp}/n_train_{n_train}/n_est_{n_est}/n_imp_{n_imp}/n_unimp_{n_unimp}/k_{k}/seed_{seed}/df_calib.csv')
        save_df_to_csv(df_est, f'./Experiments/variance/output_files/dgp_{dgp}/n_train_{n_train}/n_est_{n_est}/n_imp_{n_imp}/n_unimp_{n_unimp}/k_{k}/seed_{seed}/df_est.csv')
        save_df_to_csv(pd.DataFrame({'x_cols' : x_cols}), f'./Experiments/variance/output_files/dgp_{dgp}/n_train_{n_train}/n_est_{n_est}/n_imp_{n_imp}/n_unimp_{n_unimp}/k_{k}/seed_{seed}/x_cols.csv')
    return df_train_sub, df_calib, df_est, x_cols

def check_variance(dgp, n_train : int, n_est : int, n_imp : int, n_unimp : int, k : int, seed : int, fit : str):
    ## make data
    # df_train, df_est, df_true, x_cols = dgp_poly_basic_df(n_samples, n_imp, n_unimp, powers=[2], perc_train=None, n_train=None)
    query_x = get_est_set(dgp, n_imp, n_unimp)
    query_x_true = query_x['CATE_true']
    query_x = query_x.drop('CATE_true', axis = 1)

    np.random.seed(seed)
    # df_train, df_est, df_true, x_cols, discrete = dgp_df(dgp = dgp, n_samples = n_train + n_est, n_imp = n_imp, n_unimp=n_unimp, perc_train=None, n_train=n_train)

    ## split into train, calibrate, and estimation
    # df_train_sub, df_calib = train_test_split(df_train, test_size = 0.4, stratify = df_train['T'].values, random_state = 42)
    # print('Calib treatment:', df_calib['T'].unique())

    df_train_sub, df_calib, df_est, x_cols = get_data(dgp, n_train, n_est, n_imp, n_unimp, k, seed)
    df_train = pd.concat([df_train_sub, df_calib], axis = 0)

    if fit == 'gen_data':
        print('Done generating data.')
        exit()

    lcm = VIM(outcome = 'Y', treatment = 'T', data = df_train_sub, binary_outcome=False, random_state=None)
    if fit == 'vim':
        lcm.fit(return_scores = False)
    elif fit == 'vim_tree':
        lcm.fit(return_scores = False, model = 'tree')
    elif fit == 'vim_ensemble':
        lcm.fit(return_scores = False, model = 'ensemble')
    elif fit == 'nn_vim':
        lcm.M = np.ones(len(x_cols))
    else:
        lcm.M = np.ones(len(x_cols))
    # else:
    #     raise ValueError('oops. fit should be either vim or nn or nn_mml.')
    
    print('k', k)
    mgs, mgs_dists = lcm.create_mgs(df_estimation=df_est, k=k, return_original_idx=False, query_x = query_x)
    cate = lcm.est_cate(df_est, match_groups = mgs, match_distances = mgs_dists, k = k, diameter_prune = False, query_x = query_x)
    cate['CATE_true'] = query_x_true

    print('cate nrow:', cate.shape[0])
    print('df_est nrow:', df_est.shape[0])
    print('df_train nrow:', df_train.shape[0])

    # estimate CATE
    baseline_df_train = pd.concat([df_train, df_est])
    if fit == 'causal_forest':
        estimator = CausalForestDML()
        CATE_mean, lb, ub = get_method_CATE_error_bound(df_train = baseline_df_train, df_est = query_x, treatment_col = 'T', outcome_col = 'Y', alpha = 0.05, estimator = estimator, fit_params = {'inference' : 'auto'})
        cate['CATE_mean'] = CATE_mean
        cate['CATE_lb'] = lb
        cate['CATE_ub'] = ub
        cate['CATE_error_bound'] = np.abs(cate['CATE_ub'] - cate['CATE_lb'])/2
    elif fit == 'kernel_dml':
        estimator = KernelDML()
        CATE_mean, lb, ub = get_method_CATE_error_bound(df_train = baseline_df_train, df_est = query_x, treatment_col = 'T', outcome_col = 'Y', alpha = 0.05, estimator = estimator, fit_params = {'inference' : 'bootstrap'})
        cate['CATE_mean'] = CATE_mean
        cate['CATE_lb'] = lb
        cate['CATE_ub'] = ub
        cate['CATE_error_bound'] = np.abs(cate['CATE_ub'] - cate['CATE_lb'])/2
    elif fit == 'dr_learner':
        estimator = DRLearner()
        CATE_mean, lb, ub = get_method_CATE_error_bound(df_train = baseline_df_train, df_est = query_x, treatment_col = 'T', outcome_col = 'Y', alpha = 0.05, estimator = estimator, fit_params = {'inference' : 'auto'})
        cate['CATE_mean'] = CATE_mean
        cate['CATE_lb'] = lb
        cate['CATE_ub'] = ub
        cate['CATE_error_bound'] = np.abs(cate['CATE_ub'] - cate['CATE_lb'])/2
    elif fit == 'x_learner':
        estimator = XLearner(models = RandomForestRegressor())
        CATE_mean, lb, ub = get_method_CATE_error_bound(df_train = baseline_df_train, df_est = query_x, treatment_col = 'T', outcome_col = 'Y', alpha = 0.05, estimator = estimator, fit_params = {'inference' : 'bootstrap'})
        cate['CATE_mean'] = CATE_mean
        cate['CATE_lb'] = lb
        cate['CATE_ub'] = ub
        cate['CATE_error_bound'] = np.abs(cate['CATE_ub'] - cate['CATE_lb'])/2
    elif fit == 's_learner':
        estimator = SLearner(overall_model = RandomForestRegressor())
        CATE_mean, lb, ub = get_method_CATE_error_bound(df_train = baseline_df_train, df_est = query_x, treatment_col = 'T', outcome_col = 'Y', alpha = 0.05, estimator = estimator, fit_params = {'inference' : 'bootstrap'})
        cate['CATE_mean'] = CATE_mean
        cate['CATE_lb'] = lb
        cate['CATE_ub'] = ub
        cate['CATE_error_bound'] = np.abs(cate['CATE_ub'] - cate['CATE_lb'])/2
    elif fit == 't_learner':
        estimator = TLearner(models = RandomForestRegressor())
        CATE_mean, lb, ub = get_method_CATE_error_bound(df_train = baseline_df_train, df_est = query_x, treatment_col = 'T', outcome_col = 'Y', alpha = 0.05, estimator = estimator, fit_params = {'inference' : 'bootstrap'})
        cate['CATE_mean'] = CATE_mean
        cate['CATE_lb'] = lb
        cate['CATE_ub'] = ub
        cate['CATE_error_bound'] = np.abs(cate['CATE_ub'] - cate['CATE_lb'])/2
    elif fit == 'nn_mml':
        d = n_imp + n_unimp
        k_n = int(df_est.shape[0]**(2/(2 + d)) * (df_est['T'].mean()))
        cate['CATE_error_bound'] = get_mml_CATE_error_bound(df_est = df_est, treatment_col = 'T', outcome_col = 'Y', mgs = mgs, alpha = 0.05, T = 1, C = 0)
        cate['CATE_lb'] = cate['CATE_mean'] - cate['CATE_error_bound']
        cate['CATE_ub'] = cate['CATE_mean'] + cate['CATE_error_bound']
    elif fit == 'naive':
        cate['CATE_error_bound'] = (df_est['Y'].max() - df_est['Y'].min())/2
        cate['CATE_lb'] = cate['CATE_mean'] - cate['CATE_error_bound']
        cate['CATE_ub'] = cate['CATE_mean'] + cate['CATE_error_bound']
    elif fit == 'prog_boost_vim':
        # fit gradient boosting trees to find prognostic score
        
        gbt = GradientBoostingRegressor()
        gbt.fit(df_train_sub.loc[df_train_sub['T'] == 0, ].drop(['T', 'Y'], axis =1), y = df_train_sub.loc[df_train_sub['T'] == 0, ]['Y'].values)
        df_calib['prog'] = gbt.predict(df_calib.drop(['T', 'Y'], axis = 1))
        df_est['prog'] = gbt.predict(df_est.drop(['T', 'Y'], axis = 1))
        query_x['prog'] = gbt.predict(query_x.drop(['T', 'Y'], axis = 1))

        # do VIM on top of prognostic score
        lcm = VIM(outcome = 'Y', treatment = 'T', data = df_calib[['prog', 'T', 'Y']], binary_outcome=False, random_state=None)
        lcm.fit(return_scores = True)
        lcm.M = np.ones(lcm.M.shape[0])
        
        # calibrate distances
        mgs, mgs_dists = lcm.create_mgs(df_estimation=df_est[['prog', 'T', 'Y']], k=k, return_original_idx=False, query_x = query_x[['prog']])
        cate = lcm.est_cate(df_est, match_groups = mgs, match_distances = mgs_dists, k = k, diameter_prune = False, query_x = query_x[['prog']])
        cate['CATE_true'] = query_x_true
        
        model_dict = estimate_calibration_distance(df_calib[['prog', 'T', 'Y']], treatment_col = 'T', outcome_col = 'Y', M = [1], metric = 'cityblock')
        cate['CATE_error_bound'] = get_CATE_error_bound(mgs_dists[1], mgs_dists[0], k = k, model_dict = model_dict, T = 1, C = 0, min_y = df_train['Y'].min(), max_y = df_train['Y'].max(), alpha = 0.05)
        cate['CATE_lb'] = cate['CATE_mean'] - cate['CATE_error_bound']
        cate['CATE_ub'] = cate['CATE_mean'] + cate['CATE_error_bound']
    elif fit == 'prog_cf_vim':
        # fit gradient boosting trees to find prognostic score
        
        cf = CausalForestDML()
        cf.fit(X = df_train_sub.drop(['T', 'Y'], axis =1), T = df_train_sub['T'].values, Y = df_train_sub['Y'].values)
        df_calib['prog'] = cf.effect(df_calib.drop(['T', 'Y'], axis = 1))
        df_est['prog'] = cf.effect(df_est.drop(['T', 'Y'], axis = 1))
        query_x['prog'] = cf.effect(query_x.drop(['T', 'Y'], axis = 1))

        # do VIM on top of prognostic score
        lcm = VIM(outcome = 'Y', treatment = 'T', data = df_calib[['prog', 'T', 'Y']], binary_outcome=False, random_state=None)
        lcm.fit(return_scores = True)
        lcm.M = np.ones(lcm.M.shape[0])
        
        # calibrate distances
        mgs, mgs_dists = lcm.create_mgs(df_estimation=df_est[['prog', 'T', 'Y']], k=k, return_original_idx=False, query_x = query_x[['prog']])
        cate = lcm.est_cate(df_est, match_groups = mgs, match_distances = mgs_dists, k = k, diameter_prune = False, query_x = query_x[['prog']])
        cate['CATE_true'] = query_x_true
        
        model_dict = estimate_calibration_distance(df_calib[['prog', 'T', 'Y']], treatment_col = 'T', outcome_col = 'Y', M = [1], metric = 'cityblock')
        cate['CATE_error_bound'] = get_CATE_error_bound(mgs_dists[1], mgs_dists[0], k = k, model_dict = model_dict, T = 1, C = 0, min_y = df_train['Y'].min(), max_y = df_train['Y'].max(), alpha = 0.05)
        cate['CATE_lb'] = cate['CATE_mean'] - cate['CATE_error_bound']
        cate['CATE_ub'] = cate['CATE_mean'] + cate['CATE_error_bound']
    elif 'vim' in fit:
        model_dict = estimate_calibration_distance(df_calib, treatment_col = 'T', outcome_col = 'Y', M = lcm.M, metric = 'cityblock')
        cate['CATE_error_bound'] = get_CATE_error_bound(mgs_dists[1], mgs_dists[0], k = k, model_dict = model_dict, T = 1, C = 0,  min_y = df_train['Y'].min(), max_y = df_train['Y'].max(), alpha = 0.05)
        cate['CATE_lb'] = cate['CATE_mean'] - cate['CATE_error_bound']
        cate['CATE_ub'] = cate['CATE_mean'] + cate['CATE_error_bound']
    elif fit == 'bias_corr':
        model_dict = calib_bias(df = df_calib, treatment = 'T', outcome = 'Y', sklearn_model = RandomForestRegressor, T = 1, C = 0, args = {})

        X_NN_T = {j : df_est.loc[mgs[1][j].values].drop(['T', 'Y'], axis = 1) for j in range(k)}
        X_NN_C = {j : df_est.loc[mgs[1][j].values].drop(['T', 'Y'], axis = 1) for j in range(k)}

        cate['CATE_error_bound'] = get_CATE_bias_bound(X_NN_T = X_NN_T, X_NN_C = X_NN_C, query_x = query_x.drop(['T', 'Y'], axis = 1), k = k, model_dict = model_dict, T = 1, C = 0, min_y = df_train['Y'].min(), max_y = df_train['Y'].max(), alpha = 0.05)
        cate['CATE_lb'] = cate['CATE_mean'] - cate['CATE_error_bound']
        cate['CATE_ub'] = cate['CATE_mean'] + cate['CATE_error_bound']
    elif fit == 'boost_bias_corr':
        
        lcm = VIM(outcome = 'Y', treatment = 'T', data = df_train_sub, binary_outcome=False, random_state=None)
        lcm.fit(return_scores = False, model = 'ensemble')

        mgs, mgs_dists = lcm.create_mgs(df_estimation=df_est, k=k, return_original_idx=False, query_x = query_x)
        cate = lcm.est_cate(df_est, match_groups = mgs, match_distances = mgs_dists, k = k, diameter_prune = False, query_x = query_x)
        cate['CATE_true'] = query_x_true
        
        model_dict = calib_bias(df = df_calib, treatment = 'T', outcome = 'Y', sklearn_model = RandomForestRegressor, T = 1, C = 0, args = {})

        X_NN_T = {j : df_est.loc[mgs[1][j].values].drop(['T', 'Y'], axis = 1) for j in range(k)}
        X_NN_C = {j : df_est.loc[mgs[1][j].values].drop(['T', 'Y'], axis = 1) for j in range(k)}

        cate['CATE_error_bound'] = get_CATE_bias_bound(X_NN_T = X_NN_T, X_NN_C = X_NN_C, query_x = query_x.drop(['T', 'Y'], axis = 1), k = k, model_dict = model_dict, T = 1, C = 0, min_y = df_train['Y'].min(), max_y = df_train['Y'].max(), alpha = 0.05)
        cate['CATE_lb'] = cate['CATE_mean'] - cate['CATE_error_bound']
        cate['CATE_ub'] = cate['CATE_mean'] + cate['CATE_error_bound']
    elif fit == 'bias_corr_betting':
        lcm = VIM(outcome = 'Y', treatment = 'T', data = df_train_sub, binary_outcome=False, random_state=None)
        lcm.fit(return_scores = False, model = 'linear')

        class dgp_model:
            def __init__(self, dgp):
                self.dgp = dgp
            
            def fit(self, X_train, y_train):
                return None
            
            def predict(self, X_est, T):
                if self.dgp == 'linear':
                    n_imp = 2
                    n_unimp = X_est.shape[1] - n_imp
                    coef = np.array(list(range(n_imp)) + [0] * n_unimp)
                    Y0_true = X_est.dot(coef)
                    Y1_true = Y0_true + 1 + X_est.dot(coef)
                elif 'lihua' in self.dgp:
                    def f(X): # bounded by [0,2]
                        return 2/(1 + np.exp(-12 * (X - 0.5)))
                    Y0_true = 0
                    Y1_true = f(X_est[:, 0]) * f(X_est[:, 1])
                return Y1_true * T + Y0_true * (1 - T)
        
        model_dict = calib_bias(df = df_calib, treatment = 'T', outcome = 'Y', sklearn_model = dgp_model, T = 1, C = 0, args = {'dgp' : dgp})

        X_NN_T = {j : df_est.loc[mgs[1][j].values].drop(['T', 'Y'], axis = 1) for j in range(k)}
        X_NN_C = {j : df_est.loc[mgs[1][j].values].drop(['T', 'Y'], axis = 1) for j in range(k)}
        
        lb, ub = get_CATE_bias_betting_bound(X_NN_T = X_NN_T, X_NN_C = X_NN_C, query_x = query_x.drop(['T', 'Y'], axis = 1), k = k, model_dict = model_dict, T = 1, C = 0, df_est = df_est, Y = 'Y', tx = 'T', mgs = mgs, alpha = 0.05)
        cate['CATE_lb'] = lb
        cate['CATE_ub'] = ub
        cate['CATE_error_bound'] = np.abs(cate['CATE_ub'] - cate['CATE_lb'])/2
    else:
        raise ValueError('fit nor recognized')
    # cate['CATE_lb'] = cate['CATE_mean'] - cate['CATE_error_bound']
    # cate['CATE_ub'] = cate['CATE_mean'] + cate['CATE_error_bound']
    cate['contains_true_cate'] = (cate['CATE_lb'] <= cate['CATE_true']) * (cate['CATE_ub'] >= cate['CATE_true'])
    cate['se'] = ((cate['CATE_true'] - cate['CATE_mean'])**2)

    cate['fit'] = fit
    cate['seed'] = seed
    cate['dgp'] = dgp
    cate['n_train'] = n_train
    cate['n_est'] = n_est
    
    df_est['cate'] = cate['CATE_true']

    print('Sq Error', cate['se'].mean())
    print('Coverage', cate['contains_true_cate'].mean())
    print('1/2 width:', cate['CATE_error_bound'].mean())

    return df_train_sub, df_calib, df_est, cate, query_x #[seed, fit, cate['contains_true_cate'].mean(), cate['se'].mean()]


# vim_results = [check_variance(dgp = 'nonlinear_mml', n_samples = n_samples, n_imp = 20, n_unimp = 0, k = int(np.sqrt(n_samples)), seed  = seed, fit = True) for seed in range(100)]
# nn_results = [check_variance(dgp = 'nonlinear_mml', n_samples = n_samples, n_imp = 20, n_unimp = 0, k = int(np.sqrt(n_samples)), seed  = seed, fit = False) for seed in range(100)]

# vim_df = pd.DataFrame(vim_results, columns = ['seed', 'vim', 'coverage', 'mse'])
# nn_df = pd.DataFrame(nn_results, columns = ['seed', 'vim', 'coverage', 'mse'])

# df = pd.concat([vim_df, nn_df], axis = 0)

# import matplotlib.pyplot as plt
# plt.scatter(df['mse'], df['coverage'], col = df['vim'])
# plt.savefig('./trash.png', dpi = 150)

def main():
    # parser = ArgumentParser()
    # parser.add_argument('--dgp', type=str, default='nonlinear_mml', help='Name of DGP: nonlinear_mml, exp, sine', choices = ['nonlinear_mml', 'exp', 'sine'])  #0.88
    # parser.add_argument('--n_samples', type=int, default=20000, help='#obs')  #0.842
    # parser.add_argument('--n_imp', type=int, default=20, help='#imp covs')
    # parser.add_argument('--n_unimp', type=int, default = 0, help='#unimp covs')
    # parser.add_argument('--k', type = int, default = 10, help = '#nns')
    # parser.add_argument('--seed', type = int, default = 42, help = 'random seed')
    # parser.add_argument('--fit', type = bool, default = True, help = 'To run VIM or vanilla matching')

    # args = parser.parse_args()

    parser = ArgumentParser()
    parser.add_argument('--task_id', type = int)
    parser.add_argument('--fit', type = str)

    parser_args = parser.parse_args()
    args_df = pd.read_csv('./Experiments/variance/args.csv')
    if parser_args.task_id >= args_df.shape[0]:
        print('task id is out of bounds for args_df size:', parser_args.task_id, args_df.shape[0])
    args = args_df.loc[parser_args.task_id - 1, ] # subtract 1 because arg generator gets first array task 
    args['fit'] = parser_args.fit

    # Create a namedtuple class
    DictTuple = namedtuple('DictTuple', args.keys())

    # Convert dictionary to namedtuple
    args = DictTuple(**args)

    print(args)
    print('starting experiment')
    df_train_sub, df_calib, df_est, cate, query_x = check_variance(dgp = args.dgp, n_train = args.n_train, n_est = args.n_est, n_imp = args.n_imp, n_unimp = args.n_unimp, k = args.k, seed = args.seed, fit = args.fit)
    print('ending experiment')
    save_df_to_csv(cate, f'./Experiments/variance/output_files/dgp_{args.dgp}/n_train_{args.n_train}/n_est_{args.n_est}/n_imp_{args.n_imp}/n_unimp_{args.n_unimp}/k_{args.k}/seed_{args.seed}/{args.fit}.csv')
    

if __name__ == '__main__':
    main()



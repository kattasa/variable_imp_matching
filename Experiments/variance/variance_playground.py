import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
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
from sklearn.ensemble import GradientBoostingRegressor

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
        
        # create vector of distances btw outcomes
        X_distance_tx = pdist(X_tx, metric = metric)
        y_distance_tx = pdist(y_tx, metric = 'cityblock')

        # regress outcome distances on cov distances using flexible, non-parametric method (e.g., deep tree)
        # with warnings.catch_warnings():
        #     warnings.filterwarnings("ignore", message="X has feature names, but KNeighborsRegressor was fitted without feature names")
        model = KNeighborsRegressor(n_neighbors=int(np.sqrt(y_tx.shape[0])))
        model.fit(X_distance_tx.reshape(-1,1), y_distance_tx)

        # save distance function for this treatment arm
        model_dict[tx] = model
    return model_dict

# def get_CATE_error_bound(max_dist_T, max_dist_C, k, model_dict, T, C, max_y, alpha = 0.05):
    
#     T_uq = model_dict[T].predict(max_dist_T)
#     C_uq = model_dict[C].predict(max_dist_C)

#     delta_a = np.maximum(T_uq, C_uq)
    
#     return delta_a + 2 * np.sqrt( -2 * max_y / k * np.log((1 - alpha)/4) )

def get_CATE_error_bound(max_dist_T, max_dist_C, k, model_dict, T, C, max_y, alpha = 0.05):
    
    T_uq = model_dict[T].predict(max_dist_T)
    C_uq = model_dict[C].predict(max_dist_C)

    delta_a = np.maximum(T_uq, C_uq)
    
    return delta_a + 2 * max_y * np.sqrt( np.log(2/(1 - alpha)) / (4 * k) )

def get_mml_CATE_error_bound(df_est, treatment_col, outcome_col, mgs, alpha, T = 1, C = 0):
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
    z_score = ndtri(1 - alpha)
    var = var_treated/K_treated + var_control/K_control
    std_err = np.sqrt(var)
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
    error_bound = cate_intervals[1] - cate_pred

    return cate_pred, error_bound


# n_samples = 10000
# n_imp = 20 # does not matter for all DGPs
# n_unimp = 0
# k = 50

def check_variance(dgp, n_train : int, n_imp : int, n_unimp : int, k : int, seed : int, fit : str):
    np.random.seed(seed)
    ## make data
    # df_train, df_est, df_true, x_cols = dgp_poly_basic_df(n_samples, n_imp, n_unimp, powers=[2], perc_train=None, n_train=None)
    df_train, df_est, df_true, x_cols, discrete = dgp_df(dgp = dgp, n_samples = n_train + 1000, n_unimp=n_unimp, perc_train=None, n_train=n_train)

    ## split into train, calibrate, and estimation
    df_train_sub, df_calib, = train_test_split(df_train, test_size = 0.4, random_state = 42)

    lcm = VIM(outcome = 'Y', treatment = 'T', data = df_train_sub, binary_outcome=False, random_state=None)
    if fit == 'vim':
        lcm.fit(return_scores = False)
    elif fit == 'vim_tree':
        lcm.fit(return_scores = False, model = 'tree')
    elif fit == 'vim_ensemble':
        lcm.fit(return_scores = False, model = 'ensemble')
    elif fit == 'nn':
        lcm.M = np.ones(len(x_cols))
    else:
        lcm.M = np.ones(len(x_cols))
    # else:
    #     raise ValueError('oops. fit should be either vim or nn or nn_mml.')
        
    mgs, mgs_dists = lcm.create_mgs(df_estimation=df_est, k=k, return_original_idx=False)
    cate = lcm.est_cate(df_est, match_groups = mgs, match_distances = mgs_dists, k = k, diameter_prune = False)
    cate['CATE_true'] = df_true['TE']

    # estimate CATE
    if fit == 'causal_forest':
        estimator = CausalForestDML()
        CATE_mean, CATE_error_bound = get_method_CATE_error_bound(df_train = df_train, df_est = df_est, treatment_col = 'T', outcome_col = 'Y', alpha = 0.05, estimator = estimator, fit_params = {'inference' : 'auto'})
        cate['CATE_mean'] = CATE_mean
        cate['CATE_error_bound'] = CATE_error_bound
    elif fit == 'kernel_dml':
        estimator = KernelDML()
        CATE_mean, CATE_error_bound = get_method_CATE_error_bound(df_train = df_train, df_est = df_est, treatment_col = 'T', outcome_col = 'Y', alpha = 0.05, estimator = estimator, fit_params = {'inference' : 'bootstrap'})
        cate['CATE_mean'] = CATE_mean
        cate['CATE_error_bound'] = CATE_error_bound
    elif fit == 'dr_learner':
        estimator = DRLearner()
        CATE_mean, CATE_error_bound = get_method_CATE_error_bound(df_train = df_train, df_est = df_est, treatment_col = 'T', outcome_col = 'Y', alpha = 0.05, estimator = estimator, fit_params = {'inference' : 'auto'})
        cate['CATE_mean'] = CATE_mean
        cate['CATE_error_bound'] = CATE_error_bound
    elif fit == 'x_learner':
        estimator = XLearner(models = GradientBoostingRegressor())
        CATE_mean, CATE_error_bound = get_method_CATE_error_bound(df_train = df_train, df_est = df_est, treatment_col = 'T', outcome_col = 'Y', alpha = 0.05, estimator = estimator, fit_params = {'inference' : 'bootstrap'})
        cate['CATE_mean'] = CATE_mean
        cate['CATE_error_bound'] = CATE_error_bound
    elif fit == 's_learner':
        estimator = SLearner(overall_model = GradientBoostingRegressor())
        CATE_mean, CATE_error_bound = get_method_CATE_error_bound(df_train = df_train, df_est = df_est, treatment_col = 'T', outcome_col = 'Y', alpha = 0.05, estimator = estimator, fit_params = {'inference' : 'bootstrap'})
        cate['CATE_mean'] = CATE_mean
        cate['CATE_error_bound'] = CATE_error_bound
    elif fit == 't_learner':
        estimator = TLearner(models = GradientBoostingRegressor())
        CATE_mean, CATE_error_bound = get_method_CATE_error_bound(df_train = df_train, df_est = df_est, treatment_col = 'T', outcome_col = 'Y', alpha = 0.05, estimator = estimator, fit_params = {'inference' : 'bootstrap'})
        cate['CATE_mean'] = CATE_mean
        cate['CATE_error_bound'] = CATE_error_bound
    elif fit == 'nn_mml':
        cate['CATE_error_bound'] = get_mml_CATE_error_bound(df_est = df_est, treatment_col = 'T', outcome_col = 'Y', mgs = mgs, alpha = 0.05, T = 1, C = 0)
    elif fit == 'naive':
        cate['CATE_error_bound'] = (df_est['Y'].max() - df_est['Y'].min())/2
    else:
        model_dict = estimate_calibration_distance(df_calib, treatment_col = 'T', outcome_col = 'Y', M = lcm.M, metric = 'cityblock')
        cate['CATE_error_bound'] = get_CATE_error_bound(cate[['dist_1']], cate[['dist_0']], k = k, model_dict = model_dict, T = 1, C = 0, max_y = df_train['Y'].max(), alpha = 0.05)

    cate['CATE_lb'] = cate['CATE_mean'] - cate['CATE_error_bound']
    cate['CATE_ub'] = cate['CATE_mean'] + cate['CATE_error_bound']
    cate['contains_true_cate'] = (cate['CATE_lb'] <= cate['CATE_true']) * (cate['CATE_ub'] >= cate['CATE_true'])
    cate['se'] = ((cate['CATE_true'] - cate['CATE_mean'])**2)

    cate['fit'] = fit
    cate['seed'] = seed
    cate['dgp'] = dgp
    cate['n_train'] = n_train
    
    return cate #[seed, fit, cate['contains_true_cate'].mean(), cate['se'].mean()]


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
    parser_args = parser.parse_args()
    args_df = pd.read_csv('./Experiments/variance/args.csv')
    args = args_df.loc[parser_args.task_id - 1, ] # subtract 1 because arg generator gets first array task 

    # Create a namedtuple class
    DictTuple = namedtuple('DictTuple', args.keys())

    # Convert dictionary to namedtuple
    args = DictTuple(**args)

    print(args)
    print('starting experiment')
    cate = check_variance(dgp = args.dgp, n_train = args.n_train, n_imp = args.n_imp, n_unimp = args.n_unimp, k = args.k, seed = args.seed, fit = args.fit)
    print('ending experiment')
    save_df_to_csv(cate, f'./Experiments/variance/output_files/dgp_{args.dgp}/n_train_{args.n_train}/n_imp_{args.n_imp}/n_unimp_{args.n_unimp}/k_{args.k}/seed_{args.seed}/{args.fit}.csv')
    
if __name__ == '__main__':
    main()
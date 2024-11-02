#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data generation processes used for simulations in LCM paper.

data_generation_dense_mixed_endo() taken from code used for the original MALTS
paper.
"""
import numpy as np
import pandas as pd
import scipy
from scipy.special import expit
from sklearn.preprocessing import minmax_scale
import warnings
from scipy.spatial import ConvexHull, Delaunay


np.random.seed(0)


def is_point_in_hull(hull, points):
    """
    Check if a point is inside the convex hull.
    
    Parameters:
    hull (ConvexHull): The convex hull of the dataset.
    point (array-like): The points to check.
    
    Returns:
    bool: True if the point is inside the convex hull, False otherwise.
    """
    hull_max = hull.max(axis = 0) + 1e-6
    hull_min = hull.min(axis = 0) - 1e-6
    inside_hull = ((points <= hull_max) * (points >= hull_min)).min(axis = 1)
    return inside_hull


def construct_sec_order(arr):
    """Intermediate function used for generating second order information"""
    second_order_feature = []
    num_cov_sec = len(arr[0])
    for a in arr:
        tmp = []
        for i in range(num_cov_sec):
            for j in range(i+1, num_cov_sec):
                tmp.append( a[i] * a[j] )
        second_order_feature.append(tmp)
    return np.array(second_order_feature)


def data_generation_dense_mixed_endo(num_samples, num_cont_imp, num_disc_imp,
                                     num_cont_unimp, num_disc_unimp, std=1.5,
                                     t_imp=2, overlap=1, weights=None):
    """Quadratic DGP"""
    def u(x):
        T = []
        for row in x:
            l = scipy.special.expit(np.sum(row[:t_imp]) - t_imp +
                                    np.random.normal(0, overlap))
            t = int(l > 0.5)
            T.append(t)
        return np.array(T)
    # the dgp that we will use. include second order information
    xc = np.random.normal(1, std, size=(num_samples, num_cont_imp))
    xd = np.random.binomial(1, 0.5, size=(num_samples, num_disc_imp))
    x = np.hstack((xc, xd))

    errors_y0 = np.random.normal(0, 1, size=num_samples)
    errors_y1 = np.random.normal(0, 1, size=num_samples)

    num_cov_dense = num_cont_imp + num_disc_imp
    dense_bs_sign = np.random.choice([-1, 1], num_cov_dense)
    dense_bs = [np.random.normal(dense_bs_sign[i]*10, 9) for i in
                range(num_cov_dense)]

    if weights is not None:
        for idx, w in weights:
            dense_bs[idx] = w['control']
    y0_true = np.dot(x, np.array(dense_bs))

    treatment_eff_coef = np.random.normal(1.0, 0.25, size=num_cov_dense)
    if weights is not None:
        for idx, w in weights:
            dense_bs[idx] = 0
            treatment_eff_coef[idx] = w['treated']
    treatment_effect = np.dot(x, treatment_eff_coef)
    second = construct_sec_order(x)
    treatment_eff_sec = np.sum(second, axis=1)
    y1_true = np.dot(x, np.array(dense_bs)) + treatment_effect + \
              treatment_eff_sec

    te = y1_true - y0_true
    y0 = y0_true + errors_y0
    y1 = y1_true + errors_y1
    T = u(x)
    y = T*y1 + (1-T)*y0
    xc2 = np.random.normal(1, std, size=(num_samples, num_cont_unimp))
    xd2 = np.random.binomial(1, 0.5, size=(num_samples, num_disc_unimp))
    x2 = np.hstack((xc2, xd2))
    num_covs_unimportant = num_cont_unimp + num_disc_unimp
    df = pd.DataFrame(np.hstack([x, x2]),
                      columns=list(['X%d'%(j) for j in
                                    range(num_cov_dense + num_covs_unimportant)
                                    ]))
    df['Y'] = y
    df['T'] = T
    binary = ['X%d'%(j) for j in range(num_cont_imp,num_cov_dense)] + \
             ['X%d'%(j) for j in range(
                 num_cov_dense + num_cont_unimp, num_cov_dense +
                                                 num_covs_unimportant)]
    df_true = pd.DataFrame()
    df_true['Y1'] = y1
    df_true['Y0'] = y0
    df_true['TE'] = te
    df_true['Y0_true'] = y0_true
    df_true['Y1_true'] = y1_true
    return df, df_true, binary


def dgp_sine(n_samples, n_unimp, n_imp = 2):
    """Sine DGP"""
    x = np.random.uniform(-np.pi, np.pi, size=(n_samples, n_imp))
    y0 = np.sin(x[:, 0])
    y1 = y0 + np.sin(-x[:, 1])
    y0_errors = np.random.normal(0, 0.1, size=n_samples)
    y1_errors = np.random.normal(0, 0.1, size=n_samples)
    te = y1 - y0
    y0 = y0 + y0_errors
    y1 = y1 + y1_errors
    t = set_t(x, 2, centered=0, overlap=1)
    y = (y0 * (1 - t)) + (y1 * t)
    x_unimp = np.random.uniform(-np.pi, np.pi, size=(n_samples, n_unimp))
    X = np.concatenate([x, x_unimp], axis=1)
    return X, y.reshape(-1, 1), t.reshape(-1, 1), y0.reshape(-1, 1), \
           y1.reshape(-1, 1), te.reshape(-1, 1), \
           (y0 - y0_errors).reshape(-1, 1), (y1 - y1_errors).reshape(-1, 1)


def dgp_exp(n_samples, n_unimp):
    """Exponential DGP"""
    x = np.random.uniform(-3, 3, size=(n_samples, 4))
    y0 = 2*np.exp(x[:, 0]) - np.sum(np.exp(x[:, 1:3]), axis=1)
    y1 = y0 + np.exp(x[:, 3])
    y0_errors = np.random.normal(0, 1, size=n_samples)
    y1_errors = np.random.normal(0, 1, size=n_samples)
    te = y1 - y0
    y0 = y0 + y0_errors
    y1 = y1 + y1_errors
    t = set_t(x, 2, centered=0, overlap=1)
    y = (y0 * (1 - t)) + (y1 * t)
    x_unimp = np.random.uniform(-3, 3, size=(n_samples, n_unimp))
    X = np.concatenate([x, x_unimp], axis=1)
    return X, y.reshape(-1, 1), t.reshape(-1, 1), y0.reshape(-1, 1), \
           y1.reshape(-1, 1), te.reshape(-1, 1), \
           (y0 - y0_errors).reshape(-1, 1), (y1 - y1_errors).reshape(-1, 1)


def dgp_poly_basic(n_samples, n_imp, n_unimp, powers=[2]):
    """Basic Quadratic DGP"""
    x_imp = np.random.normal(0, 2.5, size=(n_samples, n_imp))
    t = np.random.binomial(1, 0.5, size=(n_samples,))
    eff_powers = np.random.choice(powers, size=(n_imp,))
    y0 = np.sum(x_imp ** eff_powers, axis=1)
    y1 = y0 + 10
    y0_errors = np.random.normal(0, 1, size=n_samples)
    y1_errors = np.random.normal(0, 1, size=n_samples)
    te = y1 - y0
    y0 = y0 + y0_errors
    y1 = y1 + y1_errors
    y = (y0 * (1 - t)) + (y1 * t)
    x_unimp = np.random.normal(0, 2.5, size=(n_samples, n_unimp))
    X = np.concatenate([x_imp, x_unimp], axis=1)
    return X, y.reshape(-1, 1), t.reshape(-1, 1), y0.reshape(-1, 1), \
           y1.reshape(-1, 1), te.reshape(-1, 1), \
           (y0 - y0_errors).reshape(-1, 1), (y1 - y1_errors).reshape(-1, 1)


def set_t(x, t_imp, centered=1, overlap=1):
    """Calculate the treatment for each sample given the covariates."""
    T = []
    for row in x:
        l = scipy.special.expit(np.sum(row[:t_imp]) - (t_imp*centered) +
                                np.random.normal(0, overlap))
        t = int(l > 0.5)
        T.append(t)
    return np.array(T)

def dgp_nonlinear_mml(n_samples, n_unimp, coef_seed, data_seed):
    def interact(X, d):
        p = X.shape[1]
        val = np.zeros(X.shape[0])
        for col in range(p):
            for col2 in range(p):
                val += X[:, col] * X[:, col2] * d[col, col2]
        return val
    n_imp = 20
    np.random.seed(coef_seed)

    lambda_vec = np.random.uniform(low = -4, high = 4, size=n_imp)
    beta_lin = lambda_vec
    beta_qua = np.random.uniform(low=0, high = 1, size = n_imp) + lambda_vec
    beta_cos = np.random.uniform(low=0, high = 1, size = n_imp) + lambda_vec
    delta = np.random.uniform(low = 0, high = 1, size = n_imp)
    delta_int = np.random.uniform(low = -.5, high = .5, size = [n_imp, n_imp])
    
    np.random.seed(data_seed)
    X = np.random.normal(loc = 0, scale = 1, size = [n_samples, n_imp])
    X_unimp = np.random.normal(loc = 0, scale = 1, size = [n_samples, n_unimp])
    sigma_i = np.random.uniform(low = 1, high = 2, size = n_samples)
    error = np.random.normal(loc = 0, scale = sigma_i)
    y1 = 5 + 5 + X.dot(beta_lin) + (X**2).dot(beta_qua) + np.cos(X).dot(beta_cos) + X.dot(delta) + interact(X, delta_int) + error
    y0 = 5 + X.dot(beta_lin) + (X**2).dot(beta_qua) + np.cos(X).dot(beta_cos) + error
    u_i = np.random.uniform(low = 0.1, high = 1, size = n_samples)
    te = y1 - y0
    logit_input = u_i * (y1 + y0)/(2 - error)
    prop_score = expit(logit_input)
    t = np.random.binomial(n = 1, p = prop_score)
    y = y1 * t + y0 * (1 - t)

    X = np.hstack([X, X_unimp])

    return X, y.reshape(-1, 1), t.reshape(-1, 1), y0.reshape(-1, 1), \
           y1.reshape(-1, 1), te.reshape(-1, 1), \
           (y0 - error).reshape(-1, 1), (y1 - error).reshape(-1, 1)


def dgp_piecewise_mml(n_samples, n_unimp, coef_seed = 42, data_seed = 42):
    n_imp = 20

    np.random.seed(coef_seed)
    lambda_vec = np.random.uniform(low = -4, high = 4, size=n_imp)
    beta_lin = lambda_vec
    delta = np.random.uniform(low = 0, high = 1, size = n_imp)
    
    np.random.seed(data_seed)
    X = np.random.normal(loc = 0, scale = 1, size = [n_samples, n_imp])
    X_gr_0 = (X > 0)
    X_unimp = np.random.normal(loc = 0, scale = 1, size = [n_samples, n_unimp])
    X_unimp_gr_0 = (X_unimp > 0)
    
    sigma_i = np.random.uniform(low = 1, high = 2, size = n_samples)
    error = np.random.normal(loc = 0, scale = sigma_i)
    y1 = 5 + 5 + X_gr_0.dot(beta_lin) + X_gr_0.dot(delta) + error
    y0 = 5 + X_gr_0.dot(beta_lin) + error
    u_i = np.random.uniform(low = 0.1, high = 1, size = n_samples)
    te = y1 - y0
    logit_input = u_i * (y1 + y0)/(2 - error)
    prop_score = expit(logit_input)
    t = np.random.binomial(n = 1, p = prop_score)
    y = y1 * t + y0 * (1 - t)

    X = np.hstack([X, X_unimp])

    return X, y.reshape(-1, 1), t.reshape(-1, 1), y0.reshape(-1, 1), \
           y1.reshape(-1, 1), te.reshape(-1, 1), \
           (y0 - error).reshape(-1, 1), (y1 - error).reshape(-1, 1)




def add_noise(X, heteroskedasticity):
    if type(X) is not np.ndarray:
        X = np.array(X)
    error = np.random.uniform(-1, 1, size = (X.shape[0], 1))
    if heteroskedasticity:
        error * np.log(X[:, 0]**2 + 1).reshape(-1,1)
    return error

# def dgp_lihua(n_unimp, n_samples, corr, heteroskedasticity, randomized = False):
#     n_imp = 2
#     mean = np.array([0] * (n_imp + n_unimp))
#     cov = np.ones(shape = [n_imp + n_unimp, n_imp + n_unimp]) * corr
#     np.fill_diagonal(cov, np.repeat(a = 1, repeats=n_imp + n_unimp))
#     X = np.random.multivariate_normal(mean, cov, size = n_samples)
#     xmax = 5
#     xmin = -5
#     X[X > xmax] = xmax
#     X[X < xmin] = xmin
    
#     import scipy.stats as stats
#     if not randomized:
#         X_stdized = minmax_scale(X[:, 0], feature_range = (0.25, 0.75))
#         prop = stats.beta.cdf(x = X_stdized, a = 3, b = 3)
#         print('prop min/max', prop.min(), prop.max())
#         T = np.random.binomial(n = 1, p = prop).reshape(-1,1)
#     else:
#         T = np.random.binomial(n = 1, p = 0.5, size = ())
#     def f(X): # bounded by [0,2]
#         return 2/(1 + np.exp(-12 * (X - 0.5)))
#     Y0_true = (np.array([0] * n_samples)).reshape(-1,1)
#     Y1_true = (f(X[:, 0]) * f(X[:, 1])).reshape(-1,1)
#     max_error = 1
#     min_error = -1
    
#     error = add_noise(X, heteroskedasticity=heteroskedasticity)
    
#     if heteroskedasticity:
#         max_error = max_error * np.log(np.max(np.abs([xmax, xmin]))**2 + 1)
#         min_error = min_error * np.log(np.min(np.abs([xmax, xmin]))**2 + 1)
    
#     Y1 = Y1_true + error
#     Y0 = Y0_true + error
    
#     Y = (Y1 * T) + Y0 * (1 - T)
#     TE = Y1_true - Y0_true

#     # ymax = max(y1_true + error) = max(y1_true) + max(error) = max(f(\cdot)) * max(f(\cdot)) + max(error) = 2 * 2 + max(error) 
#     ymax = 4 + max_error
#     # ymin = min(y1_true + error, y0_true + error) = min(sigmoid(\cdot) + error, 0 + error) = min(0 + error, 0 + error) = min(error)
#     ymin = min_error

#     return X, Y, T, Y0, Y1, TE, Y0_true, Y1_true, ymin, ymax



class dgp_linear:
    def __init__(self, xmin, xmax, n_imp, n_unimp, heteroskedastic):
        if n_imp < 2:
            warnings.warn('Warning. n_imp must be >= 2. Setting n_imp = 2.')
        self.n_imp = n_imp
        self.n_unimp = n_unimp
        self.heteroskedastic = heteroskedastic
        self.xmin = xmin
        self.xmax = xmax
        self.coef = np.array(list(range(1, n_imp + 1)) + [0] * n_unimp)
        if not heteroskedastic:
            self.maxerror = 1
            self.minerror = -1
        else:
            self.maxerror = np.log(max(xmax**2, xmin**2) + 1) #max(x^2) = 1 and log is monotonic, so max log(x^2 + 1) = log(max x^2 + 1) = log(2)
            self.minerror = -np.log(max(xmax**2, xmin**2) + 1)
        self.ymax = np.array([2 * xmax] * (n_imp + n_unimp)).dot(self.coef) + self.maxerror
        self.ymin = np.array([2 * xmin] * (n_imp + n_unimp)).dot(self.coef) + self.minerror
    def sample_X(self, n_samples):
        X = np.random.uniform(self.xmin, self.xmax, size = [n_samples, self.n_imp + self.n_unimp])
        x_cols = [f'X_{j}' for j in range(self.n_imp + self.n_unimp)]
        return X, x_cols
    def prop_score(self, X):
        if type(X) is not np.ndarray:
            X = np.array(X)
        return minmax_scale(X[:,0] + 3 * X[:, 1], feature_range=(0.25, 0.75))
    def sample_T(self, prop_score):
        return np.random.binomial(n = 1, p = prop_score)
    def fit(X, T, Y):
        ## useless function...
        return None
    
    def outcome_reg(self, X, T):
        if type(X) is not np.ndarray:
            X = np.array(X)
        Y0_true = X.dot(self.coef).reshape(-1,1)
        Y1_true = 2 * Y0_true + 1
        Y = T * Y1_true + (1 - T) * Y0_true
        return Y
    
    def predict(self, X, T):
        if type(X) is not np.ndarray:
            X = np.array(X)
        Y0_true = X.dot(self.coef).reshape(-1,1)
        Y1_true = 2 * Y0_true + 1
        if T == 1:
            return Y1_true
        else:
            return Y0_true
    def add_noise(self, X):
        if type(X) is not np.ndarray:
            X = np.array(X)
        error = np.random.uniform(-1, 1, size = (X.shape[0], 1))
        if self.heteroskedastic:
            error *= np.log(X[:, 0]**2 + 1).reshape(-1,1)
        return error

    def gen_resampled_dataset(self, X_df):
        x_cols = X_df.columns.tolist()
        X = np.array(X_df)

        prop_score = self.prop_score(X = X)
        T = self.sample_T(prop_score = prop_score).reshape(-1,1)

        ## generate outcomes
        Y1_true = self.outcome_reg(X = X, T = 1)
        Y0_true = self.outcome_reg(X = X, T = 0)
        Y_true = Y1_true * T + Y0_true * (1 - T)
        error = self.add_noise(X = X)
        Y = Y_true + error

        ## set up return dataframe
        return_df = X_df.copy()
        return_df['Y'] = Y.flatten()
        return_df['T'] = T.flatten()
        
        true_df = return_df.copy()
        true_df['Y1_true'] = Y1_true.flatten()
        true_df['Y0_true'] = Y0_true.flatten()
        true_df['cate_true'] = (Y1_true - Y0_true).flatten()
        return return_df, true_df
        
    def gen_raw_dataset(self, n_samples):
        X, x_cols = self.sample_X(n_samples = n_samples)
        X_df = pd.DataFrame(X, columns = x_cols)
        return_df, true_df = self.gen_resampled_dataset(X_df)
        return X_df, return_df, true_df

    def gen_train_est_query(self, n_train, n_est, n_query):
        X_train, train_df, train_true_df = self.gen_raw_dataset(n_samples = n_train)
        X_est, est_df, est_true_df = self.gen_raw_dataset(n_samples = n_est)
        X_query, query_df, query_true_df = self.gen_raw_dataset(n_samples = n_query)

        return X_train, train_df, train_true_df, X_est, est_df, est_true_df, X_query, query_df, query_true_df


class dgp_friedman:
    def __init__(self, xmin, xmax, n_imp, n_unimp, heteroskedastic):
        if n_imp < 5:
            warnings.warn('Warning. n_imp must be >= 2. Setting n_imp = 2.')
        self.n_imp = n_imp
        self.n_unimp = n_unimp
        self.heteroskedastic = heteroskedastic
        self.xmin = xmin
        self.xmax = xmax
        self.coef = np.array(list(range(1, n_imp + 1)) + [0] * n_unimp)
        if not heteroskedastic:
            self.maxerror = 1
            self.minerror = -1
        else:
            self.maxerror = np.log(max(xmax**2, xmin**2) + 1) #max(x^2) = 1 and log is monotonic, so max log(x^2 + 1) = log(max x^2 + 1) = log(2)
            self.minerror = -np.log(max(xmax**2, xmin**2) + 1)
        self.ymax = 10 + 20 * max(self.xmin - 0.5, self.xmax - 0.5)**2 + 10 * self.xmax + 5 * self.xmax + self.xmax + self.maxerror
        self.ymin = -10 - 10 * self.xmin - 5 * self.xmin - self.xmin + self.minerror
    def sample_X(self, n_samples):
        X = np.random.uniform(self.xmin, self.xmax, size = [n_samples, self.n_imp + self.n_unimp])
        x_cols = [f'X_{j}' for j in range(self.n_imp + self.n_unimp)]
        return X, x_cols
    def prop_score(self, X):
        if type(X) is not np.ndarray:
            X = np.array(X)
        return minmax_scale(X[:,0] + 3 * X[:, 1], feature_range=(0.25, 0.75))
    def sample_T(self, prop_score):
        return np.random.binomial(n = 1, p = prop_score)
    def fit(X, T, Y):
        ## useless function...
        return None
    
    def outcome_reg(self, X, T):
        if type(X) is not np.ndarray:
            X = np.array(X)
        #maxY = 10 + 20 * max(xmin - 0.5, xmax - 0.5)**2 + 10 * xmax + 5 * xmax + xmax + maxerror
        #minY = -10 - 10 * xmin - 5 * xmin - xmin + minerror
        Y0_true = 10 * np.sin(np.pi * X[:, 0] * X[:, 1]) + 20 * (X[:, 2] - 0.5)**2 + 10 * X[:, 3] + 5 * X[:, 4]
        Y1_true = Y0_true + X[:, 2] * np.cos(np.pi * X[:, 0] * X[:, 1])
        Y = T * Y1_true + (1 - T) * Y0_true
        return Y
    
    def predict(self, X, T):
        if type(X) is not np.ndarray:
            X = np.array(X)
        if T == 1:
            return self.outcome_reg(X, T == 1)
        else:
            return self.outcome_reg(X, T == 0)
    def add_noise(self, X):
        if type(X) is not np.ndarray:
            X = np.array(X)
        error = np.random.uniform(-1, 1, size = (X.shape[0], 1))
        if self.heteroskedastic:
            error *= np.log(X[:, 0]**2 + 1).reshape(-1,1)
        return error

    def gen_resampled_dataset(self, X_df):
        x_cols = X_df.columns.tolist()
        X = np.array(X_df)

        prop_score = self.prop_score(X = X)
        T = self.sample_T(prop_score = prop_score).reshape(-1,1)

        ## generate outcomes
        Y1_true = self.outcome_reg(X = X, T = 1)
        Y0_true = self.outcome_reg(X = X, T = 0)
        Y_true = Y1_true * T + Y0_true * (1 - T)
        error = self.add_noise(X = X)
        Y = Y_true + error

        ## set up return dataframe
        return_df = X_df.copy()
        return_df['Y'] = Y.flatten()
        return_df['T'] = T.flatten()
        
        true_df = return_df.copy()
        true_df['Y1_true'] = Y1_true.flatten()
        true_df['Y0_true'] = Y0_true.flatten()
        true_df['cate_true'] = (Y1_true - Y0_true).flatten()
        return return_df, true_df
        
    def gen_raw_dataset(self, n_samples):
        X, x_cols = self.sample_X(n_samples = n_samples)
        X_df = pd.DataFrame(X, columns = x_cols)
        return_df, true_df = self.gen_resampled_dataset(X_df)
        return X_df, return_df, true_df

    def gen_train_est_query(self, n_train, n_est, n_query):
        X_train, train_df, train_true_df = self.gen_raw_dataset(n_samples = n_train)
        X_est, est_df, est_true_df = self.gen_raw_dataset(n_samples = n_est)
        X_query, query_df, query_true_df = self.gen_raw_dataset(n_samples = n_query)

        return X_train, train_df, train_true_df, X_est, est_df, est_true_df, X_query, query_df, query_true_df

class dgp_lihua:
    def __init__(self, xmin, xmax, n_imp, n_unimp, heteroskedastic, corr = False):
        if n_imp < 2:
            warnings.warn('Warning. n_imp must be >= 2. Setting n_imp = 2.')
            n_imp = 2
        self.n_imp = n_imp
        self.n_unimp = n_unimp
        self.heteroskedastic = heteroskedastic
        self.xmin = xmin
        self.xmax = xmax
        self.coef = np.array(list(range(1, n_imp + 1)) + [0] * n_unimp)
        self.corr = corr
        if not heteroskedastic:
            self.maxerror = 1
            self.minerror = -1
        else:
            self.maxerror = np.log(max(xmax**2, xmin**2) + 1) #max(x^2) = 1 and log is monotonic, so max log(x^2 + 1) = log(max x^2 + 1) = log(2)
            self.minerror = -np.log(max(xmax**2, xmin**2) + 1)
        self.ymax = 4 + self.maxerror
        self.ymin = self.minerror
    def sample_X(self, n_samples):
        if self.corr:
            cov = np.ones(shape = [self.n_imp + self.n_unimp, self.n_imp + self.n_unimp]) * 0.9
            np.fill_diagonal(cov, 1)
            mean = np.array([0] * (self.n_imp + self.n_unimp))
            np.fill_diagonal(cov, np.repeat(a = 1, repeats=self.n_imp + self.n_unimp))
            X = np.random.multivariate_normal(mean, cov, size = n_samples)
            X[X > self.xmax] = self.xmax
            X[X < self.xmin] = self.xmin
        else:
            # cov = np.zeros(shape = [self.n_imp + self.n_unimp, self.n_imp + self.n_unimp])
            # np.fill_diagonal(cov, 1)
            X = np.random.uniform(self.xmin, self.xmax, size = [n_samples, self.n_imp + self.n_unimp])
        x_cols = [f'X_{j}' for j in range(self.n_imp + self.n_unimp)]
        return X, x_cols
    def prop_score(self, X):
        if type(X) is not np.ndarray:
            X = np.array(X)
        return minmax_scale(X[:,0] + 3 * X[:, 1], feature_range=(0.25, 0.75))
    def sample_T(self, prop_score):
        return np.random.binomial(n = 1, p = prop_score)
    def fit(X, T, Y):
        ## useless function...
        return None
    
    def outcome_reg(self, X, T):
        if type(X) is not np.ndarray:
            X = np.array(X)
        def f(X): # bounded by [0,2]
            return 2/(1 + np.exp(-12 * (X - 0.5)))
        Y0_true = (np.array([0] * X.shape[0])).reshape(-1,1)
        Y1_true = (f(X[:, 0]) * f(X[:, 1])).reshape(-1,1)
        Y = Y1_true * T + Y0_true * (1 - T)
        return Y.reshape(-1,1)
    
    def predict(self, X, T):
        if type(X) is not np.ndarray:
            X = np.array(X)
        if T == 1:
            return self.outcome_reg(X, T == 1)
        else:
            return self.outcome_reg(X, T == 0)
    def add_noise(self, X):
        if type(X) is not np.ndarray:
            X = np.array(X)
        error = np.random.uniform(-1, 1, size = (X.shape[0], 1))
        if self.heteroskedastic:
            error *= np.log(X[:, 0]**2 + 1).reshape(-1,1)
        return error

    def gen_resampled_dataset(self, X_df):
        x_cols = X_df.columns.tolist()
        X = np.array(X_df)

        prop_score = self.prop_score(X = X)
        T = self.sample_T(prop_score = prop_score).reshape(-1,1)

        ## generate outcomes
        Y1_true = self.outcome_reg(X = X, T = 1)
        Y0_true = self.outcome_reg(X = X, T = 0)
        Y_true = Y1_true * T + Y0_true * (1 - T)
        error = self.add_noise(X = X).reshape(-1,1)
        Y = Y_true + error

        ## set up return dataframe
        return_df = X_df.copy()
        return_df['Y'] = Y.flatten()
        return_df['T'] = T.flatten()
        
        true_df = return_df.copy()
        true_df['Y1_true'] = Y1_true.flatten()
        true_df['Y0_true'] = Y0_true.flatten()
        true_df['cate_true'] = (Y1_true - Y0_true).flatten()
        return return_df, true_df
        
    def gen_raw_dataset(self, n_samples):
        X, x_cols = self.sample_X(n_samples = n_samples)
        X_df = pd.DataFrame(X, columns = x_cols)
        return_df, true_df = self.gen_resampled_dataset(X_df)
        return X_df, return_df, true_df

    def gen_train_est_query(self, n_train, n_est, n_query):
        X_train, train_df, train_true_df = self.gen_raw_dataset(n_samples = n_train)
        X_est, est_df, est_true_df = self.gen_raw_dataset(n_samples = n_est)
        X_query, query_df, query_true_df = self.gen_raw_dataset(n_samples = n_query)

        return X_train, train_df, train_true_df, X_est, est_df, est_true_df, X_query, query_df, query_true_df

    def gen_group_ate(self, hull, n_samples=10000, max_iter=500):
        """
        Generate the average treatment effect (ATE) for a group of samples within a convex hull.

        Parameters:
        hull (numpy array): Matched group that we will use as group
        n_samples (int): The number of samples to generate. Default is 10,000.
        max_iter (int): The maximum number of iterations to attempt generating samples. Default is 500.

        Returns:
        float: The average treatment effect (ATE) for the group of samples within the convex hull.
        """
        gate_final = []  # List to store the final samples
        n_final = 0  # Counter for the number of final samples
        n_iter = 0  # Counter for the number of iterations

        # Loop until the desired number of samples is reached or the maximum iterations are exceeded
        while n_final <= n_samples and n_iter <= max_iter:
            print(n_iter, n_final)
            # Generate samples
            X, x_cols = self.sample_X(n_samples= 5 * n_samples)
            # Filter samples that are inside the convex hull
            X = X[is_point_in_hull(hull, X), ]
            if X.shape[0] > 0:
                # Append the filtered samples to the final list
                X_df = pd.DataFrame(X, columns=x_cols)
                _, true_df = self.gen_resampled_dataset(X_df)
                gate_final.append(true_df['cate_true'])
                # Update the counter for the number of final samples
                n_final += X.shape[0]
            # Increment the iteration counter
            n_iter += 1

        # Concatenate all the final samples
        gate_final = np.concatenate(gate_final, axis=0)
        gate_final = np.mean(gate_final)
        print('gate final:', gate_final)
        # Return the mean of the true conditional average treatment effect (CATE)
        return gate_final

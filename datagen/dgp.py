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

np.random.seed(0)


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


def dgp_sine(n_samples, n_unimp):
    """Sine DGP"""
    x = np.random.uniform(-np.pi, np.pi, size=(n_samples, 2))
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

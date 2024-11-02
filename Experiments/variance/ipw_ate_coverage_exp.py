import sys
sys.path.append('/usr/project/xtmp/sk787/variable_imp_matching/')
sys.path.insert(0, '/usr/project/xtmp/sk787/variable_imp_matching/')

import numpy as np
import pandas as pd
from scipy.sparse import lil_matrix
from sklearn.neighbors import NearestNeighbors, KNeighborsRegressor
from datagen.dgp_df import dgp_df

from confseq.betting import betting_ci
from sklearn.preprocessing import minmax_scale
from sklearn.ensemble import RandomForestRegressor

import warnings

from Experiments.variance.randomization_coverage_exp import knn_match
from datagen.dgp import dgp_linear, dgp_lihua

import importlib
importlib.reload(sys.modules['Experiments.variance.randomization_coverage_exp'])
from Experiments.variance.randomization_coverage_exp import knn_match


n_train = 1000
n_est = 50000
n_query = 1
train_query_seed = 42
xmin = -5
xmax = 5
n_imp = 2
n_unimp = 0
heteroskedastic = 0
corr = False

k = n_est

gen_linear_dgp = dgp_lihua(xmin = xmin, xmax = xmax, n_imp = n_imp, n_unimp=n_unimp, heteroskedastic=heteroskedastic, corr = corr)

np.random.seed(train_query_seed)
X_train, train_df, train_true_df, X_est, est_df, est_true_df, X_query, query_df, query_true_df = gen_linear_dgp.gen_train_est_query(n_train = n_train, n_est = n_est, n_query = n_query)

n_iter = 100
ub_list = []
lb_list = []

ub_clt_list = []
lb_clt_list = []

ub_hoeff_list = []
lb_hoeff_list = []

ub_bern_list = []
lb_bern_list = []

ate_est_list = []

ate_list = []
coverage = 0
coverage_clt = 0
for iter in range(n_iter):
    print(iter)
    seed = 42 * iter
    np.random.seed(seed)

    ## resample X and T
    df_iter, df_true_iter = gen_linear_dgp.gen_resampled_dataset(X_est)

    X_est_iter = X_est.copy()
    T_est_iter = df_iter['T'].values
    Y_est_iter = df_iter['Y'].values

    ## estimate confidence intervals on this iteration
    knn_match_iter = knn_match(prop_score = gen_linear_dgp.prop_score, outcome_reg = lambda X, T : np.zeros(X.shape[0]), ymin = gen_linear_dgp.ymin, ymax = gen_linear_dgp.ymax)
    knn_match_iter.learn_projection(X_train, train_df['T'].values, train_df['Y'].values, algorithm = 'true_coef', args = {'coef' : np.array([1,1])})# args = {'coef' : gen_linear_dgp.coef})
    # knn_match_iter.learn_projection(X_train, train_df['T'].values, train_df['Y'].values, algorithm = 'true_prog')
    knn_match_iter.fit(X_est_iter, T_est_iter, Y_est_iter, k = k)

    lb_iter, ub_iter, bias_iter = knn_match_iter.est_cate_conf_int(X_query, alpha = 0.05, return_bias = True)
    lb_iter = lb_iter - bias_iter
    ub_iter = ub_iter - bias_iter

    lb_clt_iter, ub_clt_iter, bias_iter = knn_match_iter.est_cate_conf_int_clt(X_query, alpha = 0.05, return_bias = True)
    lb_hoeff_iter, ub_hoeff_iter, bias_hoeff_iter = knn_match_iter.est_cate_conf_int_hoeff(X_query, alpha = 0.05, return_bias = True)
    lb_bern_iter, ub_bern_iter, bias_hoeff_iter = knn_match_iter.est_cate_conf_int_bern(X_query, alpha = 0.05, return_bias = True)

    cate_mg = knn_match_iter.est_cate_mg_or(X_query)
    cate_est = knn_match_iter.est_cate(X_query)

    ate_true = (gen_linear_dgp.outcome_reg(X_est, T = 1) - gen_linear_dgp.outcome_reg(X_est, T = 0)).mean()

    coverage += (lb_iter <= ate_true) * (ate_true <= ub_iter)
    coverage_clt += (lb_clt_iter <= ate_true) * (ate_true <= ub_clt_iter)

    lb_list.append(lb_iter[0])
    ub_list.append(ub_iter[0])
    ate_list.append(ate_true)

    lb_clt_list.append(lb_clt_iter[0])
    ub_clt_list.append(ub_clt_iter[0])
    ate_est_list.append(cate_est)

    lb_hoeff_list.append(lb_hoeff_iter[0])
    ub_hoeff_list.append(ub_hoeff_iter[0])

    lb_bern_list.append(lb_bern_iter[0])
    ub_bern_list.append(ub_bern_iter[0])

    print('bias', bias_iter)
    print('coverage', coverage/(iter + 1))
    print('coverage clt', coverage_clt/(iter + 1))

import matplotlib.pyplot as plt
# Plot lb_iter and ub_iter
plt.plot(lb_list, label='Betting Bound', color='blue')
plt.plot(ub_list, color='blue')

plt.plot(lb_bern_list, label='Emp Bern Bound', color='pink')
plt.plot(ub_bern_list, color='pink')


plt.plot(lb_clt_list, label='CLT Bound', color='red')
plt.plot(ub_clt_list, color='red')

plt.plot(lb_hoeff_list, label='Hoeffding Bound', color='orange')
plt.plot(ub_hoeff_list, color='orange')

plt.plot(ate_est_list, label = 'ATE Est', color = 'black')

# Add a dotted line for ate_true
plt.axhline(y=ate_true, color='green', linestyle='--', label='ATE True')

# Add labels and title
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Plot of lb_iter and ub_iter with ATE True Line')
plt.legend()

# Save the plot
plt.savefig('./trash_ipw_ate.png')
plt.close()  # Close the figure after saving to avoid displaying it in notebooks or interactive environments
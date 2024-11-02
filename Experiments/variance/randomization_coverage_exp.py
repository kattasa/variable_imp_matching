import numpy as np
import pandas as pd
from scipy.sparse import lil_matrix
from sklearn.neighbors import NearestNeighbors, KNeighborsRegressor
from datagen.dgp import dgp_linear, dgp_lihua

from confseq.betting import betting_ci
from confseq.predmix import predmix_empbern_twosided_cs
from scipy.special import ndtri
from sklearn.preprocessing import minmax_scale
from sklearn.ensemble import RandomForestRegressor
from econml.dml import CausalForestDML
from econml.metalearners import XLearner

from utils import save_df_to_csv

import warnings
from argparse import ArgumentParser
    
def normalize_samples(samples, min, max):
    return (samples - min)/(max - min)

def unnormalize_samples(samples, min, max):
    return samples * (max - min) + min

def get_k_neighbors_graph(query_df, estimation_df, k, distance_metric, tree_type='kd_tree'):
    """
    Creates a K-neighbors graph using KD-Tree or Ball Tree, represented as a sparse adjacency matrix.

    Parameters:
    query_df (pd.DataFrame): Query dataframe containing covariates of interest.
    estimation_df (pd.DataFrame): Estimation dataframe containing covariates and outcomes.
    k (int): Number of neighbors.
    distance_metric (str): Distance metric to use (e.g., 'euclidean', 'manhattan', etc.).
    tree_type (str): Type of tree structure to use ('kd_tree' or 'ball_tree').

    Returns:
    sparse_matrix: Sparse adjacency matrix representing K-neighbors.
    """
    # Convert dataframes to numpy arrays
    query_data = query_df.values
    estimation_data = estimation_df.values

    # Choose the appropriate tree algorithm
    algorithm = 'auto'  # Use 'auto' for automatic selection, or 'kd_tree'/'ball_tree' explicitly
    if tree_type == 'kd_tree':
        algorithm = 'kd_tree'
    elif tree_type == 'ball_tree':
        algorithm = 'ball_tree'
    
    # Initialize the NearestNeighbors model
    nbrs = NearestNeighbors(n_neighbors=k, algorithm=algorithm, metric=distance_metric)
    
    # Fit the model on the estimation data
    nbrs.fit(estimation_data)
    
    # Find the K-nearest neighbors for each query point
    distances, indices = nbrs.kneighbors(query_data)

    # Create the sparse adjacency matrix
    num_query = query_data.shape[0]
    num_estimation = estimation_data.shape[0]
    sparse_matrix = lil_matrix((num_query, num_estimation))
    
    # Populate the sparse matrix with 1's at the positions of the neighbors
    for i, neighbors in enumerate(indices):
        sparse_matrix[i, neighbors] = 1

    return sparse_matrix


class knn_match:
    def __init__(self, prop_score, outcome_reg, ymin, ymax, bias_correction = True, distance_metric = 'euclidean'):
        self.outcome_reg = outcome_reg
        self.prop_score = prop_score
        self.distance_metric = distance_metric
        self.ymin = ymin
        self.ymax = ymax
        self.projection = lambda x: x
        self.bias_correction = bias_correction
    # def learn_dist_metric(self, X_train, T_train, Y_train, algorithm = RandomForestRegressor):
    #     X_train_T = X_train.loc[T_train == 1, ]
    #     Y_train_T = Y_train[T_train == 1]

    #     X_train_C = X_train.loc[T_train == 0, ]
    #     Y_train_C = Y_train[T_train == 0]

    #     algorithm_T = algorithm.fit(X_train_T, Y_train_T)
    #     algorithm_C = algorithm.fit(X_train_C, Y_train_C)

    def learn_projection(self, X_train, T_train, Y_train, args = {}, algorithm = 'rf_prognostic_score'):
        if algorithm == 'rf_prognostic_score':
            
            # split training data into treated group and control group
            X_train_T = X_train.loc[T_train == 1, ]
            Y_train_T = Y_train[T_train == 1]

            X_train_C = X_train.loc[T_train == 0, ]
            Y_train_C = Y_train[T_train == 0]

            # fit prognostic score on treated data if there are more treated than control obs
            rf = RandomForestRegressor()
            if X_train_T.shape[0] > X_train_C.shape[0]:    
                rf.fit(X_train_T, Y_train_T)
            else:
                # fit prognostic score on control data
                rf.fit(X_train_C, Y_train_C)

            # define prognostic score average function
            self.projection = lambda x: (rf.predict(x) + rf.predict(x)).reshape(-1,1)

        elif algorithm == 'true_coef':
            coef = args['coef']
            self.projection = lambda x: x * np.array(coef)
        elif algorithm == 'true_prog':
            self.projection = lambda x: self.outcome_reg(x, T = 1)
        else:
            warnings.warn('Warning: did not recognize algorithm for learning projection. Default to no projection...')

    def fit(self, X_est, T_est, Y_est, k):
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto', metric=self.distance_metric)
        nbrs.fit(self.projection(X_est)) ## use projection function to project estimation set into a different space

        if self.prop_score is None:
            prop_score = KNeighborsRegressor(n_neighbors = k, algorithm = 'brute', metric = self.distance_metric)
            prop_score.fit(self.projection(X_est), T_est)
            self.prop_score = lambda x: prop_score.predict(self.projection(x))
        
        self.nbrs = nbrs
        self.n_est = X_est.shape[0]
        self.X_est = X_est
        self.T_est = T_est
        self.Y_est = Y_est
        self.prop_score_est = self.prop_score(X_est)
        self.k = k
        self.mu_x_T = self.outcome_reg(X_est, T = 1)
        self.mu_x_C = self.outcome_reg(X_est, T = 0)
    def find_nn(self, X_query):
        distaces, indices = self.nbrs.kneighbors(self.projection(X_query)) ## use projection function to project query set into different space
        return indices
    def est_cate(self, X_query):
        indices = self.find_nn(X_query)
        # get treatment and outcome of nearest neighbors
        T_nn = self.T_est[indices]
        Y_nn = self.Y_est[indices]

        # estimate CATE
        cate_est = ((2 * T_nn - 1) * Y_nn).mean(axis = 1)
        
        return cate_est

    def est_cate_mg_or(self, X_query):
        indices = self.find_nn(X_query)
        # get treatment and outcome of nearest neighbors
        mu_nn_T = self.mu_x_T[indices].mean(axis = 1)
        mu_nn_C = self.mu_x_C[indices].mean(axis = 1)

        return mu_nn_T - mu_nn_C
    
    def make_bernstein_ci(self, samples, min, max, alpha):
        n = len(samples)
        if n < 2:
            raise ValueError("Sample size must be at least 2 to compute variance.")
        
        M = np.max([np.abs(min), np.abs(max)])
        # Compute sample mean and sample variance
        sample_mean = np.mean(samples)
        sample_variance = np.var(samples, ddof=1)  # ddof=1 for unbiased variance
        
        # Compute the confidence bound components
        confidence_radius = np.sqrt((2 * sample_variance * np.log(2 / alpha)) / n) + (3 * M * np.log(2 / alpha) / n)
        
        # Confidence interval
        lower_bound = sample_mean - confidence_radius
        upper_bound = sample_mean + confidence_radius
        
        return lower_bound, upper_bound

    
    
    def make_betting_ci(self, samples, alpha, min, max):   
        import numpy as np

        samples_normalized = normalize_samples(samples, min, max) # (samples - y_min)/(y_max - y_min)

        # print('normalized min, max:', samples_normalized.min(), samples_normalized.max())

        # betting_cs_result = betting_ci(samples_normalized, alpha = alpha)
        betting_cs_result = predmix_empbern_twosided_cs(samples_normalized, alpha = alpha)
        # betting_cs_result = empirical_bernstein_ci(samples_normalized, max = 1, min = 0, alpha = alpha)
        betting_lb = betting_cs_result[0]
        betting_ub = betting_cs_result[1]
        if type(betting_lb) is np.ndarray:
            betting_lb = betting_lb[-1]
            betting_ub = betting_ub[-1]

        betting_lb = unnormalize_samples(betting_lb, min, max) # betting_lb * (y_max - y_min) + y_min
        betting_ub = unnormalize_samples(betting_ub, min, max) # betting_ub * (y_max - y_min) + y_min
        
        return betting_lb, betting_ub
    def make_clt_ci(self, samples, alpha):
        mean = np.mean(samples)
        std_err = np.std(samples) / np.sqrt(len(samples))
        z_score =  ndtri(1 - alpha/2)
        ci_lower = mean - z_score * std_err
        ci_upper = mean + z_score * std_err
        return ci_lower, ci_upper
    
    def make_hoeffding_ci(self, samples, min, max, alpha):
        # Hoeffding's inequality
        samples_normalized = normalize_samples(samples, min, max)
        n = len(samples_normalized)
        mean = np.mean(samples_normalized)
        ci_half_width = np.sqrt(np.log(2 / alpha) / (2 * n))
        ci_lower = mean - ci_half_width
        ci_upper = mean + ci_half_width
        ci_lower = unnormalize_samples(ci_lower, min, max)
        ci_upper = unnormalize_samples(ci_upper, min, max)
        return ci_lower, ci_upper
    
    def est_bias(self, X_query):
        mu_query_T = self.outcome_reg(X_query, T = 1)
        mu_query_C = self.outcome_reg(X_query, T = 0)


        ## get neighbors
        indices = self.find_nn(X_query)
        mu_nn_T = self.mu_x_T[indices].mean(axis = 1)
        mu_nn_C = self.mu_x_C[indices].mean(axis = 1)

        ## estimate bias
        # bias = mu_nn_C - mu_query_C - mu_nn_T + mu_query_T
        bias = mu_query_T - mu_nn_T - mu_query_C + mu_nn_C

        return bias
    def est_cate(self, X_query):
        indices = self.find_nn(X_query)
        # get treatment and outcome of nearest neighbors
        T_nn = self.T_est[indices]
        Y_nn = self.Y_est[indices]
        prop_nn = self.prop_score_est[indices]
        weight_nn = 1/prop_nn * T_nn + 1/(1 - prop_nn) * (1 - T_nn)
        Yunweighted_nn = (2 * T_nn - 1) * Y_nn
        Ynormalized_nn = Yunweighted_nn * weight_nn

        Ynormalized_nn_mean = Ynormalized_nn.mean(axis = 1)
        bias = self.est_bias(X_query).flatten()

        return Ynormalized_nn_mean + bias
    def est_cate_conf_int_clt(self, X_query, alpha = 0.05, return_bias = False):
        indices = self.find_nn(X_query)
        # get treatment and outcome of nearest neighbors
        T_nn = self.T_est[indices]
        Y_nn = self.Y_est[indices]
        prop_nn = self.prop_score_est[indices]
        weight_nn = 1/prop_nn * T_nn + 1/(1 - prop_nn) * (1 - T_nn)
        Yunweighted_nn = (2 * T_nn - 1) * Y_nn
        Ynormalized_nn = Yunweighted_nn * weight_nn

        lb = []
        ub = []
        for i_query in range(indices.shape[0]):
            
            lb_i, ub_i = self.make_clt_ci(samples = Ynormalized_nn[i_query, :], alpha = alpha)
            lb.append(lb_i)
            ub.append(ub_i)
        if self.bias_correction:
            bias = self.est_bias(X_query)
            lb = np.array(lb).reshape(bias.shape)
            ub = np.array(ub).reshape(bias.shape)
            if return_bias:
                return lb + bias, ub + bias, bias
            else:
                return lb + bias, ub + bias
        else:
            if return_bias:
                warning('Warning: bias_correction set to False. Cannot return bias. Returning 0 vector')
                return lb, ub, np.zeros(lb.shape)
            return lb, ub
        

    def est_cate_conf_int_hoeff(self, X_query, alpha = 0.05, return_bias = False):
        indices = self.find_nn(X_query)
        # get treatment and outcome of nearest neighbors
        T_nn = self.T_est[indices]
        Y_nn = self.Y_est[indices]
        prop_nn = self.prop_score_est[indices]
        weight_nn = 1/prop_nn * T_nn + 1/(1 - prop_nn) * (1 - T_nn)
        Yunweighted_nn = (2 * T_nn - 1) * Y_nn
        Ynormalized_nn = Yunweighted_nn * weight_nn

        # get \max_{j \in MG(x_i)}(2t_j - 1){ymax, ymin}
        ymin_nn = (2 * T_nn - 1) * self.ymin
        ymax_nn = (2 * T_nn - 1) * self.ymax
        ysumm_nn = np.hstack([ymin_nn, ymax_nn])
        ymax_mg = ysumm_nn.max(axis = 1)
        ymin_mg = ysumm_nn.min(axis = 1)

        # to get \max/\min_{j \in MG(x_i)}(2t_j - 1) * y * weight(x_j, t_j) --> multiply weight(x_j, t_j) with ymax_mg/ymin_mg
    
        # get min/max of weights within each MG
        weight_min = weight_nn.min(axis = 1)
        weight_max = weight_nn.max(axis = 1)

        # weight_max = 1/(0.25)
        # weight_min = 1/(0.75)
        # warnings.warn('weight_max and weight_min are being hard coded currently. remember to update this.')


        yweightmax_mg = np.hstack([(weight_max * ymax_mg).reshape(-1,1), (weight_min * ymax_mg).reshape(-1,1), (weight_max * ymin_mg).reshape(-1,1), (weight_min * ymin_mg).reshape(-1,1)]).max(axis = 1)
        yweightmin_mg = np.hstack([(weight_max * ymax_mg).reshape(-1,1), (weight_min * ymax_mg).reshape(-1,1), (weight_max * ymin_mg).reshape(-1,1), (weight_min * ymin_mg).reshape(-1,1)]).min(axis = 1)

        lb = []
        ub = []
        for i_query in range(indices.shape[0]):
            
            lb_i, ub_i = self.make_hoeffding_ci(samples = Ynormalized_nn[i_query, :], alpha = alpha, min = yweightmin_mg, max = yweightmax_mg)
            lb.append(lb_i)
            ub.append(ub_i)
        if self.bias_correction:
            bias = self.est_bias(X_query)
            lb = np.array(lb).reshape(bias.shape)
            ub = np.array(ub).reshape(bias.shape)
            if return_bias:
                return lb + bias, ub + bias, bias
            else:
                return lb + bias, ub + bias
        else:
            if return_bias:
                warning('Warning: bias_correction set to False. Cannot return bias. Returning 0 vector')
                return lb, ub, np.zeros(lb.shape)
            return lb, ub
        

    def est_cate_conf_int_bern(self, X_query, alpha = 0.05, return_bias = False):
        indices = self.find_nn(X_query)
        # get treatment and outcome of nearest neighbors
        T_nn = self.T_est[indices]
        Y_nn = self.Y_est[indices]
        prop_nn = self.prop_score_est[indices]
        weight_nn = 1/prop_nn * T_nn + 1/(1 - prop_nn) * (1 - T_nn)
        Yunweighted_nn = (2 * T_nn - 1) * Y_nn
        Ynormalized_nn = Yunweighted_nn * weight_nn

        # get \max_{j \in MG(x_i)}(2t_j - 1){ymax, ymin}
        ymin_nn = (2 * T_nn - 1) * self.ymin
        ymax_nn = (2 * T_nn - 1) * self.ymax
        ysumm_nn = np.hstack([ymin_nn, ymax_nn])
        ymax_mg = ysumm_nn.max(axis = 1)
        ymin_mg = ysumm_nn.min(axis = 1)

        # to get \max/\min_{j \in MG(x_i)}(2t_j - 1) * y * weight(x_j, t_j) --> multiply weight(x_j, t_j) with ymax_mg/ymin_mg
    
        # get min/max of weights within each MG
        # weight_min = weight_nn.min(axis = 1)
        # weight_max = weight_nn.max(axis = 1)

        weight_max = 1/(0.25)
        weight_min = 1/(0.75)
        warnings.warn('weight_max and weight_min are being hard coded currently. remember to update this.')


        yweightmax_mg = np.hstack([(weight_max * ymax_mg).reshape(-1,1), (weight_min * ymax_mg).reshape(-1,1), (weight_max * ymin_mg).reshape(-1,1), (weight_min * ymin_mg).reshape(-1,1)]).max(axis = 1)
        yweightmin_mg = np.hstack([(weight_max * ymax_mg).reshape(-1,1), (weight_min * ymax_mg).reshape(-1,1), (weight_max * ymin_mg).reshape(-1,1), (weight_min * ymin_mg).reshape(-1,1)]).min(axis = 1)

        lb = []
        ub = []
        for i_query in range(indices.shape[0]):
            
            lb_i, ub_i = self.make_bernstein_ci(samples = Ynormalized_nn[i_query, :], alpha = alpha, min = yweightmin_mg, max = yweightmax_mg)
            lb.append(lb_i)
            ub.append(ub_i)
        if self.bias_correction:
            bias = self.est_bias(X_query)
            lb = np.array(lb).reshape(bias.shape)
            ub = np.array(ub).reshape(bias.shape)
            if return_bias:
                return lb + bias, ub + bias, bias
            else:
                return lb + bias, ub + bias
        else:
            if return_bias:
                warning('Warning: bias_correction set to False. Cannot return bias. Returning 0 vector')
                return lb, ub, np.zeros(lb.shape)
            return lb, ub
    
    def est_cate_conf_int(self, X_query, alpha = 0.05, return_bias = False):
        # return self.est_cate_conf_int_normal(X_query, alpha, return_bias)
        indices = self.find_nn(X_query)
        # get treatment and outcome of nearest neighbors
        T_nn = self.T_est[indices]
        Y_nn = self.Y_est[indices]
        prop_nn = self.prop_score_est[indices]
        weight_nn = 1/prop_nn * T_nn + 1/(1 - prop_nn) * (1 - T_nn)
        Yunweighted_nn = (2 * T_nn - 1) * Y_nn
        Ynormalized_nn = Yunweighted_nn * weight_nn

        # get \max_{j \in MG(x_i)}(2t_j - 1){ymax, ymin}
        ymin_nn = (2 * T_nn - 1) * self.ymin
        ymax_nn = (2 * T_nn - 1) * self.ymax
        ysumm_nn = np.hstack([ymin_nn, ymax_nn])
        ymax_mg = ysumm_nn.max(axis = 1)
        ymin_mg = ysumm_nn.min(axis = 1)

        # to get \max/\min_{j \in MG(x_i)}(2t_j - 1) * y * weight(x_j, t_j) --> multiply weight(x_j, t_j) with ymax_mg/ymin_mg
    
        # get min/max of weights within each MG
        weight_min = weight_nn.min(axis = 1)
        weight_max = weight_nn.max(axis = 1)

        # weight_max = 1/(0.25)
        # weight_min = 1/(0.75)
        # warnings.warn('weight_max and weight_min are being hard coded currently. remember to update this.')


        yweightmax_mg = np.hstack([(weight_max * ymax_mg).reshape(-1,1), (weight_min * ymax_mg).reshape(-1,1), (weight_max * ymin_mg).reshape(-1,1), (weight_min * ymin_mg).reshape(-1,1)]).max(axis = 1)
        yweightmin_mg = np.hstack([(weight_max * ymax_mg).reshape(-1,1), (weight_min * ymax_mg).reshape(-1,1), (weight_max * ymin_mg).reshape(-1,1), (weight_min * ymin_mg).reshape(-1,1)]).min(axis = 1)

        lb = []
        ub = []
        for i_query in range(indices.shape[0]):
            
            try:
                lb_i, ub_i = self.make_betting_ci(samples = Ynormalized_nn[i_query, :], alpha = alpha, min = yweightmin_mg[i_query], max = yweightmax_mg[i_query])
            except:
                print('Ynormalized_nn[i_query, :]', Ynormalized_nn[i_query, :])
                print('yweightmin_mg[i_query]', yweightmin_mg[i_query])
                print('yweightmax_mg[i_query]', yweightmax_mg[i_query])
                print('weight_max[i_query]', weight_max[i_query])
                print('weight_min[i_query]', weight_min[i_query])
                print('self.ymin', self.ymin)
                print('self.ymax', self.ymax)
                print('ymin_mg[i_query]', ymin_mg[i_query])
                print('ymax_mg[i_query]', ymax_mg[i_query])
                print('Yunweighted_nn[i_query, :].min()', Yunweighted_nn[i_query, :].min())
                print('Yunweighted_nn[i_query, :].max()', Yunweighted_nn[i_query, :].max())
                raise ValueError('oops. something broke here...')
            lb.append(lb_i)
            ub.append(ub_i)
        if self.bias_correction:
            bias = self.est_bias(X_query)
            lb = np.array(lb).reshape(bias.shape)
            ub = np.array(ub).reshape(bias.shape)
            if return_bias:
                return lb + bias, ub + bias, bias
            else:
                return lb + bias, ub + bias
        else:
            if return_bias:
                warning('Warning: bias_correction set to False. Cannot return bias. Returning 0 vector')
                return lb, ub, np.zeros(lb.shape)
            return lb, ub

    
    def calc_group_ate(self, X_query, dgp):
        indices = self.find_nn(X_query)
        gates = []
        for i in range(X_query.shape[0]):
            mg = self.X_est.loc[indices[i], :].values
            gate_i = dgp.gen_group_ate(hull = mg)
            gates.append(gate_i)
        return gates

def knn_match_true_prop_or(dgp, X_train, T_train, Y_train, X_est_iter, T_est_iter, Y_est_iter, projection_algorithm, k, X_query, var_est = 'betting'):

    knn_match_iter = knn_match(prop_score = dgp.prop_score, outcome_reg = dgp.outcome_reg, ymin = dgp.ymin, ymax = dgp.ymax)
    ## fit projection on training set
    knn_match_iter.learn_projection(X_train, T_train, Y_train, algorithm = projection_algorithm, args = {'coef' : dgp.coef})
    ## estimate intervals on estimation set
    knn_match_iter.fit(X_est_iter, T_est_iter, Y_est_iter, k = k)

    if var_est == 'betting':
        lb_iter, ub_iter = knn_match_iter.est_cate_conf_int(X_query)
    elif var_est == 'clt':
        lb_iter, ub_iter = knn_match_iter.est_cate_conf_int_clt(X_query)
    elif var_est == 'bern':
        lb_iter, ub_iter = knn_match_iter.est_cate_conf_int_bern(X_query)
    else:
        warning.warn(f'Warning: var_est choice {var_est} not found. Defaulting to betting bound')
        lb_iter, ub_iter = knn_match_iter.est_cate_conf_int(X_query, alpha = 0.05)
    return lb_iter, ub_iter, knn_match_iter

def knn_match_est_prop_or(or_estimator, dgp, X_train, T_train, Y_train, X_est_iter, T_est_iter, Y_est_iter, projection_algorithm, k, X_query, bias_correction = True, var_est = 'betting'):
    
    class or_class:
        def __init__(self, or_estimator, or_args = {}):
            self.or_T = or_estimator(**or_args)
            self.or_C = or_estimator(**or_args)
        def fit(self, X_train, T_train, Y_train):
            X_train = np.array(X_train)
            self.or_T.fit(X_train[T_train == 1, ], Y_train[T_train == 1])
            self.or_C.fit(X_train[T_train == 0, ], Y_train[T_train == 0])
        def predict(self, X_query, T):
            if T == 1:
                return self.or_T.predict(X_query)
            elif T == 0:
                return self.or_C.predict(X_query)
            else:
                raise ValueError('T must be 0/1. Uncrecognized T:', T)
    
    outcome_reg = or_class(or_estimator)
    outcome_reg.fit(X_train, T_train, Y_train)
    # prop_score None defaults to using NNs...
    knn_match_iter = knn_match(prop_score = None, outcome_reg = outcome_reg.predict, ymin = dgp.ymin, ymax = dgp.ymax, bias_correction = bias_correction)
    
    ## fit projection on training set
    knn_match_iter.learn_projection(X_train, T_train, Y_train, algorithm = projection_algorithm)
    
    ## estimate intervals on estimation set
    knn_match_iter.fit(X_est_iter, T_est_iter, Y_est_iter, k = k)
    if var_est == 'betting':
        lb_iter, ub_iter = knn_match_iter.est_cate_conf_int(X_query)
    elif var_est == 'clt':
        lb_iter, ub_iter = knn_match_iter.est_cate_conf_int_clt(X_query)
    elif var_est == 'bern':
        lb_iter, ub_iter = knn_match_iter.est_cate_conf_int_bern(X_query)
    else:
        warning.warn(f'Warning: var_est choice {var_est} not found. Defaulting to betting bound')
        lb_iter, ub_iter = knn_match_iter.est_cate_conf_int(X_query, alpha = 0.05)
    return lb_iter, ub_iter, knn_match_iter


def get_method_CATE_error_bound(df_train, df_est, treatment_col, outcome_col, alpha, estimator, fit_params):
    X_train = df_train.drop([treatment_col, outcome_col], axis = 1)
    t_train = df_train[treatment_col]
    y_train = df_train[outcome_col]

    X_est = df_est[X_train.columns]

    # Initialize and fit the estimator
    estimator.fit(Y=y_train, T=t_train, X=X_train, **fit_params)

    # Estimate CATE for the test set
    cate_pred = estimator.effect(X_est)

    # Get confidence intervals
    
    cate_intervals = estimator.effect_interval(X_est, alpha=alpha)  # 95% confidence interval
    # error_bound = cate_intervals[1] - cate_pred

    return cate_pred, cate_intervals[0], cate_intervals[1]

def main():
    
    parser = ArgumentParser()
    parser.add_argument('--task_id', type = int)
    # parser.add_argument('--fit', type = str)

    parser_args = parser.parse_args()
    args_df = pd.read_csv('./Experiments/variance/randomization_args.csv')
    if parser_args.task_id >= args_df.shape[0]:
        print('task id is out of bounds for args_df size:', parser_args.task_id, args_df.shape[0])
    args = args_df.loc[parser_args.task_id - 1, ] # subtract 1 because arg generator gets first array task 
    # args['fit'] = parser_args.fit
    print(args)
    
    np.random.seed(args.seed)

    ## read in training and estimation datasets for this iteration
    folder = f'./Experiments/variance/randomization_files/dgp_{args.dgp}/n_train_{args.n_train}/n_est_{args.n_est}/n_imp_{args.n_imp}/n_unimp_{args.n_unimp}'
    train_iter_df = pd.read_csv(folder + f'/seed_{args.seed}/train_df.csv')
    est_iter_df = pd.read_csv(folder + f'/seed_{args.seed}/est_df.csv')
    query_true_df = pd.read_csv(folder + f'/query_true_df.csv')
    xcols = [col for col in train_iter_df.columns if col[0] == 'X']

    X_train = train_iter_df[xcols]
    T_train = train_iter_df['T'].values
    Y_train = train_iter_df['Y'].values

    X_est = est_iter_df[xcols]
    T_est = est_iter_df['T'].values
    Y_est = est_iter_df['Y'].values

    X_query = query_true_df[xcols]

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
    # elif args.dgp == 'friedman_homoskedastic':
    #     gen_dgp = dgp_friedman(xmin = -5, xmax = 5, n_imp = args.n_imp, n_unimp = args.n_unimp, heteroskedastic=False)
    # elif args.dgp == 'friedman_heteroskedastic':
    #     gen_dgp = dgp_friedman(xmin = -5, xmax = 5, n_imp = args.n_imp, n_unimp = args.n_unimp, heteroskedastic=True)


    if args.fit == 'knn_match_true_prop_true_or_coef':
        lb_iter, ub_iter, knn_match_iter = knn_match_true_prop_or(gen_dgp, X_train, T_train, Y_train, X_est, T_est, Y_est, 'true_coef', k = args.k, X_query = X_query, var_est = 'betting')
    elif args.fit == 'knn_match_true_prop_true_or_true_prog':
        lb_iter, ub_iter, knn_match_iter = knn_match_true_prop_or(gen_dgp, X_train, T_train, Y_train, X_est, T_est, Y_est, 'true_prog', k = args.k, X_query = X_query, var_est = 'betting')
    elif args.fit == 'knn_match_true_prop_true_or_true_prog_clt':
        lb_iter, ub_iter, knn_match_iter = knn_match_true_prop_or(gen_dgp, X_train, T_train, Y_train, X_est, T_est, Y_est, 'true_prog', k = args.k, X_query = X_query, var_est = 'clt')
    elif args.fit == 'knn_match_true_prop_true_or_true_prog_bern':
        lb_iter, ub_iter, knn_match_iter = knn_match_true_prop_or(gen_dgp, X_train, T_train, Y_train, X_est, T_est, Y_est, 'true_prog', k = args.k, X_query = X_query, var_est = 'bern')
    elif args.fit == 'knn_match_true_prop_true_or_rf_prog':
        lb_iter, ub_iter, knn_match_iter = knn_match_true_prop_or(gen_dgp, X_train, T_train, Y_train, X_est, T_est, Y_est, 'rf_prognostic_score', k = args.k, X_query = X_query, var_est = 'betting')
    elif args.fit == 'knn_match_est_prop_est_or_rf_prog':
        # knn_match_est_prop_or(RandomForestRegressor, gen_dgp, X_train, T_train, Y_train, X_est, T_est, Y_est, projection_algorithm = 'rf_prognostic_score', k = args.k, X_query = X_query)
        lb_iter, ub_iter, knn_match_iter = knn_match_est_prop_or(or_estimator=RandomForestRegressor, dgp = gen_dgp, X_train = X_train, T_train = T_train, Y_train = Y_train, X_est_iter = X_est, T_est_iter = T_est, Y_est_iter = Y_est, projection_algorithm = 'rf_prognostic_score', k = args.k, X_query = X_query, bias_correction = True)
    elif args.fit == 'knn_match_est_prop_est_or_rf_prog_bern':
        # knn_match_est_prop_or(RandomForestRegressor, gen_dgp, X_train, T_train, Y_train, X_est, T_est, Y_est, projection_algorithm = 'rf_prognostic_score', k = args.k, X_query = X_query)
        lb_iter, ub_iter, knn_match_iter = knn_match_est_prop_or(or_estimator=RandomForestRegressor, dgp = gen_dgp, X_train = X_train, T_train = T_train, Y_train = Y_train, X_est_iter = X_est, T_est_iter = T_est, Y_est_iter = Y_est, projection_algorithm = 'rf_prognostic_score', k = args.k, X_query = X_query, bias_correction = True, var_est = 'bern')
    elif args.fit == 'knn_match_est_prop_est_or_rf_prog_no_bias':
        # knn_match_est_prop_or(RandomForestRegressor, gen_dgp, X_train, T_train, Y_train, X_est, T_est, Y_est, projection_algorithm = 'rf_prognostic_score', k = args.k, X_query = X_query, bias_correction = False)
        lb_iter, ub_iter, knn_match_iter = knn_match_est_prop_or(or_estimator=RandomForestRegressor, dgp = gen_dgp, X_train = X_train, T_train = T_train, Y_train = Y_train, X_est_iter = X_est, T_est_iter = T_est, Y_est_iter = Y_est, projection_algorithm = 'rf_prognostic_score', k = args.k, X_query = X_query, bias_correction = False)
    elif args.fit == 'xlearner':
        df_train = pd.concat([train_iter_df[xcols + ['T', 'Y']], est_iter_df[xcols + ['T', 'Y']]], axis = 0)
        estimator = XLearner(models = RandomForestRegressor())
        cate_est, lb_iter, ub_iter  = get_method_CATE_error_bound(df_train = df_train, df_est = X_query, treatment_col = 'T', outcome_col = 'Y', alpha = 0.05, estimator = estimator, fit_params = {'inference' : 'bootstrap'})
    elif args.fit == 'causal_forest':
        df_train = pd.concat([train_iter_df[xcols + ['T', 'Y']], est_iter_df[xcols + ['T', 'Y']]], axis = 0)
        estimator = CausalForestDML()
        cate_est, lb_iter, ub_iter = get_method_CATE_error_bound(df_train = df_train, df_est = X_query, treatment_col = 'T', outcome_col = 'Y', alpha = 0.05, estimator = estimator, fit_params = {'inference' : 'auto'})
    
    true_df_iter = query_true_df.copy()
    true_df_iter['lb'] = lb_iter
    true_df_iter['ub'] = ub_iter
    true_df_iter['seed'] = args.seed
    true_df_iter['fit'] = args.fit
    true_df_iter['ymax'] = gen_dgp.ymax
    true_df_iter['ymin'] = gen_dgp.ymin
    # if 'knn' in args.fit:
    #     true_df_iter['mg_ate'] = knn_match_iter.calc_group_ate(X_query, gen_dgp)
    # else:
    true_df_iter['mg_ate'] = np.nan
    print('Coverage', ((true_df_iter['lb'] <= true_df_iter['cate_true']) * (true_df_iter['cate_true'] <= true_df_iter['ub'])).mean())
    save_df_to_csv(true_df_iter, f'./Experiments/variance/randomization_files/dgp_{args.dgp}/n_train_{args.n_train}/n_est_{args.n_est}/n_imp_{args.n_imp}/n_unimp_{args.n_unimp}/seed_{args.seed}/{args.fit}_k{args.k}.csv')


if __name__ == '__main__':
    main()



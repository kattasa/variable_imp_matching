import numpy as np
import pandas as pd
from confseq.betting import betting_ci

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
    # set Y to be -Y if in control group
    df_est[Y + '_normalized'] = df_est[Y].values * (df_est[tx].values) - df_est[Y].values * (1 - df_est[tx].values)
    
    # get N_query x K matrix of outcomes for each treatment group
    # entry (i,j) represents the normalized outcome of the j-th nearest neighbor of unit i
    Y_T = df_est[Y + '_normalized'].values[mgs[T]]
    Y_C = df_est[Y + '_normalized'].values[mgs[C]] 
    Y_stack = np.hstack([Y_T, Y_C])

    # get confidence intervals from hedging by betting sequences...
    
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
    try: # use this if you have a data structure that is split into treatment groups (e.g., T learner)
        for i in range(k):
            T_uq.append(model_dict[T].predict(X_NN_T[i], T = 1))
            C_uq.append(model_dict[C].predict(X_NN_C[i], T = 0))

        T_uq = np.array(T_uq).mean(axis = 0)
        T_query = model_dict[T].predict(query_x, T = 1)

        C_uq = np.array(C_uq).mean(axis = 0)
        C_query = model_dict[C].predict(query_x, T = 0)
    except TypeError:
        for i in range(k):
            T_uq.append(model_dict[T].predict(X_NN_T[i]))
            C_uq.append(model_dict[C].predict(X_NN_C[i]))
        
        T_uq = np.array(T_uq).mean(axis = 0)
        T_query = model_dict[T].predict(query_x)

        C_uq = np.array(C_uq).mean(axis = 0)
        C_query = model_dict[C].predict(query_x)
    bias = C_uq - C_query - T_uq + T_query

    lb, ub = get_betting_bounds(df_est = df_est, Y = Y, tx = tx, mgs = mgs, T = T, C = C, alpha = alpha, ymin = ymin, ymax = ymax)
    lb, ub = lb + bias, ub + bias
    
    return lb, ub


def get_weighted_betting_bounds(df_est, Y, tx, mgs, T, C, alpha, ymin, ymax, prop_score, min_prop, max_prop):
    df_est = df_est.copy()
    df_est[tx + '_prop'] = prop_score(df_est.drop([Y, tx], axis = 1).to_numpy())
    print('prop est range', df_est[tx + '_prop'].min(), df_est[tx + '_prop'].max())

    df_est[tx + '_weights'] = 1/df_est[tx + '_prop'].values * df_est[tx].values + 1/(1 - df_est[tx + '_prop'].values) * (1 - df_est[tx].values)
    
    df_est[Y + '_normalized'] = (2 * df_est[tx].values - 1) * df_est[Y].values
    df_est[Y + '_weighted'] =  df_est[tx + '_weights'].values * df_est[Y + '_normalized'].values

    ymin_unweighted = np.min([ymin, ymax, -ymin, -ymax])
    ymax_unweighted = np.max([ymin, ymax, -ymin, -ymax])

    # ymax_unweighted = np.max([ymax_unweighted, ymin_unweighted])
    # ymin_unweighted = np.min([ymax_unweighted, ymin_unweighted])

    weight_max = np.max([1/min_prop, 1/(1 - max_prop)])
    weight_min = np.min([1/max_prop, 1/(1 - min_prop)])

    print('prop range', min_prop, max_prop)

    Y_weighted_T = df_est[Y + '_weighted'].values[mgs[T]]
    Y_weighted_C = df_est[Y + '_weighted'].values[mgs[C]] 
    Y_weighted_stack = np.hstack([Y_weighted_T, Y_weighted_C])

    Y_T = df_est[Y + '_normalized'].values[mgs[T]]
    Y_C = df_est[Y + '_normalized'].values[mgs[C]] 
    Y_stack = np.hstack([Y_T, Y_C])

    weights_scores = df_est[tx + '_weights'].values
    weights_scores_T = weights_scores[mgs[1]]
    weights_scores_C = weights_scores[mgs[0]]
    weights_scores_stack = np.hstack([weights_scores_T, weights_scores_C])
    
    
    lb = []
    ub = []

    for i in range(Y_weighted_stack.shape[0]):
        mg_weights_scores = weights_scores_stack[i, :]
        
        k = mg_weights_scores.shape[0]

        wmax_i = mg_weights_scores.max()
        wmin_i = mg_weights_scores.min()

        ymax_i = Y_stack[i, :].max()
        ymin_i = Y_stack[i, :].min()
        
        # print('w range', wmin_i, wmax_i)

        zmax_i = weight_max * ymax_unweighted
        zmin_i = weight_min * ymin_unweighted

        # if zmax_i > 0:
        #     zmax_i *= (k + 1)/k
        # else:
        #     zmax_i *= k/(k + 1)
        
        # if zmin_i > 0:
        #     zmin_i *= k/(k + 1)
        # else:
        #     zmin_i *= (k + 1)/k

        print('y norm. range', ymin_unweighted, ymax_unweighted)
        print('z range', zmin_i, zmax_i)
        print('Y stack range', Y_weighted_stack[i, :].min(), Y_weighted_stack[i, :].max())
        print('Y unweighted min/max:', Y_stack[i, :].min(), Y_stack[i, :].max())
        print('Weights min/max:', weights_scores_stack[i, :].min(), weights_scores_stack[i, :].max())
        bounds_i = make_betting_ci(Y_weighted_stack[i, :], alpha=alpha, min = zmin_i, max = zmax_i)

        lb.append(bounds_i[0])
        ub.append(bounds_i[1])
    lb = np.array(lb)
    ub = np.array(ub)
    return lb, ub

   
def get_CATE_weighted_bias_betting_bound(X_NN_T, X_NN_C, query_x, k, model_dict, T, C, df_est, Y, tx, mgs, ymin, ymax, prop_score, min_prop, max_prop, alpha = 0.05):

    T_uq = []
    C_uq = []
    # try: # use this if you have a data structure that is split into treatment groups (e.g., T learner)
    for i in range(k):
        T_uq.append(model_dict[T].predict(X_NN_T[i], T = 1))
        C_uq.append(model_dict[C].predict(X_NN_C[i], T = 0))

    T_uq = np.array(T_uq).mean(axis = 0)
    T_query = model_dict[T].predict(query_x, T = 1)

    C_uq = np.array(C_uq).mean(axis = 0)
    C_query = model_dict[C].predict(query_x, T = 0)
    # except TypeError:
    #     for i in range(k):
    #         T_uq.append(model_dict[T].predict(X_NN_T[i]))
    #         C_uq.append(model_dict[C].predict(X_NN_C[i]))
        
    #     T_uq = np.array(T_uq).mean(axis = 0)
    #     T_query = model_dict[T].predict(query_x)

    #     C_uq = np.array(C_uq).mean(axis = 0)
    #     C_query = model_dict[C].predict(query_x)
    bias = C_uq - C_query - T_uq + T_query

    lb, ub = get_weighted_betting_bounds(df_est = df_est, Y = Y, tx = tx, mgs = mgs, T = T, C = C, alpha = alpha, ymin = ymin, ymax = ymax, prop_score = prop_score, min_prop = min_prop, max_prop = max_prop)
    lb, ub = lb + bias, ub + bias
    
    return lb, ub
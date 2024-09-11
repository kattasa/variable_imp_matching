import pandas as pd

def save_params(file):
    list_args = []
    for dgp in ['nonlinear_mml', 'piecewise_mml']:
        for n_train in [100, 1000, 5000, 7500, 10000, 20000]:
            for n_imp in [20]:
                for n_unimp in [0]:
                    for k in [int(np.sqrt(n_train))]:
                        for seed in [42 * i for i in range(10)]:
                            for fit in ['vim', 'vim_tree', 'vim_ensemble', 'nn', 'nn_mml', 'causal_forest', 'naive', 'dr_learner', 'x_learner', 's_learner', 't_learner']:
                            # for fit in ['vim', 'nn']:
                                args = {'dgp' : dgp, 'n_train' : n_train, 'n_imp' : n_imp, 'n_unimp' : n_unimp, 'k' : k, 'seed' : seed, 'fit' : fit}
                                list_args.append( args )
    args_df = pd.DataFrame(list_args)
    args_df.to_csv(file, index = False)



def main():
    save_params('./Experiments/variance/args.csv')

main()
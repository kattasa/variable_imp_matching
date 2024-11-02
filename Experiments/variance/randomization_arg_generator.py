import pandas as pd
import numpy as np
import os

def save_params(file, fits, make_data):
    
    list_args = []
    counter = 0
    for dgp in [
        # 'linear_homoskedastic', 
        # 'linear_heteroskedastic', 
        'lihua_uncorr_homoskedastic',
        'lihua_uncorr_heteroskedastic', 
        'lihua_corr_homoskedastic', 
        'lihua_corr_heteroskedastic'
        ]:
        for n_train in [10000]:
            # for n_est in [1000, 5000, 10000, 20000]:
            for n_est in [1000, 10000]:
                for n_imp in [2]:
                    for n_unimp in [10, 100]:
                        for n_query in [100]:
                            for n_iter in [10]:
                                if make_data:
                                    os.system(f'export PYTHONPATH=/usr/project/xtmp/sk787/variable_imp_matching/; python3 ./Experiments/variance/randomization_coverage_exp_datagen.py  --n_train {n_train}  --n_est {n_est}  --n_query {n_query} --n_imp {n_imp}  --n_iter {n_iter}  --n_unimp {n_unimp} --dgp {dgp}')
                                for k in [
                                    # 2 * int(np.sqrt(n_est)), 
                                    # min(4 * int(np.sqrt(n_est)), n_est), 
                                    # min(8 * int(np.sqrt(n_est)), n_est), 
                                    # min(16 * int(np.sqrt(n_est)), n_est), 
                                    # min(32 * int(np.sqrt(n_est)), n_est),
                                    # min(64 * int(np.sqrt(n_est)), n_est),
                                    n_est
                                    ]:
                                    # for fit in ['vim', 'vim_tree', 'vim_ensemble', 'nn_vim', 'nn_mml', 'causal_forest', 'naive', 'dr_learner', 'x_learner', 's_learner', 't_learner']:
                                    # for fit in ['vim', 'vim_tree', 'vim_ensemble', 'naive', 'dr_learner', 'x_learner', 's_learner', 't_learner', 'prog_boost_vim', 'prog_cf_vim']:
                                    for seed in [42 * i for i in range(n_iter)]:
                                        for fit in fits:
                                            args = {'dgp' : dgp, 'n_train' : n_train, 'n_est' : n_est, 'n_imp' : n_imp, 'n_unimp' : n_unimp, 'k' : k, 'seed' : seed, 'fit' : fit}
                                            list_args.append( args )
                                            counter += 1

    args_df = pd.DataFrame(list_args)
    args_df.to_csv(file, index = False)

    njobs = counter

    commands = []
    for fit in fits:
        commands.append(f'srun python3 /usr/project/xtmp/sk787/variable_imp_matching/Experiments/variance/randomization_coverage_exp.py --task_id $SLURM_ARRAY_TASK_ID --fit {fit}')
    commands = '\n'.join(commands)

    bash_str = f'''#!/bin/bash
#SBATCH --job-name=error_calib_variance
#SBATCH --output=array_job_%A_%a.out  # %A is the job array ID, %a is the task ID
#SBATCH --error=array_job_%A_%a.err
#SBATCH --array=1-{njobs + 1}
#SBATCH --ntasks=1                    # Number of tasks per job
#SBATCH --cpus-per-task=4             # Number of CPU cores per task
#SBATCH --mem=12G                      # Memory per task
#SBATCH --time=01:00:00               # Time limit
#SBATCH --mail-type=ALL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --exclude=linux[46],linux[1]
#SBATCH -p compsci 

export PYTHONPATH=/usr/project/xtmp/sk787/variable_imp_matching/

# Print the task ID
echo "Starting task ID: $SLURM_ARRAY_TASK_ID"

# Run your command or script here, using $SLURM_ARRAY_TASK_ID if needed
srun python3 /usr/project/xtmp/sk787/variable_imp_matching/Experiments/variance/randomization_coverage_exp.py --task_id $SLURM_ARRAY_TASK_ID

# srun Rscript /usr/project/xtmp/sk787/variable_imp_matching/Experiments/variance/mml_fixed.r --task_id $SLURM_ARRAY_TASK_ID
        '''
    f = open('./Experiments/variance/randomization_coverage_slurm.sh', 'w')
    print(bash_str, file = f)
    f.close()
    # os.system('sbatch ./Experiments/variance/randomization_coverage_slurm.sh')
    return counter


def main():
    fits = [
        # 'knn_match_true_prop_true_or_true_prog',
        # 'knn_match_true_prop_true_or_rf_prog',
        # 'knn_match_est_prop_est_or_rf_prog',
        'knn_match_est_prop_est_or_rf_prog_bern',
        # 'knn_match_est_prop_est_or_rf_prog_no_bias',
        'knn_match_true_prop_true_or_true_prog_bern',
        'causal_forest'
    ]
    save_params(file='./Experiments/variance/randomization_args.csv', fits = fits, make_data=False)

    # scate_params(file = './Experiments/variance/scate_args.csv')

main()
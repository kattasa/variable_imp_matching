import pandas as pd
import numpy as np
import os


def save_params(file, fits, dgps):
    list_args = []
    counter = 0
    for dgp in dgps:
        for n_train in [1000, 10000]:
            for n_est in [1000, 10000]:
            # for n_est in [1000, 10000]:
                for n_imp in [2]:
                    for n_unimp in [0, 10, 20]:
                        for k in [int(np.sqrt(n_est))]:
                                # for fit in ['vim', 'vim_tree', 'vim_ensemble', 'nn_vim', 'nn_mml', 'causal_forest', 'naive', 'dr_learner', 'x_learner', 's_learner', 't_learner']:
                                # for fit in ['vim', 'vim_tree', 'vim_ensemble', 'naive', 'dr_learner', 'x_learner', 's_learner', 't_learner', 'prog_boost_vim', 'prog_cf_vim']:
                                for seed in [42 * i for i in range(10)]:
                                    args = {'dgp' : dgp, 'n_train' : n_train, 'n_est' : n_est, 'n_imp' : n_imp, 'n_unimp' : n_unimp, 'k' : k, 'seed' : seed}
                                    list_args.append( args )
                                    counter += 1

    args_df = pd.DataFrame(list_args)
    args_df.to_csv(file, index = False)

    commands = []
    for fit in fits:
        commands.append(f'srun python3 /usr/project/xtmp/sk787/variable_imp_matching/Experiments/variance/variance_playground.py --task_id $SLURM_ARRAY_TASK_ID --fit {fit}')
    commands = '\n'.join(commands)

    njobs = counter

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
# srun python3 /usr/project/xtmp/sk787/variable_imp_matching/Experiments/variance/variance_playground.py --task_id $SLURM_ARRAY_TASK_ID --fit gen_data
{commands}
# srun Rscript /usr/project/xtmp/sk787/variable_imp_matching/Experiments/variance/mml_fixed.r --task_id $SLURM_ARRAY_TASK_ID
        '''
    print(bash_str, file = open('./Experiments/variance/variance_slurm.sh', 'w'))
    os.system('sbatch ./Experiments/variance/variance_slurm.sh')
    return counter

def scate_params(file):
    
    list_args = []
    counter = 0
    for dgp in ['linear', 'lihua_uncorr_homo', 'lihua_corr_homo', 'lihua_uncorr_hetero', 'lihua_corr_hetero']:
        for n_train in [10000]:
            # for n_est in [1000, 5000, 10000, 20000]:
            for n_est in [1000]:
                for n_query in [100]:
                    for n_imp in [2]:
                        for n_unimp in [10, 100]:
                            for query_seed in [100]:
                                for sample_seed in [42069]:
                                    for n_iter in [100]:
                                        os.system(f'export PYTHONPATH=/usr/project/xtmp/sk787/variable_imp_matching/; python3 ./Experiments/variance/scate_coverage_exp_datagen.py --query_seed {query_seed} --sample_seed {sample_seed}  --n_train {n_train}  --n_est {n_est}  --n_query {n_query} --n_imp {n_imp}  --n_iter {n_iter}  --n_unimp {n_unimp} --dgp {dgp}')
                                        for k in [int(np.sqrt(n_est))]:
                                                # for fit in ['vim', 'vim_tree', 'vim_ensemble', 'nn_vim', 'nn_mml', 'causal_forest', 'naive', 'dr_learner', 'x_learner', 's_learner', 't_learner']:
                                                # for fit in ['vim', 'vim_tree', 'vim_ensemble', 'naive', 'dr_learner', 'x_learner', 's_learner', 't_learner', 'prog_boost_vim', 'prog_cf_vim']:
                                                for seed in [42 * i for i in range(n_iter)]:
                                                    args = {'dgp' : dgp, 'n_train' : n_train, 'n_est' : n_est, 'n_imp' : n_imp, 'n_unimp' : n_unimp, 'k' : k, 'query_seed' : query_seed, 'sample_seed' : sample_seed, 'seed' : seed}
                                                    list_args.append( args )
                                                    counter += 1

    args_df = pd.DataFrame(list_args)
    args_df.to_csv(file, index = False)

    njobs = counter

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
srun python3 /usr/project/xtmp/sk787/variable_imp_matching/Experiments/variance/scate_coverage_exp.py --task_id $SLURM_ARRAY_TASK_ID --fit bias_corr_betting

# srun Rscript /usr/project/xtmp/sk787/variable_imp_matching/Experiments/variance/mml_fixed.r --task_id $SLURM_ARRAY_TASK_ID
        '''
    print(bash_str, file = open('./Experiments/variance/scate_coverage_slurm.sh', 'w'))
    os.system('sbatch ./Experiments/variance/scate_coverage_slurm.sh')
    return counter


def main():
    
    save_params(file='./Experiments/variance/args.csv', fits = ['bias_corr_betting', 'weighted_bias_corr_betting'], dgps = ['lihua_uncorr_homo', 'lihua_corr_homo', 'lihua_uncorr_hetero', 'lihua_corr_hetero'])

    # scate_params(file = './Experiments/variance/scate_args.csv')

main()
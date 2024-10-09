import pandas as pd
import numpy as np
import os


def save_params(file):
    list_args = []
    counter = 0
    for dgp in ['linear', 'poly']:
        for n_train in [1000, 5000, 10000, 20000]:
            # for n_est in [1000, 5000, 10000, 20000]:
            for n_est in [1000, 5000, 10000, 20000]:
                for n_imp in [20]:
                    for n_unimp in [0]:
                        for k in [int(np.sqrt(n_est))]:
                                # for fit in ['vim', 'vim_tree', 'vim_ensemble', 'nn_vim', 'nn_mml', 'causal_forest', 'naive', 'dr_learner', 'x_learner', 's_learner', 't_learner']:
                                # for fit in ['vim', 'vim_tree', 'vim_ensemble', 'naive', 'dr_learner', 'x_learner', 's_learner', 't_learner', 'prog_boost_vim', 'prog_cf_vim']:
                                for seed in [42 * i for i in range(10)]:
                                    args = {'dgp' : dgp, 'n_train' : n_train, 'n_est' : n_est, 'n_imp' : n_imp, 'n_unimp' : n_unimp, 'k' : k, 'seed' : seed}
                                    list_args.append( args )
                                    counter += 1

    args_df = pd.DataFrame(list_args)
    args_df.to_csv(file, index = False)
    return counter



def main():
    njobs = save_params(file='./Experiments/variance/args.csv')

    commands = []
    for fit in ['boost_bias_corr', 'bias_corr', 'bias_corr_betting']:
        commands.append(f'srun python3 /usr/project/xtmp/sk787/variable_imp_matching/Experiments/variance/variance_playground.py --task_id $SLURM_ARRAY_TASK_ID --fit {fit}')
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
srun python3 /usr/project/xtmp/sk787/variable_imp_matching/Experiments/variance/variance_playground.py --task_id $SLURM_ARRAY_TASK_ID --fit gen_data
{commands}
srun Rscript /usr/project/xtmp/sk787/variable_imp_matching/Experiments/variance/mml_fixed.r --task_id $SLURM_ARRAY_TASK_ID
        '''
    print(bash_str, file = open('./Experiments/variance/variance_slurm.sh', 'w'))
    os.system('sbatch ./Experiments/variance/variance_slurm.sh')

main()
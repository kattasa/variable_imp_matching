#!/bin/bash
#SBATCH --job-name=error_calib_variance
#SBATCH --output=array_job_%A_%a.out  # %A is the job array ID, %a is the task ID
#SBATCH --error=array_job_%A_%a.err
#SBATCH --array=1-481
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
srun python3 /usr/project/xtmp/sk787/variable_imp_matching/Experiments/variance/variance_playground.py --task_id $SLURM_ARRAY_TASK_ID --fit bias_corr_betting
srun python3 /usr/project/xtmp/sk787/variable_imp_matching/Experiments/variance/variance_playground.py --task_id $SLURM_ARRAY_TASK_ID --fit weighted_bias_corr_betting
# srun Rscript /usr/project/xtmp/sk787/variable_imp_matching/Experiments/variance/mml_fixed.r --task_id $SLURM_ARRAY_TASK_ID
        

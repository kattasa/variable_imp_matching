#!/bin/bash
#SBATCH --job-name=error_calib_variance
#SBATCH --output=output_files/array_job_%A_%a.out  # %A is the job array ID, %a is the task ID
#SBATCH --error=output_files/array_job_%A_%a.err
#SBATCH --array=1-700                  # Define the range of array jobs
#SBATCH --ntasks=1                    # Number of tasks per job
#SBATCH --cpus-per-task=4             # Number of CPU cores per task
#SBATCH --mem=12G                      # Memory per task
#SBATCH --time=01:00:00               # Time limit
#SBATCH --mail-type=ALL          # Mail events (NONE, BEGIN, END, FAIL, ALL)

export PYTHONPATH=/usr/project/xtmp/sk787/variable_imp_matching/

python3 /usr/project/xtmp/sk787/variable_imp_matching/Experiments/variance/arg_generator.py

# Print the task ID
echo "Starting task ID: $SLURM_ARRAY_TASK_ID"

# Run your command or script here, using $SLURM_ARRAY_TASK_ID if needed
python3 /usr/project/xtmp/sk787/variable_imp_matching/Experiments/variance/variance_playground.py --task_id $SLURM_ARRAY_TASK_ID

# python3 /usr/project/xtmp/sk787/variable_imp_matching/Experiments/variance/variance_plots.py

Rscript /usr/project/xtmp/sk787/variable_imp_matching/Experiments/variance/r_variance_plots.r
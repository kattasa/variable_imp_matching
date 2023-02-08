#!/bin/bash
#
#SBATCH --get-user-env
#SBATCH --output=/hpc/group/volfovskylab/qml/linear_coef_matching/scaling/Results/slurm_scale_accuracy.out
#SBATCH --error=/hpc/group/volfovskylab/qml/linear_coef_matching/scaling/Results/slurm_scale_accuracy.err
#SBATCH --mem=2G

export RESULTS_FOLDER=/hpc/group/volfovskylab/qml/linear_coef_matching/scaling/Results
export PYTHONPATH=/hpc/home/qml/linear_coef_matching:$PYTHONPATH
export R_HOME=/hpc/home/qml/miniconda3/envs/linear_coef_matching/lib/R
export N_REPEATS=1
export RANDOM_STATE=0

memory=$"16G"

num_covs=(0 8 24 56 120 248 504)
for n in ${num_covs[@]}; do
    echo "Running scripts for ${n}"
    mkdir "${RESULTS_FOLDER}/num_covs/${n}/lcm_accuracy"
    mkdir "${RESULTS_FOLDER}/num_covs/${n}/malts_accuracy"
    sbatch -o "${RESULTS_FOLDER}/num_covs/${n}/lcm_accuracy.txt" -e "${RESULTS_FOLDER}/num_covs/${n}/lcm_accuracy.err" --mem="$memory" --export=SAVE_FOLDER="${RESULTS_FOLDER}/num_covs/${n}",PYTHONPATH,R_HOME,N_REPEATS,RANDOM_STATE slurm_lcm_accuracy.q
    sbatch -o "${RESULTS_FOLDER}/num_covs/${n}/malts_accuracy.txt" -e "${RESULTS_FOLDER}/num_covs/${n}/malts_accuracy.err" --mem="$memory" --export=SAVE_FOLDER="${RESULTS_FOLDER}/num_covs/${n}",PYTHONPATH,R_HOME,N_REPEATS,RANDOM_STATE slurm_malts_accuracy.q
done
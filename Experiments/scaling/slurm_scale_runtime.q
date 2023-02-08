#!/bin/bash
#
#SBATCH --get-user-env
#SBATCH --output=/hpc/group/volfovskylab/qml/linear_coef_matching/scaling/Results/slurm_scale_runtimes.out
#SBATCH --error=/hpc/group/volfovskylab/qml/linear_coef_matching/scaling/Results/slurm_scale_runtimes.err
#SBATCH --mem=2G

export RESULTS_FOLDER=/hpc/group/volfovskylab/qml/linear_coef_matching/scaling/Results
export PYTHONPATH=/hpc/home/qml/linear_coef_matching:$PYTHONPATH
export R_HOME=/hpc/home/qml/miniconda3/envs/linear_coef_matching/lib/R

memory=$"16G"
iters=20

mkdir "${RESULTS_FOLDER}/num_samples"
num_samples=(256 512 1024 2048 4096 8192)
for n in ${num_samples[@]}; do
    echo "Running scripts for ${n}"
    mkdir "${RESULTS_FOLDER}/num_samples/${n}/lcm_fit_times"
    mkdir "${RESULTS_FOLDER}/num_samples/${n}/malts_fit_times"
    mkdir "${RESULTS_FOLDER}/num_samples/${n}/genmatch_fit_times"
    counter=0
    while [ $counter -lt $iters ]
    do
        sbatch -o "${RESULTS_FOLDER}/num_samples/${n}/lcm_fit_times/${counter}.txt" -e "${RESULTS_FOLDER}/num_samples/${n}/lcm_fit_times/${counter}.err" --mem="$memory" --export=SAVE_FOLDER="${RESULTS_FOLDER}/num_samples/${n}",PYTHONPATH,R_HOME slurm_lcm_scale.q
        sbatch -o "${RESULTS_FOLDER}/num_samples/${n}/malts_fit_times/${counter}.txt" -e "${RESULTS_FOLDER}/num_samples/${n}/malts_fit_times/${counter}.err" --mem="$memory" --export=SAVE_FOLDER="${RESULTS_FOLDER}/num_samples/${n}",PYTHONPATH,R_HOME slurm_malts_scale.q
        sbatch -o "${RESULTS_FOLDER}/num_samples/${n}/genmatch_fit_times/${counter}.txt" -e "${RESULTS_FOLDER}/num_samples/${n}/genmatch_fit_times/${counter}.err" --mem="$memory" --export=SAVE_FOLDER="${RESULTS_FOLDER}/num_samples/${n}",PYTHONPATH,R_HOME slurm_genmatch_scale.q
        ((counter++))
    done
done

mkdir "${RESULTS_FOLDER}/num_covs"
num_covs=(0 8 24 56 120 248 504)
for n in ${num_covs[@]}; do
    echo "Running scripts for ${n}"
    mkdir "${RESULTS_FOLDER}/num_covs/${n}/lcm_fit_times"
    mkdir "${RESULTS_FOLDER}/num_covs/${n}/malts_fit_times"
    mkdir "${RESULTS_FOLDER}/num_covs/${n}/genmatch_fit_times"
    counter=0
    while [ $counter -lt $iters ]
    do
        sbatch -o "${RESULTS_FOLDER}/num_covs/${n}/lcm_fit_times/${counter}.txt" -e "${RESULTS_FOLDER}/num_covs/${n}/lcm_fit_times/${counter}.err" --mem="$memory" --export=SAVE_FOLDER="${RESULTS_FOLDER}/num_covs/${n}",PYTHONPATH,R_HOME slurm_lcm_scale.q
        sbatch -o "${RESULTS_FOLDER}/num_covs/${n}/malts_fit_times/${counter}.txt" -e "${RESULTS_FOLDER}/num_covs/${n}/malts_fit_times/${counter}.err" --mem="$memory" --export=SAVE_FOLDER="${RESULTS_FOLDER}/num_covs/${n}",PYTHONPATH,R_HOME slurm_malts_scale.q
        sbatch -o "${RESULTS_FOLDER}/num_covs/${n}/genmatch_fit_times/${counter}.txt" -e "${RESULTS_FOLDER}/num_covs/${n}/genmatch_fit_times/${counter}.err" --mem="$memory" --export=SAVE_FOLDER="${RESULTS_FOLDER}/num_covs/${n}",PYTHONPATH,R_HOME slurm_genmatch_scale.q
        ((counter++))
    done
done
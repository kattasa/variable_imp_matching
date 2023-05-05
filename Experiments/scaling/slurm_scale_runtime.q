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
    mkdir "${RESULTS_FOLDER}/num_samples/${n}"
    echo "Running scripts for ${n}"
    mkdir "${RESULTS_FOLDER}/num_samples/${n}/lcm_fit_times"
    mkdir "${RESULTS_FOLDER}/num_samples/${n}/malts_fit_times"
    mkdir "${RESULTS_FOLDER}/num_samples/${n}/genmatch_fit_times"
    mkdir "${RESULTS_FOLDER}/num_samples/${n}/ahb_fit_times"
    counter=0
    while [ $counter -lt $iters ]
    do
        sbatch -p common-old -o "${RESULTS_FOLDER}/num_samples/${n}/lcm_fit_times/${counter}.txt" -e "${RESULTS_FOLDER}/num_samples/${n}/lcm_fit_times/${counter}.err" --mem="$memory" --export=N_SAMPLES="$n",RESULTS_FOLDER,PYTHONPATH,R_HOME slurm_lcm_scale_samples.q
        sbatch -p common-old -o "${RESULTS_FOLDER}/num_samples/${n}/malts_fit_times/${counter}.txt" -e "${RESULTS_FOLDER}/num_samples/${n}/malts_fit_times/${counter}.err" --mem="$memory" --export=N_SAMPLES="$n",RESULTS_FOLDER,PYTHONPATH,R_HOME slurm_malts_scale_samples.q
        sbatch -p common-old -o "${RESULTS_FOLDER}/num_samples/${n}/genmatch_fit_times/${counter}.txt" -e "${RESULTS_FOLDER}/num_samples/${n}/genmatch_fit_times/${counter}.err" --mem="$memory" --export=N_SAMPLES="$n",RESULTS_FOLDER,PYTHONPATH,R_HOME slurm_genmatch_scale_samples.q
        sbatch -p common-old -o "${RESULTS_FOLDER}/num_samples/${n}/ahb_fit_times/${counter}.txt" -e "${RESULTS_FOLDER}/num_samples/${n}/ahb_fit_times/${counter}.err" --mem="$memory" --export=N_SAMPLES="$n",RESULTS_FOLDER slurm_ahb_scale_samples.q
        ((counter++))
    done
done

mkdir "${RESULTS_FOLDER}/num_covs"
num_covs=(16)
for n in ${num_covs[@]}; do
    mkdir "${RESULTS_FOLDER}/num_covs/${n}"
    echo "Running scripts for ${n}"
    mkdir "${RESULTS_FOLDER}/num_covs/${n}/lcm_fit_times"
    mkdir "${RESULTS_FOLDER}/num_covs/${n}/malts_fit_times"
    mkdir "${RESULTS_FOLDER}/num_covs/${n}/genmatch_fit_times"
    mkdir "${RESULTS_FOLDER}/num_covs/${n}/ahb_fit_times"
    counter=0
    while [ $counter -lt $iters ]
    do
        sbatch -p common-old -o "${RESULTS_FOLDER}/num_covs/${n}/lcm_fit_times/${counter}.txt" -e "${RESULTS_FOLDER}/num_covs/${n}/lcm_fit_times/${counter}.err" --mem="$memory" --export=N_COVS="$n",RESULTS_FOLDER,PYTHONPATH,R_HOME slurm_lcm_scale_covs.q
        sbatch -p common-old -o "${RESULTS_FOLDER}/num_covs/${n}/malts_fit_times/${counter}.txt" -e "${RESULTS_FOLDER}/num_covs/${n}/malts_fit_times/${counter}.err" --mem="$memory" --export=N_COVS="$n",RESULTS_FOLDER,PYTHONPATH,R_HOME slurm_malts_scale_covs.q
        sbatch -p common-old -o "${RESULTS_FOLDER}/num_covs/${n}/genmatch_fit_times/${counter}.txt" -e "${RESULTS_FOLDER}/num_covs/${n}/genmatch_fit_times/${counter}.err" --mem="$memory" --export=N_COVS="$n",RESULTS_FOLDER,PYTHONPATH,R_HOME slurm_genmatch_scale_covs.q
        sbatch -p common-old -o "${RESULTS_FOLDER}/num_covs/${n}/ahb_fit_times/${counter}.txt" -e "${RESULTS_FOLDER}/num_covs/${n}/ahb_fit_times/${counter}.err" --mem="$memory" --export=N_COVS="$n",RESULTS_FOLDER slurm_ahb_scale_covs.q
        ((counter++))
    done
done
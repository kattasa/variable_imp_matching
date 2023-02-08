#!/bin/bash
#
#SBATCH --get-user-env
#SBATCH --output=/hpc/group/volfovskylab/qml/linear_coef_matching/scaling/Results/slurm_scale_df.out
#SBATCH --error=/hpc/group/volfovskylab/qml/linear_coef_matching/scaling/Results/slurm_scale_df.err
#SBATCH --mem=2G

export RESULTS_FOLDER=/hpc/group/volfovskylab/qml/linear_coef_matching/scaling/Results
export PYTHONPATH=/hpc/home/qml/linear_coef_matching:$PYTHONPATH
export R_HOME=/hpc/home/qml/miniconda3/envs/linear_coef_matching/lib/R

memory=$"16G"

mkdir "${RESULTS_FOLDER}/num_samples"
num_samples=(256 512 1024 2048 4096 8192 16384)
imp_c=8
unimp_c=56
for n in ${num_samples[@]}; do
    mkdir "${RESULTS_FOLDER}/num_samples/${n}"
    sbatch -o "${RESULTS_FOLDER}/num_samples/${n}/dgp.txt" -e "${RESULTS_FOLDER}/num_samples/${n}/dgp.err" --mem="$memory" --export=NUM_SAMPLES=$n,IMP_C=$imp_c,UNIMP_C=$unimp_c,SAVE_FOLDER="${RESULTS_FOLDER}/num_samples/${n}",PYTHONPATH,R_HOME slurm_dgp_scale.q
done

mkdir "${RESULTS_FOLDER}/num_covs"
num_covs=(0 8 24 56 120 248 504 1016)
num_samples=2048
imp_c=8
for n in ${num_covs[@]}; do
    mkdir "${RESULTS_FOLDER}/num_covs/${n}"
    sbatch -o "${RESULTS_FOLDER}/num_covs/${n}/dgp.txt" -e "${RESULTS_FOLDER}/num_covs/${n}/dgp.err" --mem="16G" --export=NUM_SAMPLES=$num_samples,IMP_C=$imp_c,UNIMP_C=$n,SAVE_FOLDER="${RESULTS_FOLDER}/num_covs/${n}",PYTHONPATH,R_HOME slurm_dgp_scale.q
done
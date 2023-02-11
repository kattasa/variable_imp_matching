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
export N_SAMPLES=1024

memory=$"64G"

sbatch -o "${RESULTS_FOLDER}/lcm_accuracy.txt" -e "${RESULTS_FOLDER}/lcm_accuracy.err" --mem="$memory" --export=RESULTS_FOLDER,N_SAMPLES,PYTHONPATH,R_HOME,N_REPEATS,RANDOM_STATE slurm_lcm_accuracy.q
sbatch -o "${RESULTS_FOLDER}/malts_accuracy.txt" -e "${RESULTS_FOLDER}/malts_accuracy.err" --mem="$memory" --export=SAVE_FOLDER=RESULTS_FOLDER,N_SAMPLES,PYTHONPATH,R_HOME,N_REPEATS,RANDOM_STATE slurm_malts_accuracy.q
sbatch -o "${RESULTS_FOLDER}/genmatch_accuracy.txt" -e "${RESULTS_FOLDER}/genmatch_accuracy.err" --mem="$memory" --export=SAVE_FOLDER=RESULTS_FOLDER,N_SAMPLES,PYTHONPATH,R_HOME,N_REPEATS,RANDOM_STATE slurm_genmatch_accuracy.q
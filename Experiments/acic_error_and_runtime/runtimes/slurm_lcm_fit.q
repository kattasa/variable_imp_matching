#!/bin/bash
#
#SBATCH --get-user-env
#SBATCH --output=/hpc/group/volfovskylab/qml/linear_coef_matching/acic_error_and_runtime/Results/slurm_%A_%a.out
#SBATCH --error=/hpc/group/volfovskylab/qml/linear_coef_matching/acic_error_and_runtime/Results/slurm_%A_%a.err
#SBATCH --mem=2G

export RESULTS_FOLDER=/hpc/group/volfovskylab/qml/linear_coef_matching/acic_error_and_runtime/Results
export ACIC_2018_FOLDER=/work/qml/acic_2018
export ACIC_2019_FOLDER=/work/qml/acic_2019
export PYTHONPATH=/hpc/home/qml/linear_coef_matching:$PYTHONPATH
export R_HOME=/hpc/home/qml/miniconda3/envs/linear_coef_matching/lib/R

folders=$(ls $RESULTS_FOLDER)

# iterate through array using a counter
for f in folders; do
    #do something to each element of array
    echo "$f"
done
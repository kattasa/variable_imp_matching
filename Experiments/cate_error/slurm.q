#!/bin/bash
#
#SBATCH --get-user-env
#SBATCH --output=/hpc/group/volfovskylab/qml/linear_coef_matching/Results/slurm_%A_%a.out
#SBATCH --error=/hpc/group/volfovskylab/qml/linear_coef_matching/Results/slurm_%A_%a.err
#SBATCH --mem=32G

export RESULTS_FOLDER=/hpc/group/volfovskylab/qml/linear_coef_matching/cate_error/Results
export ACIC_2018_FOLDER=/work/qml/acic_2018
export ACIC_2019_FOLDER=/work/qml/acic_2019
export PYTHONPATH=/hpc/home/qml/linear_coef_matching:$PYTHONPATH
export R_HOME=/hpc/home/qml/miniconda3/envs/linear_coef_matching/lib/R

source /hpc/home/qml/miniconda3/etc/profile.d/conda.sh
conda activate linear_coef_matching
python /hpc/home/qml/linear_coef_matching/Experiments/cate_error/run.py
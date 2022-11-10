#!/bin/bash
#
#SBATCH --get-user-env

export RESULTS_FOLDER=/hpc/group/volfovskylab/qml/linear_coef_matching/acic_error_and_runtime/Results
export PYTHONPATH=/hpc/home/qml/linear_coef_matching:$PYTHONPATH
export R_HOME=/hpc/home/qml/miniconda3/envs/linear_coef_matching/lib/R

source /hpc/home/qml/miniconda3/etc/profile.d/conda.sh
conda activate linear_coef_matching
python /hpc/home/qml/linear_coef_matching/Experiments/acic_erro_and_runtime/runtimes/lcm_fit_run.py
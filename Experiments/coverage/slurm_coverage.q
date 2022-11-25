#!/bin/bash
#
#SBATCH --get-user-env

source /hpc/home/qml/miniconda3/etc/profile.d/conda.sh
conda activate linear_coef_matching
python -m sklearnex /hpc/home/qml/linear_coef_matching/Experiments/coverage/coverage_and_stderr.py
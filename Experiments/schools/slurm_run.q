#!/bin/bash
#
#SBATCH --get-user-env
#SBATCH --mem=64G

export PYTHONPATH=/hpc/home/qml/linear_coef_matching:$PYTHONPATH
export R_HOME=/hpc/home/qml/miniconda3/envs/linear_coef_matching/lib/R
export SCHOOLS_FOLDER=/work/qml/schools
export SAVE_FOLDER=Results

source /hpc/home/qml/miniconda3/etc/profile.d/conda.sh
conda activate linear_coef_matching
python /hpc/home/qml/linear_coef_matching/Experiments/schools/run.py
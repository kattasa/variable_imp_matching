#!/bin/bash
#
#SBATCH --get-user-env
#SBATCH --output=/hpc/group/volfovskylab/qml/linear_coef_matching/scaling/Results/slurm_scale_df.out
#SBATCH --error=/hpc/group/volfovskylab/qml/linear_coef_matching/scaling/Results/slurm_scale_df.err
#SBATCH --mem=8G

export SAVE_FOLDER=/hpc/group/volfovskylab/qml/linear_coef_matching/scaling/Results
export PYTHONPATH=/hpc/home/qml/linear_coef_matching:$PYTHONPATH
export R_HOME=/hpc/home/qml/miniconda3/envs/linear_coef_matching/lib/R

source /hpc/home/qml/miniconda3/etc/profile.d/conda.sh
conda activate linear_coef_matching
python /hpc/home/qml/linear_coef_matching/Experiments/scaling/create_dataset.py
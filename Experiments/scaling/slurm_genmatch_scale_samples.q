#!/bin/bash
#
#SBATCH --get-user-env

source /hpc/home/qml/miniconda3/etc/profile.d/conda.sh
conda activate linear_coef_matching
Rscript /hpc/home/qml/linear_coef_matching/Experiments/scaling/genmatch_scale_samples.R
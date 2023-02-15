#!/bin/bash
#
#SBATCH --get-user-env

module load Java/11.0.8 1> /dev/null
module load CPLEX/20.1 1> /dev/null
module load R/4.1.1-rhel8 1> /dev/null

Rscript /hpc/home/qml/linear_coef_matching/Experiments/scaling/ahb_scale_covs.R
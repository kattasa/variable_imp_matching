#!/bin/bash
#
#SBATCH --get-user-env
#SBATCH --output=/hpc/group/volfovskylab/qml/linear_coef_matching/acic_error_and_runtime/Results/slurm_fit.out
#SBATCH --error=/hpc/group/volfovskylab/qml/linear_coef_matching/acic_error_and_runtime/Results/slurm_fit.err
#SBATCH --mem=2G

export RESULTS_FOLDER=/hpc/group/volfovskylab/qml/linear_coef_matching/acic_error_and_runtime/Results
export ACIC_2018_FOLDER=/work/qml/acic_2018
export ACIC_2019_FOLDER=/work/qml/acic_2019
export PYTHONPATH=/hpc/home/qml/linear_coef_matching:$PYTHONPATH
export R_HOME=/hpc/home/qml/miniconda3/envs/linear_coef_matching/lib/R

memory=$"8G"
folders=$(ls -d */ $RESULTS_FOLDER)

# iterate through array using a counter
for f in $folders; do
    n_splits=$((python -c "import json;print(json.load(open('${RESULTS_FOLDER}/${f}config.txt', 'rb'))['n_splits'])") 2>&1)
    counter=0
    while [ $counter -l 3 ]
    do
      split_num=0
      while [ $split_num -l $n_splits ]
      do
        sbatch -o "${RESULTS_FOLDER}/${f}/lcm_fit_times.txt" -e "${RESULTS_FOLDER}/${f}/lcm_fit_times.err" --open-mode=append --mem="$memory" slurm_lcm_fit.q --export=ACIC_FOLDER="$f" --export=SPLIT_NUM="$split_num"
        ((split_num++))
      done
      ((counter++))
    done
    break
done
#!/bin/bash
#
#SBATCH --get-user-env
#SBATCH --output=/hpc/group/volfovskylab/qml/linear_coef_matching/acic_error_and_runtime/Results_Final/slurm_model_fits.out
#SBATCH --error=/hpc/group/volfovskylab/qml/linear_coef_matching/acic_error_and_runtime/Results_Final/slurm_model_fits.err
#SBATCH --mem=2G

export RESULTS_FOLDER=/hpc/group/volfovskylab/qml/linear_coef_matching/acic_error_and_runtime/Results_Final
export PYTHONPATH=/hpc/home/qml/linear_coef_matching:$PYTHONPATH

memory=$"8G"
random_state=1
cd $RESULTS_FOLDER
#folders=$(ls -d */)
folders=("acic_2019-8_000")
cd -
iters=5

#for f in $folders; do
for f in ${folders[*]}; do
    echo "Running scripts for ${f}"
    n_splits=$((python -c "import json;print(json.load(open('${RESULTS_FOLDER}/${f}config.txt', 'rb'))['n_splits'])") 2>&1)
    n_splits=$(($n_splits + 0))
    echo "${n_splits} splits"
    mkdir "${RESULTS_FOLDER}/${f}/lcm_model_fit_scores"
    counter=0
    while [ $counter -lt $iters ]
    do
      split_num=0
      while [ $split_num -lt $n_splits ]
      do
        sbatch -o "${RESULTS_FOLDER}/${f}/lcm_model_fit_scores/${split_num}_${counter}.txt" -e "${RESULTS_FOLDER}/${f}/lcm_model_fit_scores/${split_num}_${counter}.err" --mem="$memory" --export=ACIC_FOLDER="$f",SPLIT_NUM=$split_num,RANDOM_STATE=$random_state,PYTHONPATH,RESULTS_FOLDER slurm_lcm_model_fit.q
        ((split_num++))
      done
      ((counter++))
    done
done
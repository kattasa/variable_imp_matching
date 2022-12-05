#!/bin/bash
#
#SBATCH --get-user-env
#SBATCH --output=/hpc/group/volfovskylab/qml/linear_coef_matching/acic_error_and_runtime/Results_Main/slurm_runtimes.out
#SBATCH --error=/hpc/group/volfovskylab/qml/linear_coef_matching/acic_error_and_runtime/Results_Main/slurm_runtimes.err
#SBATCH --mem=2G

export RESULTS_FOLDER=/hpc/group/volfovskylab/qml/linear_coef_matching/acic_error_and_runtime/Results_Main
export ACIC_2018_FOLDER=/work/qml/acic_2018
export ACIC_2019_FOLDER=/work/qml/acic_2019
export PYTHONPATH=/hpc/home/qml/linear_coef_matching:$PYTHONPATH
export R_HOME=/hpc/home/qml/miniconda3/envs/linear_coef_matching/lib/R

memory=$"8G"
k_est=60
random_state=0
cd $RESULTS_FOLDER
folders=$(ls -d */)
cd -
iters=25

for f in $folders; do
    echo "Running scripts for ${f}"
    n_splits=$((python -c "import json;print(json.load(open('${RESULTS_FOLDER}/${f}config.txt', 'rb'))['n_splits'])") 2>&1)
    n_splits=$(($n_splits + 0))
    echo "${n_splits} splits"
    mkdir "${RESULTS_FOLDER}/${f}/lcm_fit_times"
    mkdir "${RESULTS_FOLDER}/${f}/tree_fit_times"
    mkdir "${RESULTS_FOLDER}/${f}/equal_lcm_fit_times"
    mkdir "${RESULTS_FOLDER}/${f}/malts_fit_times"
    mkdir "${RESULTS_FOLDER}/${f}/prognostic_fit_times"
    mkdir "${RESULTS_FOLDER}/${f}/bart_fit_times"
    mkdir "${RESULTS_FOLDER}/${f}/causalforest_fit_times"
    mkdir "${RESULTS_FOLDER}/${f}/causalforest_dml_fit_times"
    mkdir "${RESULTS_FOLDER}/${f}/doubleml_fit_times"
    mkdir "${RESULTS_FOLDER}/${f}/drlearner_fit_times"
    counter=0
    while [ $counter -lt $iters ]
    do
      split_num=0
      while [ $split_num -lt $n_splits ]
      do
        sbatch -o "${RESULTS_FOLDER}/${f}/lcm_fit_times/${split_num}_${counter}.txt" -e "${RESULTS_FOLDER}/${f}/lcm_fit_times/${split_num}_${counter}.err" --mem="$memory" --export=ACIC_FOLDER="$f",SPLIT_NUM=$split_num,K_EST=$k_est,RANDOM_STATE=$random_state,LCM_METHOD="linear",LCM_EQUAL_WEIGHTS=0,PYTHONPATH,RESULTS_FOLDER slurm_lcm_runtime.q
        sbatch -o "${RESULTS_FOLDER}/${f}/tree_fit_times/${split_num}_${counter}.txt" -e "${RESULTS_FOLDER}/${f}/tree_fit_times/${split_num}_${counter}.err" --mem="$memory" --export=ACIC_FOLDER="$f",SPLIT_NUM=$split_num,K_EST=$k_est,RANDOM_STATE=$random_state,LCM_METHOD="tree",LCM_EQUAL_WEIGHTS=0,PYTHONPATH,RESULTS_FOLDER slurm_lcm_runtime.q
        sbatch -o "${RESULTS_FOLDER}/${f}/equal_lcm_fit_times/${split_num}_${counter}.txt" -e "${RESULTS_FOLDER}/${f}/equal_lcm_fit_times/${split_num}_${counter}.err" --mem="$memory" --export=ACIC_FOLDER="$f",SPLIT_NUM=$split_num,K_EST=$k_est,RANDOM_STATE=$random_state,LCM_METHOD="linear",LCM_EQUAL_WEIGHTS=1,PYTHONPATH,RESULTS_FOLDER slurm_lcm_runtime.q
        sbatch -o "${RESULTS_FOLDER}/${f}/malts_fit_times/${split_num}_${counter}.txt" -e "${RESULTS_FOLDER}/${f}/malts_fit_times/${split_num}_${counter}.err" --mem="$memory" --export=ACIC_FOLDER="$f",SPLIT_NUM=$split_num,K_EST=$k_est,RANDOM_STATE=$random_state,PYTHONPATH,RESULTS_FOLDER slurm_malts_runtime.q
        sbatch -o "${RESULTS_FOLDER}/${f}/prognostic_fit_times/${split_num}_${counter}.txt" -e "${RESULTS_FOLDER}/${f}/prognostic_fit_times/${split_num}_${counter}.err" --mem="$memory" --export=ACIC_FOLDER="$f",SPLIT_NUM=$split_num,K_EST=$k_est,RANDOM_STATE=$random_state,PYTHONPATH,RESULTS_FOLDER slurm_prognostic_runtime.q
        sbatch -o "${RESULTS_FOLDER}/${f}/bart_fit_times/${split_num}_${counter}.txt" -e "${RESULTS_FOLDER}/${f}/bart_fit_times/${split_num}_${counter}.err" --mem="$memory" --export=ACIC_FOLDER="$f",SPLIT_NUM=$split_num,PYTHONPATH,R_HOME,RESULTS_FOLDER slurm_bart_runtime.q
        sbatch -o "${RESULTS_FOLDER}/${f}/causalforest_fit_times/${split_num}_${counter}.txt" -e "${RESULTS_FOLDER}/${f}/causalforest_fit_times/${split_num}_${counter}.err" --mem="$memory" --export=ACIC_FOLDER="$f",SPLIT_NUM=$split_num,PYTHONPATH,R_HOME,RESULTS_FOLDER slurm_causalforest_runtime.q
        sbatch -o "${RESULTS_FOLDER}/${f}/causalforest_dml_fit_times/${split_num}_${counter}.txt" -e "${RESULTS_FOLDER}/${f}/causalforest_dml_fit_times/${split_num}_${counter}.err" --mem="$memory" --export=ACIC_FOLDER="$f",SPLIT_NUM=$split_num,RANDOM_STATE=$random_state,PYTHONPATH,R_HOME,RESULTS_FOLDER slurm_causalforest_dml_runtime.q
        sbatch -o "${RESULTS_FOLDER}/${f}/doubleml_fit_times/${split_num}_${counter}.txt" -e "${RESULTS_FOLDER}/${f}/doubleml_fit_times/${split_num}_${counter}.err" --mem="$memory" --export=ACIC_FOLDER="$f",SPLIT_NUM=$split_num,RANDOM_STATE=$random_state,PYTHONPATH,R_HOME,RESULTS_FOLDER slurm_doubleml_runtime.q
        sbatch -o "${RESULTS_FOLDER}/${f}/drlearner_fit_times/${split_num}_${counter}.txt" -e "${RESULTS_FOLDER}/${f}/drlearner_fit_times/${split_num}_${counter}.err" --mem="$memory" --export=ACIC_FOLDER="$f",SPLIT_NUM=$split_num,RANDOM_STATE=$random_state,PYTHONPATH,R_HOME,RESULTS_FOLDER slurm_drlearner_runtime.q
        ((split_num++))
      done
      ((counter++))
    done
done
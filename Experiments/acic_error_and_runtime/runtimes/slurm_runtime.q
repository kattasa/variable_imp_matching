#!/bin/bash
#
#SBATCH --get-user-env
#SBATCH --output=/hpc/group/volfovskylab/qml/linear_coef_matching/acic_error_and_runtime/Results_Final/slurm_runtimes.out
#SBATCH --error=/hpc/group/volfovskylab/qml/linear_coef_matching/acic_error_and_runtime/Results_Final/slurm_runtimes.err
#SBATCH --mem=2G

export RESULTS_FOLDER=/hpc/group/volfovskylab/qml/linear_coef_matching/acic_error_and_runtime/Results_Final
export ACIC_2018_FOLDER=/work/qml/acic_2018
export ACIC_2019_FOLDER=/work/qml/acic_2019
export PYTHONPATH=/hpc/home/qml/linear_coef_matching:$PYTHONPATH
export R_HOME=/hpc/home/qml/miniconda3/envs/linear_coef_matching/lib/R

memory=$"16G"
k_est=60
random_state=1
cd $RESULTS_FOLDER
folders=$(ls -d */)
cd -
iters=10
malts_max=5000
bart_dont_run="acic_2018-d09f96200455407db569ae33fe06b0d3_000/"
two_splits_if_below=1500
f="acic_2019-4_000/"
#for f in $folders; do
echo "Running scripts for ${f}"
n_splits=$((python -c "import json;print(json.load(open('${RESULTS_FOLDER}/${f}config.txt', 'rb'))['n_splits'])") 2>&1)
n_splits=$(($n_splits + 0))
n_samples=$(cat "${RESULTS_FOLDER}/${f}df_true.csv" | wc -l)
n_samples=$(($n_samples - 1))
if [[ $n_samples -lt $two_splits_if_below ]]
then
  n_splits=2
fi
echo "${n_splits} splits"
echo "${n_samples} samples"
mkdir "${RESULTS_FOLDER}/${f}/lcm_fit_times"
mkdir "${RESULTS_FOLDER}/${f}/tree_fit_times"
mkdir "${RESULTS_FOLDER}/${f}/equal_lcm_fit_times"
if [[ $n_samples -le $malts_max ]]
then
    mkdir "${RESULTS_FOLDER}/${f}/malts_fit_times"
fi
mkdir "${RESULTS_FOLDER}/${f}/prognostic_fit_times"
if [ "${f}" != "$bart_dont_run" ]
then
    mkdir "${RESULTS_FOLDER}/${f}/bart_fit_times"
fi
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
    if [[ $n_samples -le $malts_max ]]
    then
        sbatch -o "${RESULTS_FOLDER}/${f}/malts_fit_times/${split_num}_${counter}.txt" -e "${RESULTS_FOLDER}/${f}/malts_fit_times/${split_num}_${counter}.err" --mem="$memory" --export=ACIC_FOLDER="$f",SPLIT_NUM=$split_num,K_EST=$k_est,RANDOM_STATE=$random_state,PYTHONPATH,RESULTS_FOLDER slurm_malts_runtime.q
    fi
    sbatch -o "${RESULTS_FOLDER}/${f}/prognostic_fit_times/${split_num}_${counter}.txt" -e "${RESULTS_FOLDER}/${f}/prognostic_fit_times/${split_num}_${counter}.err" --mem="$memory" --export=ACIC_FOLDER="$f",SPLIT_NUM=$split_num,K_EST=$k_est,RANDOM_STATE=$random_state,PYTHONPATH,RESULTS_FOLDER slurm_prognostic_runtime.q
    if [ "${f}" != "$bart_dont_run" ]
    then
        sbatch -o "${RESULTS_FOLDER}/${f}/bart_fit_times/${split_num}_${counter}.txt" -e "${RESULTS_FOLDER}/${f}/bart_fit_times/${split_num}_${counter}.err" --mem="$memory" --export=ACIC_FOLDER="$f",SPLIT_NUM=$split_num,RANDOM_STATE=$random_state,PYTHONPATH,R_HOME,RESULTS_FOLDER slurm_bart_runtime.q
    fi
    sbatch -o "${RESULTS_FOLDER}/${f}/causalforest_fit_times/${split_num}_${counter}.txt" -e "${RESULTS_FOLDER}/${f}/causalforest_fit_times/${split_num}_${counter}.err" --mem="$memory" --export=ACIC_FOLDER="$f",SPLIT_NUM=$split_num,RANDOM_STATE=$random_state,PYTHONPATH,R_HOME,RESULTS_FOLDER slurm_causalforest_runtime.q
    sbatch -o "${RESULTS_FOLDER}/${f}/causalforest_dml_fit_times/${split_num}_${counter}.txt" -e "${RESULTS_FOLDER}/${f}/causalforest_dml_fit_times/${split_num}_${counter}.err" --mem="$memory" --export=ACIC_FOLDER="$f",SPLIT_NUM=$split_num,RANDOM_STATE=$random_state,PYTHONPATH,R_HOME,RESULTS_FOLDER slurm_causalforest_dml_runtime.q
    sbatch -o "${RESULTS_FOLDER}/${f}/doubleml_fit_times/${split_num}_${counter}.txt" -e "${RESULTS_FOLDER}/${f}/doubleml_fit_times/${split_num}_${counter}.err" --mem="$memory" --export=ACIC_FOLDER="$f",SPLIT_NUM=$split_num,RANDOM_STATE=$random_state,PYTHONPATH,R_HOME,RESULTS_FOLDER slurm_doubleml_runtime.q
    sbatch -o "${RESULTS_FOLDER}/${f}/drlearner_fit_times/${split_num}_${counter}.txt" -e "${RESULTS_FOLDER}/${f}/drlearner_fit_times/${split_num}_${counter}.err" --mem="$memory" --export=ACIC_FOLDER="$f",SPLIT_NUM=$split_num,RANDOM_STATE=$random_state,PYTHONPATH,R_HOME,RESULTS_FOLDER slurm_drlearner_runtime.q
    ((split_num++))
  done
  ((counter++))
done
#done
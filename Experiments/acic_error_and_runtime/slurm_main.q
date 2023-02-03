#!/bin/bash
#
#SBATCH --get-user-env
#SBATCH --output=/hpc/group/volfovskylab/qml/linear_coef_matching/acic_error_and_runtime/Results/slurm_%A.out
#SBATCH --error=/hpc/group/volfovskylab/qml/linear_coef_matching/acic_error_and_runtime/Results/slurm_%A.err
#SBATCH --mem=2G

export RESULTS_FOLDER=/hpc/group/volfovskylab/qml/linear_coef_matching/acic_error_and_runtime/Results
export ACIC_2018_FOLDER=/work/qml/acic_2018
export ACIC_2019_FOLDER=/work/qml/acic_2019
export PYTHONPATH=/hpc/home/qml/linear_coef_matching:$PYTHONPATH
export R_HOME=/hpc/home/qml/miniconda3/envs/linear_coef_matching/lib/R

source /hpc/home/qml/miniconda3/etc/profile.d/conda.sh
conda activate linear_coef_matching

memory=$"8G"
k_est_per_1000=4
k_est_max=15
n_splits=2
n_sample_per_split=2500

all_acic_2018_files=($(python -c "import glob;import os;print([f.replace('.csv', '') for f in set([c.split('/')[-1].replace('_cf', '') for c in glob.glob('${ACIC_2018_FOLDER}/*.csv')])])" | tr -d '[],'))

for acic_file in 1 2 5 6 7 8
do
  counter=0
  save_dir=$(printf "${RESULTS_FOLDER}/acic_2019-${acic_file}_%03d" $counter)
  while [ -d save_dir ]
  do
    ((counter++))
    save_dir=$(printf "${RESULTS_FOLDER}/acic_2019-${acic_file}_%03d" $counter)
  done
  mkdir $save_dir
#  if [ -f "${save_dir}/df_err.csv" ]; then
#    echo "${save_dir}/df_err.csv exists"
#  else
  sbatch -o "${save_dir}/slurm.out" -e "${save_dir}/slurm.err" --mem="$memory" --export=ACIC_YEAR="acic_2019",ACIC_FILE=$acic_file,K_EST_PER_1000=$k_est_per_1000,K_EST_MAX=$k_est_max,SAVE_FOLDER=$save_dir,N_SPLITS=$n_splits,N_SAMPLES_PER_SPLIT=$n_sample_per_split,ACIC_2018_FOLDER,ACIC_2019_FOLDER,PYTHONPATH,R_HOME slurm_cate_error.q
#  fi
  ((acic_file++))
done

for acic_file in "${all_acic_2018_files[@]}"
do
  counter=0
  save_dir=$(printf "${RESULTS_FOLDER}/acic_2018-${acic_file}_%03d" $counter | tr -d \"\')
  while [ -d save_dir ]
  do
    ((counter++))
    save_dir=$(printf "${RESULTS_FOLDER}/acic_2018-${acic_file}_%03d" $counter | tr -d \"\')
  done
  mkdir $save_dir
  sbatch -o "${save_dir}/slurm.out" -e "${save_dir}/slurm.err" --mem="$memory" --export=ACIC_YEAR="acic_2018",ACIC_FILE=$acic_file,K_EST_PER_1000=$k_est_per_1000,K_EST_MAX=$k_est_max,SAVE_FOLDER=$save_dir,N_SPLITS=$n_splits,N_SAMPLES_PER_SPLIT=$n_sample_per_split,ACIC_2018_FOLDER,ACIC_2019_FOLDER,PYTHONPATH,R_HOME slurm_cate_error.q
done
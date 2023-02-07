#!/bin/bash
#
#SBATCH --get-user-env
#SBATCH --output=/hpc/group/volfovskylab/qml/linear_coef_matching/acic_error_and_runtime/Results_Final/slurm_%A.out
#SBATCH --error=/hpc/group/volfovskylab/qml/linear_coef_matching/acic_error_and_runtime/Results_Final/slurm_%A.err
#SBATCH --mem=2G

export RESULTS_FOLDER=/hpc/group/volfovskylab/qml/linear_coef_matching/acic_error_and_runtime/Results_Final
export ACIC_2018_FOLDER=/work/qml/acic_2018
export ACIC_2019_FOLDER=/work/qml/acic_2019
export PYTHONPATH=/hpc/home/qml/linear_coef_matching:$PYTHONPATH
export R_HOME=/hpc/home/qml/miniconda3/envs/linear_coef_matching/lib/R

source /hpc/home/qml/miniconda3/etc/profile.d/conda.sh
conda activate linear_coef_matching

small_memory=$"16G"
medium_small_memory=$"24G"
medium_memory=$"32G"
large_memory=$"64G"
k_est_per_500=3
k_est_max=15
min_n_splits=2
max_n_splits=10
n_sample_per_split_2019=500
n_sample_per_split_2018=1000
n_repeats=2
malts_max=5000

large_acic_2018_files=($(python -c "import csv;file=open('${ACIC_2018_FOLDER}/acic_file_sizes/large.csv');acic = list(csv.reader(file, delimiter=','))[0];file.close();print(acic)" | tr -d '[],'))
medium_acic_2018_files=($(python -c "import csv;file=open('${ACIC_2018_FOLDER}/acic_file_sizes/medium.csv');acic = list(csv.reader(file, delimiter=','))[0];file.close();print(acic)" | tr -d '[],'))
medium_small_acic_2018_files=($(python -c "import csv;file=open('${ACIC_2018_FOLDER}/acic_file_sizes/medium_small.csv');acic = list(csv.reader(file, delimiter=','))[0];file.close();print(acic)" | tr -d '[],'))
small_acic_2018_files=($(python -c "import csv;file=open('${ACIC_2018_FOLDER}/acic_file_sizes/small.csv');acic = list(csv.reader(file, delimiter=','))[0];file.close();print(acic)" | tr -d '[],'))

for acic_file in 1 2 3 4 5 6 7 8
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
  sbatch -o "${save_dir}/slurm.out" -e "${save_dir}/slurm.err" --mem="$medium_small_memory" --export=ACIC_YEAR="acic_2019",ACIC_FILE=$acic_file,K_EST_PER_500=$k_est_per_500,K_EST_MAX=$k_est_max,SAVE_FOLDER=$save_dir,MIN_N_SPLITS=$min_n_splits,MAX_N_SPLITS=$max_n_splits,N_SAMPLES_PER_SPLIT=$n_sample_per_split_2019,N_REPEATS=$n_repeats,MALTS_MAX=$malts_max,ACIC_2018_FOLDER,ACIC_2019_FOLDER,PYTHONPATH,R_HOME slurm_cate_error.q
#  fi
  ((acic_file++))
done

for acic_file in "${small_acic_2018_files[@]}"
do
  counter=0
  save_dir=$(printf "${RESULTS_FOLDER}/acic_2018-${acic_file}_%03d" $counter | tr -d \"\')
  while [ -d save_dir ]
  do
    ((counter++))
    save_dir=$(printf "${RESULTS_FOLDER}/acic_2018-${acic_file}_%03d" $counter | tr -d \"\')
  done
  mkdir $save_dir
  sbatch -o "${save_dir}/slurm.out" -e "${save_dir}/slurm.err" --mem="$small_memory" --export=ACIC_YEAR="acic_2018",ACIC_FILE=$acic_file,K_EST_PER_500=$k_est_per_500,K_EST_MAX=$k_est_max,SAVE_FOLDER=$save_dir,MIN_N_SPLITS=$min_n_splits,MAX_N_SPLITS=$max_n_splits,N_SAMPLES_PER_SPLIT=$n_sample_per_split_2018,N_REPEATS=$n_repeats,MALTS_MAX=$malts_max,ACIC_2018_FOLDER,ACIC_2019_FOLDER,PYTHONPATH,R_HOME slurm_cate_error.q
done

for acic_file in "${medium_small_acic_2018_files[@]}"
do
  counter=0
  save_dir=$(printf "${RESULTS_FOLDER}/acic_2018-${acic_file}_%03d" $counter | tr -d \"\')
  while [ -d save_dir ]
  do
    ((counter++))
    save_dir=$(printf "${RESULTS_FOLDER}/acic_2018-${acic_file}_%03d" $counter | tr -d \"\')
  done
  mkdir $save_dir
  sbatch -o "${save_dir}/slurm.out" -e "${save_dir}/slurm.err" --mem="$medium_small_memory" --export=ACIC_YEAR="acic_2018",ACIC_FILE=$acic_file,K_EST_PER_500=$k_est_per_500,K_EST_MAX=$k_est_max,SAVE_FOLDER=$save_dir,MIN_N_SPLITS=$min_n_splits,MAX_N_SPLITS=$max_n_splits,N_SAMPLES_PER_SPLIT=$n_sample_per_split_2018,N_REPEATS=$n_repeats,MALTS_MAX=$malts_max,ACIC_2018_FOLDER,ACIC_2019_FOLDER,PYTHONPATH,R_HOME slurm_cate_error.q
done

for acic_file in "${medium_acic_2018_files[@]}"
do
  counter=0
  save_dir=$(printf "${RESULTS_FOLDER}/acic_2018-${acic_file}_%03d" $counter | tr -d \"\')
  while [ -d save_dir ]
  do
    ((counter++))
    save_dir=$(printf "${RESULTS_FOLDER}/acic_2018-${acic_file}_%03d" $counter | tr -d \"\')
  done
  mkdir $save_dir
  sbatch -o "${save_dir}/slurm.out" -e "${save_dir}/slurm.err" --mem="$medium_memory" --export=ACIC_YEAR="acic_2018",ACIC_FILE=$acic_file,K_EST_PER_500=$k_est_per_500,K_EST_MAX=$k_est_max,SAVE_FOLDER=$save_dir,MIN_N_SPLITS=$min_n_splits,MAX_N_SPLITS=$max_n_splits,N_SAMPLES_PER_SPLIT=$n_sample_per_split_2018,N_REPEATS=$n_repeats,MALTS_MAX=$malts_max,ACIC_2018_FOLDER,ACIC_2019_FOLDER,PYTHONPATH,R_HOME slurm_cate_error.q
done

for acic_file in "${large_acic_2018_files[@]}"
do
  counter=0
  save_dir=$(printf "${RESULTS_FOLDER}/acic_2018-${acic_file}_%03d" $counter | tr -d \"\')
  while [ -d save_dir ]
  do
    ((counter++))
    save_dir=$(printf "${RESULTS_FOLDER}/acic_2018-${acic_file}_%03d" $counter | tr -d \"\')
  done
  mkdir $save_dir
  sbatch -o "${save_dir}/slurm.out" -e "${save_dir}/slurm.err" --mem="$large_memory" --export=ACIC_YEAR="acic_2018",ACIC_FILE=$acic_file,K_EST_PER_500=$k_est_per_500,K_EST_MAX=$k_est_max,SAVE_FOLDER=$save_dir,MIN_N_SPLITS=$min_n_splits,MAX_N_SPLITS=$max_n_splits,N_SAMPLES_PER_SPLIT=$n_sample_per_split_2018,N_REPEATS=$n_repeats,MALTS_MAX=$malts_max,ACIC_2018_FOLDER,ACIC_2019_FOLDER,PYTHONPATH,R_HOME slurm_cate_error.q
done
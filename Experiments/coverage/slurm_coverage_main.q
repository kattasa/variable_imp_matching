#!/bin/bash
#
#SBATCH --get-user-env
#SBATCH --output=/hpc/group/volfovskylab/qml/linear_coef_matching/coverage/Results/slurm_%A.out
#SBATCH --error=/hpc/group/volfovskylab/qml/linear_coef_matching/coverage/Results/slurm_%A.err
#SBATCH --mem=2G

export RESULTS_FOLDER=/hpc/group/volfovskylab/qml/linear_coef_matching/coverage/Results
export PYTHONPATH=/hpc/home/qml/linear_coef_matching:$PYTHONPATH
export R_HOME=/hpc/home/qml/miniconda3/envs/linear_coef_matching/lib/R
export N_SPLITS=2
export N_REPEATS=4
export N_ITERS=4
export K_EST=40

source /hpc/home/qml/miniconda3/etc/profile.d/conda.sh
conda activate linear_coef_matching

memory=$"16G"

for n_samples in 250 500;
do
  save_dir=$"{RESULTS_FOLDER}/${n_samples}_samples"
  mkdir $save_dir
  sbatch -o "${save_dir}/slurm.out" -e "${save_dir}/slurm.err" --mem="$memory" --export=SAVE_FOLDER=$save_dir,N_SAMPLES=$n_samples,N_SPLITS,N_REPEATS,N_ITERS,K_EST,PYTHONPATH,R_HOME slurm_coverage.q
  ((acic_file++))
done
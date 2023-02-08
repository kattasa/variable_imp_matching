#!/bin/bash
#
#SBATCH --get-user-env
#SBATCH --output=/hpc/group/volfovskylab/qml/linear_coef_matching/scaling/Results/slurm_scaling.out
#SBATCH --error=/hpc/group/volfovskylab/qml/linear_coef_matching/scaling/Results/slurm_scaling.err
#SBATCH --mem=2G

export RESULTS_FOLDER=/hpc/group/volfovskylab/qml/linear_coef_matching/scaling/Results
export PYTHONPATH=/hpc/home/qml/linear_coef_matching:$PYTHONPATH
export R_HOME=/hpc/home/qml/miniconda3/envs/linear_coef_matching/lib/R

memory=$"16G"
iters=10

mkdir "${RESULTS_FOLDER}/num_samples"
#num_samples=(128 256 512 1024 2048 4096 8192)
num_samples=(1024)
imp_c=8
unimp_c=32
for n in ${num_samples[@]}; do
#    mkdir "${RESULTS_FOLDER}/num_samples/${n}"
#    sbatch -o "${RESULTS_FOLDER}/num_samples/${n}/dgp.txt" -e "${RESULTS_FOLDER}/num_samples/${n}/dgp.err" --mem="16G" --export=NUM_SAMPLES=$n,IMP_C=$imp_c,UNIMP_C=$unimp_c,SAVE_FOLDER="${RESULTS_FOLDER}/num_samples/${n}",PYTHONPATH,R_HOME slurm_dgp_scale.q
#    while [ ! -f "${RESULTS_FOLDER}/num_samples/${n}/df.csv" ]; do
#        sleep 10
#    done
    echo "Running scripts for ${n}"
#    mkdir "${RESULTS_FOLDER}/num_samples/${n}/lcm_fit_times"
#    mkdir "${RESULTS_FOLDER}/num_samples/${n}/malts_fit_times"
    mkdir "${RESULTS_FOLDER}/num_samples/${n}/genmatch_fit_times"
    counter=0
    while [ $counter -lt $iters ]
    do
#        sbatch -o "${RESULTS_FOLDER}/num_samples/${n}/lcm_fit_times/${counter}.txt" -e "${RESULTS_FOLDER}/num_samples/${n}/lcm_fit_times/${counter}.err" --mem="$memory" --export=SAVE_FOLDER="${RESULTS_FOLDER}/num_samples/${n}",PYTHONPATH,R_HOME slurm_lcm_scale.q
#        sbatch -o "${RESULTS_FOLDER}/num_samples/${n}/malts_fit_times/${counter}.txt" -e "${RESULTS_FOLDER}/num_samples/${n}/malts_fit_times/${counter}.err" --mem="$memory" --export=SAVE_FOLDER="${RESULTS_FOLDER}/num_samples/${n}",PYTHONPATH,R_HOME slurm_malts_scale.q
        sbatch -o "${RESULTS_FOLDER}/num_samples/${n}/genmatch_fit_times/${counter}.txt" -e "${RESULTS_FOLDER}/num_samples/${n}/genmatch_fit_times/${counter}.err" --mem="$memory" --export=SAVE_FOLDER="${RESULTS_FOLDER}/num_samples/${n}",PYTHONPATH,R_HOME slurm_genmatch_scale.q
        ((counter++))
    done
done

#mkdir "${RESULTS_FOLDER}/num_covs"
#num_covs=(0 2 4 8 16 32 64 128 256 512)
#num_samples=1024
#imp_c=8
#for n in ${num_covs[@]}; do
##    mkdir "${RESULTS_FOLDER}/num_covs/${n}"
##    sbatch -o "${RESULTS_FOLDER}/num_covs/${n}/dgp.txt" -e "${RESULTS_FOLDER}/num_covs/${n}/dgp.err" --mem="16G" --export=NUM_SAMPLES=$num_samples,IMP_C=$imp_c,UNIMP_C=$n,SAVE_FOLDER="${RESULTS_FOLDER}/num_covs/${n}",PYTHONPATH,R_HOME slurm_dgp_scale.q
##    while [ ! -f "${RESULTS_FOLDER}/num_covs/${n}/df.csv" ]; do
##        sleep 10
##    done
#    echo "Running scripts for ${n}"
##    mkdir "${RESULTS_FOLDER}/num_covs/${n}/lcm_fit_times"
##    mkdir "${RESULTS_FOLDER}/num_covs/${n}/malts_fit_times"
#    mkdir "${RESULTS_FOLDER}/num_covs/${n}/genmatch_fit_times"
#    counter=0
#    while [ $counter -lt $iters ]
#    do
##        sbatch -o "${RESULTS_FOLDER}/num_covs/${n}/lcm_fit_times/${counter}.txt" -e "${RESULTS_FOLDER}/num_covs/${n}/lcm_fit_times/${counter}.err" --mem="$memory" --export=SAVE_FOLDER="${RESULTS_FOLDER}/num_covs/${n}",PYTHONPATH,R_HOME slurm_lcm_scale.q
##        sbatch -o "${RESULTS_FOLDER}/num_covs/${n}/malts_fit_times/${counter}.txt" -e "${RESULTS_FOLDER}/num_covs/${n}/malts_fit_times/${counter}.err" --mem="$memory" --export=SAVE_FOLDER="${RESULTS_FOLDER}/num_covs/${n}",PYTHONPATH,R_HOME slurm_malts_scale.q
#        sbatch -o "${RESULTS_FOLDER}/num_covs/${n}/genmatch_fit_times/${counter}.txt" -e "${RESULTS_FOLDER}/num_covs/${n}/genmatch_fit_times/${counter}.err" --mem="$memory" --export=SAVE_FOLDER="${RESULTS_FOLDER}/num_covs/${n}",PYTHONPATH,R_HOME slurm_genmatch_scale.q
#        ((counter++))
#    done
#done
#!/usr/bin/env bash
work_path=$(dirname $0)
ROOT=../..
export PYTHONPATH=$ROOT:$PYTHONPATH

name='NSE'

MKL_SERVICE_FORCE_INTEL=1 MV2_USE_CUDA=1 MV2_ENABLE_AFFINITY=0 MV2_SMP_USE_CMA=0 srun \
    --mpi=pmi2 --gres=gpu:8 -n$1 --ntasks-per-node=8 --job-name=${name} --partition=$2 \
    "python -u $ROOT/main.py --dist_mode --config $work_path/config_NSE27.yaml"

# slurm environment command
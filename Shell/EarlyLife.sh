#!/bin/bash
#SBATCH --job-name=EarlyLife-rnn-1-32-M100-Tstep20001-deltaT0.0004
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --partition=v100
#SBATCH --gres=gpu:1
#SBATCH -o out/%j.out
#SBATCH -e out/%j.out
#SBATCH -t 13-2:00:00

# load the environment
module purge

CUDA_LAUNCH_BLOCKING=1 python3 MasterEq.py --L 3 --M 100 --Model 'EarlyLife' --net 'rnn' --lossType 'kl' --max_stepAll 5000 --max_stepLater 100 --lr 0.001 --net_depth 1 --net_width 32 --print_step 1000 --batch_size 1000 --Tstep 20001 --delta_t 0.0004 --cuda 0 --dtype float64

#exit 0


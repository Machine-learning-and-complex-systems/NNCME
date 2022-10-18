#!/bin/bash
#SBATCH --job-name=cascade1-rnn-1-32-M10-Tstep1001-dt0.01
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --partition=v100
#SBATCH --gres=gpu:1
#SBATCH -o out/%j.out
#SBATCH -e out/%j.out
#SBATCH -t 13-2:00:00

# load the environment
module purge

CUDA_LAUNCH_BLOCKING=1 python3 MasterEq.py --L 15 --M 10 --Model 'cascade1' --net 'rnn' --lossType 'kl' --max_stepAll 10000 --max_stepLater 100 --lr 0.001 --net_depth 1 --net_width 32 --print_step 1000 --batch_size 1000 --Tstep 1001 --delta_t 0.01 --cuda 0 --dtype float64

#exit 0


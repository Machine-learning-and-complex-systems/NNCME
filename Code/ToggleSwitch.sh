#!/bin/bash
#SBATCH --job-name=ToggleSwitch-rnn-1-32-M80-Tstep8001-dt0.005
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --partition=v100
#SBATCH --gres=gpu:1
#SBATCH -o out/%j.out
#SBATCH -e out/%j.out
#SBATCH -t 13-2:00:00

# load the environment
module purge

CUDA_LAUNCH_BLOCKING=1 python3 MasterEq.py --L 4 --M 80 --Model 'ToggleSwitch' --net 'rnn' --lossType 'kl' --max_stepAll 5000 --max_stepLater 100 --lr 0.001 --net_depth 1 --net_width 32 --print_step 40 --batch_size 1000 --Tstep 8001 --delta_t 0.005 --cuda 0 --dtype float64

#exit 0


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

CUDA_LAUNCH_BLOCKING=1 python3 MasterEq.py \
--species_num 4 -upper_limit 80 --reaction_num 8 --reaction_rates [50,50,1,1,1e-4,1e-4,0.1,0.1] --initial_distirbution 'delta' --initial_num [1,1,0,0] \
--reaction_matrix_left [(1,0,0,0,0,1,0,0),(0,1,0,0,1,0,0,0),(0,0,1,0,2,0,0,0),(0,0,0,1,0,2,0,0)] --reaction_matrix_right [(1,0,0,0,0,0,0,1),(0,1,0,0,0,0,1,0),(1,0,0,0,0,0,2,0),(0,1,0,0,0,0,0,2)] \
--MConstraint [2,2,80,80] --Conservation 1 \
--batch_size 1000 --training_step 8001 --deltaT 0.005 --net 'rnn'  --net_depth 1 --net_width 32 --epoch1 5000 --epoch2 100

#exit 0


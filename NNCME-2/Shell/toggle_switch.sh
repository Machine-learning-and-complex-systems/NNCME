#!/bin/bash
#SBATCH --job-name=ToggleSwitch
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --partition=v100
#SBATCH --gres=gpu:1
#SBATCH -o out/%j.out
#SBATCH -e out/%j.out
#SBATCH -t 2-2:00:00

# Define your parameters
L=4
Para=1
Model="ToggleSwitch"
net="NADE"
method="TDVP"
epoch0=50
epoch=1
lr=1
net_width=16
batch_size=2000
M=80
Tstep=8001

# Construct the job name dynamically
JOB_NAME="${Model}--${net}-width${net_width}-batch${batch_size}-${method}-epoch${epoch}-lr${lr}"

# Change the job name dynamically
scontrol update JobId=$SLURM_JOB_ID JobName=$JOB_NAME

# load the environment
module purge

# Run your Python script
CUDA_LAUNCH_BLOCKING=1 python3 MasterEq.py --L $L --L_label Gx Gy Px Py --L_plot 0 1 2 3 --M $M --Model $Model --net $net --method $method --epoch0 $epoch0 --epoch $epoch --lr $lr --net_depth 1 --net_width $net_width --batch_size $batch_size --Tstep $Tstep --delta_t 0.005 --cuda 0 --dtype float64 --num_prints 100 --num_plots 20 --Para $Para --cuda 0 --sampling 'default'

python3 PlotToggleSwitch_merge.py

#exit 0


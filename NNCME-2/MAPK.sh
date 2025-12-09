#!/bin/bash

#SBATCH --job-name=MAPK  # 这里可以根据需要设置默认的job名称
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition=v100  # 固定指定分区
#SBATCH -o out/%j.out
#SBATCH -e out/%j.err  # 修改了错误输出文件名
#SBATCH -t 13-2:00:00

# Define your parameters
Model="MAPK"
net="NADE"
method="NatGrad"
epoch0=10
epoch=5
lr=0.5
net_depth=1
net_width=16
batch_size=2000
Tstep=1000001
delta_t=0.01

# Construct the job name dynamically
JOB_NAME="${Model}-Tstep${Tstep}-batch${batch_size}-epoch0${epoch0}-epoch${epoch}-lr${lr}-width${net_width}-dt${delta_t}-M${M}-noConstrain-factor1e-2"

# Change the job name dynamically
scontrol update JobId=$SLURM_JOB_ID JobName=$JOB_NAME

# load the environment
module purge

# Run your Python script
CUDA_LAUNCH_BLOCKING=1 python3 MasterEq.py --L 16  --L_label MKP3 K Kpp --L_plot 1 2 5 --M 10 --Model $Model --net $net --method $method --epoch0 $epoch0 --epoch $epoch --lr $lr --net_depth $net_depth --net_width $net_width --batch_size $batch_size --Tstep $Tstep --delta_t $delta_t --num_prints 200 --num_plots 20 --cuda 0 --dtype float64 --sampling 'default' --ESNumber 200

python3 PlotMAPK_merge.py

#exit 0

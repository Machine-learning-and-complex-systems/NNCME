#!/bin/bash

#SBATCH --job-name=default_job_name  # 这里必须给出一个默认值, 因为 #SBATCH 指令无法动态生成
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition=v100 # 固定指定分区
#SBATCH -o out/%j.out
#SBATCH -e out/%j.err
#SBATCH -t 13-2:00:00

# Define your parameters
L=2
Sites=(2,1)
Para=1
Model="Schlogl_2d"
net='NADE'
d_model=16
d_ff=32
n_layers=2
n_heads=2
net_width=16
net_depth=1

method="NatGrad"
epoch=5
lr=0.8
batch_size=5000
sampling='random'
ESratio=0.1

dt=0.000008
Tstep=125001

# Construct the job name dynamically
JOB_NAME="${Model}-Sites${Sites}-Para${Para}-${net}-${method}-batch${batch_size}-epoch${epoch}-width${net_width}-lr${lr}-sampling${sampling}-ESratio${ESratio}-norewieght-lambd5e-3"

# Change the job name dynamically
scontrol update JobId=$SLURM_JOB_ID JobName=$JOB_NAME

# load the environment
module purge

# Run your Python script
CUDA_LAUNCH_BLOCKING=1 python3 MasterEq.py --L $L --Sites $Sites --L_label X1 --L_plot 0 --M 85 --Model $Model --net $net --method $method --lossType 'kl' --epoch0 50 --epoch $epoch --lr $lr --batch_size $batch_size --Tstep $Tstep --delta_t $dt --cuda 0 --dtype float64 --num_prints 100 --num_plots 10 --Para $Para --sampling $sampling --ESratio $ESratio --noreweight --d_model $d_model --d_ff $d_ff --n_layers $n_layers --n_heads $n_heads --net_depth $net_depth --net_width $net_width

python3 PlotSchlogl_2d.py

#exit 0


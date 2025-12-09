#!/bin/bash

#SBATCH --job-name=default_job_name
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition=fat3t
#SBATCH -o out/%j.out
#SBATCH -e out/%j.err
#SBATCH -t 13-2:00:00

# Define your parameters
L=2
Sites=2
Para=1
Model="Schlogl"
net="NADE"
net_width=16
method="SGD"
epoch=100
lr=0.005
batch_size=2000
ESNumber=500
Tstep=100001
dt=0.00001
sampling='alpha'
alpha=0.3
reweighted=true

# Construct the job name dynamically
JOB_NAME="${Model}-Sites${Sites}-Para${Para}-${net}-width${net_width}-${method}-batch${batch_size}-epoch${epoch}-lr${lr}-Tstep${Tstep}-sampling${sampling}${ESNumber}_alpha${alpha}_reweighted${reweighted}"
scontrol update JobId=$SLURM_JOB_ID JobName=$JOB_NAME

module purge

# Run your Python script
if [ "$reweighted" = true ]; then
    RW_FLAG="--reweighted"
else
    RW_FLAG="--noreweight"
fi

CUDA_LAUNCH_BLOCKING=1 python3 MasterEq.py \
    --L $L --Sites $Sites --L_label X1 X2 --L_plot 0 1 --M 85 --Model $Model \
    --net $net --method $method --lossType 'kl' --epoch0 50 --epoch $epoch \
    --d_model 4 --d_ff 8 --n_layers 2 --n_heads 2 \
    --lr $lr --net_depth 1 --net_width $net_width --batch_size $batch_size \
    --Tstep $Tstep --delta_t $dt --cuda 0 --dtype float64 --num_prints 100 \
    --num_plots 10 --Para $Para --sampling $sampling --ESNumber $ESNumber \
    --alpha $alpha $RW_FLAG

#exit 0
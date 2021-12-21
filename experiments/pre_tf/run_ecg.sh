#!/bin/bash

#SBATCH --job-name=pre_tf_ori
#SBATCH --gres=gpu:1
#SBATCH -o slurm.out
#SBATCH --time=10:00:00  # 10 hours

. /data/haenim/anaconda3/etc/profile.d/conda.sh
conda activate venv

test=1   # 0 means train the model, 1 means evaluate the model
threshold=0.003
fold_cnt=1

dataroot="../../../dataset"
model="beatgan"

cd experiments/ecg

w_adv=1
niter=100
lr=0.0001
n_aug=0
gpu_ids=2
outf="./output"

for (( i=0; i<$fold_cnt; i+=1))
do
    echo "#################################"
    echo "########  Folder $i  ############"
    if [ $test = 0 ]; then
	    python -u main.py  \
            --dataroot $dataroot \
            --model $model \
            --niter $niter \
            --lr $lr \
            --outf  $outf \
            --folder $i \
            --gpu_ids $gpu_ids \



	else
	    python -u main.py  \
            --dataroot $dataroot \
            --model $model \
            --niter $niter \
            --lr $lr \
            --outf  $outf \
            --folder $i  \
            --outf  $outf \
            --istest  \
            --threshold $threshold
    fi

done

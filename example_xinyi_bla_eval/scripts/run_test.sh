#!/bin/bash

#SBATCH --nodes=1
#SBATCH --job-name=test_wanb
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --time=01:00:00
#SBATCH --output=test_wanb.out


module load 2021
module load  Anaconda3/2021.05
source activate volta

export LC_ALL=C.UTF-8
export LANG=C.UTF-8

rm -r /home/xchen/datasets/BLA/annotations/cache
cd $HOME/volta-bla
echo "VILBERT results"

python wandb_test.py


conda deactivate

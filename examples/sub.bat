#!/bin/bash

#SBATCH --job-name=SimChign
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=42G
#SBATCH --time=4-00:00:00
#SBATCH --gres=gpu:4
#SBATCH --partition=gpu
# #SBATCH --exclude=xgpu[17-23]

echo "# what GPUs have we been assigned?"
nvidia-smi --query-gpu=index,name,serial,gpu_bus_id,memory.total --format=csv

echo "# what cores have we been assigned?"
taskset -pc $$

echo "# what node are we running on?"
hostname -s

# your commands here..
cwd=$(pwd)

# Activate virtual envirment 
cd /home/jacopo/
source miniconda/bin/activate
conda activate mdsimulations

# Run program
cd ${cwd}

mpiexec --oversubscribe -n 4 python sim-chignolin.py

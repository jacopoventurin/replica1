#!/bin/bash

#SBATCH --job-name=Simulation_0-50ns
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=42G
#SBATCH --time=4-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
# #SBATCH --exclude=xgpu[17-23]

echo "# what GPUs have we been assigned?"
nvidia-smi --query-gpu=index,name,serial,gpu_bus_id,memory.total --format=csv

echo "# what cores have we been assigned?"
taskset -pc $$

echo "# what node are we running on?"
hostname -s

# your commands here..

# Activate virtual envirment 
cd /home/jacopo/
source miniconda/bin/activate
conda activate mdsimulations

# Run program
cd /scratch/jacopo/trans_temp/example_test/replica1/

python -u sim-chignolin.py > replica.oput

#!/bin/bash

#SBATCH --job-name=corn_density_train   # job name
#SBATCH --partition=a100                # partition (queue) # gpu # a100
#SBATCH --gres=gpu:1                    # number of gpus per node # a4500 # a100_2g.10gb #a100
#SBATCH --nodes=1                       # node count
#SBATCH --ntasks=1                      # total number of tasks across all nodes
#SBATCH --mem=64G                       # total memory per node 
#SBATCH --time=12:00:00                 # total run time limit (DD-HH:MM:SS)
#SBATCH --output=slurm_output/train_%N_%j.out # output file path

module purge
module load miniconda
source ~/.bashrc
conda activate yolov8_cuda12

python "density estimation/unet_smp.py"

conda deactivate
#!/bin/bash

#SBATCH --job-name=corn_train           # job name
#SBATCH --partition=a100                # partition (queue) # gpu # a100
#SBATCH --gres=gpu:1                    # number of gpus per node # a4500 # a100_2g.10gb #a100
#SBATCH --nodes=1                       # node count
#SBATCH --ntasks=1                      # total number of tasks across all nodes
#SBATCH --mem=64G                       # total memory per node 
#SBATCH --time=06:00:00                 # total run time limit (DD-HH:MM:SS)
#SBATCH --output=slurm_output/train_%N_%j.out # output file path

module purge
module load miniconda
source ~/.bashrc
conda activate yolov8_cuda12

yolo train name=corn_kernel_sgd \
    data=corn_yolo.yaml \
    model=yolov8n.pt \
    optimizer=SGD \
    epochs=300 \
    batch=16 \
    device=0 \
    patience=50 \
    max_det=900 \
    augment=True \
    erasing=0.1 \
    degrees=180 \
    flipud=0.5 \
    fliplr=0.5 \
    hsv_h=0.1 \
    hsv_s=0.1 \
    hsv_v=0.2

conda deactivate
#!/bin/bash

#SBATCH --job-name=corn_infer           # job name
#SBATCH --partition=a100                # partition (queue) # gpu # a100
#SBATCH --gres=gpu:1                    # number of gpus per node # a4500 # a100_2g.10gb #a100
#SBATCH --nodes=1                       # node count
#SBATCH --ntasks=1                      # total number of tasks across all nodes
#SBATCH --mem=64G                       # total memory per node 
#SBATCH --time=06:00:00                 # total run time limit (DD-HH:MM:SS)
#SBATCH --output=slurm_output/pred_%N_%j.out # output file path

module purge
module load miniconda
source ~/.bashrc
conda activate yolov8_cuda12

yolo detect predict name=corn_kernel_baseline_pred_iou06 \
    model='runs/detect/corn_kernel_baseline/weights/best.pt' \
    source='datasets/corn_kernel_yolo/images/test/*.jpg' \
    max_det=500 \
    iou=0.6 \
    show_labels=False \
    line_width=1 \
    save=True \
    save_txt=True \
    classes=0

conda deactivate
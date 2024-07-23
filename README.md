# Corny
This repository contains the code and dataset used in the
[Neuromatch Academy Deep Learning Course 2024](https://neuromatch.io/deep-learning-course/) 
group project "Corny".

## Dataset
The dataset used in this project is the *Base* portion of the [Intelinair Corn Kernel Counting dataset](https://registry.opendata.aws/intelinair_corn_kernel_counting/).[^1] 
The labels have been filtered to retain only the  `Kernel` class. In order to use the dataset with [YOLOv8](https://yolov8.com/) for object detection, the labels are converted from COCO format to YOLO format using the [JSON2YOLO](https://github.com/ultralytics/JSON2YOLO) toolkit. As YOLO requires class numbers to be zero-indexed, the `Kernel` class is re-assigned the class number `0`.

## Set up conda environment
1. Depending on your CUDA version, create the [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) environment `pytorch_cuda11` or `pytorch_cuda12` from the corresponding `envs/env_cuda11.yml` or `envs/env_cuda12.yml` file. For example, to create the `pytorch_cuda11` environment: 
    ```
    conda env create -f envs/env_cuda11.yml
    ```

2. Activate the environment
    ```
    conda activate pytorch_cuda11
    ```

3. Deactivate the environment
    ```
    conda deactivate
    ```

[^1]: **Hobbs, J., Khachatryan, V., Anandan, B. S., Hovhannisyan, H., & Wilson, D. (2021).** Broad Dataset and Methods for Counting and Localization of On-Ear Corn Kernels. *Frontiers in Robotics and AI*, 8. [https://doi.org/10.3389/frobt.2021.627009](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2021.627009)
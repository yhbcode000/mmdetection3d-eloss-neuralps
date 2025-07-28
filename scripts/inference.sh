#!/bin/bash

cd /workspace/mmdetection3d

# --- Configuration ---

# 1. Path to the Point Cloud Data file (.bin)
PCD_FILE="demo/data/kitti/000008.bin"

# 2. Path to the corresponding Image file (.png)
IMAGE_FILE="demo/data/kitti/000008.png"

# 3. Path to the Ground Truth Annotation file (.pkl)
#    (Used for visualizing ground truth boxes alongside predictions)
ANNOTATION_FILE="demo/data/kitti/000008.pkl"

# 4. Path to the model's configuration file (.py)
CONFIG_FILE="configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car.py"

# 5. Path to the trained model checkpoint file (.pth)
CHECKPOINT_FILE="checkpoints/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_20220331_134606-d42d15ed.pth"

# --- Execute Demo ---

# uv run demo/multi_modality_demo.py \
#     ${PCD_FILE} \
#     ${IMAGE_FILE} \
#     ${ANNOTATION_FILE} \
#     ${CONFIG_FILE} \
#     ${CHECKPOINT_FILE} \
#     --cam-type CAM2 \
#     --show

uv run demo/pcd_demo.py \
    ${PCD_FILE} \
    ${CONFIG_FILE} \
    ${CHECKPOINT_FILE} \
    --show \

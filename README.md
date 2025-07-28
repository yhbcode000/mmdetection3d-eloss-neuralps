# Stabilizing Information Flow: Entropy Regularization for Safe and Interpretable Autonomous Driving Perception

This repository contains the official PyTorch implementation for the paper: **"Stabilizing Information Flow: Entropy Regularization for Safe and Interpretable Autonomous Driving Perception"** (NeurIPS 2025).

Our work introduces **E<sub>loss</sub>**, a novel information-theoretic regularizer that enforces stable and interpretable information flow in deep perception networks for autonomous driving.

## 🎯 Abstract

Deep perception networks in autonomous driving traditionally rely on data-intensive training and post-hoc anomaly detection, often ignoring the information-theoretic constraints that govern stable information processing. We reconceptualize deep encoders as hierarchical communication chains that compress sensory inputs into latent features. Within this framework, we introduce **E<sub>loss</sub>**, an entropy-based regularizer that encourages smooth, monotonic entropy decay with network depth. This approach unifies information-theoretic stability with standard perception tasks, enabling principled detection of anomalous sensor inputs through entropy deviations. Our experiments on KITTI and nuScenes show that models trained with E<sub>loss</sub> achieve competitive accuracy while dramatically enhancing sensitivity to anomalies by up to **two orders of magnitude**.

## 🔑 Key Idea: A Communication View of Perception

We model a deep encoder not as a single black box, but as a chain of compression layers. For a perception system to be robust, the information flow through this chain must be stable.

* **(a) Conventional View:** The encoder is treated as a single block, hiding unstable jumps in information entropy. Anomaly detection is often unreliable.

* **(b) Our Stable Compression View:** We decompose the encoder and encourage a smooth, monotonic decay of entropy across layers. Anomalous inputs (dashed red) disrupt this stable profile, making them easy to spot.

Our proposed regularizer, **E<sub>loss</sub>**, penalizes the variance of entropy drops between consecutive layers, effectively enforcing this stable behavior during training.

## ✨ Main Results

### 1. Unparalleled Sensitivity to Anomalous Inputs

E<sub>loss</sub> provides a powerful, direct signal for out-of-distribution or corrupted inputs. Unlike standard model confidence, which changes modestly, E<sub>loss</sub> spikes by orders of magnitude, making it a reliable safety indicator.

| **Model & Dataset**       | **Metric**           | **Clean Input** | **Noisy Input** | **Sensitivity (Δ)** |
| ------------------------- | -------------------- | --------------- | --------------- | ------------------- |
| PointPillars nuScenes     | Confidence           | 0.168           | 0.128           | -23.7%              |
| **PointPillars nuScenes** | **E<sub>loss</sub>** | **2.56E-4**     | **1.48E-1**     | **+57,746%** 🚀      |
| VoxelNet KITTI            | Confidence           | 0.495           | 0.248           | -49.9%              |
| **VoxelNet KITTI**        | **E<sub>loss</sub>** | **1.58E-3**     | **9.09E-3**     | **+473.5%** 🔥       |

### 2. Emergent Monotonic Compression

By regularizing the *flow* of information, E<sub>loss</sub> reshapes the network's internal representations. Feature distributions become progressively more compact and isotropic, demonstrating that the network has learned to discard irrelevant information smoothly and efficiently.

*Top row: Without E<sub>loss</sub>, feature distributions are irregular. Bottom row: With E<sub>loss</sub>, features show stable, monotonic compression.*

### 3. Competitive Detection Accuracy

While its primary benefit is robustness, E<sub>loss</sub> also achieves competitive or improved 3D object detection accuracy, particularly for smaller and less common object classes like Pedestrians and Cyclists. For full results, please see Section 4.2 of our paper.

## ⚙️ Installation

This project is built on the [MMDetection3D](https://github.com/open-mmlab/mmdetection3d) toolbox.


## 🚀 Getting Started

### Data Preparation

Please follow the official MMDetection3D guides for preparing the **KITTI** and **nuScenes** datasets.

* [KITTI Data Preparation Guide](https://mmdetection3d.readthedocs.io/en/latest/advanced_guides/datasets/kitti.html)

* [nuScenes Data Preparation Guide](https://mmdetection3d.readthedocs.io/en/latest/advanced_guides/datasets/nuscenes.html)

We recommend creating symbolic links to the dataset root directories:

```
mkdir data
ln -s /path/to/kitti data/kitti
ln -s /path/to/nuscenes data/nuscenes

```

### Training

To train a model with E<sub>loss</sub>, use the provided configuration files.

**Single-GPU Training:**

```
python tools/train.py configs/xnet/your_config_file.py

```

**Multi-GPU Training:**

```
# Example using 4 GPUs
./tools/dist_train.sh configs/xnet/your_config_file.py 4

```

Training logs and model checkpoints will be saved to the `work_dirs/` directory.

### Evaluation

To evaluate a trained model:

**Single-GPU Evaluation:**

```
python tools/test.py \
    configs/xnet/your_config_file.py \
    work_dirs/your_config_file/latest.pth \
    --eval bbox

```

**Multi-GPU Evaluation:**

```
./tools/dist_test.sh \
    configs/xnet/your_config_file.py \
    work_dirs/your_config_file/latest.pth \
    4 # Number of GPUs
    --eval bbox

```

## 💡 Core Implementation

The main files for our project are located in the following directories. These files contain the core logic for the E<sub>loss</sub> regularizer and its integration into the detection pipeline.

* `configs/xnet/`

  * This directory contains all the configuration files needed to reproduce our experiments. Each file defines the model architecture, dataset, and training schedule.

* `mmdet3d/models/detectors/x_net.py`

  * This file defines the main detector architecture (`XNet`). It integrates the E<sub>loss</sub> calculation into the forward pass of the model.

* `mmdet3d/models/losses/xnet_loss.py`

  * This file contains the standalone implementation of our proposed **E<sub>loss</sub>** regularizer. It computes the variance of entropy drops between layer features.

## 🙏 Acknowledgements

This project is built upon the excellent [MMDetection3D](https://github.com/open-mmlab/mmdetection3d) codebase. We thank the OpenMMLab team for their contributions to the community.

4090 cuda12.4
uv pip install -e . --no-build-isolation

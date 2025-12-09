# M$^3$-ReID: Unifying Multi-View, Granularity, and Modality for Video-Based Visible-Infrared Person Re-Identification

[![Paper](https://img.shields.io/badge/Paper-IEEE%20TIFS-00629B?logo=gitbook&logoColor=white)](https://ieeexplore.ieee.org/document/11275868)
[![Python](https://img.shields.io/badge/python-3.9+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green?logo=opensourceinitiative&logoColor=white)](./LICENSE)

This repository contains the **official implementation** of the paper:

[**M$^3$-ReID: Unifying Multi-View, Granularity, and Modality for Video-Based Visible-Infrared Person Re-Identification** *Accepted by IEEE Transactions on Information Forensics and Security (TIFS), 2025.*](https://ieeexplore.ieee.org/document/11275868)

By [Tengfei Liang](https://scholar.google.com/citations?user=YE6fPvgAAAAJ),
[Yi Jin](https://scholar.google.com/citations?user=NQAenU0AAAAJ),
[Zhun Zhong](https://scholar.google.com/citations?user=nZizkQ0AAAAJ),
[Xin Chen](https://scholar.google.com/citations?user=yqpHF90AAAAJ),
[Xianjia Meng](https://dblp.org/pid/201/8107.html),
[Tao Wang](https://scholar.google.com/citations?user=F3C5oAcAAAAJ),
[Yidong Li](https://scholar.google.com/citations?user=3PagRQEAAAAJ).

*Notes:*
This repository offers the complete implementation of the method, featuring a well-organized directory structure and detailed comments that facilitate model training and testing. 
Moreover, the codebase is designed to be easy to extend, allowing users to quickly adapt or build upon the existing framework. 
It is hoped that this work can serve as a **new strong and simple baseline** for video-based visible–infrared person re-identification and further contribute to the advancement of the VVI-ReID field.


## 📖 Introduction

Video-based visible-infrared person re-identification (VVI-ReID) task focuses on cross-modality retrieval of pedestrian videos, which are captured in visible and infrared modalities by non-overlapping cameras across diverse scenes, and holds significant value for security surveillance scenarios. 
It's a challenging task due to three main issues: the difficulty of capturing comprehensive spatio-temporal cues, intra-class variations within video sequences, and inter-modality discrepancies between visible
and infrared data.

To address these, we propose **M$^3$-ReID**, a unified framework that simultaneously handles:
* **Multi-View Learning (MVL):** Captures diverse spatio-temporal patterns (Spatial, Temporal-Width, Temporal-Height views) with the Diverse Attention Constraint.
* **Multi-Granularity Representation (MGR):** Optimizes features across both fine-grained frame level (with Orthogonal Frame Regularizer) and coarse-grained video level.
* **Multi-Modality Alignment (MMA):** Explicitly aligns metric learning with cross-modality retrieval goals using a retrieval-oriented contrastive objective.

![Architecture](figs/m3reid_architecture.png)

## 🧱 Code Structure

The codebase is organized into modular components to facilitate understanding and maintenance.

```text
M3-ReID/
├── data/                       # Data Pipeline
│   ├── manager.py              # DataManager: Parses raw dataset structures (HITSZ-VCM/BUPTCampus)
│   ├── dataset.py              # VideoVIDataset: Handles video sequence loading and sampling
│   ├── sampler.py              # Diverse sampling strategies (PxK, Cross-Modality, Identity-Balanced, etc.)
│   └── transform.py            # Custom transforms (Sync-Augmentation, WeightedGrayscale, StyleVariation)
├── losses/                     # Loss Functions
│   ├── mma_loss.py             # Multi-Modality Alignment (MMA) Loss
│   └── sep_loss.py             # Separation Loss (Used for OFR and DAC)
├── models/                     # Model Definitions
│   ├── backbones/              # Backbone Networks
│   │   └── resnet.py           # ResNet backbone implementation (FC layers removed)
│   ├── modules/                # Custom Modules
│   │   ├── mvl_attention.py    # Multi-View Learning (MVL) Attention module
│   │   ├── non_local.py        # Spatio-Temporal Non-Local Block
│   │   └── normalize.py        # Feature Normalization layer
│   └── model_m3reid.py         # The M3-ReID architecture definition
├── tools/                      # Utilities
│   ├── eval_metrics.py         # Evaluation (CMC, mAP, mINP)
│   └── utils.py                # Logger, random seed, path handling
├── train_m3reid.py             # Training Entry Point
└── test_m3reid.py              # Evaluation Entry Point
```

### Key Features

We have implemented several engineering optimizations to ensure robustness, efficiency, and extensibility:

* **🧩 Modular & Extensible Design**: The framework decouples the Data, Model, and Loss components. You can easily plug in new backbones, custom losses, or optimization strategies with minimal changes to the core codebase.
* **🎞️ Synchronized Video Augmentation**: Unlike previous VVI-ReID methods that often overlook temporal consistency during augmentation, our `SyncTrackTransform` ensures that random operations (e.g., cropping, flipping, erasing) are applied **identically across all frames** within a video tracklet. This preserves temporal coherence and avoids introducing artificial jitter noise that could confuse the temporal learning modules.
* **🎨 Rich Sampling Strategies**: `sampler.py` provides diverse sampling logic, including standard PxK sampling, **Identity-Balanced Cross-Modality sampling** (guaranteeing half IR / half RGB per batch), and random sampling, satisfying various training requirements.
* **🔥 Mixed Precision Training**: The training loop natively supports **Automatic Mixed Precision (AMP)** (`--fp16`), allowing for reduced GPU memory usage and faster training throughput without compromising performance.
* **📈 Comprehensive Logging**: We provide a dual-logging system that simultaneously records training progress to **console**, **text files**, and **TensorBoard**. This makes it easy to monitor loss curves and accuracy in real-time.
* **🔋 GPU-Based Metric Calculation**: The evaluation script (`eval_metrics.py`) computes CMC, mAP, and mINP entirely on the **GPU**. This significantly accelerates the evaluation process, especially for large-scale gallery sets, compared to traditional CPU-based implementations.


## 🛠️ Installation

### Requirements
* CUDA 11.3+
* Python 3.9+
* PyTorch 1.12+
* TorchVision 0.13+
* Numpy
* Pillow
* TensorBoard

*P.S. In our experiments, we used CUDA 11.3, Python 3.9, PyTorch 1.12.1, TorchVision 0.13.1, NumPy 1.26.4, Pillow 11.1.0, and TensorBoard 2.19.0. 
Since our code depends only on these common libraries and does not require any unusual packages, higher or lower versions of these packages may also be supported. 
It is not strictly necessary to follow the exact versions listed in our [requirements.txt](./requirements.txt). 
Users are free to explore different versions; if any warnings appear, they can usually be resolved with small adjustments to the code.*

### Step-by-step

#### 1. Get the Code
You can clone the repository using Git or download the ZIP archive manually.

**Option A: Using Git**
```bash
git clone https://github.com/workingcoder/M3-ReID.git
cd M3-ReID
```

**Option B: Manual Download**

1. Click the "Code" button at the top right of this page.

2. Select "Download ZIP" and extract it.

3. Enter the directory via terminal:

```bash
cd M3-ReID-main
```

#### 2. Install Dependencies

Install the required packages. 
Note that the provided [requirements.txt](./requirements.txt) is intended as a reference rather than a strict specification, and users are free to explore different versions.

```bash
pip install -r requirements.txt
```


## 📂 Data Preparation

During our experiments, we evaluated the proposed method on two publicly available datasets: **HITSZ-VCM** and **BUPTCampus**, which are commonly used benchmarks for VVI-ReID.

**Please download the corresponding datasets from their official sources.** 
Our code is designed to support their original directory structures. 
The file organization is shown below for reference:

### 1. HITSZ-VCM Dataset
```text
HITSZ-VCM/
├── Train/                  # Training data
│   ├── 0001/
│   │   ├── ir/
│   │   │   ├── D2/
│   │   │   │   ├── 1.jpg
│   │   │   │   ├── 6.jpg
│   │   │   │   └── ...
│   │   │   ├── D4/
│   │   │   └── ...
│   │   └── rgb/
│   ├── 0002/
│   ├── 0003/
│   └── ...
├── Test/                   # Testing data
│   └── ...
└── info/                   # Metadata files
    ├── track_train_info.txt
    ├── track_test_info.txt
    ├── train_name.txt
    ├── test_name.txt
    └── query_IDX.txt
```

### 2. BUPTCampus Dataset
```text
BUPTCampus/
├── DATA/
│   ├── 1/
│   │   ├── IR/
│   │   │   ├── CQ1/
│   │   │   │   ├── 1_IR_CQ1_1_8535.jpg
│   │   │   │   ├── 1_IR_CQ1_1_8536.jpg
│   │   │   │   └── ...
│   │   │   ├── G25/
│   │   │   └── ...
│   │   ├── RGB/
│   │   │   ├── CQ1/
│   │   │   │   ├── 1_RGB_CQ1_1_8535.jpg
│   │   │   │   ├── 1_RGB_CQ1_1_8536.jpg
│   │   │   │   └── ...
│   │   │   ├── G25/
│   │   │   └── ...
│   │   └── FakeIR/
│   │       ├── CQ1/
│   │       │   ├── 1_RGB_CQ1_1_8535.jpg
│   │       │   ├── 1_RGB_CQ1_1_8536.jpg
│   │       │   └── ...
│   │       ├── G25/
│   │       └── ...
│   ├── 2/
│   ├── 3/
│   └── ...
├── data_paths.json
├── train.txt
├── train_aux.txt
├── query.txt
└── gallery.txt
```


## 🚀 Training

We provide training scripts ([train_m3reid.py](./train_m3reid.py)) for both the HITSZ-VCM and BUPTCampus datasets. 
In our experiments, Mixed Precision Training (AMP) is enabled by default to reduce GPU memory consumption and accelerate training. 
Users can adjust all parameters as needed for further exploration.

### Arguments Description

We provide a comprehensive set of arguments to configure the training process.

**Data Configurations**
* `--dataset`: Target dataset name. Choices: `HITSZVCM`, `BUPTCampus`.
* `--dataset_dir`: Path to the root directory of the dataset.
* `--img_h`, `--img_w`: Input image resolution. Default: `288` x `144`.
* `--p_num`: Number of identities (P) per batch. Default: `4`.
* `--k_num`: Number of images (K) per identity. Default: `8`.
    * *Note: The total Batch Size is calculated as `p_num * k_num`.*
* `--workers`: Number of data loading threads. Default: `4`.

**Optimization**
* `--lr`: Initial learning rate. Default: `0.0002`.
* `--wd`: Weight decay. Default: `0.0005`.

**Training & Logging**
* `--log_interval`: Frequency of logging training status (in batches).
* `--test_interval`: Frequency of evaluating on the test set (in epochs).
* `--save_interval`: Frequency of saving model checkpoints (in epochs).
* `--resume`: Path to a checkpoint file to resume training from.
* `--seed`: Random seed for reproducibility. Default: `0`.
* `--fp16`: **[Flag]** Enable Automatic Mixed Precision (AMP) training to save memory and speed up.
* `--desc`: A short description string for the experiment (used for naming the log folder).
* `--gpu`: GPU device ID to use (e.g., `0`).

### 1. Train on HITSZ-VCM

```bash
python train_m3reid.py \
    --dataset HITSZVCM \
    --dataset_dir path/to/HITSZ-VCM \
    --img_h 288 \
    --img_w 144 \
    --p_num 4 \
    --k_num 8 \
    --lr 0.0002 \
    --fp16 \
    --desc M3-ReID \
    --gpu 0
```

### 2. Train on BUPTCampus

```bash
python train_m3reid.py \
    --dataset BUPTCampus \
    --dataset_dir path/to/BUPTCampus \
    --img_h 288 \
    --img_w 144 \
    --p_num 4 \
    --k_num 8 \
    --lr 0.0001 \
    --fp16 \
    --desc M3-ReID \
    --gpu 0
```

### 3. Check Logs

We provide three complementary ways to monitor the training process:

* **Console Output**: Real-time progress, loss values, and accuracy metrics are printed directly to the terminal standard output.
* **Text Log**: A complete log file is automatically saved in the `ckptlog` directory.
    * *Path Structure*: `ckptlog/<DATASET>/Time-<TIMESTAMP>_<DESC>/log_*.txt`
    * *Example*: `ckptlog/HITSZVCM/Time-2025-12-07_M3-ReID/log_Time-2025-12-07_M3-ReID.txt`
* **TensorBoard**: Visualizations of loss curves and evaluation metrics are recorded in the `tensorboard` subdirectory.

To view training curves via TensorBoard:

```bash
# Option A: Point to the specific experiment directory
tensorboard --logdir ckptlog/HITSZVCM/Time-2025-XX-XX_M3-ReID/tensorboard --port 6006

# Option B: Point to the root log directory to compare multiple experiments
tensorboard --logdir ckptlog/HITSZVCM/ --port 6006

# Then open http://localhost:6006 in your browser to view the curves
```


## ⚡ Evaluation

The testing script ([test_m3reid.py](./test_m3reid.py)) performs cross-modality retrieval evaluation on **both** "Visible-to-Infrared" (V2I) and "Infrared-to-Visible" (I2V) modes. It calculates and reports Rank-1, Rank-5, Rank-10, mAP, and mINP metrics.

### Arguments Description

**Data Configuration**
* `--dataset`: Target dataset name. Choices: `HITSZVCM`, `BUPTCampus`.
* `--dataset_dir`: Path to the root directory of the dataset.
* `--img_h`, `--img_w`: Input image resolution. Default: `288` x `144`.
* `--batch_size`: Batch size for testing. Default: `32`.
    * *Note: This refers to the number of video tracks processed simultaneously. Increasing this (e.g., to 64 or 128) can significantly speed up inference if GPU memory allows.*
* `--workers`: Number of data loading threads. Default: `4`.

**Model & Environment**
* `--resume`: **[Required]** Path to the trained model checkpoint file (e.g., `path/to/modelckpt/model_epoch-200.pth`).
* `--desc`: A short description string for this testing process (used for naming the log folder).
* `--gpu`: GPU device ID to use (e.g., `0`).

### 1. Evaluate on HITSZ-VCM

```bash
python test_m3reid.py \
    --dataset HITSZVCM \
    --dataset_dir path/to/HITSZ-VCM \
    --img_h 288 \
    --img_w 144 \
    --batch_size 32 \
    --resume ckptlog/HITSZVCM/Time-Your_Desc/modelckpt/model_epoch-200.pth \
    --desc M3-ReID \
    --gpu 0
```

### 2. Evaluate on BUPTCampus

```bash
python test_m3reid.py \
    --dataset BUPTCampus \
    --dataset_dir path/to/BUPTCampus \
    --img_h 288 \
    --img_w 144 \
    --batch_size 32 \
    --resume ckptlog/BUPTCampus/Time-Your_Desc/modelckpt/model_epoch-200.pth \
    --desc M3-ReID \
    --gpu 0
```


## 📊 Results

Our method achieves good performance on VVI-ReID benchmarks. 
Detailed cross-modality retrieval performance is reported below.

### 1. Performance on HITSZ-VCM

| Method | Mode | Rank-1 | Rank-5 | Rank-10 | Rank-20 | mAP |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **M$^3$-ReID** | **Visible to Infrared** | **75.31%** | **85.17%** | **88.66%** | **91.43%** | **60.51%** |
| **M$^3$-ReID** | **Infrared to Visible** | **73.21%** | **82.81%** | **86.74%** | **89.70%** | **59.02%** |

### 2. Performance on BUPTCampus

| Method | Mode | Rank-1 | Rank-5 | Rank-10 | Rank-20 | mAP |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **M$^3$-ReID** | **Visible to Infrared** | **70.74%** | **85.19%** | **89.44%** | **92.04%** | **63.75%** |
| **M$^3$-ReID** | **Infrared to Visible** | **68.28%** | **85.45%** | **88.99%** | **91.79%** | **64.96%** |

*(Results cited from Table I and Table II of the [original paper](https://ieeexplore.ieee.org/document/11275868))*


## 🔧 Quickstart: How to Develop Your Own Method

Our framework is designed to be easily extensible. 
Follow these steps to implement a new VVI-ReID method (e.g., `YourMethod`).

### Step 1: Define Your Model
Create a new model file `models/model_yourmethod.py`. You can refer to `models/model_m3reid.py` as a template.

1.  **Inherit `nn.Module`**: Define your class (e.g., `class YourMethod(nn.Module)`).
2.  **Backbone Integration**: Initialize your backbone in `__init__`.
    * *Flexibility*: While we use ResNet-50 by default, you can easily integrate other backbones (e.g., **ViT**, **Swin Transformer**, or **ConvNeXt**) by adding their implementations to `models/backbones/` and importing them here.
3.  **Custom Modules**: Add your specific components (e.g., Attention mechanisms, Temporal Aggregation) in `models/modules/` and assemble them in your model.
4.  **Forward Pass**: Implement the feature extraction flow. Ensure `forward()` returns the necessary tensors for loss calculation (during training) or the final embedding (during evaluation).

### Step 2: Create Training Script
Create a new training script `train_yourmethod.py`. You can duplicate `train_m3reid.py` as a robust starting point and modify:

1.  **Import Model**:
    ```python
    # from models.model_m3reid import M3ReID  <-- Replace with your model
    from models.model_yourmethod import YourMethod
    ```
2.  **Custom Data Pipeline (Samplers & Transforms)**:
    * You can customize augmentations in `data/transform.py` or design new sampling strategies in `data/sampler.py`.
    * *Note*: Our existing samplers already support **balanced batch sampling (half IR, half RGB)**. Even if not used in our default configuration, you can easily enable them to get separate samples for each modality using the `track_mid` (modality labels) from a batch returned by the dataloader.
3.  **Define Custom Losses**:
    * If your method requires unique loss functions, implement them in the `losses/` directory (e.g., `losses/your_new_loss.py`).
    * Import and initialize them in your training script:
        ```python
        from losses.your_new_loss import YourNewLoss
        # ... inside the script ...
        criterion_new = YourNewLoss().cuda()
        ```
4.  **Initialize Model**:
    ```python
    model = YourMethod(sample_seq_num, num_train_class).cuda()
    ```
5.  **Optimization Loop**:
    * Update the forward/backward logic inside the training loop (`for batch_idx, batch_data in enumerate(train_loader):`).
    * *Flexibility*: You are free to customize the optimization flow to support complex paradigms, such as **Generative Adversarial Networks (GANs)** (alternating optimizer steps) or multi-stage training, by modifying how `optimizer.step()` and gradients are handled here.
6.  **Custom Logging & Visualization**:
    * **Text Logs**: Since `sys.stdout` is redirected to the `Logger` class, any `print()` statement inside the training loop will automatically be saved to the log file in `ckptlog/`. You can modify the formatted string in the loop to include your new losses or any other values.
    * **TensorBoard**: Use the initialized `writer` object to track your new metrics. Simply add `writer.add_scalar('group/metric_name', value, iter_num)` alongside the existing logging code to visualize your custom curves in real-time.

### Step 3: Create Testing Script
Create a new testing script `test_yourmethod.py`. Duplicate `test_m3reid.py` and update the model import:

```python
# from models.model_m3reid import M3ReID  <-- Replace with your model
from models.model_yourmethod import YourMethod

# ... inside the script ...
model = YourMethod(sample_seq_num, num_train_class).cuda()
```

### Step 4: Run you method
Now you can run your own method using:

```bash
python train_yourmethod.py --dataset HITSZVCM ...
```

```bash
python test_yourmethod.py --dataset HITSZVCM ...
```

Enjoy your research journey!


## 📝 Citation

If you find this code or paper useful for your research, please cite:

```bibtex
@article{TIFS_M3ReID,
  author  = {Tengfei Liang and 
             Yi Jin and
             Zhun Zhong and
             Xin Chen and
             Xianjia Meng and
             Tao Wang and
             Yidong Li},
  title   = {M$^3$-ReID: Unifying Multi-View, Granularity, and Modality for Video-Based Visible-Infrared Person Re-Identification},
  journal = {IEEE Transactions on Information Forensics and Security},
  year    = {2025}
}
```


## 📄 License

This repository is released under the MIT license. Please see the [LICENSE](./LICENSE) file for more information.

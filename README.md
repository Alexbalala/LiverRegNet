# Accurate Liver Registration of 3D Ultrasound and CT Volume: An Open Dataset and a Model Fusion Method

This repository contains the code and dataset for the paper **"Accurate Liver Registration of 3D Ultrasound and CT Volume: An Open Dataset and a Model Fusion Method"**.

---

## Overview

Liver registration between ultrasound (US) and computed tomography (CT) is critical for various medical applications, such as surgical navigation and interventional guidance. This work introduces:

- **A new open dataset**: Featuring paired 3D ultrasound and CT volumes with accurate liver segmentation annotations.
- **A model fusion-based registration method**: Combining deep learning and traditional optimization approaches for robust and accurate results.

---

## Dataset

The dataset includes:

1. **3D Ultrasound (US) Volumes**: High-quality, segmented US images with corresponding labels.
2. **3D Computed Tomography (CT) Volumes**: Paired CT images with liver segmentation ground truth.

### Data Format

- **US Data**: Provided in NIfTI (.nii) format.
- **CT Data**: Provided in NIfTI (.nii) format.
- **Annotations**: Liver segmentations are included as binary masks.

### Access

Please [request access](mailto:your-email@example.com) to download the dataset.

---

## Code

This repository provides the implementation of our proposed registration pipeline:

1. **Preprocessing**: Normalization, cropping, and resampling of US and CT data.
2. **Model Fusion Registration**:
   - **Deep Learning Component**: Pre-trained feature extraction for initial alignment.
   - **Optimization Component**: Fine-tuning registration using similarity measures (e.g., mutual information).
3. **Evaluation Metrics**: Dice Similarity Coefficient (DSC), Hausdorff Distance, etc.

---

### Setup

#### Prerequisites

- Python >= 3.8
- PyTorch >= 1.10
- Numpy, Scipy, SimpleITK, and other dependencies listed in `requirements.txt`.

#### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/3D-US-CT-Liver-Registration.git
   cd 3D-US-CT-Liver-Registration

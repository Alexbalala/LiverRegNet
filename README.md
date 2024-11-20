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

1. **3D Ultrasound (US) Volumes**
2. **3D Computed Tomography (CT) Volumes**

### Data Format

- **US Data**: Provided in NIfTI (.nii) format.
- **CT Data**: Provided in NIfTI (.nii) format.
- **label**: All data are registered in pairs, and you can transform them freely. The transformation matrix is ​​your label.

### Access

Please [request access](mailto:yw.xu1@siat.ac.cn) to download the dataset.

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

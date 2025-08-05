# Multi-Disease Lung Segmentation and Reconstruction

<div align="center">

![Research Banner](https://img.shields.io/badge/Research-IIT%20Kharagpur-orange?style=for-the-badge)
![Deep Learning](https://img.shields.io/badge/Deep%20Learning-U--Net-blue?style=for-the-badge)
![3D Reconstruction](https://img.shields.io/badge/3D%20Reconstruction-Surface%20Topology-green?style=for-the-badge)
![Medical AI](https://img.shields.io/badge/Medical%20AI-Lung%20Analysis-red?style=for-the-badge)

**Advancing Medical Imaging with AI-Powered 3D Lung Analysis**

*Developed at Indian Institute of Technology Kharagpur*

</div>

---

## 🔬 Research Overview

This repository presents a groundbreaking approach to medical image analysis that combines the power of deep learning with advanced surface topology algorithms to revolutionize lung disease diagnosis and treatment planning. Our research introduces a unified pipeline that integrates 2D CT-slice segmentation with 3D surface reconstruction, producing anatomically accurate, color-coded lung models that highlight infection severity and disease progression.

The work addresses critical challenges in medical imaging including variability in CT image quality, difficulties in handling pathological cases, limited annotated datasets, high computational demands, and the absence of standardized AI models. By leveraging U-Net's encoder-decoder architecture for precise segmentation and combining it with surface topology-based reconstruction algorithms, we achieve unprecedented accuracy in identifying lung structures and pathologies.

Our methodology represents a significant advancement in automated lung analysis, enhancing diagnostic accuracy and clinical decision-making through high-quality 3D reconstructions. The integration of deep learning with topology-driven reconstruction provides a robust foundation for improved medical imaging applications, supporting precise disease assessment and facilitating applications in detection, surgical planning, education, and computational medicine.


### Key Features:

*   **U-Net Based Segmentation:** Utilizes a U-Net based deep learning model for precise segmentation of COVID-19 lesions, pneumonia, and lung cancer from 2D CT slices.
*   **3D Surface Reconstruction:** Employs the Marching Cubes algorithm to convert segmented voxel data into continuous surface meshes, followed by Radial Basis Function (RBF) interpolation and the Ball-Pivot Algorithm (BPA) for smoothing and watertight mesh generation.
*   **Hounsfield Unit (HU) based Color Coding:** Visualizes disease severity by color-coding 3D lung models based on Hounsfield Unit thresholds (green for less severe, yellow for mildly infected, red for severely infected regions).
*   **Multi-Disease Analysis:** Capable of segmenting and analyzing both lung cancer and pneumonia/COVID-19 infections.
*   **Robustness:** Designed to handle variability in CT image quality and pathological cases, ensuring robust segmentation even with lung deformations.

---

## 🚀 Technical Details

### Segmentation Module (`Pnuemonia_and_Cancer_Segmentation`)

This module focuses on the 2D segmentation of lung CT slices using U-Net models. It includes:

*   `Models/unet_family_models.py` and `Models/unet_model.py`: Implementations of U-Net architectures used for segmentation.
*   `dataloader.py`: Handles data loading and preprocessing for both Lung Cancer and COVID-19 datasets.
*   `train_code.py` and `test_code.py`: Scripts for training and evaluating the segmentation models.
*   `cancer and covid training.ipynb`: Jupyter notebook for interactive training and experimentation.

### Reconstruction Module (`Pnuemonia_and_Cancer_Reconstruction`)

This module is responsible for generating 3D lung models from the 2D segmented slices and performing surface reconstruction. Key components include:

*   `main.py`: Orchestrates the entire 3D reconstruction pipeline, from downloading datasets to generating STL files and point clouds.
*   `3dMesh_visualization.py`: Utilities for visualizing 3D meshes.
*   `rbf_helper_functions.py`: Helper functions for Radial Basis Function (RBF) interpolation, crucial for smoothing reconstructed surfaces.

### Classification Module (`Pnuemonia_Classification`)

This module focuses on the classification of pneumonia using various deep learning models. It includes:

*   `Models/mobilenetv2_model.py`, `Models/resnet_model.py`, `Models/shufflenet_model.py`: Implementations of different classification models.
*   `dataloader.py` and `dataset_creator.py`: Handle data loading and combining multiple pneumonia datasets.
*   `train_code.py`: Script for training the classification models.

---

## 📂 Repository Structure

```
Multi-Disease-Lung-Segmentation-and-Reconstruction/
├── Pnuemonia_and_Cancer_Reconstruction/
│   ├── __init__.py
│   ├── 3dMesh_visualization.py
│   ├── main.py
│   └── rbf_helper_functions.py
├── Pnuemonia_and_Cancer_Segmentation/
│   ├── Models/
│   │   ├── __init__.py
│   │   ├── unet_family_models.py
│   │   └── unet_model.py
│   ├── __init__.py
│   ├── cancer and covid training.ipynb
│   ├── dataloader.py
│   ├── main.py
│   ├── metrics.py
│   ├── test_code.py
│   └── train_code.py
├── Pnuemonia_Classification/
│   ├── Models/
│   │   ├── __init__.py
│   │   ├── mobilenetv2_model.py
│   │   ├── resnet_model.py
│   │   └── shufflenet_model.py
│   ├── __init__.py
│   ├── dataloader.py
│   ├── dataset_creator.py
│   ├── main.py
│   └── train_code.py
├── LICENSE
├── README.md
└── requirements.txt
```

---

## 📊 Results and Visualizations

Our experimental results on 20 CT volumes demonstrate that the hybrid approach preserves fine anatomical details and enables intuitive visualization of disease progression. The U-Net model achieved Dice scores of 0.8837 for cancer and 0.6168 for pneumonia segmentation on 2D data.

### Sample Visualizations:


---

## 🎓 Academic Contributions

This work was conducted as part of an internship at IIT Kharagpur, under the guidance of:

*   **Subhamoy Mandal Sir**
*   **Partha Acharya** (IIT KGP PhD)
*   **Suman Chakraborty Sir** (IIT KGP Director)

### Citation

*(Paper not yet published. Citation and paper link will be provided here upon publication.)*

---

## 🤝 Contributing

We welcome contributions to this project! Please feel free to fork the repository, open issues, or submit pull requests.

---

## 📄 License

This project is licensed under the [LICENSE](LICENSE) file.


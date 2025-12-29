# Satellite Image Land-Cover Segmentation Using Deep Learning

## Project Overview

This project implements a deep learning–based **semantic land-cover segmentation system** using high-resolution optical satellite imagery. The system classifies each pixel in a satellite image into one of several land-cover categories such as urban areas, agriculture, forest, water, and barren land.

A U-Net convolutional neural network architecture is trained on the **DeepGlobe Land Cover Classification dataset** and deployed through a user-friendly graphical interface that allows users to upload satellite images and visualize segmentation results.

This project focuses on **model application, evaluation, and interpretation**, following industry-standard workflows rather than developing novel architectures from scratch.

---

## Objectives

- Train a deep learning model for pixel-wise land-cover classification  
- Apply semantic segmentation to real satellite imagery  
- Visualize segmentation outputs clearly with color-coded class labels  
- Evaluate model performance and understand its limitations  
- Demonstrate a real-world land-cover analysis use case  

---

## Dataset Description

The project uses the **DeepGlobe Land Cover Classification Dataset**, which consists of high-resolution RGB satellite images annotated with pixel-level land-cover labels.

### Dataset Source

- **Official Challenge**: https://www.deepglobe.org/
- **Kaggle Dataset**:  
  https://www.kaggle.com/datasets/balraj98/deepglobe-land-cover-classification-dataset

### Dataset Characteristics

- Source: DeepGlobe 2018 Challenge  
- Image resolution: 2448 × 2448 pixels  
- Spatial resolution: 50 cm per pixel  
- Image type: RGB optical satellite imagery  
- Number of classes: 7  

### Land-Cover Classes

| Class ID | Class Name     |
|--------:|----------------|
| 0       | Unknown        |
| 1       | Urban          |
| 2       | Agriculture    |
| 3       | Rangeland      |
| 4       | Forest         |
| 5       | Water          |
| 6       | Barren         |

The dataset is split into training, validation, and test sets. Only the training and validation sets include ground-truth masks.

---

## Model Architecture

The segmentation model is based on the U-Net architecture, which is well suited for semantic segmentation tasks due to its encoder–decoder structure and skip connections.
Key Characteristics: 
- Encoder–decoder CNN with skip connections
- Input size: 512 × 512 RGB images
- Output: Pixel-wise class predictions
- Loss function: Cross-Entropy Loss
- Optimizer: Adam

---

## Usage: Land-Cover Segmentation Demo

A graphical user interface is provided to demonstrate land-cover segmentation on single satellite images.
Steps:
1. Run usage_single_image.py
2. Upload a satellite image (RGB, optical)
3. Click Process
4. View the segmented land-cover output
5. Save the result if required

Interface Displays:
- Raw satellite image
- Segmented land-cover map
- Color legend explaining each class

---

## Dataset Citation

Any use of the DeepGlobe dataset must cite the following publication:

```bibtex
@InProceedings{DeepGlobe18,
  author    = {Demir, Ilke and Koperski, Krzysztof and Lindenbaum, David and Pang, Guan and Huang, Jing and Basu, Saikat and Hughes, Forest and Tuia, Devis and Raskar, Ramesh},
  title     = {DeepGlobe 2018: A Challenge to Parse the Earth Through Satellite Images},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
  month     = {June},
  year      = {2018}
}

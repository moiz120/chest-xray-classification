
# Chest X-Ray Image Classification

A computer vision assignment comparing classical feature-based machine learning against a deep learning approach for classifying chest X-ray images into three categories: **Normal**, **Pneumonia**, and **COVID-19**.

## Overview

This notebook explores two different approaches to medical image classification:

1. **Classical ML pipeline** — hand-crafted feature extraction followed by traditional classifiers
2. **Deep learning pipeline** — a Convolutional Neural Network (CNN) trained directly on image data

The goal is to compare how well classical computer vision techniques hold up against deep learning on a real-world medical imaging task.

## Methods

**Feature Extraction**
- **HOG (Histogram of Oriented Gradients)** — captures shape and edge structure
- **SIFT (Scale-Invariant Feature Transform)** — captures local keypoint features robust to scale and rotation

**Classifiers**
- **KNN (K-Nearest Neighbors)** — trained on HOG/SIFT features
- **SVM-RBF (Support Vector Machine, RBF kernel)** — trained on HOG/SIFT features
- **CNN (Convolutional Neural Network)** — trained end-to-end directly on raw image data

## Evaluation

Models were evaluated and compared using:
- Accuracy
- Classification reports (precision, recall, F1-score per class)
- Confusion matrices

## Tech Stack

- Python
- OpenCV
- scikit-learn
- HOG / SIFT (via OpenCV)

## Running the Notebook

This notebook was built and run in Google Colab.

1. Open `chest_xray_classification.ipynb` in Google Colab
2. Upload/mount the chest X-ray dataset (Normal / Pneumonia / COVID-19 classes)
3. Run all cells in order — feature extraction, model training, and evaluation are laid out sequentially

## Notes

This was completed as part of a Computer Vision course assignment, focused on comparing classical feature-engineering-based approaches against deep learning for medical image classification.

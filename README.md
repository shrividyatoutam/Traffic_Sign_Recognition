## Overview

This project implements a Convolutional Neural Network (CNN) for classifying
43 German traffic sign categories using the GTSRB dataset.
The trained model supports both single-image prediction and
real-time webcam inference.

## Features

- Classifies **43 German traffic sign classes**
- Image preprocessing pipeline for improved accuracy
- Data augmentation during training
- Convolutional Neural Network built using TensorFlow/Keras
- Real-time webcam prediction using OpenCV
- Single image prediction support
- Training performance visualization

## Dataset

The project uses the **German Traffic Sign Recognition Benchmark (GTSRB)** dataset.

Dataset includes:

- 43 traffic sign categories
- Thousands of labeled traffic sign images
- Images resized to **32 × 32 pixels**

## Image Preprocessing

Before training, every image undergoes the following preprocessing steps:

- Convert RGB image to Grayscale
- Histogram Equalization
- Pixel Normalization
- Image Resizing (32 × 32)

These preprocessing techniques improve image quality and help the model generalize better.

## Data Augmentation

To reduce overfitting and improve robustness, the following augmentation techniques are applied during training:

- Rotation
- Width Shift
- Height Shift
- Zoom
- Shear

## Training

Model training was performed using **Google Colab GPU**.

Training configuration:

- Epochs: 10
- Batch Size: 50
- Optimizer: Adam
- Learning Rate: 0.001


## Project structure

```
├── myData/              # 43 class folders (0-42) of training images
├── labels/labels.csv    # class ID -> traffic sign name mapping
├── train.py             # loads data, builds and trains the CNN, saves the model
├── test.py               # real-time webcam classification using the trained model
├── predict_image.py     # classify a single image file (no webcam needed)
└── requirements.txt
```

## Running the Project

### 1. Clone the repository

```bash
git clone <repository-url>
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
``

### 3. Train the model
`python train.py` — trains the CNN and saves `model_trained.keras`.
   The full dataset (~34,800 images) with the default settings
   (`steps_per_epoch_val=2000`, `epochs_val=10`) is CPU-heavy; **use a GPU
   (e.g. Google Colab, free tier) or it may take a long time.** For a quick
   local smoke test, lower `steps_per_epoch_val` and `epochs_val` at the top
   of the file.
    This generates:

- model_trained.keras
- training graphs
- sample images


### 4. Run real-time prediction
 `python test.py` — opens your webcam and classifies signs live.
   `python predict_image.py path/to/photo.png` — classify a single photo instead.

### 5. Predict a single image

```bash
python Predict_image.py
```
## Model

A 4-conv-layer CNN (60 filters → 60 filters → 30 filters → 30 filters, with
max-pooling and dropout) followed by a 500-unit dense layer, trained on
grayscale, histogram-equalized 32×32 images with on-the-fly data augmentation
(shift, zoom, shear, rotation).


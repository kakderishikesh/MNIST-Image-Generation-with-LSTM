# MNIST Image Generation with LSTM

## Overview
This project implements a **Recurrent Neural Network (RNN)** using **LSTM or GRU** to generate the missing bottom half of MNIST digit images given the top half. The model is trained on sequences of **vectorized image patches** and learns to predict the next sequence step-by-step. The trained network is capable of reconstructing digit images using a generative approach.

## Problem Statement
- The **MNIST dataset** consists of **28×28 grayscale images** of handwritten digits (0-9).
- Each image is **split into 16 smaller patches** of **7×7 pixels**.
- The patches are **flattened** into **49-dimensional vectors** and treated as a sequence.
- The first 8 patches (top half of the image) are used as **input**, and the model predicts the next 8 patches (bottom half).
- The **goal** is to train an RNN (LSTM/GRU) to generate the missing bottom half based on the given top half.

## Dataset
- Uses the **MNIST dataset** available in `torchvision.datasets.MNIST`.
- **Training Set**: 50,000 images
- **Test Set**: 10,000 images
- **Batch Size**: 100
- **Input Shape**: `16 × 49` (16 sequential patches, each with 49 features)

## Model Architecture
- **Input Layer**: Sequences of **(batch_size, 16, 49)**
- **Recurrent Layer**: LSTM/GRU with **2 layers of 64 hidden units**
- **Dense Layer**: Fully connected layer mapping hidden state back to **49 features** per patch
- **Loss Function**: Mean Squared Error (MSE)
- **Optimizer**: Adam
- **Training**: The model predicts the **next patch** based on previous patches in the sequence.

## Training Process
1. **Data Preprocessing**:
   - Convert MNIST images into **sequences of 16 patches**.
   - Flatten each patch into a **49-dimensional vector**.
   - Create a dataset of **sequences** for training.

2. **Training the RNN**:
   - Input: First 8 patches
   - Target: Next 8 patches
   - Train using **Mean Squared Error (MSE) loss**.
   - Validate using **test set images** at each epoch.

3. **Generation Phase**:
   - Feed the **top half (8 patches)** of a test image into the trained model.
   - The model **predicts the next 8 patches** (bottom half).
   - The final **28×28 image is reconstructed** by combining the known and generated patches.

## Results
- The model successfully generates realistic-looking digits for most cases.
- Some failure cases occur when ambiguous top-half structures lead to incorrect bottom-half predictions (e.g., a "3" might be completed as a "2").

## Future Improvements
- Implement **GAN-based** image completion for better quality.
- Use **CNN-LSTM hybrid models** for feature extraction.
- Experiment with **attention mechanisms** to improve predictions.
- Train on **higher-resolution datasets** for better generalization.


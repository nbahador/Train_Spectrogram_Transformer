# Spectrogram Transformer

This repository contains a PyTorch implementation of a Vision Transformer (ViT) model fine-tuned for regression tasks using LoRA (Low-Rank Adaptation). The model is designed to extract the characteristics of chirp patterns within spectrogram images.

---

## Overview

The goal of this project is to predict three continuous values from spectrogram images:

- **Chirp Start Time**
- **Chirp Start Frequency**
- **Chirp End Frequency**

The model uses a pre-trained Vision Transformer (ViT) as a backbone and fine-tunes it using LoRA for parameter-efficient adaptation. The training process includes mixed precision training, early stopping, and learning rate scheduling.

---

## Features

### Vision Transformer (ViT)
- Utilizes a pre-trained ViT model for feature extraction.

### LoRA Fine-Tuning
- Applies Low-Rank Adaptation for efficient fine-tuning of the ViT model.

### Mixed Precision Training
- Uses `torch.cuda.amp` for faster training and reduced memory usage.

### Early Stopping
- Prevents overfitting by monitoring validation loss.

### Learning Rate Scheduling
- Adjusts the learning rate based on validation performance.

### Natural Language Descriptions
- Generates human-readable descriptions of predicted chirp patterns.

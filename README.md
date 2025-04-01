## Chirp Spectrogram Transformer

This repository contains a PyTorch implementation of a Vision Transformer (ViT) model fine-tuned for regression tasks using LoRA (Low-Rank Adaptation). The model is designed to extract the characteristics of chirp patterns within spectrogram images.

---

### Overview

The goal of this project is to predict three continuous values from spectrogram images:

- **Chirp Start Time**
- **Chirp Start Frequency**
- **Chirp End Frequency**

The model uses a pre-trained Vision Transformer (ViT) as a backbone and fine-tunes it using LoRA for parameter-efficient adaptation. The training process includes mixed precision training, early stopping, and learning rate scheduling.


| Feature                      | Description                                                                 |
|------------------------------|-----------------------------------------------------------------------------|
| **Vision Transformer (ViT)** | Utilizes a pre-trained ViT model for feature extraction.                   |
| **LoRA Fine-Tuning**         | Applies Low-Rank Adaptation for efficient fine-tuning of the ViT model.    |
| **Mixed Precision Training** | Uses `torch.cuda.amp` for faster training and reduced memory usage.        |
| **Early Stopping**           | Prevents overfitting by monitoring validation loss.                        |
| **Learning Rate Scheduling** | Adjusts the learning rate based on validation performance.                 |
| **Natural Language Descriptions** | Generates human-readable descriptions of predicted chirp patterns.    |

---

| Resource | Description | Link |
|----------|-------------|------|
| Trained Vision Transformer Model | Access to a pre-trained Vision Transformer model fine-tuned on synthetic spectrograms for chirp localization | [HuggingFace Model Hub](https://huggingface.co/nubahador/Fine_Tuned_Transformer_Model_for_Chirp_Localization/tree/main) |
| Synthetic Spectrogram Dataset | Download link for 100,000 synthetic spectrograms with corresponding labels for chirp localization | [HuggingFace Dataset Hub](https://huggingface.co/datasets/nubahador/ChirpLoc100K___A_Synthetic_Spectrogram_Dataset_for_Chirp_Localization/tree/main) |
| PyTorch Implementation | Repository containing the PyTorch code for fine-tuning the Vision Transformer on spectrograms | [GitHub Repository](https://github.com/nbahador/Train_Spectrogram_Transformer) |
| Synthetic Chirp Generator | Python package for generating synthetic chirp spectrograms (images with corresponding labels) | [arXiv Paper](https://arxiv.org/pdf/2503.22713) |

---

### Dataset Details

#### [100,000 Labeled Chirp Spectrogram Images – Download on Hugging Face!](https://huggingface.co/datasets/nubahador/ChirpLoc100K___A_Synthetic_Spectrogram_Dataset_for_Chirp_Localization/blob/main/README.md)

<table>
<tr>
<td style="vertical-align: top; width: 25%">
  
**Curated by**  
Nooshin Bahador

</td>
<td style="vertical-align: top; width: 20%">
  
**Funded by**  
Canadian Neuroanalytics Scholars Program

</td>
<td style="vertical-align: top; width: 30%">
  
**Citation**  
Bahador, N., & Lankarany, M. (2025). Chirp localization via fine-tuned transformer model: A proof-of-concept study. arXiv preprint arXiv:2503.22713. [[PDF]](https://arxiv.org/pdf/2503.22713)

</td>
</tr>
</table>

#### Sample Spectrogram Image

<div style="display: flex; justify-content: space-between; gap: 20px;">
    <img src="https://github.com/nbahador/chirp_spectrogram_generator/blob/main/Usage_Example/spectrogram_4.png" alt="Sample Generated Spectrogram" width="300" height="200" />
    <img src="https://github.com/nbahador/chirp_spectrogram_generator/blob/main/Usage_Example/Samples.jpg" alt="Sample Generated Spectrograms" width="400" height="200" />
</div>

#### Sample Label

| Chirp Start Time (s) | Chirp Start Freq (Hz) | Chirp End Freq (Hz) | Chirp Duration (s) | Chirp Type   |
|----------------------|-----------------------|---------------------|--------------------|--------------|
| 38.92107594          | 14.58740744           | 36.84728556         | 10.80687464        | exponential  |

---

### Vision Transformer (ViT) with LoRA for Spectrogram Regression

---
### Fine-Tuning Details

| Category              | Specification                                                                                     |
|-----------------------|---------------------------------------------------------------------------------------------------|
| **Framework**         | PyTorch                                                                                          |
| **Architecture**      | Pre-trained Vision Transformer (ViT)                                                             |
| **Adaptation Method** | LoRA (Low-Rank Adaptation)                                                                        |
| **Task**             | Regression on time-frequency representations                                                      |
| **Target Variables**  | 1. Chirp start time (s)<br>2. Start frequency (Hz)<br>3. End frequency (Hz)                   |
| **Training Protocol** | • Automatic Mixed Precision (AMP)<br>• Early stopping<br>• Learning Rate scheduling              |
| **Output**           | Quantitative predictions + optional natural language descriptions                                 |

---
### Resource Details

| Resource | Description | Link |
|----------|-------------|------|
| Trained Vision Transformer Model | Access to a pre-trained Vision Transformer model fine-tuned on synthetic spectrograms for chirp localization | [HuggingFace Model Hub](https://huggingface.co/nubahador/Fine_Tuned_Transformer_Model_for_Chirp_Localization/tree/main) |
| Synthetic Spectrogram Dataset | Download link for 100,000 synthetic spectrograms with corresponding labels for chirp localization | [HuggingFace Dataset Hub](https://huggingface.co/datasets/nubahador/ChirpLoc100K___A_Synthetic_Spectrogram_Dataset_for_Chirp_Localization/tree/main) |
| PyTorch Implementation | Repository containing the PyTorch code for fine-tuning the Vision Transformer on spectrograms | [Implementation GitHub Repository](https://github.com/nbahador/Train_Spectrogram_Transformer) |
| Synthetic Chirp Generator | Python package for generating synthetic chirp spectrograms (images with corresponding labels) | [Dataset GitHub Repository](https://github.com/nbahador/chirp_spectrogram_generator) |

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

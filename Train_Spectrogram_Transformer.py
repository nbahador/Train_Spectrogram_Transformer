import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from PIL import Image
import torch.nn as nn
from transformers import ViTModel
from torch.optim import AdamW
from peft import LoraConfig, get_peft_model
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
from scipy.stats import pearsonr
from torch.cuda.amp import GradScaler, autocast  # Correct import for mixed precision
from multiprocessing import freeze_support  # Required for Windows

# Directory containing spectrogram images and labels
data_dir = r"Enter the path to the data directory"

# Load labels from CSV
print("Loading labels from CSV...")
labels_df = pd.read_csv(os.path.join(data_dir, "labels.csv"))

# Prepare labels
print("Preparing and normalizing labels...")
labels = labels_df[["Chirp_Start_Time", "Chirp_Start_Freq", "Chirp_End_Freq"]].values

# Normalize labels
label_mean = labels.mean(axis=0)
label_std = labels.std(axis=0)
labels = (labels - label_mean) / label_std  # Standardize labels

# Split data into training and testing sets
print("Splitting data into training and testing sets...")
train_indices, test_indices = train_test_split(range(len(labels_df)), test_size=0.2, random_state=42)

# Custom Dataset class for lazy loading
class SpectrogramDataset(Dataset):
    def __init__(self, data_dir, indices, labels):
        self.data_dir = data_dir
        self.indices = indices
        self.labels = labels

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # Get the actual index from the split indices
        actual_idx = self.indices[idx]
        # Load the spectrogram image
        img_path = os.path.join(self.data_dir, f"spectrogram_{actual_idx + 1}.png")
        img = Image.open(img_path).convert("RGB")  # Convert to RGB (ViT expects 3 channels)
        img = img.resize((224, 224))  # Resize to (224, 224) for ViT
        img = np.array(img) / 255.0  # Convert to NumPy array and normalize to [0, 1]
        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)  # Convert to tensor and permute to (C, H, W)
        # Get the corresponding label
        label = torch.tensor(self.labels[actual_idx], dtype=torch.float32)
        return img, label

# Create datasets and dataloaders with optimized data loading
print("Creating datasets and dataloaders...")
train_dataset = SpectrogramDataset(data_dir, train_indices, labels)
test_dataset = SpectrogramDataset(data_dir, test_indices, labels)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

# Define a Vision Transformer (ViT) model for regression with LoRA
print("Initializing the Vision Transformer (ViT) model...")
class ViTForRegression(nn.Module):
    def __init__(self, pretrained_model_name="google/vit-base-patch16-224"):
        super(ViTForRegression, self).__init__()
        self.vit = ViTModel.from_pretrained(pretrained_model_name)
        # Apply LoRA for Parameter-Efficient Fine-Tuning
        lora_config = LoraConfig(
            r=8,  # Rank of the low-rank adaptation
            lora_alpha=16,  # Scaling factor
            target_modules=["query", "value"],  # Apply LoRA to specific layers
            lora_dropout=0.1,
            bias="none"
        )
        self.vit = get_peft_model(self.vit, lora_config)
        self.regression_head = nn.Sequential(
            nn.Linear(self.vit.config.hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3)  # Output 3 values for regression
        )

    def forward(self, x):
        # Pass input through ViT
        outputs = self.vit(pixel_values=x)
        # Use the [CLS] token representation for regression
        cls_output = outputs.last_hidden_state[:, 0, :]
        regression_output = self.regression_head(cls_output)
        return regression_output

# Initialize the model
print("Moving model to device...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ViTForRegression().to(device)

# Set up optimizer and loss function
print("Setting up optimizer, loss function, and learning rate scheduler...")
optimizer = AdamW(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()  # Use MSE loss for regression
scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=3, verbose=True)

# Mixed precision training setup
print("Initializing GradScaler for mixed precision training...")
scaler = GradScaler()  # Correct initialization without device_type

# Training function with mixed precision
def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for batch_idx, batch in enumerate(dataloader):
        spectrograms, labels = batch
        spectrograms, labels = spectrograms.to(device), labels.to(device)

        optimizer.zero_grad()

        with autocast():  # Mixed precision
            outputs = model(spectrograms)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()  # Scale loss and backward pass
        scaler.step(optimizer)  # Step optimizer
        scaler.update()  # Update the scale for next iteration

        total_loss += loss.item()

        # Print progress every 10 batches
        if batch_idx % 10 == 0:
            print(f"  Batch {batch_idx + 1}/{len(dataloader)}, Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(dataloader)
    return avg_loss

# Function to calculate correlation between predicted and actual labels
def calculate_correlation(predicted, actual):
    correlations = []
    for i in range(predicted.shape[1]):
        corr, _ = pearsonr(predicted[:, i], actual[:, i])
        correlations.append(corr)
    return correlations

# Function to measure inference speed
def measure_inference_speed(model, dataloader, device):
    model.eval()
    start_time = time.time()
    with torch.no_grad():
        for batch in dataloader:
            spectrograms, _ = batch
            spectrograms = spectrograms.to(device)
            _ = model(spectrograms)
    end_time = time.time()
    inference_time = end_time - start_time
    return inference_time

# Evaluation function
def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            spectrograms, labels = batch
            spectrograms, labels = spectrograms.to(device), labels.to(device)

            # Forward pass
            outputs = model(spectrograms)
            loss = criterion(outputs, labels)

            # Compute metrics
            total_loss += loss.item()
            all_preds.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

            # Print progress every 10 batches
            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx + 1}/{len(dataloader)}, Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(dataloader)
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # Calculate correlation
    correlations = calculate_correlation(all_preds, all_labels)

    # Measure inference speed
    inference_time = measure_inference_speed(model, dataloader, device)

    return avg_loss, correlations, inference_time

# Main function to run the script
def main():
    # Fine-tune the model with early stopping
    num_epochs = 100
    best_test_loss = float("inf")
    patience = 4  # Early stopping patience
    epochs_without_improvement = 0

    print("Starting training...")
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("Training...")
        train_loss = train(model, train_loader, optimizer, criterion, device)
        print("Evaluating...")
        test_loss, correlations, inference_time = evaluate_model(model, test_loader, criterion, device)
        scheduler.step(test_loss)  # Adjust learning rate based on test loss

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Correlations: {correlations}")
        print(f"Inference Time: {inference_time:.4f} seconds")
        print("---")

        # Early stopping logic
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), os.path.join(data_dir, "best_vit_regression.pth"))
            print("Saved new best model.")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print("Early stopping triggered")
                break

    # Load the best model
    print("Loading the best model...")
    model.load_state_dict(torch.load(os.path.join(data_dir, "best_vit_regression.pth")))

    # Evaluate the best model on the test set
    print("Evaluating the best model on the test set...")
    test_loss, correlations, inference_time = evaluate_model(model, test_loader, criterion, device)

    # Save the metrics to a file
    print("Saving metrics to file...")
    metrics = {
        "test_loss": test_loss,
        "correlations": correlations,
        "inference_time": inference_time
    }

    with open(os.path.join(data_dir, "metrics.txt"), "w") as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")

    # Function to generate natural language descriptions
    def generate_description(predicted_values):
        chirp_start_time, chirp_start_freq, chirp_end_freq = predicted_values
        description = f"A chirp pattern was observed starting at time {chirp_start_time:.2f} with a start frequency of {chirp_start_freq:.2f} Hz and an end frequency of {chirp_end_freq:.2f} Hz."
        return description

    # Example usage: Predict and generate description for a test sample
    print("Generating predictions for a test sample...")
    sample_idx = test_indices[0]  # Use the first test sample
    sample_spectrogram, _ = test_dataset[0]  # Get the spectrogram
    sample_spectrogram = sample_spectrogram.unsqueeze(0).to(device)  # Add batch dimension

    # Get the predicted values
    predicted_values = model(sample_spectrogram).detach().cpu().numpy()[0]

    # Denormalize predicted values
    predicted_values = predicted_values * label_std + label_mean

    # Get the real label for the test sample
    real_label = labels[sample_idx] * label_std + label_mean  # Denormalize

    # Generate descriptions
    predicted_description = generate_description(predicted_values)
    real_description = generate_description(real_label)

    # Print results
    print("Predicted Values:")
    print(predicted_description)
    print("\nReal Label:")
    print(real_description)

# Entry point for the script
if __name__ == '__main__':
    freeze_support()  # Required for multiprocessing on Windows
    main()
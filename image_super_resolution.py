import os
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Define the path to the JPEG images
dataset_path = "/kaggle/input/dlw4-213138787/VOCdevkit/VOC2007/JPEGImages"

# Get all image file paths
image_paths = sorted(glob.glob(os.path.join(dataset_path, "*.jpg")))  # Load only first 10 images

# Print sample paths
print("First 5 image paths:\n", image_paths[:5])


class SuperResolutionDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load Image
        img = Image.open(self.image_paths[idx]).convert("RGB")

        # Resize Images
        img_large = img.resize((288, 288), Image.BICUBIC)  # Target Large Image
        img_mid = img.resize((144, 144), Image.BICUBIC)    # Intermediate Image
        img_small = img.resize((72, 72), Image.BICUBIC)    # Low-Resolution Input

        # Convert to Tensor
        transform = transforms.ToTensor()
        img_small = transform(img_small)
        img_mid = transform(img_mid)
        img_large = transform(img_large)

        return img_small, img_mid, img_large


# Create dataset instance
sr_dataset = SuperResolutionDataset(image_paths)

# Create DataLoader
sr_loader = DataLoader(sr_dataset, batch_size=2, shuffle=True)

# Check the shapes of images in a batch
small, mid, large = next(iter(sr_loader))
print(f"Small Image Shape: {small.shape}")  # Expected: (Batch, 3, 72, 72)
print(f"Mid Image Shape: {mid.shape}")      # Expected: (Batch, 3, 144, 144)
print(f"Large Image Shape: {large.shape}")  # Expected: (Batch, 3, 288, 288)


small, mid, large = next(iter(sr_loader))

def show_images(small, mid, large, image_paths):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Extract filenames for better labeling
    image_name = os.path.basename(image_paths[0])

    # Titles
    titles = [
        f"X_train[0]\nX image = {image_name}\nsize 72x72",
        f"y_mid_train[0]\nY_mid image = {image_name}\nsize 144x144",
        f"y_large_train[0]\nY_large image = {image_name}\nsize 288x288"
    ]

    # Display images
    axes[0].imshow(small[0].permute(1, 2, 0))
    axes[1].imshow(mid[0].permute(1, 2, 0))
    axes[2].imshow(large[0].permute(1, 2, 0))

    # Set titles
    for ax, title in zip(axes, titles):
        ax.set_title(title, fontsize=12, color='red')
        ax.axis("on")

    plt.tight_layout()
    plt.show()

# Call the function with the first batch
show_images(small, mid, large, image_paths)



!pip install pytorch-lightning

!pip install mlflow

from torch.utils.data import random_split, DataLoader
# Split dataset into training and validation sets
train_size = int(0.8 * len(sr_dataset))
val_size = len(sr_dataset) - train_size
train_dataset, val_dataset = random_split(sr_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torch.optim as optim
import matplotlib.pyplot as plt
import mlflow
import imageio
import numpy as np
import time
from pytorch_lightning.loggers import TensorBoardLogger, MLFlowLogger
from torch.utils.data import random_split, DataLoader

def psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0  
    return 20 * torch.log10(max_pixel / torch.sqrt(mse))

class SuperResolutionModel(pl.LightningModule):
    def __init__(self, lr=1e-3):
        super(SuperResolutionModel, self).__init__()

        # Define the model architecture
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=1)
        
        # Learning rate
        self.lr = lr
        self.criterion = nn.MSELoss()
        self.predictions = []
        self.train_losses = []
        self.val_losses = []
        self.train_psnr = []
        self.val_psnr = []

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.upsample(x)
        x = self.conv3(x)
        return x

    def training_step(self, batch, batch_idx):
        x_small, y_mid, _ = batch  # Use X_train (small) and y_mid_train (mid)
        y_pred = self(x_small)
        loss = self.criterion(y_pred, y_mid)
        psnr_value = psnr(y_pred, y_mid).item()
        
        self.log("train_loss", loss, prog_bar=True, logger=True)
        self.log("train_psnr", psnr_value, prog_bar=True, logger=True)
        
        if batch_idx == 0:
            self.train_losses.append(loss.item())
            self.train_psnr.append(psnr_value)
        return loss

    def validation_step(self, batch, batch_idx):
        x_small, y_mid, _ = batch  # Use X_train (small) and y_mid_train (mid)
        y_pred = self(x_small)
        loss = self.criterion(y_pred, y_mid)
        psnr_value = psnr(y_pred, y_mid).item()
        
        self.log("val_loss", loss, prog_bar=True, logger=True)
        self.log("val_psnr", psnr_value, prog_bar=True, logger=True)
        
        if batch_idx == 0:
            self.val_losses.append(loss.item())
            self.val_psnr.append(psnr_value)

        # Save predictions for visualization every 5 epochs
        if batch_idx == 0 and (len(self.train_losses) % 5 == 0):
            self.predictions.append((x_small[0].cpu(), y_pred[0].cpu(), y_mid[0].cpu()))
        
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)

    def plot_progress(self, save_gif=False):
        """Visualizes improvements every 5 epochs for one validation image."""
        num_epochs = len(self.predictions)
        fig, axes = plt.subplots(num_epochs, 3, figsize=(10, num_epochs * 3))
        fig.suptitle('Model Improvement Over Epochs (Every 5 Epochs)', fontsize=16)
        frames = []
        
        for i, (input_img, pred_img, target_img) in enumerate(self.predictions):
            axes[i, 0].imshow(input_img.permute(1, 2, 0).numpy())
            axes[i, 0].set_title(f"Epoch {i*5+1} - Input (72x72)")
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(pred_img.permute(1, 2, 0).numpy())
            axes[i, 1].set_title(f"Epoch {i*5+1} - Prediction (144x144)")
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(target_img.permute(1, 2, 0).numpy())
            axes[i, 2].set_title(f"Epoch {i*5+1} - Target (144x144)")
            axes[i, 2].axis('off')
            
            plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()
    def plot_loss_curves(self):
        """Plots training and validation loss and PSNR curves."""
        epochs = range(1, len(self.train_losses) + 1)
        if len(self.val_losses) > len(self.train_losses):
            self.val_losses = self.val_losses[:len(self.train_losses)]
            self.val_psnr = self.val_psnr[:len(self.train_psnr)]
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.train_losses, label='Train Loss')
        plt.plot(epochs, self.val_losses, label='Validation Loss')
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss Over Epochs")
        plt.legend()
        plt.grid()

        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.train_psnr, label='Train PSNR')
        plt.plot(epochs, self.val_psnr, label='Validation PSNR')
        plt.xlabel("Epochs")
        plt.ylabel("PSNR (dB)")
        plt.title("Training and Validation PSNR Over Epochs")
        plt.legend()
        plt.grid()

        plt.show()
    
    def print_metrics_per_epoch(self):
        """Prints the training and validation loss and PSNR per epoch."""
        num_epochs = min(len(self.train_losses), len(self.val_losses))
        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}: Train Loss = {self.train_losses[epoch]:.4f}, Val Loss = {self.val_losses[epoch]:.4f}, Train PSNR = {self.train_psnr[epoch]:.2f}, Val PSNR = {self.val_psnr[epoch]:.2f}")


# Set up logging
mlflow_logger = MLFlowLogger(experiment_name="SuperResolutionExperiment")
tensorboard_logger = TensorBoardLogger("tb_logs", name="SuperResolution")

# Instantiate the model
basic_model = SuperResolutionModel()

# Define the trainer
start_time = time.time()
trainer = pl.Trainer(max_epochs=10, accelerator="auto", logger=[mlflow_logger, tensorboard_logger])

# Train the model
trainer.fit(basic_model, train_loader, val_loader)
end_time = time.time()

# Print metrics per epoch
basic_model.print_metrics_per_epoch()

# Print total training time
print(f"Total Training Time: {end_time - start_time:.2f} seconds")

# Plot loss curves
basic_model.plot_loss_curves()


basic_model.plot_progress()

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torch.optim as optim
import matplotlib.pyplot as plt
import mlflow
import imageio
import numpy as np
import time
from pytorch_lightning.loggers import TensorBoardLogger, MLFlowLogger
from torch.utils.data import random_split, DataLoader

# PSNR Calculation
def psnr(img1, img2):
    mse = F.mse_loss(img1, img2)
    if mse == 0:
        return 100  # Ideal case
    return 20 * torch.log10(255.0 / torch.sqrt(mse))

class SuperResolutionModel(pl.LightningModule):
    def __init__(self, lr=1e-3):
        super(SuperResolutionModel, self).__init__()

        # Define the model architecture
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv3_mid = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=1)
        
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv3_large = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=1)
        
        # Learning rate
        self.lr = lr
        self.criterion = nn.MSELoss()
        self.predictions = []
        self.train_losses = []
        self.val_losses = []
        self.train_psnrs = []
        self.val_psnrs = []

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x_mid = self.upsample1(x)
        x_mid_output = self.conv3_mid(x_mid)

        x_large = self.upsample2(x_mid)
        x_large_output = self.conv3_large(x_large)
        
        return x_mid_output, x_large_output

    def training_step(self, batch, batch_idx):
        x_small, y_mid, y_large = batch
        y_pred_mid, y_pred_large = self(x_small)
        loss_mid = self.criterion(y_pred_mid, y_mid)
        loss_large = self.criterion(y_pred_large, y_large)

        psnr_mid = psnr(y_pred_mid, y_mid)
        psnr_large = psnr(y_pred_large, y_large)
        
        self.log("train_loss_mid", loss_mid, prog_bar=True, logger=True)
        self.log("train_loss_large", loss_large, prog_bar=True, logger=True)
        self.log("train_psnr_mid", psnr_mid, prog_bar=True, logger=True)
        self.log("train_psnr_large", psnr_large, prog_bar=True, logger=True)

        if batch_idx == 0:
            self.train_losses.append((loss_mid.item(), loss_large.item()))
            self.train_psnrs.append((psnr_mid.item(), psnr_large.item()))

        return loss_mid + loss_large

    def validation_step(self, batch, batch_idx):
        x_small, y_mid, y_large = batch
        y_pred_mid, y_pred_large = self(x_small)
        loss_mid = self.criterion(y_pred_mid, y_mid)
        loss_large = self.criterion(y_pred_large, y_large)

        psnr_mid = psnr(y_pred_mid, y_mid)
        psnr_large = psnr(y_pred_large, y_large)

        self.log("val_loss_mid", loss_mid, prog_bar=True, logger=True)
        self.log("val_loss_large", loss_large, prog_bar=True, logger=True)
        self.log("val_psnr_mid", psnr_mid, prog_bar=True, logger=True)
        self.log("val_psnr_large", psnr_large, prog_bar=True, logger=True)

        if batch_idx == 0:
            self.val_losses.append((loss_mid.item(), loss_large.item()))
            self.val_psnrs.append((psnr_mid.item(), psnr_large.item()))
        if batch_idx == 0 and (len(self.train_losses) % 5 == 0):
            self.predictions.append((x_small[0].cpu(), y_pred_mid[0].cpu(), y_mid[0].cpu(), y_pred_large[0].cpu(), y_large[0].cpu()))

        return loss_mid + loss_large

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)

    def print_loss_per_epoch(self):
        """Prints the training and validation loss and PSNR per epoch."""
        for epoch in range(len(self.train_losses)):
            train_mid, train_large = self.train_losses[epoch]
            val_mid, val_large = self.val_losses[epoch]
            psnr_train_mid, psnr_train_large = self.train_psnrs[epoch]
            psnr_val_mid, psnr_val_large = self.val_psnrs[epoch]
            print(f"Epoch {epoch+1}: Train Loss Mid = {train_mid:.4f}, Train Loss Large = {train_large:.4f}, "
                  f"Validation Loss Mid = {val_mid:.4f}, Validation Loss Large = {val_large:.4f}, "
                  f"Train PSNR Mid = {psnr_train_mid:.2f}, Train PSNR Large = {psnr_train_large:.2f}, "
                  f"Validation PSNR Mid = {psnr_val_mid:.2f}, Validation PSNR Large = {psnr_val_large:.2f}")

    def plot_loss_curves(self):
        """Plots separate training and validation losses and PSNRs for mid and large outputs."""
        epochs = range(1, len(self.train_losses) + 1)
        if len(self.val_losses) > len(self.train_losses):
            self.val_losses = self.val_losses[:len(self.train_losses)]
            self.val_psnrs = self.val_psnrs[:len(self.train_psnrs)]

        train_mid_losses, train_large_losses = zip(*self.train_losses)
        val_mid_losses, val_large_losses = zip(*self.val_losses)
        train_mid_psnr, train_large_psnr = zip(*self.train_psnrs)
        val_mid_psnr, val_large_psnr = zip(*self.val_psnrs)

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_mid_losses, label='Train Loss Mid')
        plt.plot(epochs, val_mid_losses, label='Validation Loss Mid')
        plt.plot(epochs, train_large_losses, label='Train Loss Large')
        plt.plot(epochs, val_large_losses, label='Validation Loss Large')
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss Over Epochs")
        plt.legend()
        plt.grid()

        plt.subplot(1, 2, 2)
        plt.plot(epochs, train_mid_psnr, label='Train PSNR Mid')
        plt.plot(epochs, val_mid_psnr, label='Validation PSNR Mid')
        plt.plot(epochs, train_large_psnr, label='Train PSNR Large')
        plt.plot(epochs, val_large_psnr, label='Validation PSNR Large')
        plt.xlabel("Epochs")
        plt.ylabel("PSNR (dB)")
        plt.title("Training and Validation PSNR Over Epochs")
        plt.legend()
        plt.grid()

        plt.show()
        
    def plot_progress(self, save_gif=False):
        """Visualizes improvements every 5 epochs for one validation image."""
        num_epochs = len(self.predictions)
        fig, axes = plt.subplots(num_epochs, 5, figsize=(15, num_epochs * 3))
        fig.suptitle('Model Improvement Over Epochs (Every 5 Epochs)', fontsize=16)
        frames = []
        
        for i, (input_img, pred_img_mid, target_img_mid, pred_img_large, target_img_large) in enumerate(self.predictions):
            axes[i, 0].imshow(input_img.permute(1, 2, 0).numpy())
            axes[i, 0].set_title(f"Epoch {i*5+1} - Input (72x72)")
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(pred_img_mid.permute(1, 2, 0).numpy())
            axes[i, 1].set_title(f"Epoch {i*5+1} - Pred Mid (144x144)")
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(target_img_mid.permute(1, 2, 0).numpy())
            axes[i, 2].set_title(f"Epoch {i*5+1} - Target Mid (144x144)")
            axes[i, 2].axis('off')
            
            axes[i, 3].imshow(pred_img_large.permute(1, 2, 0).numpy())
            axes[i, 3].set_title(f"Epoch {i*5+1} - Pred Large (288x288)")
            axes[i, 3].axis('off')
            
            axes[i, 4].imshow(target_img_large.permute(1, 2, 0).numpy())
            axes[i, 4].set_title(f"Epoch {i*5+1} - Target Large (288x288)")
            axes[i, 4].axis('off')
            
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            fig.canvas.draw()
            frame = np.array(fig.canvas.renderer.buffer_rgba())
            frames.append(frame)
        
        if save_gif:
            imageio.mimsave('training_progress.gif', frames, fps=2)
        plt.show()


# Split dataset into training and validation sets
train_size = int(0.8 * len(sr_dataset))
val_size = len(sr_dataset) - train_size
train_dataset, val_dataset = random_split(sr_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Set up logging
mlflow_logger = MLFlowLogger(experiment_name="SuperResolutionExperiment")
tensorboard_logger = TensorBoardLogger("tb_logs", name="SuperResolution")

# Instantiate the model
middle_and_large_model = SuperResolutionModel()

# Measure training time
start_time = time.time()

# Define the trainer
trainer = pl.Trainer(max_epochs=10, accelerator="auto", logger=[mlflow_logger, tensorboard_logger])

# Train the model
trainer.fit(middle_and_large_model, train_loader, val_loader)

# End time
end_time = time.time()

# Print total training time
print(f"\nTotal Training Time: {end_time - start_time:.2f} seconds")

# Print loss per epoch
middle_and_large_model.print_loss_per_epoch()

# Plot loss and PSNR curves
middle_and_large_model.plot_loss_curves()


middle_and_large_model.plot_progress()

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torch.optim as optim
import matplotlib.pyplot as plt
import mlflow
import imageio
import numpy as np
import time
from pytorch_lightning.loggers import TensorBoardLogger, MLFlowLogger
from torch.utils.data import DataLoader

def psnr(pred, target, max_pixel=255.0):
    """Compute Peak Signal-to-Noise Ratio (PSNR)."""
    mse = F.mse_loss(pred, target)
    return 20 * torch.log10(max_pixel / torch.sqrt(mse))

class ResidualBlock(nn.Module):
    """Residual block with Conv2D layers and skip connections."""
    def __init__(self, channels=32):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.activation = nn.ReLU()

    def forward(self, x):
        residual = x  # Skip connection
        out = self.activation(self.conv1(x))
        out = self.conv2(out)
        return self.activation(out + residual)  # Element-wise addition (skip connection)

class SuperResolutionModel(pl.LightningModule):
    def __init__(self, lr=1e-3):
        super(SuperResolutionModel, self).__init__()

        # Initial convolution (Input: 3 channels, Output: 32 feature maps)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=1)

        # Residual blocks with 32 filters each
        self.residual_block1 = ResidualBlock(32)
        self.residual_block2 = ResidualBlock(32)

        # Skip Connection
        self.final_conv = nn.Conv2d(32, 32, kernel_size=3, padding=1)  

        # First Upsampling
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv_mid = nn.Conv2d(32, 3, kernel_size=1)  # 3 filters for output

        # Residual block after first upsampling
        self.residual_block3 = ResidualBlock(32)

        # Second Upsampling
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv_large = nn.Conv2d(32, 3, kernel_size=1)  # 3 filters for large output

        # Learning rate and loss function
        self.lr = lr
        self.criterion = nn.MSELoss()

        # Tracking losses and PSNRs for visualization
        self.predictions = []
        self.train_losses = []
        self.val_losses = []
        self.train_psnrs = []
        self.val_psnrs = []

    def forward(self, x):
        x = self.conv1(x)  # Initial convolution (maps from 3 → 32 channels)
        x = self.residual_block1(x)  # First residual block
        x = self.residual_block2(x)  # Second residual block

        # Skip connection: Adding original feature maps before ReLU activation
        x = x + self.final_conv(x)  
        x = F.relu(x)

        # First upsampling
        x_mid = self.upsample1(x)
        x_mid_output = self.conv_mid(x_mid)  # Mid-resolution output (3 channels)

        # Residual block after first upsampling
        x_mid = self.residual_block3(x_mid)

        # Second upsampling
        x_large = self.upsample2(x_mid)
        x_large_output = self.conv_large(x_large)  # Large-resolution output (3 channels)

        return x_mid_output, x_large_output

    def training_step(self, batch, batch_idx):
        x_small, y_mid, y_large = batch
        y_pred_mid, y_pred_large = self(x_small)
        loss_mid = self.criterion(y_pred_mid, y_mid)
        loss_large = self.criterion(y_pred_large, y_large)

        psnr_mid = psnr(y_pred_mid, y_mid)
        psnr_large = psnr(y_pred_large, y_large)
        
        self.log("train_loss_mid", loss_mid, prog_bar=True)
        self.log("train_loss_large", loss_large, prog_bar=True)
        self.log("train_psnr_mid", psnr_mid, prog_bar=True)
        self.log("train_psnr_large", psnr_large, prog_bar=True)

        if batch_idx == 0:
            self.train_losses.append((loss_mid.item(), loss_large.item()))
            self.train_psnrs.append((psnr_mid.item(), psnr_large.item()))

        return loss_mid + loss_large

    def validation_step(self, batch, batch_idx):
        x_small, y_mid, y_large = batch
        y_pred_mid, y_pred_large = self(x_small)
        loss_mid = self.criterion(y_pred_mid, y_mid)
        loss_large = self.criterion(y_pred_large, y_large)

        psnr_mid = psnr(y_pred_mid, y_mid)
        psnr_large = psnr(y_pred_large, y_large)

        self.log("val_loss_mid", loss_mid, prog_bar=True)
        self.log("val_loss_large", loss_large, prog_bar=True)
        self.log("val_psnr_mid", psnr_mid, prog_bar=True)
        self.log("val_psnr_large", psnr_large, prog_bar=True)

        if batch_idx == 0:
            self.val_losses.append((loss_mid.item(), loss_large.item()))
            self.val_psnrs.append((psnr_mid.item(), psnr_large.item()))
        if batch_idx == 0 and (len(self.train_losses) % 5 == 0):
            self.predictions.append((x_small[0].cpu(), y_pred_mid[0].cpu(), y_mid[0].cpu(), y_pred_large[0].cpu(), y_large[0].cpu()))

        return loss_mid + loss_large

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)

    def print_loss_per_epoch(self):
        """Prints the training and validation loss and PSNR per epoch."""
        for epoch in range(len(self.train_losses)):
            train_mid, train_large = self.train_losses[epoch]
            val_mid, val_large = self.val_losses[epoch]
            psnr_train_mid, psnr_train_large = self.train_psnrs[epoch]
            psnr_val_mid, psnr_val_large = self.val_psnrs[epoch]
            print(f"Epoch {epoch+1}: Train Loss Mid = {train_mid:.4f}, Train Loss Large = {train_large:.4f}, "
                  f"Validation Loss Mid = {val_mid:.4f}, Validation Loss Large = {val_large:.4f}, "
                  f"Train PSNR Mid = {psnr_train_mid:.2f}, Train PSNR Large = {psnr_train_large:.2f}, "
                  f"Validation PSNR Mid = {psnr_val_mid:.2f}, Validation PSNR Large = {psnr_val_large:.2f}")
    def plot_loss_curves(self):
        """Plots separate training and validation losses and PSNRs for mid and large outputs."""
        epochs = range(1, len(self.train_losses) + 1)
        if len(self.val_losses) > len(self.train_losses):
            self.val_losses = self.val_losses[:len(self.train_losses)]
            self.val_psnrs = self.val_psnrs[:len(self.train_psnrs)]

        train_mid_losses, train_large_losses = zip(*self.train_losses)
        val_mid_losses, val_large_losses = zip(*self.val_losses)
        train_mid_psnr, train_large_psnr = zip(*self.train_psnrs)
        val_mid_psnr, val_large_psnr = zip(*self.val_psnrs)

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_mid_losses, label='Train Loss Mid')
        plt.plot(epochs, val_mid_losses, label='Validation Loss Mid')
        plt.plot(epochs, train_large_losses, label='Train Loss Large')
        plt.plot(epochs, val_large_losses, label='Validation Loss Large')
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss Over Epochs")
        plt.legend()
        plt.grid()

        plt.subplot(1, 2, 2)
        plt.plot(epochs, train_mid_psnr, label='Train PSNR Mid')
        plt.plot(epochs, val_mid_psnr, label='Validation PSNR Mid')
        plt.plot(epochs, train_large_psnr, label='Train PSNR Large')
        plt.plot(epochs, val_large_psnr, label='Validation PSNR Large')
        plt.xlabel("Epochs")
        plt.ylabel("PSNR (dB)")
        plt.title("Training and Validation PSNR Over Epochs")
        plt.legend()
        plt.grid()

        plt.show()
        
    def plot_progress(self, save_gif=False):
        """Visualizes improvements every 5 epochs for one validation image."""
        num_epochs = len(self.predictions)
        fig, axes = plt.subplots(num_epochs, 5, figsize=(15, num_epochs * 3))
        fig.suptitle('Model Improvement Over Epochs (Every 5 Epochs)', fontsize=16)
        frames = []
        
        for i, (input_img, pred_img_mid, target_img_mid, pred_img_large, target_img_large) in enumerate(self.predictions):
            axes[i, 0].imshow(input_img.permute(1, 2, 0).numpy())
            axes[i, 0].set_title(f"Epoch {i*5+1} - Input (72x72)")
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(pred_img_mid.permute(1, 2, 0).numpy())
            axes[i, 1].set_title(f"Epoch {i*5+1} - Pred Mid (144x144)")
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(target_img_mid.permute(1, 2, 0).numpy())
            axes[i, 2].set_title(f"Epoch {i*5+1} - Target Mid (144x144)")
            axes[i, 2].axis('off')
            
            axes[i, 3].imshow(pred_img_large.permute(1, 2, 0).numpy())
            axes[i, 3].set_title(f"Epoch {i*5+1} - Pred Large (288x288)")
            axes[i, 3].axis('off')
            
            axes[i, 4].imshow(target_img_large.permute(1, 2, 0).numpy())
            axes[i, 4].set_title(f"Epoch {i*5+1} - Target Large (288x288)")
            axes[i, 4].axis('off')
            
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            fig.canvas.draw()
            frame = np.array(fig.canvas.renderer.buffer_rgba())
            frames.append(frame)
        
        if save_gif:
            imageio.mimsave('training_progress.gif', frames, fps=2)
        plt.show()

# Set up logging
mlflow_logger = MLFlowLogger(experiment_name="SuperResolutionExperiment")
tensorboard_logger = TensorBoardLogger("tb_logs", name="SuperResolution")

# Instantiate the model
residual_block_model = SuperResolutionModel()

# Measure training time
start_time = time.time()

# Define the trainer
trainer = pl.Trainer(max_epochs=10, accelerator="auto", logger=[mlflow_logger, tensorboard_logger])

# Train the model
trainer.fit(residual_block_model, train_loader, val_loader)

# End time
end_time = time.time()

# Print total training time
print(f"\nTotal Training Time: {end_time - start_time:.2f} seconds")

# Print loss per epoch
residual_block_model.print_loss_per_epoch()

residual_block_model.plot_loss_curves()

residual_block_model.plot_progress()

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torch.optim as optim
import matplotlib.pyplot as plt
import mlflow
import imageio
import numpy as np
import time
from pytorch_lightning.loggers import TensorBoardLogger, MLFlowLogger
from torch.utils.data import DataLoader

def psnr(pred, target, max_pixel=255.0):
    """Compute Peak Signal-to-Noise Ratio (PSNR)."""
    mse = F.mse_loss(pred, target)
    return 20 * torch.log10(max_pixel / torch.sqrt(mse))

class DilatedConvBlock(nn.Module):
    """Dilated (Atrous) Convolution Block"""
    def __init__(self, channels=32):
        super(DilatedConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, dilation=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=2, dilation=2)
        self.conv3 = nn.Conv2d(channels, channels, kernel_size=3, padding=4, dilation=4)
        self.final_conv = nn.Conv2d(channels * 3, channels, kernel_size=3, padding=1)  # Final output
        self.activation = nn.LeakyReLU()

    def forward(self, x):
        d1 = self.conv1(x)
        d2 = self.conv2(x)
        d3 = self.conv3(x)
        out = torch.cat([d1, d2, d3], dim=1)  # Concatenate along channel dimension
        out = self.final_conv(out)  # Reduce back to 32 channels
        return self.activation(out)

class SuperResolutionModel(pl.LightningModule):
    def __init__(self, lr=1e-3):
        super(SuperResolutionModel, self).__init__()

        # Initial convolution (Input: 3 channels, Output: 32 feature maps)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=1)

        # Dilated convolution blocks replacing residual blocks
        self.dilated_block1 = DilatedConvBlock(32)
        self.dilated_block2 = DilatedConvBlock(32)

        # Skip Connection
        self.final_conv = nn.Conv2d(32, 32, kernel_size=3, padding=1)  

        # First Upsampling
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv_mid = nn.Conv2d(32, 3, kernel_size=1)  # 3 filters for output

        # Dilated convolution block after first upsampling
        self.dilated_block3 = DilatedConvBlock(32)

        # Second Upsampling
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv_large = nn.Conv2d(32, 3, kernel_size=1)  # 3 filters for large output

        # Learning rate and loss function
        self.lr = lr
        self.criterion = nn.MSELoss()

        # Tracking losses and PSNRs for visualization
        self.predictions = []
        self.train_losses = []
        self.val_losses = []
        self.train_psnrs = []
        self.val_psnrs = []

    def forward(self, x):
        x = self.conv1(x)  # Initial convolution (maps from 3 → 32 channels)
        x = self.dilated_block1(x)  # First dilated convolution block
        x = self.dilated_block2(x)  # Second dilated convolution block

        # Skip connection: Adding original feature maps before activation
        x = x + self.final_conv(x)  
        x = F.leaky_relu(x)

        # First upsampling
        x_mid = self.upsample1(x)
        x_mid_output = self.conv_mid(x_mid)  # Mid-resolution output (3 channels)

        # Dilated block after first upsampling
        x_mid = self.dilated_block3(x_mid)

        # Second upsampling
        x_large = self.upsample2(x_mid)
        x_large_output = self.conv_large(x_large)  # Large-resolution output (3 channels)

        return x_mid_output, x_large_output

    def training_step(self, batch, batch_idx):
        x_small, y_mid, y_large = batch
        y_pred_mid, y_pred_large = self(x_small)
        loss_mid = self.criterion(y_pred_mid, y_mid)
        loss_large = self.criterion(y_pred_large, y_large)

        psnr_mid = psnr(y_pred_mid, y_mid)
        psnr_large = psnr(y_pred_large, y_large)
        
        self.log("train_loss_mid", loss_mid, prog_bar=True)
        self.log("train_loss_large", loss_large, prog_bar=True)
        self.log("train_psnr_mid", psnr_mid, prog_bar=True)
        self.log("train_psnr_large", psnr_large, prog_bar=True)

        if batch_idx == 0:
            self.train_losses.append((loss_mid.item(), loss_large.item()))
            self.train_psnrs.append((psnr_mid.item(), psnr_large.item()))

        return loss_mid + loss_large

    def validation_step(self, batch, batch_idx):
        x_small, y_mid, y_large = batch
        y_pred_mid, y_pred_large = self(x_small)
        loss_mid = self.criterion(y_pred_mid, y_mid)
        loss_large = self.criterion(y_pred_large, y_large)

        psnr_mid = psnr(y_pred_mid, y_mid)
        psnr_large = psnr(y_pred_large, y_large)

        self.log("val_loss_mid", loss_mid, prog_bar=True)
        self.log("val_loss_large", loss_large, prog_bar=True)
        self.log("val_psnr_mid", psnr_mid, prog_bar=True)
        self.log("val_psnr_large", psnr_large, prog_bar=True)

        if batch_idx == 0:
            self.val_losses.append((loss_mid.item(), loss_large.item()))
            self.val_psnrs.append((psnr_mid.item(), psnr_large.item()))
        if batch_idx == 0 and (len(self.train_losses) % 5 == 0):
            self.predictions.append((x_small[0].cpu(), y_pred_mid[0].cpu(), y_mid[0].cpu(), y_pred_large[0].cpu(), y_large[0].cpu()))

        return loss_mid + loss_large

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)

    def print_loss_per_epoch(self):
        """Prints the training and validation loss and PSNR per epoch."""
        for epoch in range(len(self.train_losses)):
            train_mid, train_large = self.train_losses[epoch]
            val_mid, val_large = self.val_losses[epoch]
            psnr_train_mid, psnr_train_large = self.train_psnrs[epoch]
            psnr_val_mid, psnr_val_large = self.val_psnrs[epoch]
            print(f"Epoch {epoch+1}: Train Loss Mid = {train_mid:.4f}, Train Loss Large = {train_large:.4f}, "
                  f"Validation Loss Mid = {val_mid:.4f}, Validation Loss Large = {val_large:.4f}, "
                  f"Train PSNR Mid = {psnr_train_mid:.2f}, Train PSNR Large = {psnr_train_large:.2f}, "
                  f"Validation PSNR Mid = {psnr_val_mid:.2f}, Validation PSNR Large = {psnr_val_large:.2f}")
    def plot_loss_curves(self):
        """Plots separate training and validation losses and PSNRs for mid and large outputs."""
        epochs = range(1, len(self.train_losses) + 1)
        if len(self.val_losses) > len(self.train_losses):
            self.val_losses = self.val_losses[:len(self.train_losses)]
            self.val_psnrs = self.val_psnrs[:len(self.train_psnrs)]

        train_mid_losses, train_large_losses = zip(*self.train_losses)
        val_mid_losses, val_large_losses = zip(*self.val_losses)
        train_mid_psnr, train_large_psnr = zip(*self.train_psnrs)
        val_mid_psnr, val_large_psnr = zip(*self.val_psnrs)

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_mid_losses, label='Train Loss Mid')
        plt.plot(epochs, val_mid_losses, label='Validation Loss Mid')
        plt.plot(epochs, train_large_losses, label='Train Loss Large')
        plt.plot(epochs, val_large_losses, label='Validation Loss Large')
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss Over Epochs")
        plt.legend()
        plt.grid()

        plt.subplot(1, 2, 2)
        plt.plot(epochs, train_mid_psnr, label='Train PSNR Mid')
        plt.plot(epochs, val_mid_psnr, label='Validation PSNR Mid')
        plt.plot(epochs, train_large_psnr, label='Train PSNR Large')
        plt.plot(epochs, val_large_psnr, label='Validation PSNR Large')
        plt.xlabel("Epochs")
        plt.ylabel("PSNR (dB)")
        plt.title("Training and Validation PSNR Over Epochs")
        plt.legend()
        plt.grid()

        plt.show()
        
    def plot_progress(self, save_gif=False):
        """Visualizes improvements every 5 epochs for one validation image."""
        num_epochs = len(self.predictions)
        fig, axes = plt.subplots(num_epochs, 5, figsize=(15, num_epochs * 3))
        fig.suptitle('Model Improvement Over Epochs (Every 5 Epochs)', fontsize=16)
        frames = []
        
        for i, (input_img, pred_img_mid, target_img_mid, pred_img_large, target_img_large) in enumerate(self.predictions):
            axes[i, 0].imshow(input_img.permute(1, 2, 0).numpy())
            axes[i, 0].set_title(f"Epoch {i*5+1} - Input (72x72)")
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(pred_img_mid.permute(1, 2, 0).numpy())
            axes[i, 1].set_title(f"Epoch {i*5+1} - Pred Mid (144x144)")
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(target_img_mid.permute(1, 2, 0).numpy())
            axes[i, 2].set_title(f"Epoch {i*5+1} - Target Mid (144x144)")
            axes[i, 2].axis('off')
            
            axes[i, 3].imshow(pred_img_large.permute(1, 2, 0).numpy())
            axes[i, 3].set_title(f"Epoch {i*5+1} - Pred Large (288x288)")
            axes[i, 3].axis('off')
            
            axes[i, 4].imshow(target_img_large.permute(1, 2, 0).numpy())
            axes[i, 4].set_title(f"Epoch {i*5+1} - Target Large (288x288)")
            axes[i, 4].axis('off')
            
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            fig.canvas.draw()
            frame = np.array(fig.canvas.renderer.buffer_rgba())
            frames.append(frame)
        
        if save_gif:
            imageio.mimsave('training_progress.gif', frames, fps=2)
        plt.show()

# Set up logging
mlflow_logger = MLFlowLogger(experiment_name="SuperResolutionExperiment")
tensorboard_logger = TensorBoardLogger("tb_logs", name="SuperResolution")

# Instantiate the model
dilated_sr_model = SuperResolutionModel()

# Measure training time
start_time = time.time()

# Define the trainer
trainer = pl.Trainer(max_epochs=10, accelerator="auto", logger=[mlflow_logger, tensorboard_logger])

# Train the model
trainer.fit(dilated_sr_model, train_loader, val_loader)

# End time
end_time = time.time()

# Print total training time
print(f"\nTotal Training Time: {end_time - start_time:.2f} seconds")

# Print loss per epoch
dilated_sr_model.print_loss_per_epoch()
dilated_sr_model.plot_loss_curves()

dilated_sr_model.plot_progress()

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torch.optim as optim
import matplotlib.pyplot as plt
import mlflow
import imageio
import numpy as np
import time
from torchvision import models
from pytorch_lightning.loggers import TensorBoardLogger, MLFlowLogger
from torch.utils.data import DataLoader

def psnr(pred, target, max_pixel=255.0):
    """Compute Peak Signal-to-Noise Ratio (PSNR)."""
    mse = F.mse_loss(pred, target)
    return 20 * torch.log10(max_pixel / torch.sqrt(mse))

class SuperResolutionModel(pl.LightningModule):
    def __init__(self, lr=1e-3):
        super(SuperResolutionModel, self).__init__()

        # Pretrained VGG16 feature extractor (Block1_Conv2)
        vgg16 = models.vgg16(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(vgg16.features.children())[:4])  # Extracting block1_conv2 layer
        for param in self.feature_extractor.parameters():
            param.requires_grad = False  # Freeze VGG16 layers

        # Main convolutional path
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)

        # Concatenation will increase channels
        self.conv_fusion = nn.Conv2d(64 + 64, 64, kernel_size=3, padding=1)

        # First Upsampling
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv_mid = nn.Conv2d(64, 3, kernel_size=1)  # 3 filters for output

        # Second Upsampling
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv_large = nn.Conv2d(64, 3, kernel_size=1)  # 3 filters for large output

        # Learning rate and loss function
        self.lr = lr
        self.criterion = nn.MSELoss()

        # Tracking losses and PSNRs for visualization
        self.predictions = []
        self.train_losses = []
        self.val_losses = []
        self.train_psnrs = []
        self.val_psnrs = []

    def forward(self, x):
        # Extract features using VGG16
        features = self.feature_extractor(x)

        # Pass input through Conv layers
        x = self.conv1(x)
        x = F.relu(self.conv2(x))

        # Concatenate VGG16 features with Conv output
        x = torch.cat([x, features], dim=1)
        x = F.relu(self.conv_fusion(x))

        # First upsampling
        x_mid = self.upsample1(x)
        x_mid_output = self.conv_mid(x_mid)  # Mid-resolution output (3 channels)

        # Second upsampling
        x_large = self.upsample2(x_mid)
        x_large_output = self.conv_large(x_large)  # Large-resolution output (3 channels)

        return x_mid_output, x_large_output

    def training_step(self, batch, batch_idx):
        x_small, y_mid, y_large = batch
        y_pred_mid, y_pred_large = self(x_small)
        loss_mid = self.criterion(y_pred_mid, y_mid)
        loss_large = self.criterion(y_pred_large, y_large)

        psnr_mid = psnr(y_pred_mid, y_mid)
        psnr_large = psnr(y_pred_large, y_large)
        
        self.log("train_loss_mid", loss_mid, prog_bar=True)
        self.log("train_loss_large", loss_large, prog_bar=True)
        self.log("train_psnr_mid", psnr_mid, prog_bar=True)
        self.log("train_psnr_large", psnr_large, prog_bar=True)

        if batch_idx == 0:
            self.train_losses.append((loss_mid.item(), loss_large.item()))
            self.train_psnrs.append((psnr_mid.item(), psnr_large.item()))

        return loss_mid + loss_large

    def validation_step(self, batch, batch_idx):
        x_small, y_mid, y_large = batch
        y_pred_mid, y_pred_large = self(x_small)
        loss_mid = self.criterion(y_pred_mid, y_mid)
        loss_large = self.criterion(y_pred_large, y_large)

        psnr_mid = psnr(y_pred_mid, y_mid)
        psnr_large = psnr(y_pred_large, y_large)

        self.log("val_loss_mid", loss_mid, prog_bar=True)
        self.log("val_loss_large", loss_large, prog_bar=True)
        self.log("val_psnr_mid", psnr_mid, prog_bar=True)
        self.log("val_psnr_large", psnr_large, prog_bar=True)

        if batch_idx == 0:
            self.val_losses.append((loss_mid.item(), loss_large.item()))
            self.val_psnrs.append((psnr_mid.item(), psnr_large.item()))
        if batch_idx == 0 and (len(self.train_losses) % 5 == 0):
            self.predictions.append((x_small[0].cpu(), y_pred_mid[0].cpu(), y_mid[0].cpu(), y_pred_large[0].cpu(), y_large[0].cpu()))

        return loss_mid + loss_large

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)

    def print_loss_per_epoch(self):
        """Prints the training and validation loss and PSNR per epoch."""
        for epoch in range(len(self.train_losses)):
            train_mid, train_large = self.train_losses[epoch]
            val_mid, val_large = self.val_losses[epoch]
            psnr_train_mid, psnr_train_large = self.train_psnrs[epoch]
            psnr_val_mid, psnr_val_large = self.val_psnrs[epoch]
            print(f"Epoch {epoch+1}: Train Loss Mid = {train_mid:.4f}, Train Loss Large = {train_large:.4f}, "
                  f"Validation Loss Mid = {val_mid:.4f}, Validation Loss Large = {val_large:.4f}, "
                  f"Train PSNR Mid = {psnr_train_mid:.2f}, Train PSNR Large = {psnr_train_large:.2f}, "
                  f"Validation PSNR Mid = {psnr_val_mid:.2f}, Validation PSNR Large = {psnr_val_large:.2f}")
    def plot_loss_curves(self):
        """Plots separate training and validation losses and PSNRs for mid and large outputs."""
        epochs = range(1, len(self.train_losses) + 1)
        if len(self.val_losses) > len(self.train_losses):
            self.val_losses = self.val_losses[:len(self.train_losses)]
            self.val_psnrs = self.val_psnrs[:len(self.train_psnrs)]

        train_mid_losses, train_large_losses = zip(*self.train_losses)
        val_mid_losses, val_large_losses = zip(*self.val_losses)
        train_mid_psnr, train_large_psnr = zip(*self.train_psnrs)
        val_mid_psnr, val_large_psnr = zip(*self.val_psnrs)

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_mid_losses, label='Train Loss Mid')
        plt.plot(epochs, val_mid_losses, label='Validation Loss Mid')
        plt.plot(epochs, train_large_losses, label='Train Loss Large')
        plt.plot(epochs, val_large_losses, label='Validation Loss Large')
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss Over Epochs")
        plt.legend()
        plt.grid()

        plt.subplot(1, 2, 2)
        plt.plot(epochs, train_mid_psnr, label='Train PSNR Mid')
        plt.plot(epochs, val_mid_psnr, label='Validation PSNR Mid')
        plt.plot(epochs, train_large_psnr, label='Train PSNR Large')
        plt.plot(epochs, val_large_psnr, label='Validation PSNR Large')
        plt.xlabel("Epochs")
        plt.ylabel("PSNR (dB)")
        plt.title("Training and Validation PSNR Over Epochs")
        plt.legend()
        plt.grid()

        plt.show()
        
    def plot_progress(self, save_gif=False):
        """Visualizes improvements every 5 epochs for one validation image."""
        num_epochs = len(self.predictions)
        fig, axes = plt.subplots(num_epochs, 5, figsize=(15, num_epochs * 3))
        fig.suptitle('Model Improvement Over Epochs (Every 5 Epochs)', fontsize=16)
        frames = []
        
        for i, (input_img, pred_img_mid, target_img_mid, pred_img_large, target_img_large) in enumerate(self.predictions):
            axes[i, 0].imshow(input_img.permute(1, 2, 0).numpy())
            axes[i, 0].set_title(f"Epoch {i*5+1} - Input (72x72)")
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(pred_img_mid.permute(1, 2, 0).numpy())
            axes[i, 1].set_title(f"Epoch {i*5+1} - Pred Mid (144x144)")
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(target_img_mid.permute(1, 2, 0).numpy())
            axes[i, 2].set_title(f"Epoch {i*5+1} - Target Mid (144x144)")
            axes[i, 2].axis('off')
            
            axes[i, 3].imshow(pred_img_large.permute(1, 2, 0).numpy())
            axes[i, 3].set_title(f"Epoch {i*5+1} - Pred Large (288x288)")
            axes[i, 3].axis('off')
            
            axes[i, 4].imshow(target_img_large.permute(1, 2, 0).numpy())
            axes[i, 4].set_title(f"Epoch {i*5+1} - Target Large (288x288)")
            axes[i, 4].axis('off')
            
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            fig.canvas.draw()
            frame = np.array(fig.canvas.renderer.buffer_rgba())
            frames.append(frame)
        
        if save_gif:
            imageio.mimsave('training_progress.gif', frames, fps=2)
        plt.show()

# Set up logging
mlflow_logger = MLFlowLogger(experiment_name="SuperResolutionExperiment")
tensorboard_logger = TensorBoardLogger("tb_logs", name="SuperResolution")

# Instantiate the model
vgg_sr_model = SuperResolutionModel()

# Measure training time
start_time = time.time()

# Define the trainer
trainer = pl.Trainer(max_epochs=10, accelerator="auto", logger=[mlflow_logger, tensorboard_logger])

# Train the model
trainer.fit(vgg_sr_model, train_loader, val_loader)

# End time
end_time = time.time()

# Print total training time
print(f"\nTotal Training Time: {end_time - start_time:.2f} seconds")

# Print loss per epoch
vgg_sr_model.print_loss_per_epoch()
vgg_sr_model.plot_loss_curves()

vgg_sr_model.plot_progress()

!pip install openai-clip

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torch.optim as optim
import matplotlib.pyplot as plt
import mlflow
import seaborn as sns
import numpy as np
import time
import clip
from torchvision import transforms
from PIL import Image
from pytorch_lightning.loggers import TensorBoardLogger, MLFlowLogger
from torch.utils.data import DataLoader

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model_clip, preprocess_clip = clip.load("ViT-B/32", device=device)

# Ensure CLIP runs in FP32 mode for compatibility
model_clip.float()


### 1️⃣ Standalone CLIP Feature Extractor ###
class CLIPFeatureExtractor(nn.Module):
    def __init__(self):
        super(CLIPFeatureExtractor, self).__init__()
        self.model = model_clip.visual  # Use CLIP's vision encoder
        for param in self.model.parameters():
            param.requires_grad = False  # Freeze CLIP weights

    def forward(self, x):
        # Convert input to PIL Images, preprocess for CLIP
        images = torch.stack([preprocess_clip(transforms.ToPILImage()(img.cpu())) for img in x])
        images = images.to(device)

        # Extract CLIP visual features
        with torch.no_grad():
            features = self.model(images).float()  # Ensure FP32 compatibility

        return features


### 2️⃣ Super-Resolution Model ###
class SuperResolutionModel(pl.LightningModule):
    def __init__(self, lr=1e-3):
        super(SuperResolutionModel, self).__init__()

        # CLIP Feature Extractor (Standalone Module)
        self.clip_extractor = CLIPFeatureExtractor()

        # Main convolutional path
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)

        # Fusion Layer (Concatenates CLIP's 512-dim vector)
        self.fusion = nn.Linear(512, 64)  # Maps CLIP vector to match spatial feature maps

        # Upsampling Layers
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv_mid = nn.Conv2d(64, 3, kernel_size=1)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv_large = nn.Conv2d(64, 3, kernel_size=1)

        # Loss and Learning Rate
        self.lr = lr
        self.criterion = nn.MSELoss()
        self.predictions=[]
        # Tracking losses and PSNRs
        self.train_losses = []
        self.val_losses = []
        self.train_psnrs = []
        self.val_psnrs = []

    def forward(self, x):
        # Extract CLIP Features
        clip_features = self.clip_extractor(x)  # (batch_size, 512)

        # Pass through CNN
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        # Expand CLIP features and fuse with CNN output
        clip_features = self.fusion(clip_features)  # Map to (batch_size, 64)
        clip_features = clip_features.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, x.shape[2], x.shape[3])  # Reshape

        # Fuse with main path
        x = x + clip_features  # Simple element-wise addition for conditioning

        # Upsampling
        x_mid = self.upsample1(x)
        x_mid_output = self.conv_mid(x_mid)
        x_large = self.upsample2(x_mid)
        x_large_output = self.conv_large(x_large)

        return x_mid_output, x_large_output

    def training_step(self, batch, batch_idx):
        x_small, y_mid, y_large = batch
        y_pred_mid, y_pred_large = self(x_small)
        loss_mid = self.criterion(y_pred_mid, y_mid)
        loss_large = self.criterion(y_pred_large, y_large)

        psnr_mid = psnr(y_pred_mid, y_mid)
        psnr_large = psnr(y_pred_large, y_large)
        
        self.log("train_loss_mid", loss_mid, prog_bar=True)
        self.log("train_loss_large", loss_large, prog_bar=True)
        self.log("train_psnr_mid", psnr_mid, prog_bar=True)
        self.log("train_psnr_large", psnr_large, prog_bar=True)

        if batch_idx == 0:
            self.train_losses.append((loss_mid.item(), loss_large.item()))
            self.train_psnrs.append((psnr_mid.item(), psnr_large.item()))

        return loss_mid + loss_large

    def validation_step(self, batch, batch_idx):
        x_small, y_mid, y_large = batch
        y_pred_mid, y_pred_large = self(x_small)
        loss_mid = self.criterion(y_pred_mid, y_mid)
        loss_large = self.criterion(y_pred_large, y_large)

        psnr_mid = psnr(y_pred_mid, y_mid)
        psnr_large = psnr(y_pred_large, y_large)

        self.log("val_loss_mid", loss_mid, prog_bar=True)
        self.log("val_loss_large", loss_large, prog_bar=True)
        self.log("val_psnr_mid", psnr_mid, prog_bar=True)
        self.log("val_psnr_large", psnr_large, prog_bar=True)

        if batch_idx == 0:
            self.val_losses.append((loss_mid.item(), loss_large.item()))
            self.val_psnrs.append((psnr_mid.item(), psnr_large.item()))
        if batch_idx == 0 and (len(self.train_losses) % 5 == 0):
            self.predictions.append((x_small[0].cpu(), y_pred_mid[0].cpu(), y_mid[0].cpu(), y_pred_large[0].cpu(), y_large[0].cpu()))

        return loss_mid + loss_large

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)

    def print_loss_per_epoch(self):
        """Prints the training and validation loss and PSNR per epoch."""
        for epoch in range(len(self.train_losses)):
            train_mid, train_large = self.train_losses[epoch]
            val_mid, val_large = self.val_losses[epoch]
            psnr_train_mid, psnr_train_large = self.train_psnrs[epoch]
            psnr_val_mid, psnr_val_large = self.val_psnrs[epoch]
            print(f"Epoch {epoch+1}: Train Loss Mid = {train_mid:.4f}, Train Loss Large = {train_large:.4f}, "
                  f"Validation Loss Mid = {val_mid:.4f}, Validation Loss Large = {val_large:.4f}, "
                  f"Train PSNR Mid = {psnr_train_mid:.2f}, Train PSNR Large = {psnr_train_large:.2f}, "
                  f"Validation PSNR Mid = {psnr_val_mid:.2f}, Validation PSNR Large = {psnr_val_large:.2f}")
    def plot_loss_curves(self):
        """Plots separate training and validation losses and PSNRs for mid and large outputs."""
        epochs = range(1, len(self.train_losses) + 1)
        if len(self.val_losses) > len(self.train_losses):
            self.val_losses = self.val_losses[:len(self.train_losses)]
            self.val_psnrs = self.val_psnrs[:len(self.train_psnrs)]

        train_mid_losses, train_large_losses = zip(*self.train_losses)
        val_mid_losses, val_large_losses = zip(*self.val_losses)
        train_mid_psnr, train_large_psnr = zip(*self.train_psnrs)
        val_mid_psnr, val_large_psnr = zip(*self.val_psnrs)

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_mid_losses, label='Train Loss Mid')
        plt.plot(epochs, val_mid_losses, label='Validation Loss Mid')
        plt.plot(epochs, train_large_losses, label='Train Loss Large')
        plt.plot(epochs, val_large_losses, label='Validation Loss Large')
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss Over Epochs")
        plt.legend()
        plt.grid()

        plt.subplot(1, 2, 2)
        plt.plot(epochs, train_mid_psnr, label='Train PSNR Mid')
        plt.plot(epochs, val_mid_psnr, label='Validation PSNR Mid')
        plt.plot(epochs, train_large_psnr, label='Train PSNR Large')
        plt.plot(epochs, val_large_psnr, label='Validation PSNR Large')
        plt.xlabel("Epochs")
        plt.ylabel("PSNR (dB)")
        plt.title("Training and Validation PSNR Over Epochs")
        plt.legend()
        plt.grid()

        plt.show()
        
    def plot_progress(self, save_gif=False):
        """Visualizes improvements every 5 epochs for one validation image."""
        num_epochs = len(self.predictions)
        fig, axes = plt.subplots(num_epochs, 5, figsize=(15, num_epochs * 3))
        fig.suptitle('Model Improvement Over Epochs (Every 5 Epochs)', fontsize=16)
        frames = []
        
        for i, (input_img, pred_img_mid, target_img_mid, pred_img_large, target_img_large) in enumerate(self.predictions):
            axes[i, 0].imshow(input_img.permute(1, 2, 0).numpy())
            axes[i, 0].set_title(f"Epoch {i*5+1} - Input (72x72)")
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(pred_img_mid.permute(1, 2, 0).numpy())
            axes[i, 1].set_title(f"Epoch {i*5+1} - Pred Mid (144x144)")
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(target_img_mid.permute(1, 2, 0).numpy())
            axes[i, 2].set_title(f"Epoch {i*5+1} - Target Mid (144x144)")
            axes[i, 2].axis('off')
            
            axes[i, 3].imshow(pred_img_large.permute(1, 2, 0).numpy())
            axes[i, 3].set_title(f"Epoch {i*5+1} - Pred Large (288x288)")
            axes[i, 3].axis('off')
            
            axes[i, 4].imshow(target_img_large.permute(1, 2, 0).numpy())
            axes[i, 4].set_title(f"Epoch {i*5+1} - Target Large (288x288)")
            axes[i, 4].axis('off')
            
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            fig.canvas.draw()
            frame = np.array(fig.canvas.renderer.buffer_rgba())
            frames.append(frame)
        
        if save_gif:
            imageio.mimsave('training_progress.gif', frames, fps=2)
        plt.show()


### 3️⃣ Training Setup ###
mlflow_logger = MLFlowLogger(experiment_name="SuperResolutionExperiment")
tensorboard_logger = TensorBoardLogger("tb_logs", name="SuperResolution")

# Instantiate Model
clip_sr_model = SuperResolutionModel().to(device)

# Measure training time
start_time = time.time()

trainer = pl.Trainer(max_epochs=10, accelerator="auto", logger=[mlflow_logger, tensorboard_logger])
trainer.fit(clip_sr_model, train_loader, val_loader)

end_time = time.time()
print(f"\nTotal Training Time: {end_time - start_time:.2f} seconds")

# Print loss per epoch
clip_sr_model.print_loss_per_epoch()
clip_sr_model.plot_loss_curves()



clip_sr_model.plot_progress()

import matplotlib.pyplot as plt
import torch

# Ensure val_dataset is properly defined and accessible
num_samples = 10  # Number of images to extract

# Extract the first 10 high-resolution (y_large) images
y_large_images = [val_dataset[i][2] for i in range(num_samples)]  # Index 2 corresponds to y_large

# Convert to numpy for visualization (assuming images are in CHW format)
y_large_images_np = [img.permute(1, 2, 0).cpu().numpy() for img in y_large_images]

# Plot the images
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
fig.suptitle("First 10 y_large Images from Validation Dataset", fontsize=16)

for i, ax in enumerate(axes.flat):
    ax.imshow(y_large_images_np[i])
    ax.set_title(f"Image {i+1}")
    ax.axis("off")

plt.show()


import torch
import clip
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms

# Ensure everything is moved to the correct device and dtype
device = "cuda" if torch.cuda.is_available() else "cpu"
model_clip, preprocess_clip = clip.load("ViT-B/32", device=device)

# Ensure CLIP is in float32 mode
model_clip = model_clip.float()

# Move Super-Resolution Model to the same device
clip_sr_model = clip_sr_model.to(device)

# Ensure dataset is moved to the correct device
num_samples = 10
x_small_images = torch.stack([val_dataset[i][0].to(device) for i in range(num_samples)])  # Move images to GPU

# Generate model predictions
with torch.no_grad():
    _, y_large_pred = clip_sr_model(x_small_images)

# Convert predictions to CLIP-preprocessed format
image_tensors = torch.stack([preprocess_clip(transforms.ToPILImage()(img.cpu())).to(device) for img in y_large_pred])

# Move tensors to the correct device and ensure float precision
image_tensors = image_tensors.to(device).float()

# Compute CLIP features
with torch.no_grad():
    image_features = model_clip.encode_image(image_tensors).float()  # Ensure consistency

# Compute text features
descriptions = [
    "Blue and white train on railway tracks.",
    "Fishing boat sailing on a calm ocean.",
    "Cartoon-faced steam train arriving at platform.",
    "Horse and rider jump over obstacle.",
    "Living room with furniture, guitar, and chandelier.",
    "Black-and-white retro bus with 'Turismo' sign.",
    "Historic building with grand arched entrance.",
    "Red and white seaplane floating on water.",
    "Traffic light glowing red at night.",
    "Cyclist in red racing on a winding road."
]

text_tokens = clip.tokenize(descriptions).to(device)

with torch.no_grad():
    text_features = model_clip.encode_text(text_tokens).float()  # Ensure float32

# Normalize feature vectors
image_features /= image_features.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)

# Compute similarity matrix
similarity_matrix = (text_features @ image_features.T).cpu().numpy()

# Convert predicted images to NumPy format for display
y_large_pred_np = [img.permute(1, 2, 0).cpu().numpy() for img in y_large_pred]

# Create the figure and grid layout
fig, ax = plt.subplots(figsize=(15, 10))

# Plot the similarity matrix heatmap
sns.heatmap(
    similarity_matrix, annot=True, cmap="viridis",
    ax=ax, yticklabels=descriptions, xticklabels=False,
    linewidths=1, linecolor="black"  # Thicker gridlines
)

# Adjust title spacing
ax.set_title("CLIP Cosine Similarity between Descriptions and Model Predictions", fontsize=16, pad=50)

# Add images above each column
image_size = 0.09  # Adjusted image size for better alignment
y_offset = 0.92  # Move images higher to avoid overlap

for idx, img in enumerate(y_large_pred_np):
    ax_img = fig.add_axes([0.1 + idx * 0.06, y_offset, image_size, image_size], anchor='NE', zorder=1)
    ax_img.imshow(img)
    ax_img.set_xticks([])
    ax_img.set_yticks([])
    ax_img.axis('off')

# Set axis labels
ax.set_xlabel("Generated Images", fontsize=14, labelpad=10)
ax.set_ylabel("Text Descriptions", fontsize=14)

plt.show()

import torch
import torch_directml
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils import trainModel
from models.unet import UNet
from dataset.sketch_dataset import SketchToImageDataset


torch.manual_seed(42)  # Set random seed for reproducibility

# Device configuration
device = torch.device('cuda') 

# Transform sketch and original images to tensors for training
train_transform = A.Compose([
    A.Resize(256, 256),
    A.HorizontalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ElasticTransform(p=0.3),
    A.ColorJitter(p=0.4),
    A.GaussNoise(p=0.3),
    A.CoarseDropout(num_holes_range=(1, 8), hole_height_range=(8, 32), hole_width_range=(8, 32), fill=0, p=0.2),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2(),
], additional_targets={'mask': 'image'})

# train_transform = transforms.Compose([
#     transforms.Resize((256, 256)),
#     transforms.ToTensor(),  # Converts to [0, 1] and moves channel to first dimension
#     transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # Scales to [-1, 1]
# ])

# Transform sketch and original images to tensors for testing
test_transform = A.Compose([
    A.Resize(256, 256),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2(),
], additional_targets={'mask': 'image'})
# test_transform = transforms.Compose([
#     transforms.Resize((256, 256)),
#     transforms.ToTensor(),  # Converts to [0, 1] and moves channel to first dimension
#     transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # Scales to [-1, 1]
# ])

# Create train dataset
train_dataset = SketchToImageDataset(
    sketch_dir='./data/sketches_training',
    original_dir='./data/originals_training',
    transform=train_transform
)

# Create test dataset
test_dataset = SketchToImageDataset(
    sketch_dir='./data/sketches_testing',
    original_dir='./data/originals_testing',
    transform=test_transform
)

# Create training DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Create testing DataLoader
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Initialize the model
model = UNet().to(device)

# Loss function
loss_fn = nn.L1Loss()

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=2e-4, betas=(0.5, 0.999))

# Scheduler
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=2)

epochs = 500 # Number of epochs
# Train the model
trainModel(epochs, model, optimizer, scheduler, loss_fn, train_loader, test_loader, device)

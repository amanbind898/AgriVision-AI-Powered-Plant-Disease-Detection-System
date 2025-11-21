"""
Train Plant Disease Detection Model using PyTorch
Works with NVIDIA GPU on Windows
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import os
import time
from datetime import datetime

print("="*80)
print("PLANT DISEASE DETECTION - PYTORCH TRAINING")
print("="*80)

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDevice: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
else:
    print("WARNING: No GPU detected! Training will be very slow.")
    print("Install PyTorch with CUDA:")
    print("pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
    response = input("\nContinue with CPU? (y/n): ")
    if response.lower() != 'y':
        exit(0)

# Configuration
IMG_SIZE = 224  # Using 224 for pretrained models
BATCH_SIZE = 32
EPOCHS = 8  # PlantVillage reaches 98-99% in 5-8 epochs
LEARNING_RATE = 0.001

TRAIN_DIR = "./data/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train"
VALID_DIR = "./data/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/valid"

print("\n" + "="*80)
print("CONFIGURATION")
print("="*80)
print(f"Image size: {IMG_SIZE}x{IMG_SIZE}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Epochs: {EPOCHS}")
print(f"Learning rate: {LEARNING_RATE}")

# Data transforms
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

valid_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load datasets
print("\n" + "="*80)
print("LOADING DATASET")
print("="*80)

train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
valid_dataset = datasets.ImageFolder(VALID_DIR, transform=valid_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

num_classes = len(train_dataset.classes)
print(f"\nClasses: {num_classes}")
print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(valid_dataset)}")
print(f"Training batches: {len(train_loader)}")
print(f"Validation batches: {len(valid_loader)}")

# Build model using pretrained EfficientNet
print("\n" + "="*80)
print("BUILDING MODEL")
print("="*80)
print("Using pretrained EfficientNet-B0 with ImageNet weights")

model = models.efficientnet_b0(pretrained=True)

# Modify final layer for our number of classes
num_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_features, num_classes)

model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)

print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# Training function
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, labels) in enumerate(loader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        if (batch_idx + 1) % 100 == 0:
            print(f'  Batch [{batch_idx+1}/{len(loader)}] Loss: {loss.item():.4f} Acc: {100.*correct/total:.2f}%')
    
    epoch_loss = running_loss / len(loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

# Validation function
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / len(loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

# Training loop
print("\n" + "="*80)
print("TRAINING MODEL")
print("="*80)
print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("\nPress Ctrl+C to stop training\n")

best_acc = 0.0
patience_counter = 0
max_patience = 3  # Reduced since model converges quickly

try:
    for epoch in range(EPOCHS):
        epoch_start = time.time()
        
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print("-" * 80)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc = validate(model, valid_loader, criterion, device)
        
        # Update learning rate
        scheduler.step(val_acc)
        
        epoch_time = time.time() - epoch_start
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print(f"  Time: {epoch_time:.1f}s")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'class_names': train_dataset.classes
            }, './models/best_model_pytorch.pth')
            print(f"  [SAVED] New best model! Val Acc: {val_acc:.2f}%")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"  No improvement. Patience: {patience_counter}/{max_patience}")
        
        # Early stopping
        if patience_counter >= max_patience:
            print(f"\n[EARLY STOPPING] No improvement for {max_patience} epochs")
            break
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"\nBest Validation Accuracy: {best_acc:.2f}%")
    print(f"Model saved to: ./models/best_model_pytorch.pth")
    
    if best_acc < 80:
        print("\n[WARNING] Accuracy below 80%. Consider training longer.")
    else:
        print("\n[SUCCESS] Model trained successfully!")

except KeyboardInterrupt:
    print("\n\n[INTERRUPTED] Training stopped by user")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'class_names': train_dataset.classes
    }, './models/interrupted_model_pytorch.pth')
    print("Model saved to: ./models/interrupted_model_pytorch.pth")

print("\n" + "="*80)
print("DONE")
print("="*80)

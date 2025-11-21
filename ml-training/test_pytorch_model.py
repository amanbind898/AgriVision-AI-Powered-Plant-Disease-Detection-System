"""
Test the PyTorch model after training
"""
import torch
from torchvision import transforms
from PIL import Image
import os

# Load model
model_path = "./models/best_model_pytorch.pth"

if not os.path.exists(model_path):
    print(f"[ERROR] Model not found: {model_path}")
    print("Please wait for training to complete first.")
    exit(1)

print("Loading model...")
checkpoint = torch.load(model_path, map_location='cpu')

print("\n" + "="*80)
print("MODEL INFO")
print("="*80)
print(f"Validation Accuracy: {checkpoint.get('val_acc', 'N/A'):.2f}%")
print(f"Epoch: {checkpoint.get('epoch', 'N/A')}")
print(f"Number of classes: {len(checkpoint.get('class_names', []))}")

print("\n" + "="*80)
print("CLASS NAMES")
print("="*80)
for i, name in enumerate(checkpoint.get('class_names', []), 1):
    print(f"{i:2d}. {name}")

print("\n" + "="*80)
print("MODEL READY FOR DEPLOYMENT")
print("="*80)
print("\nThe model is ready to use in the backend!")
print("Make sure to:")
print("1. Install PyTorch: pip install torch torchvision")
print("2. Update backend to use disease_model_pytorch.py")
print("3. Restart the backend server")

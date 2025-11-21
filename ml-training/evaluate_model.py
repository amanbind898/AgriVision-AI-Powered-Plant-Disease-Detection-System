"""
Comprehensive Model Evaluation Script
Calculates all metrics, generates graphs, confusion matrix, ROC curves
"""
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    roc_curve, auc, precision_recall_fscore_support,
    accuracy_score
)
from sklearn.preprocessing import label_binarize
import json
import os
from datetime import datetime

print("="*80)
print("COMPREHENSIVE MODEL EVALUATION")
print("="*80)

# Configuration
IMG_SIZE = 224
BATCH_SIZE = 32
MODEL_PATH = "./models/best_model_pytorch.pth"
TRAIN_DIR = "./data/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train"
VALID_DIR = "./data/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/valid"
OUTPUT_DIR = "./evaluation_results"

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDevice: {device}")

# Load model
print("\n" + "="*80)
print("LOADING MODEL")
print("="*80)

checkpoint = torch.load(MODEL_PATH, map_location=device)
class_names = checkpoint['class_names']
num_classes = len(class_names)

model = models.efficientnet_b0(pretrained=False)
num_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_features, num_classes)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

print(f"Model loaded successfully")
print(f"Number of classes: {num_classes}")
print(f"Validation accuracy from training: {checkpoint.get('val_acc', 'N/A'):.2f}%")


# Data transforms
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load datasets
print("\n" + "="*80)
print("LOADING DATASETS")
print("="*80)

train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=transform)
valid_dataset = datasets.ImageFolder(VALID_DIR, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(valid_dataset)}")

# Evaluation function
def evaluate_model(model, loader, device, dataset_name="Dataset"):
    """Evaluate model and return predictions and labels"""
    print(f"\n{'='*80}")
    print(f"EVALUATING ON {dataset_name.upper()}")
    print("="*80)
    
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
            if (i + 1) % 50 == 0:
                print(f"  Processed {(i+1)*BATCH_SIZE}/{len(loader.dataset)} samples")
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    return all_preds, all_labels, all_probs

# Evaluate on both datasets
train_preds, train_labels, train_probs = evaluate_model(model, train_loader, device, "Training Set")
valid_preds, valid_labels, valid_probs = evaluate_model(model, valid_loader, device, "Validation Set")


# Calculate metrics
print("\n" + "="*80)
print("CALCULATING METRICS")
print("="*80)

def calculate_metrics(labels, preds, probs, dataset_name):
    """Calculate all metrics"""
    accuracy = accuracy_score(labels, preds) * 100
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, preds, average='weighted', zero_division=0
    )
    
    print(f"\n{dataset_name} Metrics:")
    print(f"  Accuracy:  {accuracy:.2f}%")
    print(f"  Precision: {precision*100:.2f}%")
    print(f"  Recall:    {recall*100:.2f}%")
    print(f"  F1-Score:  {f1*100:.2f}%")
    
    return {
        'accuracy': accuracy,
        'precision': precision * 100,
        'recall': recall * 100,
        'f1_score': f1 * 100
    }

train_metrics = calculate_metrics(train_labels, train_preds, train_probs, "Training")
valid_metrics = calculate_metrics(valid_labels, valid_preds, valid_probs, "Validation")

# Save metrics to JSON
metrics_data = {
    'training': train_metrics,
    'validation': valid_metrics,
    'num_classes': num_classes,
    'num_train_samples': len(train_dataset),
    'num_valid_samples': len(valid_dataset),
    'evaluation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
}

with open(f"{OUTPUT_DIR}/metrics.json", 'w') as f:
    json.dump(metrics_data, f, indent=2)

print(f"\n✅ Metrics saved to {OUTPUT_DIR}/metrics.json")


# Generate Confusion Matrix
print("\n" + "="*80)
print("GENERATING CONFUSION MATRIX")
print("="*80)

cm = confusion_matrix(valid_labels, valid_preds)

plt.figure(figsize=(20, 18))
sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names,
            cbar_kws={'label': 'Count'})
plt.title('Confusion Matrix - Validation Set', fontsize=16, pad=20)
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.xticks(rotation=90, ha='right', fontsize=8)
plt.yticks(rotation=0, fontsize=8)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/confusion_matrix.png", dpi=300, bbox_inches='tight')
plt.close()

print(f"✅ Confusion matrix saved to {OUTPUT_DIR}/confusion_matrix.png")

# Generate normalized confusion matrix
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(20, 18))
sns.heatmap(cm_normalized, annot=False, fmt='.2f', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names,
            cbar_kws={'label': 'Proportion'}, vmin=0, vmax=1)
plt.title('Normalized Confusion Matrix - Validation Set', fontsize=16, pad=20)
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.xticks(rotation=90, ha='right', fontsize=8)
plt.yticks(rotation=0, fontsize=8)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/confusion_matrix_normalized.png", dpi=300, bbox_inches='tight')
plt.close()

print(f"✅ Normalized confusion matrix saved to {OUTPUT_DIR}/confusion_matrix_normalized.png")


# Generate Per-Class Metrics
print("\n" + "="*80)
print("CALCULATING PER-CLASS METRICS")
print("="*80)

precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
    valid_labels, valid_preds, average=None, zero_division=0
)

per_class_metrics = []
for i, class_name in enumerate(class_names):
    per_class_metrics.append({
        'class': class_name,
        'precision': precision_per_class[i] * 100,
        'recall': recall_per_class[i] * 100,
        'f1_score': f1_per_class[i] * 100,
        'support': int(support_per_class[i])
    })

# Sort by F1-score
per_class_metrics_sorted = sorted(per_class_metrics, key=lambda x: x['f1_score'], reverse=True)

print("\nTop 10 Classes by F1-Score:")
for i, metrics in enumerate(per_class_metrics_sorted[:10], 1):
    print(f"{i:2d}. {metrics['class']:40s} F1: {metrics['f1_score']:5.2f}%")

print("\nBottom 10 Classes by F1-Score:")
for i, metrics in enumerate(per_class_metrics_sorted[-10:], 1):
    print(f"{i:2d}. {metrics['class']:40s} F1: {metrics['f1_score']:5.2f}%")

# Save per-class metrics
with open(f"{OUTPUT_DIR}/per_class_metrics.json", 'w') as f:
    json.dump(per_class_metrics_sorted, f, indent=2)

print(f"\n✅ Per-class metrics saved to {OUTPUT_DIR}/per_class_metrics.json")


# Generate ROC Curves (One-vs-Rest for multiclass)
print("\n" + "="*80)
print("GENERATING ROC CURVES")
print("="*80)

# Binarize labels for ROC curve
valid_labels_bin = label_binarize(valid_labels, classes=range(num_classes))

# Calculate ROC curve and AUC for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(valid_labels_bin[:, i], valid_probs[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Calculate micro-average ROC curve and AUC
fpr["micro"], tpr["micro"], _ = roc_curve(valid_labels_bin.ravel(), valid_probs.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Calculate macro-average ROC curve and AUC
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))
mean_tpr = np.zeros_like(all_fpr)
for i in range(num_classes):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
mean_tpr /= num_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

print(f"Micro-average AUC: {roc_auc['micro']:.4f}")
print(f"Macro-average AUC: {roc_auc['macro']:.4f}")

# Plot ROC curves (macro and micro average)
plt.figure(figsize=(10, 8))
plt.plot(fpr["micro"], tpr["micro"],
         label=f'Micro-average ROC (AUC = {roc_auc["micro"]:.4f})',
         color='deeppink', linestyle=':', linewidth=4)
plt.plot(fpr["macro"], tpr["macro"],
         label=f'Macro-average ROC (AUC = {roc_auc["macro"]:.4f})',
         color='navy', linestyle=':', linewidth=4)
plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curves - Validation Set', fontsize=14)
plt.legend(loc="lower right", fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/roc_curve.png", dpi=300, bbox_inches='tight')
plt.close()

print(f"✅ ROC curve saved to {OUTPUT_DIR}/roc_curve.png")

# Save AUC scores
auc_data = {
    'micro_average': roc_auc['micro'],
    'macro_average': roc_auc['macro'],
    'per_class': {class_names[i]: roc_auc[i] for i in range(num_classes)}
}

with open(f"{OUTPUT_DIR}/auc_scores.json", 'w') as f:
    json.dump(auc_data, f, indent=2)

print(f"✅ AUC scores saved to {OUTPUT_DIR}/auc_scores.json")


# Generate Classification Report
print("\n" + "="*80)
print("CLASSIFICATION REPORT")
print("="*80)

report = classification_report(valid_labels, valid_preds, target_names=class_names, digits=4)
print(report)

with open(f"{OUTPUT_DIR}/classification_report.txt", 'w') as f:
    f.write("Classification Report - Validation Set\n")
    f.write("="*80 + "\n\n")
    f.write(report)

print(f"\n✅ Classification report saved to {OUTPUT_DIR}/classification_report.txt")

# Summary
print("\n" + "="*80)
print("EVALUATION COMPLETE")
print("="*80)
print(f"\nAll results saved to: {OUTPUT_DIR}/")
print("\nGenerated files:")
print("  1. metrics.json - Overall metrics")
print("  2. per_class_metrics.json - Per-class performance")
print("  3. confusion_matrix.png - Confusion matrix")
print("  4. confusion_matrix_normalized.png - Normalized confusion matrix")
print("  5. roc_curve.png - ROC curves")
print("  6. auc_scores.json - AUC scores")
print("  7. classification_report.txt - Detailed classification report")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"\nTraining Set:")
print(f"  Samples: {len(train_dataset)}")
print(f"  Accuracy: {train_metrics['accuracy']:.2f}%")
print(f"  Precision: {train_metrics['precision']:.2f}%")
print(f"  Recall: {train_metrics['recall']:.2f}%")
print(f"  F1-Score: {train_metrics['f1_score']:.2f}%")

print(f"\nValidation Set:")
print(f"  Samples: {len(valid_dataset)}")
print(f"  Accuracy: {valid_metrics['accuracy']:.2f}%")
print(f"  Precision: {valid_metrics['precision']:.2f}%")
print(f"  Recall: {valid_metrics['recall']:.2f}%")
print(f"  F1-Score: {valid_metrics['f1_score']:.2f}%")

print(f"\nROC AUC:")
print(f"  Micro-average: {roc_auc['micro']:.4f}")
print(f"  Macro-average: {roc_auc['macro']:.4f}")

print("\n" + "="*80)


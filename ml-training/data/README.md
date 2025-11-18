# Dataset Information

## 📊 New Plant Diseases Dataset

**Source**: [Kaggle - New Plant Diseases Dataset](https://www.kaggle.com/vipoooool/new-plant-diseases-dataset)

### Overview
This dataset contains ~70,000 images of healthy and diseased plant leaves collected under controlled conditions. The images are organized by plant species and disease type.

### Dataset Structure
```
PlantVillage/
├── train/
│   ├── Apple___Apple_scab/
│   ├── Apple___Black_rot/
│   ├── Apple___Cedar_apple_rust/
│   ├── Apple___healthy/
│   ├── ...
│   └── Tomato___healthy/
├── valid/
│   └── [same structure as train]
└── test/
    └── [same structure as train]
```

### Statistics
- **Total Images**: ~70,000
- **Training Set**: ~54,000 images (70%)
- **Validation Set**: ~8,000 images (15%)
- **Test Set**: ~8,000 images (15%)
- **Image Format**: JPG
- **Image Size**: 256x256 pixels (resized to 128x128 for training)
- **Color Space**: RGB

### Classes (38 total)

#### Apple (4 classes)
- Apple scab
- Black rot
- Cedar apple rust
- Healthy

#### Blueberry (1 class)
- Healthy

#### Cherry (2 classes)
- Powdery mildew
- Healthy

#### Corn/Maize (4 classes)
- Cercospora leaf spot (Gray leaf spot)
- Common rust
- Northern Leaf Blight
- Healthy

#### Grape (4 classes)
- Black rot
- Esca (Black Measles)
- Leaf blight (Isariopsis Leaf Spot)
- Healthy

#### Orange (1 class)
- Huanglongbing (Citrus greening)

#### Peach (2 classes)
- Bacterial spot
- Healthy

#### Pepper/Bell (2 classes)
- Bacterial spot
- Healthy

#### Potato (3 classes)
- Early blight
- Late blight
- Healthy

#### Raspberry (1 class)
- Healthy

#### Soybean (1 class)
- Healthy

#### Squash (1 class)
- Powdery mildew

#### Strawberry (2 classes)
- Leaf scorch
- Healthy

#### Tomato (10 classes)
- Bacterial spot
- Early blight
- Late blight
- Leaf Mold
- Septoria leaf spot
- Spider mites (Two-spotted spider mite)
- Target Spot
- Tomato Yellow Leaf Curl Virus
- Tomato mosaic virus
- Healthy

## 📥 Download Instructions

### Method 1: Using kagglehub (Recommended)
```python
import kagglehub

# Download dataset
path = kagglehub.dataset_download("vipoooool/new-plant-diseases-dataset")
print("Dataset downloaded to:", path)
```

### Method 2: Manual Download
1. Go to [Kaggle Dataset Page](https://www.kaggle.com/vipoooool/new-plant-diseases-dataset)
2. Click "Download" button
3. Extract to `ml-training/data/` directory

### Method 3: Kaggle API
```bash
# Install Kaggle API
pip install kaggle

# Download dataset
kaggle datasets download -d vipoooool/new-plant-diseases-dataset

# Extract
unzip new-plant-diseases-dataset.zip -d ml-training/data/
```

## 🔍 Data Preprocessing

### Applied Transformations
1. **Resize**: 256x256 → 128x128 pixels
2. **Normalization**: Pixel values scaled to [0, 1]
3. **Augmentation** (training only):
   - Random rotation (±20°)
   - Random horizontal flip
   - Random brightness adjustment
   - Random zoom (±10%)

### Code Example
```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    brightness_range=[0.8, 1.2]
)

test_datagen = ImageDataGenerator(rescale=1./255)
```

## 📊 Class Distribution

The dataset is relatively balanced across classes, with each class containing approximately 1,500-2,500 images.

## ⚠️ Important Notes

1. **Dataset Size**: ~3GB compressed, ~4GB uncompressed
2. **License**: Check Kaggle dataset page for license information
3. **Citation**: If using this dataset, please cite the original source
4. **Not Included in Repo**: Due to size, dataset is not included in the repository. Download separately.

## 🎯 Data Quality

- **High Quality**: Images captured under controlled conditions
- **Consistent**: Uniform background and lighting
- **Labeled**: Manually verified labels
- **Diverse**: Multiple angles and disease stages

## 📝 Usage in Project

The training notebook automatically downloads and prepares the dataset. No manual intervention required!

```python
# In Plant_Disease_Prediction.ipynb
import kagglehub
path = kagglehub.dataset_download("vipoooool/new-plant-diseases-dataset")
```

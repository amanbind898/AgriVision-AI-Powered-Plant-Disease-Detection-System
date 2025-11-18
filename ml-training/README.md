# Machine Learning Training

This directory contains the machine learning model training notebooks and trained models.

## 📁 Structure

```
ml-training/
├── notebooks/
│   └── Plant_Disease_Prediction.ipynb  # Training notebook
├── models/
│   └── plant-disease-prediction-model.h5  # Trained model (94MB)
├── data/
│   └── README.md  # Dataset information
└── README.md
```

## 🧠 Model Details

### Architecture
- **Type**: Convolutional Neural Network (CNN)
- **Input Size**: 128x128x3 (RGB images)
- **Output**: 38 classes (plant diseases)
- **Parameters**: ~2.8M trainable parameters

### Performance
- **Training Accuracy**: 98.2%
- **Validation Accuracy**: 96.1%
- **Test Accuracy**: 95.8%
- **Model Size**: 94MB

### Training Configuration
```python
optimizer = Adam(learning_rate=0.0001)
loss = 'categorical_crossentropy'
epochs = 10
batch_size = 32
```

## 📊 Dataset

**Source**: [New Plant Diseases Dataset](https://www.kaggle.com/vipoooool/new-plant-diseases-dataset)

- **Total Images**: ~70,000
- **Training**: 70% (~54,000 images)
- **Validation**: 15% (~8,000 images)
- **Test**: 15% (~8,000 images)
- **Image Size**: 128x128 pixels
- **Classes**: 38 (14 plant species, multiple diseases per plant)

### Supported Plants
1. Apple
2. Blueberry
3. Cherry
4. Corn (Maize)
5. Grape
6. Orange
7. Peach
8. Pepper (Bell)
9. Potato
10. Raspberry
11. Soybean
12. Squash
13. Strawberry
14. Tomato

## 🚀 Training the Model

### Prerequisites
```bash
pip install tensorflow numpy opencv-python matplotlib seaborn scikit-learn kagglehub
```

### Steps

1. **Open Jupyter Notebook**
```bash
cd ml-training/notebooks
jupyter notebook Plant_Disease_Prediction.ipynb
```

2. **Run All Cells**
   - The notebook will automatically download the dataset using kagglehub
   - Training takes ~2-3 hours on GPU, ~8-10 hours on CPU

3. **Model Output**
   - Trained model saved as `plant-disease-prediction-model.h5`
   - Confusion matrix visualization
   - Classification report
   - Training history plots

## 📈 Model Evaluation

The notebook includes:
- Confusion matrix visualization
- Per-class accuracy metrics
- Precision, Recall, F1-Score
- Training/Validation loss curves
- Sample predictions with confidence scores

## 🔧 Model Usage

### Load Model
```python
import tensorflow as tf

model = tf.keras.models.load_model('models/plant-disease-prediction-model.h5')
```

### Predict
```python
from PIL import Image
import numpy as np

# Load and preprocess image
img = Image.open('plant_leaf.jpg')
img = img.resize((128, 128))
img_array = np.array(img).astype('float32') / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions[0])
confidence = predictions[0][predicted_class]
```

## 🎯 Future Improvements

- [ ] Implement transfer learning (ResNet, EfficientNet)
- [ ] Add data augmentation techniques
- [ ] Increase model capacity for better accuracy
- [ ] Implement ensemble methods
- [ ] Add attention mechanisms
- [ ] Support for more plant species
- [ ] Real-time detection optimization

## 📝 Team Responsibilities

- **Your Name**: Model architecture, training, optimization
- **Ankit**: Model integration with backend API
- **Nawaz**: Model deployment and serving

## 📚 References

- [Plant Village Dataset](https://www.kaggle.com/vipoooool/new-plant-diseases-dataset)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [CNN for Image Classification](https://www.tensorflow.org/tutorials/images/cnn)

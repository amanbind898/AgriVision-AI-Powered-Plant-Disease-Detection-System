import tensorflow as tf
import numpy as np
from PIL import Image
from typing import Dict, List
import os

class DiseaseModel:
    """Plant Disease Detection Model"""
    
    CLASS_NAMES = [
        'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
        'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
        'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
        'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
        'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
        'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
        'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
        'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
        'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
        'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
        'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
        'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
        'Tomato___healthy'
    ]
    
    def __init__(self, model_path: str = "../ml-training/models/plant-disease-prediction-model.h5"):
        """Initialize the model"""
        self.model_path = model_path
        self.class_names = self.CLASS_NAMES
        self.model = self._load_model()
    
    def _build_model(self):
        """Build the model architecture"""
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(128, 128, 3)),
            
            # Block 1
            tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            
            # Block 2
            tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            
            # Block 3
            tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            
            # Block 4
            tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
            tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            
            # Block 5
            tf.keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu'),
            tf.keras.layers.Conv2D(512, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            
            # Classifier
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1500, activation='relu'),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(38, activation='softmax')
        ])
        return model
    
    def _load_model(self):
        """Load the trained model"""
        try:
            # Try standard loading first
            try:
                model = tf.keras.models.load_model(self.model_path, compile=False)
                print(f"[OK] Model loaded successfully from {self.model_path}")
            except (TypeError, AttributeError) as e:
                # Handle compatibility issues by rebuilding and loading weights
                print(f"[WARNING] Compatibility issue detected, loading weights only...")
                model = self._build_model()
                model.load_weights(self.model_path)
                print(f"[OK] Model weights loaded successfully from {self.model_path}")
            
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            return model
        except Exception as e:
            print(f"[ERROR] Error loading model: {e}")
            raise
    
    def preprocess_image(self, image: Image.Image) -> np.ndarray:
        """Preprocess image for model prediction"""
        # Resize to model input size
        img = image.resize((128, 128))
        
        # Convert to array and normalize
        img_array = np.array(img).astype('float32') / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    def format_disease_name(self, class_name: str) -> tuple:
        """Format disease name for better readability"""
        parts = class_name.split('___')
        plant = parts[0].replace('_', ' ').replace(',', '')
        disease = parts[1].replace('_', ' ')
        return plant, disease
    
    def predict(self, image: Image.Image) -> Dict:
        """
        Predict plant disease from image
        
        Returns:
        - predicted_class: Full class name
        - plant_name: Plant type
        - disease_name: Disease name
        - confidence: Confidence score (0-100)
        - is_healthy: Boolean
        - top_5_predictions: List of top 5 predictions with confidence
        """
        # Preprocess image
        processed_image = self.preprocess_image(image)
        
        # Make prediction
        predictions = self.model.predict(processed_image, verbose=0)
        
        # Get top prediction
        predicted_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_idx] * 100)
        predicted_class = self.class_names[predicted_idx]
        
        # Format names
        plant_name, disease_name = self.format_disease_name(predicted_class)
        is_healthy = "healthy" in disease_name.lower()
        
        # Get top 5 predictions
        top_5_idx = np.argsort(predictions[0])[-5:][::-1]
        top_5_predictions = []
        
        for idx in top_5_idx:
            class_name = self.class_names[idx]
            prob = float(predictions[0][idx] * 100)
            plant, disease = self.format_disease_name(class_name)
            
            top_5_predictions.append({
                "plant_name": plant,
                "disease_name": disease,
                "confidence": round(prob, 2)
            })
        
        return {
            "predicted_class": predicted_class,
            "plant_name": plant_name,
            "disease_name": disease_name,
            "confidence": round(confidence, 2),
            "is_healthy": is_healthy,
            "top_5_predictions": top_5_predictions
        }

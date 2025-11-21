"""
PyTorch-based Plant Disease Detection Model
"""
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
from typing import Dict

class DiseaseModel:
    """Plant Disease Detection Model using PyTorch"""
    
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
    
    def __init__(self, model_path: str = "../ml-training/models/best_model_pytorch.pth"):
        """Initialize the model"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.class_names = self.CLASS_NAMES
        self.model = self._load_model()
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def _load_model(self):
        """Load the trained PyTorch model"""
        try:
            # Load checkpoint
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Build model architecture
            model = models.efficientnet_b0(pretrained=False)
            num_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(num_features, len(self.CLASS_NAMES))
            
            # Load weights
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(self.device)
            model.eval()
            
            print(f"[OK] Model loaded from {self.model_path}")
            print(f"[OK] Using device: {self.device}")
            
            return model
        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")
            raise
    
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
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
        
        # Get predictions
        probs = probabilities.cpu().numpy()[0]
        predicted_idx = np.argmax(probs)
        confidence = float(probs[predicted_idx] * 100)
        predicted_class = self.class_names[predicted_idx]
        
        # Format names
        plant_name, disease_name = self.format_disease_name(predicted_class)
        is_healthy = "healthy" in disease_name.lower()
        
        # Get top 5 predictions
        top_5_idx = np.argsort(probs)[-5:][::-1]
        top_5_predictions = []
        
        for idx in top_5_idx:
            class_name = self.class_names[idx]
            prob = float(probs[idx] * 100)
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

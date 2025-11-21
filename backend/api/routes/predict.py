from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import io
import numpy as np
from typing import List, Dict
import json
import sys
import os

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, parent_dir)

from models.disease_model_pytorch import DiseaseModel
from utils.recommendations import get_treatment_recommendations

router = APIRouter()

# Initialize model
model = DiseaseModel()

@router.post("/")
async def predict_disease(file: UploadFile = File(...)):
    """
    Predict plant disease from uploaded image
    
    Returns:
    - predicted_class: Disease name
    - confidence: Prediction confidence (0-100)
    - plant_name: Plant type
    - disease_name: Disease name
    - is_healthy: Boolean
    - top_5_predictions: List of top 5 predictions
    - recommendations: Treatment suggestions
    """
    try:
        # Validate file type
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and process image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Convert to RGB if necessary
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Get prediction
        result = model.predict(image)
        
        # Get treatment recommendations
        recommendations = get_treatment_recommendations(
            result["plant_name"],
            result["disease_name"],
            result["is_healthy"]
        )
        
        # Add recommendations to result
        result["recommendations"] = recommendations
        
        return JSONResponse(content=result)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@router.get("/classes")
async def get_classes():
    """Get all supported plant disease classes"""
    return {
        "classes": model.class_names,
        "total": len(model.class_names)
    }

@router.get("/supported-plants")
async def get_supported_plants():
    """Get list of supported plant types"""
    plants = set()
    for class_name in model.class_names:
        plant = class_name.split("___")[0].replace("_", " ")
        plants.add(plant)
    
    return {
        "plants": sorted(list(plants)),
        "total": len(plants)
    }

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import json
import os

router = APIRouter()

# Simple file-based storage (can be replaced with database)
HISTORY_FILE = "data/prediction_history.json"

class PredictionRecord(BaseModel):
    id: str
    timestamp: str
    plant_name: str
    disease_name: str
    confidence: float
    is_healthy: bool
    image_name: Optional[str] = None

class HistoryResponse(BaseModel):
    predictions: List[PredictionRecord]
    total: int

def load_history() -> List[dict]:
    """Load prediction history from file"""
    if not os.path.exists(HISTORY_FILE):
        os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
        return []
    
    try:
        with open(HISTORY_FILE, 'r') as f:
            return json.load(f)
    except:
        return []

def save_history(history: List[dict]):
    """Save prediction history to file"""
    os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=2)

@router.post("/add")
async def add_to_history(record: PredictionRecord):
    """Add a new prediction to history"""
    try:
        history = load_history()
        history.append(record.dict())
        save_history(history)
        
        return {"message": "Added to history", "id": record.id}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save history: {str(e)}")

@router.get("/", response_model=HistoryResponse)
async def get_history(limit: int = 50):
    """Get prediction history"""
    try:
        history = load_history()
        
        # Sort by timestamp (newest first)
        history.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        
        # Limit results
        history = history[:limit]
        
        return HistoryResponse(
            predictions=history,
            total=len(history)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load history: {str(e)}")

@router.delete("/{prediction_id}")
async def delete_from_history(prediction_id: str):
    """Delete a prediction from history"""
    try:
        history = load_history()
        history = [h for h in history if h.get("id") != prediction_id]
        save_history(history)
        
        return {"message": "Deleted from history"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete: {str(e)}")

@router.delete("/")
async def clear_history():
    """Clear all history"""
    try:
        save_history([])
        return {"message": "History cleared"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear history: {str(e)}")

@router.get("/stats")
async def get_statistics():
    """Get statistics from prediction history"""
    try:
        history = load_history()
        
        if not history:
            return {
                "total_predictions": 0,
                "healthy_count": 0,
                "diseased_count": 0,
                "most_common_disease": None,
                "average_confidence": 0
            }
        
        healthy_count = sum(1 for h in history if h.get("is_healthy", False))
        diseased_count = len(history) - healthy_count
        
        # Most common disease
        diseases = [h.get("disease_name") for h in history if not h.get("is_healthy", False)]
        most_common = max(set(diseases), key=diseases.count) if diseases else None
        
        # Average confidence
        confidences = [h.get("confidence", 0) for h in history]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        return {
            "total_predictions": len(history),
            "healthy_count": healthy_count,
            "diseased_count": diseased_count,
            "most_common_disease": most_common,
            "average_confidence": round(avg_confidence, 2)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")

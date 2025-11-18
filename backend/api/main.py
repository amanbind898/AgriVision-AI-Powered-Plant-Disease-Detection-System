from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from typing import Optional
import os
import sys

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from api.routes import predict, chat, history

# Initialize FastAPI app
app = FastAPI(
    title="AgriVision API",
    description="AI-Powered Plant Disease Detection System",
    version="1.0.0"
)

# CORS middleware for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],  # Next.js dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(predict.router, prefix="/api/predict", tags=["Prediction"])
app.include_router(chat.router, prefix="/api/chat", tags=["Chatbot"])
app.include_router(history.router, prefix="/api/history", tags=["History"])

@app.get("/")
async def root():
    return {
        "message": "AgriVision API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "predict": "/api/predict/",
            "chat": "/api/chat/",
            "history": "/api/history/",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

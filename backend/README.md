# AgriVision Backend API

FastAPI backend for plant disease detection system.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Setup Environment

```bash
cp .env.example .env
# Edit .env and add your Perplexity API key
```

### 3. Run Server

```bash
python api/main.py
```

Or with uvicorn:

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Server will start at: `http://localhost:8000`

API Documentation: `http://localhost:8000/docs`

## 📡 API Endpoints

### Prediction

- `POST /api/predict/` - Upload image and get disease prediction
- `GET /api/predict/classes` - Get all supported disease classes
- `GET /api/predict/supported-plants` - Get list of supported plants

### Chatbot

- `POST /api/chat/` - Ask agricultural questions
- `POST /api/chat/explain` - Get explanation for a prediction

### History

- `POST /api/history/add` - Add prediction to history
- `GET /api/history/` - Get prediction history
- `GET /api/history/stats` - Get statistics
- `DELETE /api/history/{id}` - Delete specific prediction
- `DELETE /api/history/` - Clear all history

## 🔑 Environment Variables

```env
PERPLEXITY_API_KEY=your_api_key_here
MODEL_PATH=../ml-training/models/plant-disease-prediction-model.h5
HOST=0.0.0.0
PORT=8000
```

## 📦 Project Structure

```
backend/
├── api/
│   ├── routes/
│   │   ├── predict.py      # Disease prediction
│   │   ├── chat.py         # AI chatbot
│   │   └── history.py      # Prediction history
│   └── main.py             # FastAPI app
├── models/
│   └── disease_model.py    # ML model wrapper
├── utils/
│   ├── recommendations.py  # Treatment database
│   └── perplexity_client.py # Perplexity API client
├── data/                   # History storage
├── requirements.txt
└── .env.example
```

## 🧪 Testing

Test the API using the interactive docs at `/docs` or with curl:

```bash
# Health check
curl http://localhost:8000/health

# Predict disease
curl -X POST "http://localhost:8000/api/predict/" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@plant_image.jpg"

# Chat
curl -X POST "http://localhost:8000/api/chat/" \
  -H "Content-Type: application/json" \
  -d '{"message": "How to treat tomato blight?", "language": "en"}'
```

## 🔧 Development

### Add New Disease Treatment

Edit `utils/recommendations.py` and add to `DISEASE_TREATMENTS` dictionary:

```python
"New Disease": {
    "fungicides": ["Treatment 1", "Treatment 2"],
    "precautions": ["Step 1", "Step 2"],
    "organic_options": ["Organic option 1"]
}
```

### Modify Model

Edit `models/disease_model.py` to change model loading or preprocessing logic.


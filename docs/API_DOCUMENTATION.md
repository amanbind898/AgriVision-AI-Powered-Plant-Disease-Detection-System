# 📡 AgriVision API Documentation

Complete API reference for the AgriVision backend.

## Base URL

```
http://localhost:8000
```

## Authentication

Currently, no authentication is required. Future versions may implement API keys.

## Response Format

All responses are in JSON format.

### Success Response
```json
{
  "data": { ... },
  "status": "success"
}
```

### Error Response
```json
{
  "detail": "Error message",
  "status": "error"
}
```

## Endpoints

### 1. Health Check

Check if the API is running.

**Endpoint**: `GET /health`

**Response**:
```json
{
  "status": "healthy"
}
```

---

### 2. Predict Disease

Upload an image and get disease prediction.

**Endpoint**: `POST /api/predict/`

**Request**:
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body: Form data with `file` field

**Example (cURL)**:
```bash
curl -X POST "http://localhost:8000/api/predict/" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@plant_image.jpg"
```

**Example (JavaScript)**:
```javascript
const formData = new FormData()
formData.append('file', imageFile)

const response = await fetch('http://localhost:8000/api/predict/', {
  method: 'POST',
  body: formData
})

const result = await response.json()
```

**Response**:
```json
{
  "predicted_class": "Tomato___Late_blight",
  "plant_name": "Tomato",
  "disease_name": "Late blight",
  "confidence": 98.45,
  "is_healthy": false,
  "top_5_predictions": [
    {
      "plant_name": "Tomato",
      "disease_name": "Late blight",
      "confidence": 98.45
    },
    {
      "plant_name": "Tomato",
      "disease_name": "Early blight",
      "confidence": 1.23
    },
    ...
  ],
  "recommendations": {
    "status": "diseased",
    "message": "Disease detected in Tomato: Late blight",
    "fungicides": [
      "Chlorothalonil",
      "Mancozeb",
      "Cymoxanil"
    ],
    "precautions": [
      "Use certified disease-free seed potatoes",
      "Destroy volunteer plants",
      "Apply fungicide preventively in wet weather",
      ...
    ],
    "organic_options": [
      "Copper-based fungicides",
      "Bordeaux mixture"
    ],
    "note": "Always follow label instructions when applying any treatment"
  }
}
```

**Status Codes**:
- `200`: Success
- `400`: Invalid file format
- `500`: Prediction failed

---

### 3. Get All Classes

Get list of all supported disease classes.

**Endpoint**: `GET /api/predict/classes`

**Response**:
```json
{
  "classes": [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    ...
  ],
  "total": 38
}
```

---

### 4. Get Supported Plants

Get list of all supported plant types.

**Endpoint**: `GET /api/predict/supported-plants`

**Response**:
```json
{
  "plants": [
    "Apple",
    "Blueberry",
    "Cherry",
    "Corn (maize)",
    ...
  ],
  "total": 14
}
```

---

### 5. Chat with AI

Ask agricultural questions to the AI assistant.

**Endpoint**: `POST /api/chat/`

**Request**:
```json
{
  "message": "How to treat tomato late blight?",
  "plant_name": "Tomato",
  "disease_name": "Late blight",
  "language": "en"
}
```

**Parameters**:
- `message` (required): User's question
- `plant_name` (optional): Plant context
- `disease_name` (optional): Disease context
- `language` (required): "en" or "hi"

**Example (cURL)**:
```bash
curl -X POST "http://localhost:8000/api/chat/" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "How to prevent this disease?",
    "plant_name": "Tomato",
    "disease_name": "Late blight",
    "language": "en"
  }'
```

**Response**:
```json
{
  "response": "To prevent tomato late blight, follow these steps:\n1. Use certified disease-free seeds\n2. Apply fungicides preventively\n3. Improve air circulation\n4. Avoid overhead watering\n5. Remove infected plants immediately",
  "language": "en"
}
```

**Status Codes**:
- `200`: Success
- `500`: Chat failed

---

### 6. Explain Prediction

Get explanation for why the model made a prediction.

**Endpoint**: `POST /api/chat/explain`

**Query Parameters**:
- `plant_name` (required): Plant name
- `disease_name` (required): Disease name
- `confidence` (required): Confidence score (0-100)
- `language` (optional): "en" or "hi" (default: "en")

**Example (cURL)**:
```bash
curl -X POST "http://localhost:8000/api/chat/explain?plant_name=Tomato&disease_name=Late%20blight&confidence=98.5&language=en"
```

**Response**:
```json
{
  "explanation": "The model identified Late blight in tomato with high confidence because it detected characteristic symptoms including dark, water-soaked lesions on leaves, white fungal growth on leaf undersides, and brown spots on stems. These visual patterns are distinctive markers of Phytophthora infestans infection.",
  "language": "en"
}
```

---

### 7. Add to History

Add a prediction to history.

**Endpoint**: `POST /api/history/add`

**Request**:
```json
{
  "id": "pred_123456",
  "timestamp": "2024-11-18T10:30:00Z",
  "plant_name": "Tomato",
  "disease_name": "Late blight",
  "confidence": 98.45,
  "is_healthy": false,
  "image_name": "tomato_leaf.jpg"
}
```

**Response**:
```json
{
  "message": "Added to history",
  "id": "pred_123456"
}
```

---

### 8. Get History

Retrieve prediction history.

**Endpoint**: `GET /api/history/`

**Query Parameters**:
- `limit` (optional): Number of records to return (default: 50)

**Example**:
```bash
curl "http://localhost:8000/api/history/?limit=10"
```

**Response**:
```json
{
  "predictions": [
    {
      "id": "pred_123456",
      "timestamp": "2024-11-18T10:30:00Z",
      "plant_name": "Tomato",
      "disease_name": "Late blight",
      "confidence": 98.45,
      "is_healthy": false,
      "image_name": "tomato_leaf.jpg"
    },
    ...
  ],
  "total": 10
}
```

---

### 9. Get History Statistics

Get statistics from prediction history.

**Endpoint**: `GET /api/history/stats`

**Response**:
```json
{
  "total_predictions": 150,
  "healthy_count": 45,
  "diseased_count": 105,
  "most_common_disease": "Late blight",
  "average_confidence": 94.32
}
```

---

### 10. Delete from History

Delete a specific prediction from history.

**Endpoint**: `DELETE /api/history/{prediction_id}`

**Example**:
```bash
curl -X DELETE "http://localhost:8000/api/history/pred_123456"
```

**Response**:
```json
{
  "message": "Deleted from history"
}
```

---

### 11. Clear History

Clear all prediction history.

**Endpoint**: `DELETE /api/history/`

**Example**:
```bash
curl -X DELETE "http://localhost:8000/api/history/"
```

**Response**:
```json
{
  "message": "History cleared"
}
```

---

## Error Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Bad Request - Invalid input |
| 404 | Not Found - Resource doesn't exist |
| 500 | Internal Server Error |

## Rate Limiting

Currently, no rate limiting is implemented. Future versions may add rate limits.

## CORS

CORS is enabled for:
- `http://localhost:3000`
- `http://localhost:3001`

To add more origins, update `backend/api/main.py`:
```python
allow_origins=["http://localhost:3000", "https://yourdomain.com"]
```

## Interactive Documentation

FastAPI provides interactive API documentation:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Code Examples

### Python
```python
import requests

# Predict disease
with open('plant_image.jpg', 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:8000/api/predict/', files=files)
    result = response.json()
    print(result)

# Chat with AI
data = {
    'message': 'How to treat this disease?',
    'language': 'en'
}
response = requests.post('http://localhost:8000/api/chat/', json=data)
print(response.json())
```

### JavaScript/TypeScript
```typescript
// Predict disease
const formData = new FormData()
formData.append('file', imageFile)

const response = await fetch('http://localhost:8000/api/predict/', {
  method: 'POST',
  body: formData
})
const result = await response.json()

// Chat with AI
const chatResponse = await fetch('http://localhost:8000/api/chat/', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    message: 'How to treat this disease?',
    language: 'en'
  })
})
const chatResult = await chatResponse.json()
```

### cURL
```bash
# Predict disease
curl -X POST "http://localhost:8000/api/predict/" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@plant_image.jpg"

# Chat with AI
curl -X POST "http://localhost:8000/api/chat/" \
  -H "Content-Type: application/json" \
  -d '{"message": "How to treat this?", "language": "en"}'

# Get history
curl "http://localhost:8000/api/history/?limit=10"
```

## Webhooks

Currently not implemented. Future versions may support webhooks for:
- Prediction completion
- Batch processing
- Model updates

## Versioning

Current version: `v1.0.0`

API versioning will be implemented in future releases:
- `/api/v1/predict/`
- `/api/v2/predict/`

## Support

For API issues or questions:
- Check [Backend README](../backend/README.md)
- Review [Setup Guide](../SETUP_GUIDE.md)
- Contact: Ankit (Backend Developer)

---

**Last Updated**: November 2024

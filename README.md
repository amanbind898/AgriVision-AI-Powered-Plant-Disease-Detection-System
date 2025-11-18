# 🌱 AgriVision - AI-Powered Plant Disease Detection System

> An intelligent web-based platform leveraging deep learning to detect plant diseases with 96% accuracy, providing instant diagnosis and comprehensive treatment recommendations in multiple languages.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Next.js](https://img.shields.io/badge/Next.js-14-black)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green)
![Accuracy](https://img.shields.io/badge/Accuracy-96%25-brightgreen)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Technology Stack](#technology-stack)
- [System Architecture](#system-architecture)
- [Model Performance](#model-performance)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [API Documentation](#api-documentation)
- [Screenshots](#screenshots)

---

## 🎯 Overview

AgriVision is a comprehensive plant disease detection system developed as a BTech 4th year minor project. The system addresses the critical challenge of timely disease identification in agriculture by providing farmers with instant, accurate diagnoses through an accessible web interface.

### Problem Statement

Agricultural diseases cause billions of dollars in crop losses annually, threatening global food security. Farmers often lack access to timely expert diagnosis, leading to delayed treatment and reduced yields. Traditional laboratory testing is expensive and time-consuming.

### Solution

AgriVision provides instant, accurate disease detection accessible through any device with a camera, enabling early intervention and reducing crop losses through timely treatment. The system supports multiple languages, making it accessible to farmers across different regions.

---

## ✨ Key Features

### Core Functionality
- **🎯 High-Accuracy Detection**: 96% accuracy across 38 disease classes and 14 plant species
- **📸 Multiple Input Methods**: Image upload (drag-and-drop) and real-time camera capture
- **⚡ Fast Processing**: ~50ms inference time on CPU, ~15ms on GPU
- **📊 Detailed Results**: Confidence scores and top 5 alternative predictions

### Advanced Features
- **💊 Treatment Recommendations**: Comprehensive guidance including:
  - Chemical fungicides with application instructions
  - Organic treatment alternatives
  - Preventive measures and best practices
  - Disease-specific precautions

- **🤖 AI-Powered Chatbot**: 
  - Context-aware agricultural advice using Perplexity API
  - Real-time responses to farming queries
  - Disease management guidance
  - Explainable AI for prediction reasoning

- **🌍 Multilingual Support**:
  - Complete English and Hindi interface
  - Language-specific AI responses
  - Accessible to diverse user base

- **📱 Responsive Design**:
  - Mobile-first approach
  - Tablet and desktop optimized
  - Cross-browser compatibility

- **📈 Additional Features**:
  - Prediction history tracking
  - Statistics and analytics
  - Real-time camera detection
  - Offline-capable model inference

---

## �️Q Technology Stack

### Frontend
```
Framework:        Next.js 14 (React 18)
Language:         TypeScript
Styling:          Tailwind CSS
UI Components:    Custom components with Framer Motion
State Management: React Hooks + Context API
HTTP Client:      Axios
File Upload:      React Dropzone
Animations:       Framer Motion
Icons:            Lucide React
```

### Backend
```
Framework:        FastAPI (Python)
Server:           Uvicorn (ASGI)
ML Framework:     TensorFlow 2.15 / Keras
Image Processing: OpenCV, Pillow
AI Integration:   Perplexity API (llama-3.1-sonar)
HTTP Client:      HTTPX (async)
Environment:      Python-dotenv
```

### Machine Learning
```
Model:            Custom CNN Architecture
Training:         TensorFlow/Keras
Dataset:          PlantVillage (~70,000 images)
Classes:          38 disease classes
Plant Species:    14 types
Input Size:       128x128 RGB
Parameters:       2.8M trainable parameters
Model Size:       94MB
```

### Development Tools
```
Version Control:  Git, Git LFS
Package Manager:  npm (frontend), pip (backend)
Code Quality:     TypeScript, Python type hints
API Testing:      FastAPI interactive docs
```

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        User Interface                        │
│                    (Next.js + TypeScript)                    │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │   Home   │  │ Predict  │  │   Chat   │  │   Team   │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
└────────────────────────┬────────────────────────────────────┘
                         │ HTTP/REST API
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                      Backend API Layer                       │
│                        (FastAPI)                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │ Predict  │  │   Chat   │  │ History  │  │  Utils   │   │
│  │ Endpoint │  │ Endpoint │  │ Endpoint │  │          │   │
│  └────┬─────┘  └────┬─────┘  └──────────┘  └──────────┘   │
└───────┼─────────────┼────────────────────────────────────────┘
        │             │
        ▼             ▼
┌──────────────┐  ┌──────────────────┐
│  ML Model    │  │  Perplexity API  │
│ (TensorFlow) │  │   (AI Chatbot)   │
│              │  │                  │
│ • CNN        │  │ • Context-aware  │
│ • 96% Acc    │  │ • Multilingual   │
│ • 50ms       │  │ • Real-time      │
└──────┬───────┘  └────────┬─────────┘
       │                   │
       ▼                   ▼
┌──────────────────────────────────┐
│     Treatment Database &         │
│     Response Generation          │
└──────────────────────────────────┘
```

### Data Flow

1. **User Input**: Image upload or camera capture
2. **Preprocessing**: Resize to 128x128, normalize to [0,1]
3. **Model Inference**: CNN prediction with confidence scores
4. **Post-processing**: Top-5 predictions, disease classification
5. **Treatment Lookup**: Database query for recommendations
6. **Response**: JSON with predictions, treatments, and metadata
7. **AI Chat** (Optional): Context-aware agricultural advice

---

## 📊 Model Performance

### Accuracy Metrics

| Metric | Training | Validation | Test |
|--------|----------|------------|------|
| **Accuracy** | 98.2% | 96.1% | 95.8% |
| **Precision** | 98.1% | 96.3% | 95.9% |
| **Recall** | 98.0% | 96.0% | 95.7% |
| **F1-Score** | 98.1% | 96.1% | 95.8% |

### Performance Benchmarks

- **Inference Time**: ~50ms per image (CPU), ~15ms (GPU)
- **Model Size**: 94MB
- **Total Parameters**: 2,847,334 (2,845,286 trainable)
- **Training Time**: ~2-3 hours on GPU
- **Dataset**: 70,000+ images across 38 classes

### Supported Plants & Diseases

**14 Plant Species**:
Apple, Blueberry, Cherry, Corn (Maize), Grape, Orange, Peach, Pepper (Bell), Potato, Raspberry, Soybean, Squash, Strawberry, Tomato

**38 Disease Classes** including:
- Apple: Scab, Black rot, Cedar apple rust, Healthy
- Tomato: Bacterial spot, Early blight, Late blight, Leaf Mold, Septoria leaf spot, Spider mites, Target Spot, Yellow Leaf Curl Virus, Mosaic virus, Healthy
- Potato: Early blight, Late blight, Healthy
- Corn: Cercospora leaf spot, Common rust, Northern Leaf Blight, Healthy
- And more...

---

## 🚀 Installation

### Prerequisites

- **Python**: 3.8 or higher
- **Node.js**: 18.0 or higher
- **npm**: 9.0 or higher
- **Git**: Latest version

### Step 1: Clone Repository

```bash
git clone <repository-url>
cd agrivision-plant-disease-detection
```

### Step 2: Backend Setup

```bash
# Navigate to backend
cd backend

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Setup environment variables
cp .env.example .env
# Edit .env and add your Perplexity API key

# Run backend server
python run.py
```

Backend will be available at: `http://localhost:8000`

### Step 3: Frontend Setup

```bash
# Open new terminal
cd frontend

# Install dependencies
npm install

# Setup environment variables
cp .env.local.example .env.local

# Run development server
npm run dev
```

Frontend will be available at: `http://localhost:3000`

### Step 4: Verify Installation

1. Open `http://localhost:3000` in your browser
2. Navigate to "Detect Disease" page
3. Upload a plant leaf image
4. Verify you receive prediction results

---

## 💻 Usage

### Web Interface

#### 1. Disease Detection

**Upload Method:**
1. Go to `http://localhost:3000/predict`
2. Click "Upload Image" tab
3. Drag & drop or click to select a plant leaf image
4. Click "Analyze Image"
5. View results with confidence scores and treatment recommendations

**Camera Method:**
1. Go to `http://localhost:3000/predict`
2. Click "Use Camera" tab
3. Click "Start Camera" and allow camera access
4. Position plant leaf in frame
5. Click "Capture & Analyze"
6. View instant results

#### 2. AI Chatbot

**From Predict Page:**
- After getting a prediction, chatbot sidebar opens automatically
- Ask questions about the detected disease
- Get context-aware treatment advice

**Dedicated Chat Page:**
1. Go to `http://localhost:3000/chat`
2. Type your agricultural question
3. Get AI-powered responses
4. Switch between English and Hindi

#### 3. Language Switching

- Click the globe icon (🌐) in the navigation bar
- Toggle between English and हिंदी
- Entire interface updates instantly

### API Usage

#### Predict Disease

```bash
curl -X POST "http://localhost:8000/api/predict/" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@plant_image.jpg"
```

#### Chat with AI

```bash
curl -X POST "http://localhost:8000/api/chat/" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "How to treat tomato blight?",
    "language": "en"
  }'
```

#### Get Supported Plants

```bash
curl "http://localhost:8000/api/predict/supported-plants"
```

---

## 📁 Project Structure

```
agrivision/
├── backend/                      # FastAPI Backend
│   ├── api/
│   │   ├── routes/
│   │   │   ├── predict.py       # Disease prediction endpoint
│   │   │   ├── chat.py          # AI chatbot endpoint
│   │   │   └── history.py       # Prediction history
│   │   └── main.py              # FastAPI application
│   ├── models/
│   │   └── disease_model.py     # ML model wrapper
│   ├── utils/
│   │   ├── recommendations.py   # Treatment database
│   │   └── perplexity_client.py # AI client
│   ├── requirements.txt
│   ├── run.py                   # Server entry point
│   └── .env.example
│
├── frontend/                     # Next.js Frontend
│   ├── src/
│   │   ├── app/
│   │   │   ├── page.tsx         # Home page
│   │   │   ├── predict/         # Detection page
│   │   │   ├── chat/            # Chat page
│   │   │   └── team/            # Team page
│   │   ├── components/
│   │   │   ├── Navbar.tsx
│   │   │   ├── PredictionCard.tsx
│   │   │   ├── ChatBotSidebar.tsx
│   │   │   └── ImageUpload.tsx
│   │   ├── contexts/
│   │   │   └── LanguageContext.tsx
│   │   ├── lib/
│   │   │   ├── api.ts           # API client
│   │   │   └── translations.ts  # i18n
│   │   └── types/
│   ├── package.json
│   └── tailwind.config.ts
│
├── ml-training/                  # Machine Learning
│   ├── notebooks/
│   │   └── Plant_Disease_Prediction.ipynb
│   ├── models/
│   │   └── plant-disease-prediction-model.h5
│   └── data/
│       └── README.md            # Dataset info
│
├── docs/
│   └── API_DOCUMENTATION.md     # Complete API reference
│
├── .gitignore
├── .gitattributes
└── README.md
```

---

## 📡 API Documentation

### Base URL
```
http://localhost:8000
```

### Endpoints

#### 1. Health Check
```http
GET /health
```

#### 2. Predict Disease
```http
POST /api/predict/
Content-Type: multipart/form-data

Body: file (image file)
```

**Response:**
```json
{
  "predicted_class": "Tomato___Late_blight",
  "plant_name": "Tomato",
  "disease_name": "Late blight",
  "confidence": 98.45,
  "is_healthy": false,
  "top_5_predictions": [...],
  "recommendations": {
    "fungicides": [...],
    "precautions": [...],
    "organic_options": [...]
  }
}
```

#### 3. AI Chat
```http
POST /api/chat/
Content-Type: application/json

{
  "message": "string",
  "plant_name": "string (optional)",
  "disease_name": "string (optional)",
  "language": "en|hi"
}
```

#### 4. Get History
```http
GET /api/history/?limit=50
```

For complete API documentation, visit: `http://localhost:8000/docs`

---

## 📸 Screenshots

### Home Page
Modern landing page with feature highlights, statistics, and call-to-action.

### Disease Detection
- Dual-mode interface: Upload or Camera
- Real-time preview and analysis
- Detailed results with confidence scores
- Treatment recommendations

### AI Chatbot
- Context-aware responses
- Multilingual support
- Quick question suggestions
- Formatted responses with proper styling

### Responsive Design
- Mobile-optimized interface
- Tablet-friendly layout
- Desktop full-screen experience

---

## 🎓 Academic Context

**Project Type**: BTech 4th Year Minor Project  
**Duration**: 6 weeks (November - December 2024)  
**Institution**: [Your Institution Name]  
**Course**: Computer Science & Engineering

### Learning Outcomes

- Full-stack web development with modern frameworks
- Deep learning model training and deployment
- RESTful API design and implementation
- Multilingual application development
- Cloud deployment and DevOps practices
- Team collaboration and project management

---

## 🔮 Future Enhancements

### Short-term Goals
- [ ] User authentication and profiles
- [ ] Database integration (PostgreSQL)
- [ ] Advanced history dashboard with analytics
- [ ] Export prediction reports (PDF)
- [ ] Mobile application (React Native)

### Long-term Vision
- [ ] Support for 50+ plant species and 100+ diseases
- [ ] Pest and insect detection
- [ ] Disease severity assessment
- [ ] Offline mode with local inference
- [ ] Weather integration for disease prediction
- [ ] Community forum for farmers
- [ ] Expert consultation booking
- [ ] Soil health analysis
- [ ] Regional language expansion (Tamil, Telugu, Punjabi, Bengali)
- [ ] GPS-based field mapping and disease tracking

---

## 🐛 Known Limitations

1. **Dataset Constraints**: Trained on PlantVillage dataset with controlled conditions; may have reduced accuracy in real field conditions
2. **Plant Coverage**: Currently supports only 14 plant species
3. **Single Leaf Analysis**: Cannot detect diseases affecting entire plants or root systems
4. **Internet Dependency**: Requires internet for AI chatbot functionality
5. **Image Quality**: Performance degrades with poor lighting, blur, or extreme angles

---

## 🔧 Troubleshooting

### Backend Issues

**Port 8000 already in use:**
```bash
# Change port in backend/run.py or kill the process
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

**Model not found:**
```bash
# Verify model file exists
ls ml-training/models/plant-disease-prediction-model.h5
```

### Frontend Issues

**Port 3000 already in use:**
```bash
# Use different port
npm run dev -- -p 3001
```

**API connection error:**
- Verify backend is running: `http://localhost:8000/health`
- Check `.env.local` has correct API URL
- Check CORS settings in backend

### Camera Issues

**Black screen:**
- Allow camera permissions in browser
- Check if camera is in use by another application
- Try refreshing the page
- Use Chrome/Firefox/Safari for best compatibility

---

## 📄 License

This project is developed for educational purposes as part of BTech curriculum.

---

## 🙏 Acknowledgments

- **PlantVillage Dataset**: For providing comprehensive plant disease images
- **TensorFlow Team**: For the excellent deep learning framework
- **Next.js Team**: For the powerful React framework
- **FastAPI Team**: For the modern Python web framework
- **Perplexity AI**: For the intelligent chatbot capabilities

---

## 📞 Support

For questions, issues, or feedback regarding this project:

- Check the [API Documentation](docs/API_DOCUMENTATION.md)
- Review the [Setup Guide](SETUP_GUIDE.md)
- Open an issue in the repository

---

**Built with ❤️ for farmers and agriculture**

*AgriVision - Empowering farmers with AI-driven plant disease detection*

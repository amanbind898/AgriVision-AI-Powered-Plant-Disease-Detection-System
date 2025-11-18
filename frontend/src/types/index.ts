export interface PredictionResult {
  predicted_class: string
  plant_name: string
  disease_name: string
  confidence: number
  is_healthy: boolean
  top_5_predictions: Array<{
    plant_name: string
    disease_name: string
    confidence: number
  }>
  recommendations?: {
    status: string
    message: string
    fungicides?: string[]
    precautions?: string[]
    organic_options?: string[]
    preventive_measures?: string[]
    note?: string
  }
}

export interface ChatRequest {
  message: string
  plant_name?: string
  disease_name?: string
  language: string
}

export interface ChatResponse {
  response: string
  language: string
}

export interface PredictionRecord {
  id: string
  timestamp: string
  plant_name: string
  disease_name: string
  confidence: number
  is_healthy: boolean
  image_name?: string
}

export interface HistoryResponse {
  predictions: PredictionRecord[]
  total: number
}

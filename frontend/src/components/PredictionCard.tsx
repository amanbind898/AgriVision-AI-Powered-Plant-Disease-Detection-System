'use client'

import { AlertCircle, CheckCircle, TrendingUp } from 'lucide-react'
import type { PredictionResult } from '@/types'
import { motion } from 'framer-motion'
import { useLanguage } from '@/contexts/LanguageContext'

interface PredictionCardProps {
  prediction: PredictionResult
}

export default function PredictionCard({ prediction }: PredictionCardProps) {
  const { t } = useLanguage()
  const { plant_name, disease_name, confidence, is_healthy, top_5_predictions, recommendations } = prediction

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.3 }}
      className="space-y-6"
    >
      {/* Main Result */}
      <div
        className={`rounded-lg p-6 ${
          is_healthy
            ? 'bg-green-50 border-2 border-green-500'
            : 'bg-red-50 border-2 border-red-500'
        }`}
      >
        <div className="flex items-start gap-4">
          {is_healthy ? (
            <CheckCircle className="w-8 h-8 text-green-600 flex-shrink-0" />
          ) : (
            <AlertCircle className="w-8 h-8 text-red-600 flex-shrink-0" />
          )}
          <div className="flex-1">
            <h3 className={`text-2xl font-bold mb-2 ${is_healthy ? 'text-green-900' : 'text-red-900'}`}>
              {is_healthy ? t.predict.result.healthy : t.predict.result.diseased}
            </h3>
            <p className="text-lg mb-1">
              <span className="font-semibold">{t.predict.result.plant}:</span> {plant_name}
            </p>
            <p className="text-lg mb-1">
              <span className="font-semibold">{is_healthy ? t.predict.result.status : t.predict.result.disease}:</span> {disease_name}
            </p>
            <p className="text-lg">
              <span className="font-semibold">{t.predict.result.confidence}:</span> {confidence.toFixed(2)}%
            </p>
          </div>
        </div>
      </div>

      {/* Top 5 Predictions */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <div className="flex items-center gap-2 mb-4">
          <TrendingUp className="w-5 h-5 text-gray-600" />
          <h3 className="text-xl font-bold text-gray-900">{t.predict.result.top5}</h3>
        </div>
        <div className="space-y-3">
          {top_5_predictions.map((pred, index) => (
            <div key={index} className="space-y-1">
              <div className="flex justify-between text-sm">
                <span className="font-medium text-gray-700">
                  {pred.plant_name} - {pred.disease_name}
                </span>
                <span className="text-gray-600">{pred.confidence.toFixed(2)}%</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div
                  className="bg-green-600 h-2 rounded-full transition-all"
                  style={{ width: `${pred.confidence}%` }}
                />
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Recommendations */}
      {recommendations && (
        <div className="bg-white rounded-lg shadow-md p-6">
          <h3 className="text-xl font-bold text-gray-900 mb-4">
            {is_healthy ? t.predict.result.preventive : t.predict.result.recommendations}
          </h3>

          {is_healthy ? (
            <div className="space-y-2">
              <p className="text-green-700 font-medium mb-3">{recommendations.message}</p>
              <ul className="space-y-2">
                {recommendations.preventive_measures?.map((measure: string, index: number) => (
                  <li key={index} className="flex items-start gap-2 text-gray-700">
                    <span className="text-green-500 mt-1">✓</span>
                    <span>{measure}</span>
                  </li>
                ))}
              </ul>
            </div>
          ) : (
            <div className="space-y-4">
              <p className="text-red-700 font-medium">{recommendations.message}</p>

              {/* Fungicides */}
              {recommendations.fungicides && recommendations.fungicides.length > 0 && (
                <div>
                  <h4 className="font-semibold text-gray-900 mb-2">{t.predict.result.fungicides}:</h4>
                  <ul className="space-y-1">
                    {recommendations.fungicides.map((fungicide: string, index: number) => (
                      <li key={index} className="text-gray-700 ml-4">
                        • {fungicide}
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {/* Precautions */}
              {recommendations.precautions && recommendations.precautions.length > 0 && (
                <div>
                  <h4 className="font-semibold text-gray-900 mb-2">{t.predict.result.precautions}:</h4>
                  <ul className="space-y-1">
                    {recommendations.precautions.map((precaution: string, index: number) => (
                      <li key={index} className="text-gray-700 ml-4">
                        • {precaution}
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {/* Organic Options */}
              {recommendations.organic_options && recommendations.organic_options.length > 0 && (
                <div>
                  <h4 className="font-semibold text-gray-900 mb-2">{t.predict.result.organic}:</h4>
                  <ul className="space-y-1">
                    {recommendations.organic_options.map((option: string, index: number) => (
                      <li key={index} className="text-gray-700 ml-4">
                        • {option}
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {recommendations.note && (
                <p className="text-sm text-gray-600 italic mt-4">
                  Note: {recommendations.note}
                </p>
              )}
            </div>
          )}
        </div>
      )}
    </motion.div>
  )
}

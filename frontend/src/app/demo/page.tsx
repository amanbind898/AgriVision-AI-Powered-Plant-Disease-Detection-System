'use client'

import { useState, useRef } from 'react'
import { Camera, X, Loader2 } from 'lucide-react'
import PredictionCard from '@/components/PredictionCard'
import { predictDisease } from '@/lib/api'
import type { PredictionResult } from '@/types'

export default function DemoPage() {
  const [stream, setStream] = useState<MediaStream | null>(null)
  const [prediction, setPrediction] = useState<PredictionResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)

  const startCamera = async () => {
    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'environment' }
      })
      setStream(mediaStream)
      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream
      }
      setError(null)
    } catch (err) {
      setError('Failed to access camera. Please check permissions.')
    }
  }

  const stopCamera = () => {
    if (stream) {
      stream.getTracks().forEach(track => track.stop())
      setStream(null)
    }
  }

  const captureImage = async () => {
    if (!videoRef.current || !canvasRef.current) return

    const video = videoRef.current
    const canvas = canvasRef.current
    const context = canvas.getContext('2d')

    if (!context) return

    // Set canvas size to video size
    canvas.width = video.videoWidth
    canvas.height = video.videoHeight

    // Draw video frame to canvas
    context.drawImage(video, 0, 0, canvas.width, canvas.height)

    // Convert canvas to blob
    canvas.toBlob(async (blob) => {
      if (!blob) return

      setLoading(true)
      setError(null)

      try {
        const file = new File([blob], 'capture.jpg', { type: 'image/jpeg' })
        const result = await predictDisease(file)
        setPrediction(result)
        stopCamera()
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Prediction failed')
      } finally {
        setLoading(false)
      }
    }, 'image/jpeg')
  }

  const reset = () => {
    setPrediction(null)
    setError(null)
    startCamera()
  }

  return (
    <div className="min-h-screen bg-gray-50 py-12 px-4">
      <div className="container mx-auto max-w-6xl">
        <div className="text-center mb-12">
          <h1 className="text-4xl font-bold text-gray-900 mb-4">
            Live Camera Detection
          </h1>
          <p className="text-lg text-gray-600">
            Use your camera to detect plant diseases in real-time
          </p>
        </div>

        <div className="grid lg:grid-cols-2 gap-8">
          {/* Camera Section */}
          <div className="bg-white rounded-lg shadow-md p-6">
            <h2 className="text-2xl font-bold text-gray-900 mb-4">Camera</h2>

            {!stream && !prediction && (
              <div className="text-center py-12">
                <Camera className="w-16 h-16 mx-auto mb-4 text-gray-400" />
                <p className="text-gray-600 mb-6">
                  Start your camera to capture plant images
                </p>
                <button
                  onClick={startCamera}
                  className="bg-green-600 text-white px-6 py-3 rounded-lg font-semibold hover:bg-green-700 transition-colors"
                >
                  Start Camera
                </button>
              </div>
            )}

            {stream && (
              <div className="space-y-4">
                <div className="relative rounded-lg overflow-hidden bg-black">
                  <video
                    ref={videoRef}
                    autoPlay
                    playsInline
                    className="w-full h-auto"
                  />
                  <button
                    onClick={stopCamera}
                    className="absolute top-2 right-2 bg-red-500 text-white p-2 rounded-full hover:bg-red-600 transition-colors"
                  >
                    <X className="w-5 h-5" />
                  </button>
                </div>

                <button
                  onClick={captureImage}
                  disabled={loading}
                  className={`w-full py-4 rounded-lg font-semibold text-lg transition-colors ${
                    loading
                      ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                      : 'bg-green-600 text-white hover:bg-green-700'
                  }`}
                >
                  {loading ? (
                    <span className="flex items-center justify-center gap-2">
                      <Loader2 className="w-5 h-5 animate-spin" />
                      Analyzing...
                    </span>
                  ) : (
                    'Capture & Analyze'
                  )}
                </button>
              </div>
            )}

            {prediction && (
              <div className="text-center py-8">
                <p className="text-gray-600 mb-4">
                  Image captured and analyzed successfully!
                </p>
                <button
                  onClick={reset}
                  className="bg-green-600 text-white px-6 py-3 rounded-lg font-semibold hover:bg-green-700 transition-colors"
                >
                  Capture Another
                </button>
              </div>
            )}

            {error && (
              <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-lg">
                <p className="text-red-600">{error}</p>
              </div>
            )}

            {/* Hidden canvas for capture */}
            <canvas ref={canvasRef} className="hidden" />
          </div>

          {/* Results Section */}
          <div>
            {prediction && <PredictionCard prediction={prediction} />}

            {!prediction && !loading && (
              <div className="bg-white rounded-lg shadow-md p-8 text-center">
                <p className="text-gray-500">
                  Capture an image to see prediction results
                </p>
              </div>
            )}
          </div>
        </div>

        {/* Instructions */}
        <div className="mt-12 bg-blue-50 rounded-lg p-6">
          <h3 className="text-xl font-bold text-gray-900 mb-4">Tips for Best Results</h3>
          <ul className="space-y-2 text-gray-700">
            <li className="flex items-start gap-2">
              <span className="text-blue-600 mt-1">•</span>
              <span>Ensure good lighting conditions</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-blue-600 mt-1">•</span>
              <span>Focus on a single leaf with clear symptoms</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-blue-600 mt-1">•</span>
              <span>Hold the camera steady and close to the leaf</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-blue-600 mt-1">•</span>
              <span>Avoid shadows and reflections</span>
            </li>
          </ul>
        </div>
      </div>
    </div>
  )
}

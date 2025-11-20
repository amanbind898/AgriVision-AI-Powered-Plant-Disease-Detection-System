'use client'

import React, { useState, useRef, useCallback, useEffect } from 'react'
import { Upload, Camera, X, Loader2, MessageSquare, ChevronRight, ChevronLeft } from 'lucide-react'
import { useDropzone } from 'react-dropzone'
import Image from 'next/image'
import PredictionCard from '@/components/PredictionCard'
import ChatBotSidebar from '@/components/ChatBotSidebar'
import { predictDisease } from '@/lib/api'
import type { PredictionResult } from '@/types'
import { useLanguage } from '@/contexts/LanguageContext'

export default function PredictPage() {
  const { t } = useLanguage()
  const [activeTab, setActiveTab] = useState<'upload' | 'camera'>('upload')
  const [prediction, setPrediction] = useState<PredictionResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [preview, setPreview] = useState<string | null>(null)
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [stream, setStream] = useState<MediaStream | null>(null)
  const [chatOpen, setChatOpen] = useState(false)
  
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)

  // Effect to handle stream changes
  useEffect(() => {
    if (stream && videoRef.current) {
      console.log('🔄 Stream changed, updating video element')
      videoRef.current.srcObject = stream
      videoRef.current.muted = true
      videoRef.current.play().catch(err => console.error('Play error:', err))
    }
  }, [stream])

  // Image Upload
  const onDrop = useCallback((acceptedFiles: File[]) => {
    const file = acceptedFiles[0]
    if (file) {
      setSelectedFile(file)
      const reader = new FileReader()
      reader.onloadend = () => {
        setPreview(reader.result as string)
      }
      reader.readAsDataURL(file)
      setActiveTab('upload')
    }
  }, [])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpeg', '.jpg', '.png']
    },
    multiple: false,
    disabled: loading
  })

  // Camera
  const startCamera = async () => {
    try {
      setError(null)
      console.log('🎥 Starting camera...')
      
      // Check if mediaDevices is supported
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        setError('Camera not supported in this browser. Please use Chrome, Firefox, or Safari.')
        return
      }

      console.log('📹 Requesting camera access...')
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: { 
          facingMode: 'user', // Try 'user' first (front camera)
          width: { ideal: 1280 },
          height: { ideal: 720 }
        }
      })
      
      console.log('✅ Camera access granted')
      console.log('Stream tracks:', mediaStream.getTracks())
      
      setStream(mediaStream)
      
      // Use setTimeout to ensure video element is rendered
      setTimeout(() => {
        if (videoRef.current) {
          console.log('📺 Setting video source...')
          videoRef.current.srcObject = mediaStream
          videoRef.current.muted = true
          
          videoRef.current.onloadedmetadata = () => {
            console.log('📹 Video metadata loaded')
            videoRef.current?.play()
              .then(() => console.log('▶️ Video playing'))
              .catch(err => {
                console.error('❌ Play error:', err)
                setError('Failed to start video. Please refresh and try again.')
              })
          }
        } else {
          console.error('❌ Video ref is null')
        }
      }, 100)
      
    } catch (err: any) {
      console.error('❌ Camera error:', err)
      let errorMessage = 'Failed to access camera. '
      
      if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
        errorMessage += 'Please allow camera access in your browser settings.'
      } else if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
        errorMessage += 'No camera found on this device.'
      } else if (err.name === 'NotReadableError' || err.name === 'TrackStartError') {
        errorMessage += 'Camera is already in use by another application.'
      } else {
        errorMessage += 'Please check your camera permissions and try again.'
      }
      
      setError(errorMessage)
    }
  }

  const stopCamera = () => {
    if (stream) {
      stream.getTracks().forEach(track => track.stop())
      setStream(null)
    }
  }

  const captureImage = async () => {
    if (!videoRef.current || !canvasRef.current) {
      setError('Camera not ready. Please try again.')
      return
    }

    const video = videoRef.current
    const canvas = canvasRef.current
    
    // Check if video is playing
    if (video.readyState !== video.HAVE_ENOUGH_DATA) {
      setError('Video not ready. Please wait a moment and try again.')
      return
    }
    
    const context = canvas.getContext('2d')
    if (!context) {
      setError('Canvas not supported.')
      return
    }

    try {
      // Set canvas size to video size
      canvas.width = video.videoWidth || 640
      canvas.height = video.videoHeight || 480
      
      // Draw video frame to canvas
      context.drawImage(video, 0, 0, canvas.width, canvas.height)

      // Convert to blob
      canvas.toBlob(async (blob) => {
        if (!blob) {
          setError('Failed to capture image. Please try again.')
          return
        }

        const file = new File([blob], 'capture.jpg', { type: 'image/jpeg' })
        setSelectedFile(file)
        setPreview(canvas.toDataURL())
        stopCamera()
        await handleAnalyze(file)
      }, 'image/jpeg', 0.95)
    } catch (err) {
      console.error('Capture error:', err)
      setError('Failed to capture image. Please try again.')
    }
  }

  // Analyze
  const handleAnalyze = async (file?: File) => {
    const fileToAnalyze = file || selectedFile
    if (!fileToAnalyze) return

    setLoading(true)
    setError(null)
    setPrediction(null)

    try {
      const result = await predictDisease(fileToAnalyze)
      setPrediction(result)
      setChatOpen(true) // Auto-open chat after prediction
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Prediction failed')
    } finally {
      setLoading(false)
    }
  }

  const handleClear = () => {
    setPreview(null)
    setSelectedFile(null)
    setPrediction(null)
    setError(null)
    stopCamera()
  }

  return (
    <div className="min-h-screen bg-gray-50 py-8 px-4">
      <div className="container mx-auto max-w-7xl">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-4">
            {t.predict.title}
          </h1>
          <p className="text-lg text-gray-600">
            {t.predict.subtitle}
          </p>
        </div>

        <div className="flex gap-6">
          {/* Main Content */}
          <div className={`flex-1 transition-all duration-300 ${chatOpen && prediction ? 'lg:mr-0' : ''}`}>
            <div className="grid lg:grid-cols-2 gap-6">
              {/* Left Column - Upload/Camera */}
              <div className="bg-white rounded-lg shadow-md p-6">
                {/* Tabs */}
                <div className="flex gap-2 mb-6 border-b">
                  <button
                    onClick={() => {
                      setActiveTab('upload')
                      stopCamera()
                    }}
                    className={`flex items-center gap-2 px-4 py-2 font-medium transition-colors border-b-2 ${
                      activeTab === 'upload'
                        ? 'border-green-600 text-green-600'
                        : 'border-transparent text-gray-600 hover:text-green-600'
                    }`}
                  >
                    <Upload className="w-4 h-4" />
                    {t.predict.uploadTab}
                  </button>
                  <button
                    onClick={() => {
                      setActiveTab('camera')
                      handleClear()
                    }}
                    className={`flex items-center gap-2 px-4 py-2 font-medium transition-colors border-b-2 ${
                      activeTab === 'camera'
                        ? 'border-green-600 text-green-600'
                        : 'border-transparent text-gray-600 hover:text-green-600'
                    }`}
                  >
                    <Camera className="w-4 h-4" />
                    {t.predict.cameraTab}
                  </button>
                </div>

                {/* Upload Tab */}
                {activeTab === 'upload' && (
                  <>
                    {!preview ? (
                      <div
                        {...getRootProps()}
                        className={`border-2 border-dashed rounded-lg p-12 text-center cursor-pointer transition-colors ${
                          isDragActive
                            ? 'border-green-500 bg-green-50'
                            : 'border-gray-300 hover:border-green-400 hover:bg-gray-50'
                        } ${loading ? 'opacity-50 cursor-not-allowed' : ''}`}
                      >
                        <input {...getInputProps()} />
                        <Upload className="w-16 h-16 mx-auto mb-4 text-gray-400" />
                        {isDragActive ? (
                          <p className="text-lg text-green-600">{t.predict.dragDrop}</p>
                        ) : (
                          <>
                            <p className="text-lg text-gray-700 mb-2">
                              {t.predict.dragDrop}
                            </p>
                            <p className="text-sm text-gray-500">
                              {t.predict.supports}
                            </p>
                          </>
                        )}
                      </div>
                    ) : (
                      <div className="space-y-4">
                        <div className="relative rounded-lg overflow-hidden border-2 border-gray-200">
                          <img
                            src={preview || ''}
                            alt="Preview"
                            className="w-full h-auto max-h-96 object-contain"
                          />
                          {!loading && (
                            <button
                              onClick={handleClear}
                              className="absolute top-2 right-2 bg-red-500 text-black p-2 rounded-full hover:bg-red-600 transition-colors"
                            >
                              <X className="w-5 h-5" />
                            </button>
                          )}
                        </div>

                        <button
                          onClick={() => handleAnalyze()}
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
                              {t.predict.analyzing}
                            </span>
                          ) : (
                            t.predict.analyze
                          )}
                        </button>

                        {!loading && (
                          <button
                            onClick={handleClear}
                            className="w-full py-3 rounded-lg font-medium text-gray-600 border-2 border-gray-300 hover:bg-gray-50 transition-colors"
                          >
                            {t.predict.uploadAnother}
                          </button>
                        )}
                      </div>
                    )}
                  </>
                )}

                {/* Camera Tab */}
                {activeTab === 'camera' && (
                  <>
                    {!stream && !preview ? (
                      <div className="text-center py-12">
                        <Camera className="w-16 h-16 mx-auto mb-4 text-gray-400" />
                        <p className="text-gray-600 mb-6">
                          {t.predict.startCamera}
                        </p>
                        <button
                          onClick={startCamera}
                          className="bg-green-600 text-white px-6 py-3 rounded-lg font-semibold hover:bg-green-700 transition-colors"
                        >
                          {t.predict.startCamera}
                        </button>
                      </div>
                    ) : stream ? (
                      <div className="space-y-4">
                        <div className="relative rounded-lg overflow-hidden bg-black">
                          <video
                            ref={videoRef}
                            autoPlay
                            playsInline
                            muted
                            className="w-full h-auto min-h-[300px]"
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
                              {t.predict.analyzing}
                            </span>
                          ) : (
                            t.predict.capture
                          )}
                        </button>
                      </div>
                    ) : (
                      <div className="space-y-4">
                        <div className="relative rounded-lg overflow-hidden border-2 border-gray-200">
                          <img
                            src={preview || ''}
                            alt="Captured"
                            className="w-full h-auto max-h-96 object-contain"
                          />
                        </div>
                        <button
                          onClick={handleClear}
                          className="w-full py-3 rounded-lg font-medium text-gray-600 border-2 border-gray-300 hover:bg-gray-50 transition-colors"
                        >
                          {t.predict.startCamera}
                        </button>
                      </div>
                    )}
                  </>
                )}

                {error && (
                  <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-lg">
                    <p className="text-red-600">{error}</p>
                  </div>
                )}

                <canvas ref={canvasRef} className="hidden" />
              </div>

              {/* Right Column - Results */}
              <div>
                {prediction ? (
                  <PredictionCard prediction={prediction} />
                ) : (
                  <div className="bg-white rounded-lg shadow-md p-8 text-center">
                    <p className="text-gray-500">
                      {t.predict.uploadToSee}
                    </p>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Chatbot Sidebar */}
          {prediction && (
            <>
              {/* Toggle Button */}
              <button
                onClick={() => setChatOpen(!chatOpen)}
                className="fixed right-0 top-1/2 -translate-y-1/2 bg-green-600 text-white p-3 rounded-l-lg shadow-lg hover:bg-green-700 transition-all z-40 lg:hidden"
              >
                {chatOpen ? <ChevronRight className="w-5 h-5" /> : <MessageSquare className="w-5 h-5" />}
              </button>

              {/* Sidebar */}
              <div
                className={`fixed lg:relative top-0 right-0 h-screen lg:h-auto bg-white shadow-2xl lg:shadow-md rounded-l-lg lg:rounded-lg transition-all duration-300 z-30 ${
                  chatOpen ? 'translate-x-0 w-96' : 'translate-x-full lg:translate-x-0 lg:w-0 lg:opacity-0'
                }`}
              >
                {chatOpen && (
                  <ChatBotSidebar
                    plantName={prediction.plant_name}
                    diseaseName={prediction.disease_name}
                    onClose={() => setChatOpen(false)}
                  />
                )}
              </div>

              {/* Overlay for mobile */}
              {chatOpen && (
                <div
                  className="fixed inset-0 bg-black bg-opacity-50 z-20 lg:hidden"
                  onClick={() => setChatOpen(false)}
                />
              )}
            </>
          )}
        </div>
      </div>
    </div>
  )
}

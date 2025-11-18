'use client'

import { useCallback, useState } from 'react'
import { useDropzone } from 'react-dropzone'
import { Upload, X, Loader2 } from 'lucide-react'
import Image from 'next/image'

interface ImageUploadProps {
  onImageUpload: (file: File) => void
  loading?: boolean
}

export default function ImageUpload({ onImageUpload, loading }: ImageUploadProps) {
  const [preview, setPreview] = useState<string | null>(null)
  const [selectedFile, setSelectedFile] = useState<File | null>(null)

  const onDrop = useCallback((acceptedFiles: File[]) => {
    const file = acceptedFiles[0]
    if (file) {
      setSelectedFile(file)
      const reader = new FileReader()
      reader.onloadend = () => {
        setPreview(reader.result as string)
      }
      reader.readAsDataURL(file)
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

  const handleAnalyze = () => {
    if (selectedFile) {
      onImageUpload(selectedFile)
    }
  }

  const handleClear = () => {
    setPreview(null)
    setSelectedFile(null)
  }

  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <h2 className="text-2xl font-bold text-gray-900 mb-4">Upload Image</h2>

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
            <p className="text-lg text-green-600">Drop the image here...</p>
          ) : (
            <>
              <p className="text-lg text-gray-700 mb-2">
                Drag & drop an image here, or click to select
              </p>
              <p className="text-sm text-gray-500">
                Supports: JPG, JPEG, PNG
              </p>
            </>
          )}
        </div>
      ) : (
        <div className="space-y-4">
          {/* Image Preview */}
          <div className="relative rounded-lg overflow-hidden border-2 border-gray-200">
            <img
              src={preview}
              alt="Preview"
              className="w-full h-auto max-h-96 object-contain"
            />
            {!loading && (
              <button
                onClick={handleClear}
                className="absolute top-2 right-2 bg-red-500 text-white p-2 rounded-full hover:bg-red-600 transition-colors"
              >
                <X className="w-5 h-5" />
              </button>
            )}
          </div>

          {/* Analyze Button */}
          <button
            onClick={handleAnalyze}
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
              'Analyze Image'
            )}
          </button>

          {/* Upload Another */}
          {!loading && (
            <button
              onClick={handleClear}
              className="w-full py-3 rounded-lg font-medium text-gray-600 border-2 border-gray-300 hover:bg-gray-50 transition-colors"
            >
              Upload Another Image
            </button>
          )}
        </div>
      )}
    </div>
  )
}

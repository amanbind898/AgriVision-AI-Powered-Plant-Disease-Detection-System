'use client'

import { useState } from 'react'
import { MessageSquare, Send, Loader2, X } from 'lucide-react'
import { chatWithAI } from '@/lib/api'
import { useLanguage } from '@/contexts/LanguageContext'

interface ChatBotSidebarProps {
  plantName?: string
  diseaseName?: string
  onClose?: () => void
}

interface Message {
  role: 'user' | 'assistant'
  content: string
}

// Format message with proper HTML
const formatMessage = (text: string) => {
  return text
    // Bold text with **
    .replace(/\*\*(.*?)\*\*/g, '<strong class="font-bold text-gray-900">$1</strong>')
    // Numbered lists
    .replace(/^(\d+)\.\s+\*\*(.*?)\*\*/gm, '<div class="mt-3 mb-2"><strong class="font-bold text-gray-900">$1. $2</strong></div>')
    .replace(/^(\d+)\.\s+(.*?)$/gm, '<div class="mt-2"><span class="font-semibold text-gray-800">$1.</span> $2</div>')
    // Bullet points
    .replace(/^-\s+(.*?)$/gm, '<div class="ml-4 mt-1">• $1</div>')
    // Headings with ###
    .replace(/^###\s+(.*?)$/gm, '<h3 class="text-base font-bold text-gray-900 mt-4 mb-2">$1</h3>')
    // Line breaks
    .replace(/\n\n/g, '<br/><br/>')
    .replace(/\n/g, '<br/>')
}

export default function ChatBotSidebar({ plantName, diseaseName, onClose }: ChatBotSidebarProps) {
  const { language, t } = useLanguage()
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)

  const handleSend = async () => {
    if (!input.trim() || loading) return

    const userMessage = input.trim()
    setInput('')
    setMessages(prev => [...prev, { role: 'user', content: userMessage }])
    setLoading(true)

    try {
      const response = await chatWithAI({
        message: userMessage,
        plant_name: plantName,
        disease_name: diseaseName,
        language
      })

      setMessages(prev => [...prev, { role: 'assistant', content: response.response }])
    } catch (error) {
      setMessages(prev => [
        ...prev,
        {
          role: 'assistant',
          content: language === 'hi' 
            ? 'क्षमा करें, एक त्रुटि हुई। कृपया पुनः प्रयास करें।'
            : 'Sorry, I encountered an error. Please try again.'
        }
      ])
    } finally {
      setLoading(false)
    }
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  const quickQuestions = language === 'hi'
    ? [
        'इस बीमारी को कैसे रोकें?',
        'सर्वोत्तम जैविक उपचार?',
        'कवकनाशी कब लगाएं?'
      ]
    : [
        'How to prevent this disease?',
        'Best organic treatment?',
        'When to apply fungicide?'
      ]

  return (
    <div className="flex flex-col h-full w-96">
      {/* Header */}
      <div className="bg-green-600 text-white p-4 flex items-center justify-between rounded-t-lg">
        <div className="flex items-center gap-2">
          <MessageSquare className="w-5 h-5" />
          <h3 className="font-semibold">{t.chat.title}</h3>
        </div>
        {onClose && (
          <button
            onClick={onClose}
            className="lg:hidden text-white hover:bg-green-700 p-1 rounded"
          >
            <X className="w-5 h-5" />
          </button>
        )}
      </div>

      {/* Context Info */}
      {plantName && diseaseName && (
        <div className="bg-green-50 p-3 border-b text-sm">
          <p className="text-gray-600">
            <span className="font-semibold">{t.chat.context}:</span> {plantName} - {diseaseName}
          </p>
        </div>
      )}

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4 bg-gray-50">
        {messages.length === 0 && (
          <div className="text-center py-8 text-gray-500">
            <MessageSquare className="w-12 h-12 mx-auto mb-3 text-gray-400" />
            <p className="mb-4">{t.chat.askAnything}</p>
            
            {/* Quick Questions */}
            <div className="space-y-2">
              <p className="text-sm font-medium text-gray-600">
                {t.chat.quickQuestions}
              </p>
              <div className="flex flex-col gap-2">
                {quickQuestions.map((question, index) => (
                  <button
                    key={index}
                    onClick={() => setInput(question)}
                    className="px-3 py-2 text-sm bg-white hover:bg-gray-100 rounded-lg transition-colors text-left"
                  >
                    {question}
                  </button>
                ))}
              </div>
            </div>
          </div>
        )}

        {messages.map((message, index) => (
          <div
            key={index}
            className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            <div
              className={`max-w-[85%] rounded-lg p-4 ${
                message.role === 'user'
                  ? 'bg-green-600 text-white'
                  : 'bg-white text-gray-900 shadow-sm border border-gray-200'
              }`}
            >
              <div 
                className="prose prose-sm max-w-none"
                dangerouslySetInnerHTML={{ 
                  __html: formatMessage(message.content) 
                }}
              />
            </div>
          </div>
        ))}
        
        {loading && (
          <div className="flex justify-start">
            <div className="bg-white rounded-lg p-3 shadow-sm">
              <Loader2 className="w-5 h-5 animate-spin text-gray-600" />
            </div>
          </div>
        )}
      </div>

      {/* Input */}
      <div className="p-4 bg-white border-t rounded-b-lg">
        <div className="flex gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder={t.chat.typeQuestion}
            className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-green-500 text-sm"
            disabled={loading}
          />
          <button
            onClick={handleSend}
            disabled={loading || !input.trim()}
            className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors"
          >
            <Send className="w-5 h-5" />
          </button>
        </div>
      </div>
    </div>
  )
}

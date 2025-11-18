'use client'

import { useState } from 'react'
import { MessageSquare, Send, Loader2, Globe } from 'lucide-react'
import { chatWithAI } from '@/lib/api'

interface ChatBotProps {
  plantName?: string
  diseaseName?: string
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

export default function ChatBot({ plantName, diseaseName }: ChatBotProps) {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [language, setLanguage] = useState<'en' | 'hi'>('en')
  const [isOpen, setIsOpen] = useState(false)

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
          content: 'Sorry, I encountered an error. Please try again.'
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

  return (
    <div className="bg-white rounded-lg shadow-md overflow-hidden">
      {/* Header */}
      <div
        className="bg-green-600 text-white p-4 flex items-center justify-between cursor-pointer"
        onClick={() => setIsOpen(!isOpen)}
      >
        <div className="flex items-center gap-2">
          <MessageSquare className="w-5 h-5" />
          <h3 className="font-semibold">AI Agricultural Assistant</h3>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={(e) => {
              e.stopPropagation()
              setLanguage(language === 'en' ? 'hi' : 'en')
            }}
            className="flex items-center gap-1 px-3 py-1 bg-white/20 rounded-full text-sm hover:bg-white/30 transition-colors text-white"
          >
            <Globe className="w-4 h-4 text-white" />
            <span className="text-white font-medium">{language === 'en' ? 'English' : 'हिंदी'}</span>
          </button>
          <span className="text-2xl">{isOpen ? '−' : '+'}</span>
        </div>
      </div>

      {/* Chat Area */}
      {isOpen && (
        <div className="p-4">
          {/* Welcome Message */}
          {messages.length === 0 && (
            <div className="text-center py-8 text-gray-500">
              <MessageSquare className="w-12 h-12 mx-auto mb-3 text-gray-400" />
              <p className="mb-2">
                {language === 'en'
                  ? 'Ask me anything about plant diseases and treatments!'
                  : 'पौधों की बीमारियों और उपचार के बारे में कुछ भी पूछें!'}
              </p>
              {plantName && diseaseName && (
                <p className="text-sm text-gray-400">
                  {language === 'en'
                    ? `Context: ${plantName} - ${diseaseName}`
                    : `संदर्भ: ${plantName} - ${diseaseName}`}
                </p>
              )}
            </div>
          )}

          {/* Messages */}
          <div className="space-y-4 mb-4 max-h-96 overflow-y-auto">
            {messages.map((message, index) => (
              <div
                key={index}
                className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                <div
                  className={`max-w-[80%] rounded-lg p-4 ${
                    message.role === 'user'
                      ? 'bg-green-600 text-white'
                      : 'bg-gray-50 text-gray-900 border border-gray-200 shadow-sm'
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
                <div className="bg-gray-100 rounded-lg p-3">
                  <Loader2 className="w-5 h-5 animate-spin text-gray-600" />
                </div>
              </div>
            )}
          </div>

          {/* Input */}
          <div className="flex gap-2">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder={
                language === 'en'
                  ? 'Type your question...'
                  : 'अपना सवाल लिखें...'
              }
              className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-green-500"
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

          {/* Quick Questions */}
          {messages.length === 0 && (
            <div className="mt-4 space-y-2">
              <p className="text-sm text-gray-600 font-medium">
                {language === 'en' ? 'Quick questions:' : 'त्वरित प्रश्न:'}
              </p>
              <div className="flex flex-wrap gap-2">
                {(language === 'en'
                  ? [
                      'How to prevent this disease?',
                      'Best organic treatment?',
                      'When to apply fungicide?'
                    ]
                  : [
                      'इस बीमारी को कैसे रोकें?',
                      'सर्वोत्तम जैविक उपचार?',
                      'कवकनाशी कब लगाएं?'
                    ]
                ).map((question, index) => (
                  <button
                    key={index}
                    onClick={() => setInput(question)}
                    className="px-3 py-1 text-sm bg-gray-100 hover:bg-gray-200 rounded-full transition-colors"
                  >
                    {question}
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

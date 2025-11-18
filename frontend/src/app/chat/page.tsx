'use client'

import { useState } from 'react'
import { MessageSquare, Send, Loader2, Leaf, CheckCircle } from 'lucide-react'
import { chatWithAI } from '@/lib/api'
import { useLanguage } from '@/contexts/LanguageContext'
import { motion } from 'framer-motion'

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
    .replace(/^###\s+(.*?)$/gm, '<h3 class="text-lg font-bold text-gray-900 mt-4 mb-2">$1</h3>')
    // Line breaks
    .replace(/\n\n/g, '<br/><br/>')
    .replace(/\n/g, '<br/>')
}

export default function ChatPage() {
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
        'टमाटर की बीमारियों के बारे में बताएं',
        'जैविक कीटनाशक कैसे बनाएं?',
        'फसल चक्र क्या है?',
        'मिट्टी की उर्वरता कैसे बढ़ाएं?'
      ]
    : [
        'Tell me about tomato diseases',
        'How to make organic pesticides?',
        'What is crop rotation?',
        'How to improve soil fertility?'
      ]

  return (
    <div className="min-h-screen bg-gradient-to-br from-green-50 to-emerald-100 py-8 px-4">
      <div className="container mx-auto max-w-5xl">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-8"
        >
          <div className="flex justify-center mb-4">
            <div className="bg-green-600 p-4 rounded-full">
              <MessageSquare className="w-8 h-8 text-white" />
            </div>
          </div>
          <h1 className="text-4xl font-bold text-gray-900 mb-4">
            {t.chat.chatPage.title}
          </h1>
          <p className="text-lg text-gray-600">
            {t.chat.chatPage.subtitle}
          </p>
        </motion.div>

        {/* Chat Container */}
        <div className="bg-white rounded-2xl shadow-xl overflow-hidden">
          {/* Welcome Message */}
          {messages.length === 0 && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="p-8 bg-gradient-to-r from-green-50 to-emerald-50"
            >
              <div className="max-w-2xl mx-auto">
                <div className="flex items-center gap-3 mb-4">
                  <Leaf className="w-6 h-6 text-green-600" />
                  <h2 className="text-2xl font-bold text-gray-900">
                    {t.chat.chatPage.welcome}
                  </h2>
                </div>
                <p className="text-gray-700 mb-4">
                  {t.chat.chatPage.welcomeDesc}
                </p>
                <ul className="space-y-2 mb-6">
                  {t.chat.chatPage.features.map((feature, index) => (
                    <li key={index} className="flex items-start gap-2">
                      <CheckCircle className="w-5 h-5 text-green-600 mt-0.5 flex-shrink-0" />
                      <span className="text-gray-700">{feature}</span>
                    </li>
                  ))}
                </ul>
                
                {/* Quick Questions */}
                <div>
                  <p className="text-sm font-semibold text-black mb-3">
                    {t.chat.quickQuestions}
                  </p>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                    {quickQuestions.map((question, index) => (
                      <button
                        key={index}
                        onClick={() => setInput(question)}
                        className="px-4 py-3 text-sm bg-white text-black hover:bg-green-50 border border-green-200 rounded-lg transition-colors text-left"
                      >
                        {question}
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            </motion.div>
          )}

          {/* Messages */}
          <div className="h-[500px] overflow-y-auto p-6 space-y-4">
            {messages.map((message, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                <div
                  className={`max-w-[75%] rounded-2xl p-5 ${
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
              </motion.div>
            ))}
            
            {loading && (
              <div className="flex justify-start">
                <div className="bg-gray-100 rounded-2xl p-4">
                  <Loader2 className="w-6 h-6 animate-spin text-gray-600" />
                </div>
              </div>
            )}
          </div>

          {/* Input */}
          <div className="p-6 bg-gray-50 text-black border-t">
            <div className="flex gap-3">
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder={t.chat.typeQuestion}
                className="flex-1 px-6 py-4 border border-gray-300 rounded-full focus:outline-none focus:ring-2 focus:ring-green-500 focus:border-transparent"
                disabled={loading}
              />
              <button
                onClick={handleSend}
                disabled={loading || !input.trim()}
                className="px-8 py-4 bg-green-600 text-white rounded-full hover:bg-green-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors font-semibold"
              >
                {loading ? (
                  <Loader2 className="w-6 h-6 animate-spin" />
                ) : (
                  <Send className="w-6 h-6" />
                )}
              </button>
            </div>
            <p className="text-xs text-gray-500 mt-2 text-center">
              {language === 'hi' 
                ? 'Enter दबाएं या भेजें बटन क्लिक करें'
                : 'Press Enter or click Send button'}
            </p>
          </div>
        </div>

        {/* Info Cards */}
        <div className="grid md:grid-cols-3 gap-4 mt-8">
          {[
            {
              icon: <MessageSquare className="w-6 h-6" />,
              title: language === 'hi' ? 'तत्काल उत्तर' : 'Instant Answers',
              desc: language === 'hi' 
                ? 'AI द्वारा संचालित त्वरित प्रतिक्रियाएं'
                : 'AI-powered quick responses'
            },
            {
              icon: <Leaf className="w-6 h-6" />,
              title: language === 'hi' ? 'विशेषज्ञ सलाह' : 'Expert Advice',
              desc: language === 'hi'
                ? 'कृषि विशेषज्ञों से सलाह'
                : 'Advice from agricultural experts'
            },
            {
              icon: <CheckCircle className="w-6 h-6" />,
              title: language === 'hi' ? '24/7 उपलब्ध' : '24/7 Available',
              desc: language === 'hi'
                ? 'कभी भी, कहीं भी मदद'
                : 'Help anytime, anywhere'
            }
          ].map((card, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.1 }}
              className="bg-white rounded-lg p-6 shadow-md"
            >
              <div className="text-green-600 mb-3">{card.icon}</div>
              <h3 className="font-semibold text-gray-900 mb-2">{card.title}</h3>
              <p className="text-sm text-gray-600">{card.desc}</p>
            </motion.div>
          ))}
        </div>
      </div>
    </div>
  )
}

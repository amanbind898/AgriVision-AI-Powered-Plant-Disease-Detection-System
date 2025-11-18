'use client'

import Link from 'next/link'
import { Leaf, Camera, MessageSquare, Globe, TrendingUp, Shield, Sparkles, CheckCircle, ArrowRight } from 'lucide-react'
import { motion } from 'framer-motion'
import { useLanguage } from '@/contexts/LanguageContext'

export default function Home() {
  const { t } = useLanguage()
  
  const features = [
    {
      icon: <Camera className="w-8 h-8" />,
      title: t.home.features.aiDetection.title,
      description: t.home.features.aiDetection.desc
    },
    {
      icon: <MessageSquare className="w-8 h-8" />,
      title: t.home.features.chatbot.title,
      description: t.home.features.chatbot.desc
    },
    {
      icon: <Globe className="w-8 h-8" />,
      title: t.home.features.multilingual.title,
      description: t.home.features.multilingual.desc
    },
    {
      icon: <TrendingUp className="w-8 h-8" />,
      title: t.home.features.treatment.title,
      description: t.home.features.treatment.desc
    },
    {
      icon: <Shield className="w-8 h-8" />,
      title: t.home.features.explainable.title,
      description: t.home.features.explainable.desc
    },
    {
      icon: <Leaf className="w-8 h-8" />,
      title: t.home.features.plants.title,
      description: t.home.features.plants.desc
    }
  ]
  
  const steps = [
    { 
      step: '1', 
      title: t.home.steps.upload.title, 
      desc: t.home.steps.upload.desc 
    },
    { 
      step: '2', 
      title: t.home.steps.analyze.title, 
      desc: t.home.steps.analyze.desc 
    },
    { 
      step: '3', 
      title: t.home.steps.results.title, 
      desc: t.home.steps.results.desc 
    }
  ]
  
  const stats = [
    { value: '96%', label: t.home.stats.accuracy },
    { value: '38+', label: t.home.stats.diseases },
    { value: '12+', label: t.home.stats.plants },
    { value: '70K+', label: t.home.stats.images }
  ]

  return (
    <div className="min-h-screen">
      {/* Hero Section */}
      <section className="relative bg-gradient-to-br from-green-50 via-emerald-50 to-teal-50 py-24 px-4 overflow-hidden">
        {/* Background Pattern */}
        <div className="absolute inset-0 opacity-10">
          <div className="absolute top-10 left-10 w-72 h-72 bg-green-400 rounded-full blur-3xl"></div>
          <div className="absolute bottom-10 right-10 w-96 h-96 bg-emerald-400 rounded-full blur-3xl"></div>
        </div>
        
        <div className="container mx-auto max-w-6xl relative z-10">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="text-center"
          >
            {/* Logo with Animation */}
            <motion.div 
              className="flex justify-center mb-6"
              animate={{ 
                scale: [1, 1.1, 1],
                rotate: [0, 5, -5, 0]
              }}
              transition={{ 
                duration: 3,
                repeat: Infinity,
                repeatDelay: 2
              }}
            >
              <div className="bg-white p-4 rounded-full shadow-lg">
                <Leaf className="w-16 h-16 text-green-600" />
              </div>
            </motion.div>
            
            <motion.h1 
              className="text-5xl md:text-7xl font-bold text-gray-900 mb-6"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 }}
            >
              {t.home.title}
            </motion.h1>
            
            <motion.p 
              className="text-xl md:text-2xl text-gray-700 mb-4 font-semibold"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.3 }}
            >
              {t.home.subtitle}
            </motion.p>
            
            <motion.p 
              className="text-lg text-gray-600 mb-10 max-w-3xl mx-auto leading-relaxed"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.4 }}
            >
              {t.home.description}
            </motion.p>
            
            <motion.div 
              className="flex flex-col sm:flex-row gap-4 justify-center"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.5 }}
            >
              <Link
                href="/predict"
                className="group bg-green-600 text-white px-8 py-4 rounded-xl font-semibold hover:bg-green-700 transition-all text-lg shadow-lg hover:shadow-xl flex items-center justify-center gap-2"
              >
                {t.home.startDetection}
                <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
              </Link>
              <Link
                href="/chat"
                className="bg-white text-green-600 px-8 py-4 rounded-xl font-semibold hover:bg-gray-50 transition-all border-2 border-green-600 text-lg shadow-lg hover:shadow-xl"
              >
                {t.home.liveDemo}
              </Link>
            </motion.div>
          </motion.div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-20 px-4 bg-white">
        <div className="container mx-auto max-w-6xl">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <h2 className="text-4xl md:text-5xl font-bold text-gray-900 mb-4">
              {t.home.powerfulFeatures}
            </h2>
            <div className="w-24 h-1 bg-green-600 mx-auto rounded-full"></div>
          </motion.div>
          
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
            {features.map((feature, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: index * 0.1 }}
                viewport={{ once: true }}
                className="group p-8 rounded-2xl border-2 border-gray-200 hover:border-green-500 hover:shadow-2xl transition-all bg-gradient-to-br from-white to-green-50/30"
              >
                <div className="text-green-600 mb-4 transform group-hover:scale-110 transition-transform">
                  {feature.icon}
                </div>
                <h3 className="text-xl font-bold mb-3 text-gray-900">
                  {feature.title}
                </h3>
                <p className="text-gray-600 leading-relaxed">{feature.description}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* How It Works */}
      <section className="py-20 px-4 bg-gradient-to-br from-gray-50 to-green-50/30">
        <div className="container mx-auto max-w-6xl">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <h2 className="text-4xl md:text-5xl font-bold text-gray-900 mb-4">
              {t.home.howItWorks}
            </h2>
            <div className="w-24 h-1 bg-green-600 mx-auto rounded-full"></div>
          </motion.div>
          
          <div className="grid md:grid-cols-3 gap-12">
            {steps.map((item, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, scale: 0.9 }}
                whileInView={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.5, delay: index * 0.2 }}
                viewport={{ once: true }}
                className="text-center relative"
              >
                {/* Connector Line */}
                {index < steps.length - 1 && (
                  <div className="hidden md:block absolute top-8 left-[60%] w-[80%] h-0.5 bg-green-300"></div>
                )}
                
                <motion.div 
                  className="w-20 h-20 bg-gradient-to-br from-green-600 to-emerald-600 text-white rounded-full flex items-center justify-center text-3xl font-bold mx-auto mb-6 shadow-lg relative z-10"
                  whileHover={{ scale: 1.1, rotate: 5 }}
                >
                  {item.step}
                </motion.div>
                <h3 className="text-2xl font-bold mb-3 text-gray-900">{item.title}</h3>
                <p className="text-gray-600 leading-relaxed">{item.desc}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Stats Section */}
      <section className="py-20 px-4 bg-gradient-to-br from-green-600 to-emerald-700 text-white relative overflow-hidden">
        {/* Background Pattern */}
        <div className="absolute inset-0 opacity-10">
          <div className="absolute top-0 left-0 w-96 h-96 bg-white rounded-full blur-3xl"></div>
          <div className="absolute bottom-0 right-0 w-96 h-96 bg-white rounded-full blur-3xl"></div>
        </div>
        
        <div className="container mx-auto max-w-6xl relative z-10">
          <div className="grid md:grid-cols-4 gap-8 text-center">
            {stats.map((stat, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: index * 0.1 }}
                viewport={{ once: true }}
                className="p-6"
              >
                <motion.div 
                  className="text-6xl md:text-7xl font-bold mb-3"
                  whileHover={{ scale: 1.1 }}
                >
                  {stat.value}
                </motion.div>
                <div className="text-green-100 text-xl font-medium">{stat.label}</div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-24 px-4 bg-gradient-to-br from-white to-green-50">
        <div className="container mx-auto max-w-4xl text-center">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
          >
            <Sparkles className="w-16 h-16 text-green-600 mx-auto mb-6" />
            <h2 className="text-4xl md:text-5xl font-bold mb-6 text-gray-900">
              {t.home.readyToProtect}
            </h2>
            <p className="text-xl text-gray-600 mb-10 max-w-2xl mx-auto">
              {t.home.description}
            </p>
            <Link
              href="/predict"
              className="inline-flex items-center gap-3 bg-green-600 text-white px-12 py-5 rounded-xl font-bold hover:bg-green-700 transition-all text-xl shadow-2xl hover:shadow-3xl hover:scale-105"
            >
              {t.home.getStartedFree}
              <ArrowRight className="w-6 h-6" />
            </Link>
          </motion.div>
        </div>
      </section>
    </div>
  )
}

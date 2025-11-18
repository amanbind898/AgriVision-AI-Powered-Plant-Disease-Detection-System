'use client'

import { Github, Linkedin, Mail } from 'lucide-react'
import { motion } from 'framer-motion'
import { useLanguage } from '@/contexts/LanguageContext'

export default function TeamPage() {
  const { t } = useLanguage()
  const team = [
    {
      name: 'Aman Kumar Bind',
      role: 'FullStack & ML Engineer',
      responsibilities: [
        'Model training & optimization',
        'ML pipeline setup',
        'Model deployment',
        'Documentation'
      ],
      github: 'https://github.com/amanbind898',
      linkedin: 'https://linkedin.com/in/amankumarbind',
      email: 'aman.2201086cs@iiitbh.ac.in'
    },
    {
      name: 'Ankit',
      role: 'Frontend Developer',
      responsibilities: [
        'FastAPI setup',
        'API endpoints development',
        'Model integration',
        'Database setup'
      ],
      github: 'https://github.com/ankit',
      linkedin: 'https://linkedin.com/in/ankit',
      email: 'ankit@example.com'
    },
    {
      name: 'Sahil Morwal',
      role: 'Frontend Developer',
      responsibilities: [
        'Next.js setup',
        'UI/UX design',
        'Component development',
        'Responsive design'
      ],
      github: 'https://github.com/sahil',
      linkedin: 'https://linkedin.com/in/sahil',
      email: 'sahil@example.com'
    },
    {
      name: 'Nawazish hassan',
      role: 'Full Stack Developer',
      responsibilities: [
        'Frontend-Backend integration',
        'Perplexity API integration',
        'Deployment',
        'Testing'
      ],
      github: 'https://github.com/nawaz',
      linkedin: 'https://linkedin.com/in/nawaz',
      email: 'nawaz@example.com'
    }
  ]

  return (
    <div className="min-h-screen bg-gray-50 py-12 px-4">
      <div className="container mx-auto max-w-6xl">
        {/* Header */}
        <div className="text-center mb-16">
          <h1 className="text-5xl font-bold text-gray-900 mb-4">
            {t.team.title}
          </h1>
          <p className="text-xl text-gray-600">
            {t.team.subtitle}
          </p>
        </div>

        {/* Team Grid */}
        <div className="grid md:grid-cols-2 gap-8 mb-16">
          {team.map((member, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: index * 0.1 }}
              className="bg-white rounded-xl shadow-lg p-8 hover:shadow-xl transition-shadow"
            >
              <div className="flex items-start gap-6">
                {/* Avatar */}
                <div className="w-20 h-20 bg-gradient-to-br from-green-400 to-green-600 rounded-full flex items-center justify-center text-white text-2xl font-bold flex-shrink-0">
                  {member.name.charAt(0)}
                </div>

                {/* Info */}
                <div className="flex-1">
                  <h3 className="text-2xl font-bold text-gray-900 mb-1">
                    {member.name}
                  </h3>
                  <p className="text-green-600 font-semibold mb-4">
                    {member.role}
                  </p>

                  {/* Responsibilities */}
                  <div className="mb-4">
                    <h4 className="text-sm font-semibold text-gray-700 mb-2">
                      {t.team.responsibilities}:
                    </h4>
                    <ul className="space-y-1">
                      {member.responsibilities.map((resp, idx) => (
                        <li key={idx} className="text-sm text-gray-600 flex items-start">
                          <span className="text-green-500 mr-2">•</span>
                          {resp}
                        </li>
                      ))}
                    </ul>
                  </div>

                  {/* Social Links */}
                  <div className="flex gap-3">
                    <a
                      href={member.github}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-gray-600 hover:text-gray-900 transition-colors"
                    >
                      <Github className="w-5 h-5" />
                    </a>
                    <a
                      href={member.linkedin}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-gray-600 hover:text-blue-600 transition-colors"
                    >
                      <Linkedin className="w-5 h-5" />
                    </a>
                    <a
                      href={`mailto:${member.email}`}
                      className="text-gray-600 hover:text-red-600 transition-colors"
                    >
                      <Mail className="w-5 h-5" />
                    </a>
                  </div>
                </div>
              </div>
            </motion.div>
          ))}
        </div>

        {/* Project Info */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.4 }}
          className="bg-gradient-to-br from-green-50 to-emerald-100 rounded-xl p-8"
        >
          <h2 className="text-3xl font-bold text-gray-900 mb-4">
            {t.team.about}
          </h2>
          <p className="text-gray-700 mb-4">
            {t.team.aboutDesc}
          </p>
          <div className="grid md:grid-cols-2 gap-4 mt-6">
            <div className="bg-white rounded-lg p-4">
              <h3 className="font-semibold text-gray-900 mb-2">{t.team.techStack}</h3>
              <p className="text-sm text-gray-600">
                Next.js, FastAPI, TensorFlow, Perplexity AI, Tailwind CSS
              </p>
            </div>
            <div className="bg-white rounded-lg p-4">
              <h3 className="font-semibold text-gray-900 mb-2">{t.team.duration}</h3>
              <p className="text-sm text-gray-600">
                6 weeks (August 2025 - November 2025)
              </p>
            </div>
          </div>
        </motion.div>
      </div>
    </div>
  )
}

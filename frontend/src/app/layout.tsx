import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import './globals.css'
import { LanguageProvider } from '@/contexts/LanguageContext'
import Navbar from '@/components/Navbar'

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata = {
  title: 'AgriVision - AI Plant Disease Detection',
  description: 'AI-powered plant disease detection system for farmers',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className={inter.className}>
        <LanguageProvider>
          <Navbar />
          <main className="min-h-screen">
            {children}
          </main>
          <footer className="bg-gray-900 text-white py-8 mt-20">
            <div className="container mx-auto px-4 text-center">
              <p className="text-gray-400">
                © 2024 AgriVision | BTech Minor Project
              </p>
              <p className="text-sm text-gray-500 mt-2">
                Team: Your Name, Ankit, Sahil, Nawaz
              </p>
            </div>
          </footer>
        </LanguageProvider>
      </body>
    </html>
  )
}

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
    <html lang="en" className="h-full">
      <body className={`${inter.className} flex flex-col min-h-screen`}>
        <LanguageProvider>
          <Navbar />
          <main className="flex-grow">
            {children}
          </main>
          <footer className="bg-gray-900 text-white py-8">
            <div className="container mx-auto px-4 text-center">
              <p className="text-gray-400">
                © 2025 AgriVision | B.Tech Minor Project
              </p>
              <p className="text-sm text-gray-500 mt-2">
                Team: Aman, Ankit, Sahil, Nawazish
              </p>
            </div>
          </footer>
        </LanguageProvider>
      </body>
    </html>
  )
}

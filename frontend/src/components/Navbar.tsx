'use client'

import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { Leaf, Menu, X, Globe } from 'lucide-react'
import { useState } from 'react'
import { useLanguage } from '@/contexts/LanguageContext'

export default function Navbar() {
  const pathname = usePathname()
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)
  const { language, setLanguage, t } = useLanguage()

  const links = [
    { href: '/', label: t.nav.home },
    { href: '/predict', label: t.nav.detect },
    { href: '/chat', label: t.nav.chat },
    { href: '/team', label: t.nav.team },
  ]

  return (
    <nav className="bg-white shadow-md sticky top-0 z-50">
      <div className="container mx-auto px-4">
        <div className="flex justify-between items-center h-16">
          {/* Logo */}
          <Link href="/" className="flex items-center gap-2 font-bold text-xl text-green-600">
            <Leaf className="w-6 h-6" />
            <span>AgriVision</span>
          </Link>

          {/* Desktop Navigation */}
          <div className="hidden md:flex items-center gap-6">
            {links.map((link) => (
              <Link
                key={link.href}
                href={link.href}
                className={`font-medium transition-colors ${
                  pathname === link.href
                    ? 'text-green-600'
                    : 'text-gray-600 hover:text-green-600'
                }`}
              >
                {link.label}
              </Link>
            ))}
            
            {/* Language Switcher */}
            <button
              onClick={() => setLanguage(language === 'en' ? 'hi' : 'en')}
              className="flex items-center gap-2 px-3 py-2 rounded-lg bg-green-50 text-green-600 hover:bg-green-100 transition-colors"
            >
              <Globe className="w-4 h-4" />
              <span className="text-sm font-medium">
                {language === 'en' ? 'हिंदी' : 'English'}
              </span>
            </button>
          </div>

          {/* Mobile Menu Button */}
          <button
            className="md:hidden text-gray-600"
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
          >
            {mobileMenuOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
          </button>
        </div>

        {/* Mobile Navigation */}
        {mobileMenuOpen && (
          <div className="md:hidden py-4 border-t">
            {links.map((link) => (
              <Link
                key={link.href}
                href={link.href}
                className={`block py-2 font-medium transition-colors ${
                  pathname === link.href
                    ? 'text-green-600'
                    : 'text-gray-600 hover:text-green-600'
                }`}
                onClick={() => setMobileMenuOpen(false)}
              >
                {link.label}
              </Link>
            ))}
            
            {/* Mobile Language Switcher */}
            <button
              onClick={() => {
                setLanguage(language === 'en' ? 'hi' : 'en')
                setMobileMenuOpen(false)
              }}
              className="flex items-center gap-2 px-3 py-2 mt-2 rounded-lg bg-green-50 text-green-600 w-full"
            >
              <Globe className="w-4 h-4" />
              <span className="text-sm font-medium">
                {language === 'en' ? 'हिंदी' : 'English'}
              </span>
            </button>
          </div>
        )}
      </div>
    </nav>
  )
}

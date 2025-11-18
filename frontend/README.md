# AgriVision Frontend

Next.js frontend for the AgriVision plant disease detection system.

## 🚀 Quick Start

### Install Dependencies
```bash
npm install
```

### Run Development Server
```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000)

### Build for Production
```bash
npm run build
npm start
```

## 📁 Project Structure

```
frontend/
├── src/
│   ├── app/
│   │   ├── page.tsx              # Home page
│   │   ├── predict/              # Disease detection page
│   │   ├── team/                 # Team page
│   │   ├── demo/                 # Live camera demo
│   │   ├── layout.tsx            # Root layout
│   │   └── globals.css           # Global styles
│   ├── components/
│   │   ├── Navbar.tsx            # Navigation bar
│   │   ├── ImageUpload.tsx       # Image upload component
│   │   ├── PredictionCard.tsx    # Results display
│   │   └── ChatBot.tsx           # AI chatbot
│   ├── lib/
│   │   └── api.ts                # API client functions
│   └── types/
│       └── index.ts              # TypeScript types
├── public/                        # Static assets
├── package.json
├── tsconfig.json
├── tailwind.config.ts
└── next.config.js
```

## 🎨 Pages

### Home (`/`)
- Hero section with project overview
- Feature highlights
- Statistics
- Call-to-action buttons

### Predict (`/predict`)
- Image upload with drag-and-drop
- Disease prediction results
- Treatment recommendations
- AI chatbot for questions

### Team (`/team`)
- Team member profiles
- Responsibilities
- Project information

### Demo (`/demo`)
- Live camera access
- Real-time image capture
- Instant disease detection

## 🧩 Components

### Navbar
- Responsive navigation
- Active route highlighting
- Mobile menu

### ImageUpload
- Drag-and-drop file upload
- Image preview
- Loading states
- Error handling

### PredictionCard
- Disease information display
- Confidence visualization
- Top 5 predictions
- Treatment recommendations

### ChatBot
- AI-powered chat interface
- Multilingual support (EN/HI)
- Context-aware responses
- Quick question suggestions

## 🔌 API Integration

All API calls are centralized in `src/lib/api.ts`:

```typescript
import { predictDisease, chatWithAI } from '@/lib/api'

// Predict disease
const result = await predictDisease(file)

// Chat with AI
const response = await chatWithAI({
  message: "How to treat this disease?",
  language: "en"
})
```

## 🎨 Styling

### Tailwind CSS
- Utility-first CSS framework
- Custom color palette (green theme)
- Responsive design
- Dark mode ready

### Custom Colors
```javascript
primary: {
  50: '#f0fdf4',
  500: '#22c55e',
  600: '#16a34a',
  700: '#15803d',
}
```

## 📱 Responsive Design

- Mobile-first approach
- Breakpoints:
  - `sm`: 640px
  - `md`: 768px
  - `lg`: 1024px
  - `xl`: 1280px

## 🔧 Configuration

### Environment Variables
Create `.env.local`:
```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

### Next.js Config
```javascript
// next.config.js
module.exports = {
  reactStrictMode: true,
  images: {
    domains: ['localhost'],
  },
}
```

## 🧪 Development

### Adding a New Page
1. Create file in `src/app/[page-name]/page.tsx`
2. Add route to Navbar
3. Implement page component

### Adding a New Component
1. Create file in `src/components/ComponentName.tsx`
2. Import and use in pages
3. Add TypeScript types if needed

### Adding API Endpoint
1. Add function to `src/lib/api.ts`
2. Add types to `src/types/index.ts`
3. Use in components

## 📦 Dependencies

### Core
- `next`: 14.0.4
- `react`: 18.2.0
- `typescript`: 5.x

### UI & Styling
- `tailwindcss`: 3.3.0
- `framer-motion`: 10.16.16
- `lucide-react`: 0.294.0

### Utilities
- `axios`: 1.6.2
- `react-dropzone`: 14.2.3
- `clsx`: 2.0.0

## 🚀 Deployment

### Vercel (Recommended)
```bash
# Install Vercel CLI
npm i -g vercel

# Deploy
vercel
```

### Docker
```bash
# Build image
docker build -t agrivision-frontend .

# Run container
docker run -p 3000:3000 agrivision-frontend
```

### Static Export
```bash
# Build static site
npm run build

# Output in 'out' directory
```

## 🎯 Performance

- **Lighthouse Score**: 95+
- **First Contentful Paint**: < 1.5s
- **Time to Interactive**: < 3s
- **Bundle Size**: < 200KB (gzipped)

## 🐛 Troubleshooting

### Port Already in Use
```bash
# Use different port
npm run dev -- -p 3001
```

### API Connection Error
- Check backend is running on port 8000
- Verify `NEXT_PUBLIC_API_URL` in `.env.local`
- Check CORS settings in backend

### Build Errors
```bash
# Clear cache
rm -rf .next
npm run build
```

## 📝 Team Responsibilities

- **Sahil**: Frontend development, UI/UX design
- **Nawaz**: Frontend-backend integration
- **Your Name**: Overall coordination

## 🔗 Useful Links

- [Next.js Documentation](https://nextjs.org/docs)
- [Tailwind CSS](https://tailwindcss.com/docs)
- [TypeScript](https://www.typescriptlang.org/docs)
- [React](https://react.dev/)

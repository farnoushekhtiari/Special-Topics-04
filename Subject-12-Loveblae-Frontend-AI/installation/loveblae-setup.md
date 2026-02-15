# Loveblae Frontend AI Setup

## Overview
This guide will help you set up your development environment for building AI-integrated frontend applications. We'll cover the essential tools, frameworks, and APIs needed for modern AI web development.

## Prerequisites

### System Requirements
- Node.js 18+ and npm
- Modern web browser (Chrome 90+, Firefox 88+, Safari 14+)
- Git for version control
- Code editor (VS Code recommended)

### Required Knowledge
- JavaScript/TypeScript fundamentals
- React or Vue.js basics
- HTML5 and CSS3
- Basic understanding of APIs

## Installation Steps

### Step 1: Node.js and npm Setup

```bash
# Check if Node.js is installed
node --version
npm --version

# If not installed, download from https://nodejs.org/
# Recommended: LTS version
```

### Step 2: Create Project Structure

```bash
# Create main project directory
mkdir loveblae-frontend-ai
cd loveblae-frontend-ai

# Initialize project
npm init -y

# Install core dependencies
npm install react react-dom next
npm install --save-dev typescript @types/react @types/node
npm install --save-dev tailwindcss postcss autoprefixer
npm install --save-dev eslint prettier

# Initialize TypeScript
npx tsc --init

# Initialize Tailwind CSS
npx tailwindcss init -p

# Create basic folder structure
mkdir -p components pages api utils hooks types lib public
mkdir -p components/ai components/ui components/chat components/vision
```

### Step 3: AI SDKs and Libraries Setup

```bash
# OpenAI SDK
npm install openai

# Hugging Face Transformers (for browser)
npm install @huggingface/transformers

# TensorFlow.js for browser-based ML
npm install @tensorflow/tfjs @tensorflow-models/coco-ssd @tensorflow-models/mobilenet

# Speech recognition and synthesis
npm install @microsoft/speech-sdk

# WebRTC for real-time communication
npm install simple-peer

# Socket.io for real-time AI communication
npm install socket.io-client

# Additional AI utilities
npm install axios lodash uuid
npm install --save-dev @types/lodash @types/uuid
```

### Step 4: Development Tools Configuration

#### TypeScript Configuration (`tsconfig.json`)

```json
{
  "compilerOptions": {
    "target": "es5",
    "lib": ["dom", "dom.iterable", "es6"],
    "allowJs": true,
    "skipLibCheck": true,
    "strict": true,
    "noEmit": true,
    "esModuleInterop": true,
    "module": "esnext",
    "moduleResolution": "bundler",
    "resolveJsonModule": true,
    "isolatedModules": true,
    "jsx": "preserve",
    "incremental": true,
    "plugins": [
      {
        "name": "next"
      }
    ],
    "baseUrl": ".",
    "paths": {
      "@/*": ["./*"],
      "@/components/*": ["components/*"],
      "@/pages/*": ["pages/*"],
      "@/utils/*": ["utils/*"],
      "@/types/*": ["types/*"]
    }
  },
  "include": ["next-env.d.ts", "**/*.ts", "**/*.tsx", ".next/types/**/*.ts"],
  "exclude": ["node_modules"]
}
```

#### Tailwind CSS Configuration (`tailwind.config.js`)

```javascript
/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './pages/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
    './app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        'ai-primary': '#3b82f6',
        'ai-secondary': '#8b5cf6',
        'ai-accent': '#06b6d4',
        'ai-success': '#10b981',
        'ai-warning': '#f59e0b',
        'ai-error': '#ef4444',
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'bounce-slow': 'bounce 2s infinite',
      }
    },
  },
  plugins: [],
}
```

#### ESLint Configuration (`.eslintrc.json`)

```json
{
  "extends": [
    "next/core-web-vitals",
    "eslint:recommended",
    "@typescript-eslint/recommended"
  ],
  "parser": "@typescript-eslint/parser",
  "plugins": ["@typescript-eslint"],
  "rules": {
    "@typescript-eslint/no-unused-vars": "error",
    "@typescript-eslint/no-explicit-any": "warn",
    "react-hooks/exhaustive-deps": "warn",
    "no-console": "warn"
  }
}
```

### Step 5: Environment Configuration

#### Create Environment Files

```bash
# .env.local (for local development)
touch .env.local
```

Add to `.env.local`:
```env
# OpenAI API
OPENAI_API_KEY=your_openai_api_key_here

# Hugging Face API
HUGGINGFACE_API_KEY=your_huggingface_api_key_here

# Application Settings
NEXT_PUBLIC_APP_NAME=Loveblae AI
NEXT_PUBLIC_API_URL=http://localhost:3000/api

# Development
NODE_ENV=development
DEBUG=true
```

#### Next.js Configuration (`next.config.js`)

```javascript
/** @type {import('next').NextConfig} */
const nextConfig = {
  experimental: {
    appDir: true,
  },
  images: {
    domains: ['localhost'],
  },
  async headers() {
    return [
      {
        source: '/api/:path*',
        headers: [
          { key: 'Access-Control-Allow-Origin', value: '*' },
          { key: 'Access-Control-Allow-Methods', value: 'GET,POST,PUT,DELETE' },
          { key: 'Access-Control-Allow-Headers', value: 'Content-Type, Authorization' },
        ],
      },
    ]
  },
  webpack: (config, { isServer }) => {
    // WebAssembly support for AI models
    config.experiments = {
      ...config.experiments,
      asyncWebAssembly: true,
      layers: true,
    }

    // Fallback for Node.js modules in browser
    if (!isServer) {
      config.resolve.fallback = {
        ...config.resolve.fallback,
        fs: false,
        path: false,
        crypto: false,
      }
    }

    return config
  },
}

module.exports = nextConfig
```

### Step 6: API Keys and Services Setup

#### OpenAI API Setup
1. Visit [OpenAI Platform](https://platform.openai.com/)
2. Create an account and generate API key
3. Add the key to your `.env.local` file

#### Hugging Face Setup
1. Visit [Hugging Face](https://huggingface.co/)
2. Create an account and get API token
3. Add the token to your `.env.local` file

#### Web Speech API (Browser Native)
No setup required - available in modern browsers.

### Step 7: Basic Project Structure

#### Create Basic Components

```typescript
// components/layout/Layout.tsx
import { ReactNode } from 'react'

interface LayoutProps {
  children: ReactNode
  title?: string
}

export default function Layout({ children, title = 'Loveblae AI' }: LayoutProps) {
  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-purple-50">
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-4">
            <h1 className="text-2xl font-bold text-gray-900">{title}</h1>
            <nav className="space-x-4">
              <a href="/" className="text-gray-600 hover:text-gray-900">Home</a>
              <a href="/chat" className="text-gray-600 hover:text-gray-900">Chat</a>
              <a href="/vision" className="text-gray-600 hover:text-gray-900">Vision</a>
            </nav>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {children}
      </main>
    </div>
  )
}
```

#### Create Utility Functions

```typescript
// utils/ai-api.ts
import OpenAI from 'openai'

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
  dangerouslyAllowBrowser: true // Only for client-side usage
})

export const aiAPI = {
  async generateText(prompt: string, model = 'gpt-3.5-turbo') {
    const response = await openai.chat.completions.create({
      model,
      messages: [{ role: 'user', content: prompt }],
      temperature: 0.7,
      max_tokens: 1000,
    })

    return response.choices[0].message.content
  },

  async analyzeImage(imageUrl: string, prompt: string) {
    const response = await openai.chat.completions.create({
      model: 'gpt-4-vision-preview',
      messages: [
        {
          role: 'user',
          content: [
            { type: 'text', text: prompt },
            { type: 'image_url', image_url: { url: imageUrl } }
          ]
        }
      ],
      max_tokens: 500,
    })

    return response.choices[0].message.content
  }
}
```

#### Create Type Definitions

```typescript
// types/ai.ts
export interface AIResponse {
  success: boolean
  data?: any
  error?: string
  timestamp: Date
}

export interface ChatMessage {
  id: string
  role: 'user' | 'assistant' | 'system'
  content: string
  timestamp: Date
  metadata?: Record<string, any>
}

export interface VisionAnalysis {
  description: string
  objects: string[]
  colors: string[]
  sentiment?: 'positive' | 'negative' | 'neutral'
  confidence: number
}

export interface AudioTranscription {
  text: string
  confidence: number
  language?: string
  duration: number
}
```

### Step 8: Development Scripts

#### Update `package.json`

```json
{
  "name": "loveblae-frontend-ai",
  "version": "0.1.0",
  "private": true,
  "scripts": {
    "dev": "next dev",
    "build": "next build",
    "start": "next start",
    "lint": "next lint",
    "type-check": "tsc --noEmit",
    "test": "jest",
    "test:watch": "jest --watch",
    "storybook": "storybook dev -p 6006",
    "build-storybook": "storybook build"
  },
  "dependencies": {
    "next": "13.4.0",
    "react": "18.2.0",
    "react-dom": "18.2.0",
    "@tensorflow/tfjs": "^4.10.0",
    "@tensorflow-models/coco-ssd": "^2.2.2",
    "@tensorflow-models/mobilenet": "^2.1.1",
    "openai": "^4.0.0",
    "@huggingface/transformers": "^2.6.0",
    "axios": "^1.4.0",
    "socket.io-client": "^4.7.0",
    "simple-peer": "^9.11.1",
    "lodash": "^4.17.21",
    "uuid": "^9.0.0"
  },
  "devDependencies": {
    "@types/node": "^20.0.0",
    "@types/react": "^18.2.0",
    "@types/react-dom": "^18.2.0",
    "@types/lodash": "^4.14.195",
    "@types/uuid": "^9.0.1",
    "typescript": "^5.0.0",
    "tailwindcss": "^3.3.0",
    "postcss": "^8.4.0",
    "autoprefixer": "^10.4.0",
    "eslint": "^8.40.0",
    "eslint-config-next": "^13.4.0",
    "@typescript-eslint/eslint-plugin": "^5.59.0",
    "@typescript-eslint/parser": "^5.59.0",
    "prettier": "^2.8.0",
    "jest": "^29.5.0",
    "@testing-library/react": "^14.0.0",
    "@testing-library/jest-dom": "^5.16.0",
    "storybook": "^7.0.0"
  }
}
```

### Step 9: Testing Setup

#### Jest Configuration (`jest.config.js`)

```javascript
const nextJest = require('next/jest')

const createJestConfig = nextJest({
  dir: './',
})

const customJestConfig = {
  setupFilesAfterEnv: ['<rootDir>/jest.setup.js'],
  moduleNameMapping: {
    '^@/(.*)$': '<rootDir>/$1',
  },
  testEnvironment: 'jest-environment-jsdom',
  collectCoverageFrom: [
    'components/**/*.{js,jsx,ts,tsx}',
    'pages/**/*.{js,jsx,ts,tsx}',
    'utils/**/*.{js,ts}',
    '!**/*.d.ts',
  ],
  testMatch: [
    '<rootDir>/**/__tests__/**/*.{js,jsx,ts,tsx}',
    '<rootDir>/**/*.{test,spec}.{js,jsx,ts,tsx}',
  ],
}

module.exports = createJestConfig(customJestConfig)
```

#### Jest Setup (`jest.setup.js`)

```javascript
import '@testing-library/jest-dom'

// Mock fetch for API calls
global.fetch = jest.fn()

// Mock Web APIs
Object.defineProperty(window, 'matchMedia', {
  writable: true,
  value: jest.fn().mockImplementation(query => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: jest.fn(),
    removeListener: jest.fn(),
    addEventListener: jest.fn(),
    removeEventListener: jest.fn(),
    dispatchEvent: jest.fn(),
  })),
})

// Mock IntersectionObserver
global.IntersectionObserver = class IntersectionObserver {
  constructor() {}
  observe() {}
  unobserve() {}
  disconnect() {}
}

// Mock ResizeObserver
global.ResizeObserver = class ResizeObserver {
  constructor() {}
  observe() {}
  unobserve() {}
  disconnect() {}
}
```

### Step 10: Run Development Server

```bash
# Start development server
npm run dev

# The application should be available at http://localhost:3000
```

### Step 11: Verify Setup

#### Create a Simple Test Page

```typescript
// pages/index.tsx
import { useState } from 'react'
import Layout from '@/components/layout/Layout'

export default function Home() {
  const [message, setMessage] = useState('')

  const testAI = async () => {
    try {
      const response = await fetch('/api/test-ai')
      const data = await response.json()
      setMessage(data.message || 'AI integration working!')
    } catch (error) {
      setMessage('AI integration needs configuration')
    }
  }

  return (
    <Layout>
      <div className="text-center">
        <h1 className="text-4xl font-bold text-gray-900 mb-8">
          Welcome to Loveblae Frontend AI
        </h1>

        <div className="bg-white rounded-lg shadow-md p-6 max-w-md mx-auto">
          <h2 className="text-xl font-semibold mb-4">Setup Verification</h2>

          <button
            onClick={testAI}
            className="bg-ai-primary hover:bg-blue-700 text-white font-bold py-2 px-4 rounded"
          >
            Test AI Integration
          </button>

          {message && (
            <p className="mt-4 text-sm text-gray-600">{message}</p>
          )}
        </div>

        <div className="mt-8 grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="bg-white rounded-lg shadow-md p-6">
            <h3 className="text-lg font-semibold mb-2">💬 Chat AI</h3>
            <p className="text-gray-600">Natural language conversations</p>
          </div>

          <div className="bg-white rounded-lg shadow-md p-6">
            <h3 className="text-lg font-semibold mb-2">👁️ Vision AI</h3>
            <p className="text-gray-600">Image analysis and understanding</p>
          </div>

          <div className="bg-white rounded-lg shadow-md p-6">
            <h3 className="text-lg font-semibold mb-2">🎵 Audio AI</h3>
            <p className="text-gray-600">Speech recognition and synthesis</p>
          </div>
        </div>
      </div>
    </Layout>
  )
}
```

#### Create Test API Route

```typescript
// pages/api/test-ai.ts
import type { NextApiRequest, NextApiResponse } from 'next'

export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse
) {
  if (req.method !== 'GET') {
    return res.status(405).json({ error: 'Method not allowed' })
  }

  try {
    // Simple test - check if environment variables are set
    const hasOpenAI = !!process.env.OPENAI_API_KEY
    const hasHuggingFace = !!process.env.HUGGINGFACE_API_KEY

    res.status(200).json({
      message: 'AI integration configured successfully!',
      openai: hasOpenAI ? 'configured' : 'missing API key',
      huggingface: hasHuggingFace ? 'configured' : 'missing API key',
      timestamp: new Date().toISOString()
    })
  } catch (error) {
    res.status(500).json({
      error: 'AI integration test failed',
      details: error.message
    })
  }
}
```

## Troubleshooting

### Common Issues

1. **Module not found errors**
   ```bash
   rm -rf node_modules package-lock.json
   npm install
   ```

2. **TypeScript errors**
   ```bash
   npm run type-check
   ```

3. **Environment variables not loading**
   - Ensure `.env.local` is in the project root
   - Restart the development server

4. **API key issues**
   - Verify API keys are correctly set in `.env.local`
   - Check API key permissions and quotas

### Performance Tips

- Use `next/image` for optimized image loading
- Implement code splitting for large AI libraries
- Use service workers for caching AI models
- Optimize bundle size by lazy loading AI features

## Next Steps

Once your environment is set up, you can proceed to:
1. [Tutorial 01: Modern Frontend with AI Integration](../tutorials/01-modern-frontend-ai.md)
2. [Workshop 01: Basic AI Chat Interface](../workshops/workshop-01-basic-ai-chat.md)

## Resources

- [Next.js Documentation](https://nextjs.org/docs)
- [OpenAI API Documentation](https://platform.openai.com/docs)
- [TensorFlow.js Guide](https://www.tensorflow.org/js/guide)
- [Hugging Face JavaScript](https://huggingface.co/docs/transformers.js/index)
- [Web APIs MDN](https://developer.mozilla.org/en-US/docs/Web/API)
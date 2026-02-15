# Tutorial 01: Modern Frontend Architecture with AI Integration

## Overview
This tutorial introduces modern frontend architecture patterns optimized for AI integration. You'll learn how to structure React/Next.js applications that seamlessly incorporate AI capabilities, handle asynchronous AI operations, and maintain good user experience during AI processing.

## Learning Objectives
- Understand modern frontend architecture for AI applications
- Implement proper state management for AI operations
- Handle loading states and error boundaries for AI features
- Optimize component design for AI-driven interactions

## Core Concepts

### AI-First Architecture Principles

#### 1. Asynchronous AI Operations
AI operations are inherently asynchronous. Your frontend must handle:
- API call delays
- Streaming responses
- Background processing
- Error recovery

```typescript
// hooks/useAI.ts
import { useState, useCallback } from 'react'

interface AIState<T> {
  data: T | null
  loading: boolean
  error: string | null
}

export function useAI<T>() {
  const [state, setState] = useState<AIState<T>>({
    data: null,
    loading: false,
    error: null
  })

  const execute = useCallback(async (operation: () => Promise<T>) => {
    setState({ data: null, loading: true, error: null })

    try {
      const result = await operation()
      setState({ data: result, loading: false, error: null })
      return result
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'AI operation failed'
      setState({ data: null, loading: false, error: errorMessage })
      throw error
    }
  }, [])

  const reset = useCallback(() => {
    setState({ data: null, loading: false, error: null })
  }, [])

  return { ...state, execute, reset }
}
```

#### 2. Progressive AI Enhancement
Don't make AI a hard requirement. Design your app to work without AI and enhance with it:

```typescript
// components/AIEnhancedComponent.tsx
import { useState, useEffect } from 'react'

interface AIEnhancedComponentProps {
  children: React.ReactNode
  aiFeature?: () => Promise<any>
  fallback?: React.ReactNode
}

export default function AIEnhancedComponent({
  children,
  aiFeature,
  fallback
}: AIEnhancedComponentProps) {
  const [aiEnabled, setAiEnabled] = useState(false)
  const [aiData, setAiData] = useState(null)

  useEffect(() => {
    // Check if AI features are available
    const checkAI = async () => {
      if (!aiFeature) return

      try {
        // Test AI availability
        const testResult = await aiFeature()
        setAiEnabled(true)
        setAiData(testResult)
      } catch (error) {
        console.warn('AI feature not available:', error)
        setAiEnabled(false)
      }
    }

    checkAI()
  }, [aiFeature])

  if (aiEnabled && aiData) {
    return <div className="ai-enhanced">{children}</div>
  }

  return (
    <div className="basic-version">
      {fallback || children}
    </div>
  )
}
```

### State Management for AI Applications

#### AI Context Provider
Create a centralized AI state management system:

```typescript
// contexts/AIContext.tsx
import React, { createContext, useContext, useReducer, ReactNode } from 'react'

interface AIState {
  capabilities: {
    chat: boolean
    vision: boolean
    speech: boolean
    realtime: boolean
  }
  activeOperations: string[]
  preferences: {
    model: string
    temperature: number
    maxTokens: number
  }
  errors: string[]
}

type AIAction =
  | { type: 'SET_CAPABILITY'; capability: keyof AIState['capabilities']; enabled: boolean }
  | { type: 'START_OPERATION'; operationId: string }
  | { type: 'END_OPERATION'; operationId: string }
  | { type: 'SET_PREFERENCE'; key: keyof AIState['preferences']; value: any }
  | { type: 'ADD_ERROR'; error: string }
  | { type: 'CLEAR_ERRORS' }

const initialState: AIState = {
  capabilities: {
    chat: false,
    vision: false,
    speech: false,
    realtime: false
  },
  activeOperations: [],
  preferences: {
    model: 'gpt-3.5-turbo',
    temperature: 0.7,
    maxTokens: 1000
  },
  errors: []
}

function aiReducer(state: AIState, action: AIAction): AIState {
  switch (action.type) {
    case 'SET_CAPABILITY':
      return {
        ...state,
        capabilities: {
          ...state.capabilities,
          [action.capability]: action.enabled
        }
      }

    case 'START_OPERATION':
      return {
        ...state,
        activeOperations: [...state.activeOperations, action.operationId]
      }

    case 'END_OPERATION':
      return {
        ...state,
        activeOperations: state.activeOperations.filter(id => id !== action.operationId)
      }

    case 'SET_PREFERENCE':
      return {
        ...state,
        preferences: {
          ...state.preferences,
          [action.key]: action.value
        }
      }

    case 'ADD_ERROR':
      return {
        ...state,
        errors: [...state.errors, action.error]
      }

    case 'CLEAR_ERRORS':
      return {
        ...state,
        errors: []
      }

    default:
      return state
  }
}

const AIContext = createContext<{
  state: AIState
  dispatch: React.Dispatch<AIAction>
} | null>(null)

export function AIProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(aiReducer, initialState)

  return (
    <AIContext.Provider value={{ state, dispatch }}>
      {children}
    </AIContext.Provider>
  )
}

export function useAI() {
  const context = useContext(AIContext)
  if (!context) {
    throw new Error('useAI must be used within an AIProvider')
  }
  return context
}
```

### Error Boundaries for AI Operations

```typescript
// components/AIErrorBoundary.tsx
import React from 'react'

interface Props {
  children: React.ReactNode
  fallback?: React.ComponentType<{ error: Error; retry: () => void }>
  onError?: (error: Error) => void
}

interface State {
  hasError: boolean
  error: Error | null
}

export class AIErrorBoundary extends React.Component<Props, State> {
  constructor(props: Props) {
    super(props)
    this.state = { hasError: false, error: null }
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error }
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    console.error('AI Error Boundary caught an error:', error, errorInfo)
    this.props.onError?.(error)
  }

  retry = () => {
    this.setState({ hasError: false, error: null })
  }

  render() {
    if (this.state.hasError && this.state.error) {
      const Fallback = this.props.fallback || DefaultAIFallback
      return <Fallback error={this.state.error} retry={this.retry} />
    }

    return this.props.children
  }
}

function DefaultAIFallback({ error, retry }: { error: Error; retry: () => void }) {
  return (
    <div className="ai-error-fallback bg-red-50 border border-red-200 rounded-lg p-4 m-4">
      <div className="flex">
        <div className="flex-shrink-0">
          <svg className="h-5 w-5 text-red-400" viewBox="0 0 20 20" fill="currentColor">
            <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
          </svg>
        </div>
        <div className="ml-3">
          <h3 className="text-sm font-medium text-red-800">
            AI Feature Error
          </h3>
          <div className="mt-2 text-sm text-red-700">
            <p>{error.message}</p>
          </div>
          <div className="mt-4">
            <div className="-mx-2 -my-1.5 flex">
              <button
                onClick={retry}
                className="bg-red-50 px-2 py-1.5 rounded-md text-sm font-medium text-red-800 hover:bg-red-100 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-red-50 focus:ring-red-600"
              >
                Retry
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
```

### Loading States and Skeleton Screens

```typescript
// components/AILoading.tsx
import { useEffect, useState } from 'react'

interface AILoadingProps {
  type?: 'spinner' | 'pulse' | 'typing' | 'shimmer'
  message?: string
  size?: 'sm' | 'md' | 'lg'
}

export default function AILoading({
  type = 'spinner',
  message = 'AI is thinking...',
  size = 'md'
}: AILoadingProps) {
  const [dots, setDots] = useState('')

  useEffect(() => {
    if (type === 'typing') {
      const interval = setInterval(() => {
        setDots(prev => prev.length >= 3 ? '' : prev + '.')
      }, 500)
      return () => clearInterval(interval)
    }
  }, [type])

  const sizeClasses = {
    sm: 'w-4 h-4',
    md: 'w-8 h-8',
    lg: 'w-12 h-12'
  }

  if (type === 'spinner') {
    return (
      <div className="flex flex-col items-center justify-center p-4">
        <div className={`${sizeClasses[size]} border-4 border-ai-primary border-t-transparent rounded-full animate-spin`}></div>
        <p className="mt-2 text-sm text-gray-600">{message}</p>
      </div>
    )
  }

  if (type === 'pulse') {
    return (
      <div className="flex flex-col items-center justify-center p-4">
        <div className={`${sizeClasses[size]} bg-ai-primary rounded-full animate-pulse`}></div>
        <p className="mt-2 text-sm text-gray-600">{message}</p>
      </div>
    )
  }

  if (type === 'typing') {
    return (
      <div className="flex flex-col items-center justify-center p-4">
        <div className="flex space-x-1">
          <div className="w-2 h-2 bg-ai-primary rounded-full animate-bounce"></div>
          <div className="w-2 h-2 bg-ai-primary rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
          <div className="w-2 h-2 bg-ai-primary rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
        </div>
        <p className="mt-2 text-sm text-gray-600">{message}{dots}</p>
      </div>
    )
  }

  if (type === 'shimmer') {
    return (
      <div className="animate-pulse">
        <div className="h-4 bg-gray-200 rounded w-3/4 mb-2"></div>
        <div className="h-4 bg-gray-200 rounded w-1/2 mb-2"></div>
        <div className="h-4 bg-gray-200 rounded w-5/6"></div>
        <p className="mt-2 text-xs text-gray-500">{message}</p>
      </div>
    )
  }

  return null
}

// Skeleton components for different AI content types
export function AISkeleton({ type }: { type: 'chat' | 'image' | 'text' }) {
  if (type === 'chat') {
    return (
      <div className="space-y-4 p-4">
        <div className="flex items-center space-x-2">
          <div className="w-8 h-8 bg-gray-200 rounded-full animate-pulse"></div>
          <div className="h-4 bg-gray-200 rounded w-24 animate-pulse"></div>
        </div>
        <div className="space-y-2 ml-10">
          <div className="h-4 bg-gray-200 rounded w-full animate-pulse"></div>
          <div className="h-4 bg-gray-200 rounded w-3/4 animate-pulse"></div>
        </div>
      </div>
    )
  }

  if (type === 'image') {
    return (
      <div className="w-full h-64 bg-gray-200 rounded-lg animate-pulse flex items-center justify-center">
        <svg className="w-12 h-12 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
        </svg>
      </div>
    )
  }

  // Default text skeleton
  return (
    <div className="space-y-2">
      <div className="h-4 bg-gray-200 rounded w-full animate-pulse"></div>
      <div className="h-4 bg-gray-200 rounded w-5/6 animate-pulse"></div>
      <div className="h-4 bg-gray-200 rounded w-4/6 animate-pulse"></div>
    </div>
  )
}
```

### Optimizing AI Component Performance

```typescript
// hooks/useAIOptimization.ts
import { useCallback, useRef, useEffect } from 'react'

interface CacheEntry {
  key: string
  data: any
  timestamp: number
  ttl: number
}

export function useAICache(maxSize: number = 100) {
  const cache = useRef<Map<string, CacheEntry>>(new Map())

  const get = useCallback((key: string) => {
    const entry = cache.current.get(key)
    if (!entry) return null

    // Check if expired
    if (Date.now() - entry.timestamp > entry.ttl) {
      cache.current.delete(key)
      return null
    }

    return entry.data
  }, [])

  const set = useCallback((key: string, data: any, ttl: number = 300000) => { // 5 minutes default
    if (cache.current.size >= maxSize) {
      // Remove oldest entry
      const firstKey = cache.current.keys().next().value
      cache.current.delete(firstKey)
    }

    cache.current.set(key, {
      key,
      data,
      timestamp: Date.now(),
      ttl
    })
  }, [maxSize])

  const clear = useCallback(() => {
    cache.current.clear()
  }, [])

  return { get, set, clear }
}

// Debounced AI requests
export function useAIDebounce(delay: number = 500) {
  const timeoutRef = useRef<NodeJS.Timeout>()

  const debounce = useCallback((func: Function) => {
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current)
    }

    timeoutRef.current = setTimeout(func, delay)
  }, [delay])

  useEffect(() => {
    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current)
      }
    }
  }, [])

  return debounce
}

// AI request batching
export function useAIBatch() {
  const batchRef = useRef<any[]>([])
  const timeoutRef = useRef<NodeJS.Timeout>()

  const addToBatch = useCallback((request: any) => {
    batchRef.current.push(request)

    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current)
    }

    timeoutRef.current = setTimeout(() => {
      // Process batch
      processBatch(batchRef.current)
      batchRef.current = []
    }, 100) // 100ms batching window
  }, [])

  return addToBatch
}

async function processBatch(requests: any[]) {
  // Implement batch processing logic
  console.log(`Processing batch of ${requests.length} AI requests`)
}
```

### Real-time AI Communication Patterns

```typescript
// hooks/useRealtimeAI.ts
import { useEffect, useRef, useState } from 'react'
import { io, Socket } from 'socket.io-client'

interface RealtimeAIMessage {
  id: string
  type: 'text' | 'image' | 'audio'
  content: any
  timestamp: Date
}

export function useRealtimeAI(endpoint: string) {
  const [connected, setConnected] = useState(false)
  const [messages, setMessages] = useState<RealtimeAIMessage[]>([])
  const socketRef = useRef<Socket>()

  useEffect(() => {
    // Connect to real-time AI service
    socketRef.current = io(endpoint, {
      transports: ['websocket', 'polling']
    })

    socketRef.current.on('connect', () => {
      setConnected(true)
    })

    socketRef.current.on('disconnect', () => {
      setConnected(false)
    })

    socketRef.current.on('ai_response', (message: RealtimeAIMessage) => {
      setMessages(prev => [...prev, message])
    })

    socketRef.current.on('ai_error', (error: any) => {
      console.error('Real-time AI error:', error)
    })

    return () => {
      socketRef.current?.disconnect()
    }
  }, [endpoint])

  const sendMessage = (message: Partial<RealtimeAIMessage>) => {
    if (socketRef.current && connected) {
      socketRef.current.emit('user_message', {
        ...message,
        timestamp: new Date()
      })
    }
  }

  const clearMessages = () => {
    setMessages([])
  }

  return {
    connected,
    messages,
    sendMessage,
    clearMessages
  }
}
```

### Implementing the Complete Architecture

```typescript
// pages/_app.tsx
import type { AppProps } from 'next/app'
import { AIProvider } from '@/contexts/AIContext'
import { AIErrorBoundary } from '@/components/AIErrorBoundary'
import '@/styles/globals.css'

export default function App({ Component, pageProps }: AppProps) {
  return (
    <AIErrorBoundary>
      <AIProvider>
        <Component {...pageProps} />
      </AIProvider>
    </AIErrorBoundary>
  )
}
```

```typescript
// components/layout/AILayout.tsx
import { ReactNode } from 'react'
import { useAI } from '@/contexts/AIContext'
import AILoading from '@/components/AILoading'

interface AILayoutProps {
  children: ReactNode
  aiRequired?: boolean
  loadingMessage?: string
}

export default function AILayout({
  children,
  aiRequired = false,
  loadingMessage = 'Initializing AI features...'
}: AILayoutProps) {
  const { state } = useAI()

  // Show loading if AI is required but not ready
  if (aiRequired && state.activeOperations.length > 0) {
    return <AILoading message={loadingMessage} />
  }

  // Show errors if any AI operations failed
  if (state.errors.length > 0) {
    return (
      <div className="ai-error-container p-4 bg-red-50 border border-red-200 rounded-lg">
        <h3 className="text-red-800 font-semibold">AI Service Issues</h3>
        <ul className="mt-2 text-red-700">
          {state.errors.map((error, index) => (
            <li key={index} className="text-sm">• {error}</li>
          ))}
        </ul>
      </div>
    )
  }

  return (
    <div className="ai-layout">
      {/* AI capability indicators */}
      <div className="ai-status-bar fixed top-0 right-0 p-2 flex space-x-2">
        {Object.entries(state.capabilities).map(([capability, enabled]) => (
          <div
            key={capability}
            className={`w-2 h-2 rounded-full ${
              enabled ? 'bg-green-500' : 'bg-gray-300'
            }`}
            title={`${capability}: ${enabled ? 'enabled' : 'disabled'}`}
          />
        ))}
      </div>

      {children}
    </div>
  )
}
```

## Best Practices

### 1. Graceful Degradation
Always ensure your application works without AI features:

```typescript
// utils/featureDetection.ts
export const AIFeatures = {
  async checkChatSupport(): Promise<boolean> {
    try {
      // Test OpenAI API
      const response = await fetch('/api/ai/chat/test')
      return response.ok
    } catch {
      return false
    }
  },

  async checkVisionSupport(): Promise<boolean> {
    try {
      // Check WebGL support for TensorFlow.js
      const canvas = document.createElement('canvas')
      const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl')
      return !!gl
    } catch {
      return false
    }
  },

  async checkSpeechSupport(): Promise<boolean> {
    return !!(window.SpeechRecognition || (window as any).webkitSpeechRecognition)
  }
}
```

### 2. Performance Monitoring

```typescript
// utils/performanceMonitor.ts
export class AIPerformanceMonitor {
  private metrics: Map<string, number[]> = new Map()

  startTiming(operation: string): () => void {
    const start = performance.now()

    return () => {
      const duration = performance.now() - start
      this.recordMetric(operation, duration)
    }
  }

  recordMetric(operation: string, value: number) {
    if (!this.metrics.has(operation)) {
      this.metrics.set(operation, [])
    }

    const values = this.metrics.get(operation)!
    values.push(value)

    // Keep only last 100 measurements
    if (values.length > 100) {
      values.shift()
    }
  }

  getStats(operation: string) {
    const values = this.metrics.get(operation) || []
    if (values.length === 0) return null

    return {
      count: values.length,
      avg: values.reduce((a, b) => a + b, 0) / values.length,
      min: Math.min(...values),
      max: Math.max(...values),
      p95: values.sort((a, b) => a - b)[Math.floor(values.length * 0.95)]
    }
  }

  getAllStats() {
    const stats: Record<string, any> = {}
    for (const [operation, values] of this.metrics.entries()) {
      stats[operation] = this.getStats(operation)
    }
    return stats
  }
}

// Global performance monitor
export const aiPerformanceMonitor = new AIPerformanceMonitor()
```

### 3. Memory Management

```typescript
// utils/memoryManager.ts
export class AIMemoryManager {
  private cache = new Map<string, { data: any; size: number; lastUsed: number }>()
  private maxMemoryUsage = 50 * 1024 * 1024 // 50MB

  estimateSize(obj: any): number {
    // Rough estimation of object size in bytes
    const str = JSON.stringify(obj)
    return str.length * 2 // UTF-16 encoding
  }

  set(key: string, data: any) {
    const size = this.estimateSize(data)
    const currentUsage = Array.from(this.cache.values())
      .reduce((total, item) => total + item.size, 0)

    // Evict old entries if needed
    while (currentUsage + size > this.maxMemoryUsage && this.cache.size > 0) {
      const oldestKey = Array.from(this.cache.entries())
        .sort(([,a], [,b]) => a.lastUsed - b.lastUsed)[0][0]
      this.cache.delete(oldestKey)
    }

    this.cache.set(key, {
      data,
      size,
      lastUsed: Date.now()
    })
  }

  get(key: string): any | undefined {
    const entry = this.cache.get(key)
    if (entry) {
      entry.lastUsed = Date.now()
      return entry.data
    }
  }

  clear() {
    this.cache.clear()
  }

  getMemoryUsage() {
    return Array.from(this.cache.values())
      .reduce((total, item) => total + item.size, 0)
  }
}
```

## Summary

Modern frontend architecture for AI applications requires:

1. **Asynchronous Operation Handling**: Use custom hooks and proper loading states
2. **Progressive Enhancement**: Design apps that work without AI and enhance with it
3. **Error Boundaries**: Implement comprehensive error handling for AI operations
4. **State Management**: Use context providers for AI state and capabilities
5. **Performance Optimization**: Cache results, debounce requests, and monitor performance
6. **Memory Management**: Implement intelligent caching and cleanup
7. **Real-time Communication**: Support WebSocket connections for streaming AI responses

This architecture ensures your AI-integrated frontend is robust, performant, and provides excellent user experience even when AI services are unavailable or slow.

## Next Steps

- [Tutorial 02: Natural Language Processing Frontend](../tutorials/02-nlp-frontend.md)
- [Workshop 01: Basic AI Chat Interface](../workshops/workshop-01-basic-ai-chat.md)

## Resources

- [React Patterns for AI Applications](https://example.com/react-ai-patterns)
- [Next.js Performance Optimization](https://nextjs.org/docs/advanced-features/measuring-performance)
- [Web Performance API](https://developer.mozilla.org/en-US/docs/Web/API/Performance_API)
- [React Error Boundaries](https://reactjs.org/docs/error-boundaries.html)
# Workshop 01: Basic AI Chat Interface

## Overview
This workshop implements a complete AI chat interface using modern React patterns and OpenAI's API. You'll build a responsive chat UI with real-time messaging, typing indicators, message history, and proper error handling.

## Prerequisites
- Completed [Modern Frontend with AI Integration Tutorial](../tutorials/01-modern-frontend-ai.md)
- OpenAI API key configured
- Basic React and TypeScript knowledge

## Learning Objectives
- Implement real-time AI chat interface
- Handle streaming responses from AI APIs
- Create responsive chat UI components
- Manage chat state and message history
- Implement proper error handling and loading states

## Implementation Steps

### Step 1: Create Chat Components

#### Message Component

```typescript
// components/chat/MessageBubble.tsx
import { memo } from 'react'
import { ChatMessage } from '@/types/ai'

interface MessageBubbleProps {
  message: ChatMessage
  isUser: boolean
}

function MessageBubble({ message, isUser }: MessageBubbleProps) {
  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'} mb-4`}>
      <div
        className={`max-w-xs lg:max-w-md px-4 py-2 rounded-lg ${
          isUser
            ? 'bg-ai-primary text-white rounded-br-none'
            : 'bg-gray-100 text-gray-800 rounded-bl-none'
        }`}
      >
        <p className="text-sm whitespace-pre-wrap">{message.content}</p>
        <p className={`text-xs mt-1 ${isUser ? 'text-blue-100' : 'text-gray-500'}`}>
          {message.timestamp.toLocaleTimeString()}
        </p>
      </div>
    </div>
  )
}

export default memo(MessageBubble)
```

#### Typing Indicator Component

```typescript
// components/chat/TypingIndicator.tsx
import { memo } from 'react'

interface TypingIndicatorProps {
  isTyping: boolean
  user?: string
}

function TypingIndicator({ isTyping, user = 'AI' }: TypingIndicatorProps) {
  if (!isTyping) return null

  return (
    <div className="flex justify-start mb-4">
      <div className="bg-gray-100 px-4 py-2 rounded-lg rounded-bl-none max-w-xs">
        <div className="flex items-center space-x-2">
          <div className="flex space-x-1">
            <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
            <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
            <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
          </div>
          <span className="text-xs text-gray-500">{user} is typing...</span>
        </div>
      </div>
    </div>
  )
}

export default memo(TypingIndicator)
```

#### Message Input Component

```typescript
// components/chat/MessageInput.tsx
import { useState, KeyboardEvent } from 'react'

interface MessageInputProps {
  onSendMessage: (message: string) => void
  disabled?: boolean
  placeholder?: string
}

export default function MessageInput({
  onSendMessage,
  disabled = false,
  placeholder = "Type your message..."
}: MessageInputProps) {
  const [message, setMessage] = useState('')

  const handleSend = () => {
    if (message.trim() && !disabled) {
      onSendMessage(message.trim())
      setMessage('')
    }
  }

  const handleKeyPress = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  return (
    <div className="border-t bg-white p-4">
      <div className="flex items-end space-x-2">
        <textarea
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder={placeholder}
          disabled={disabled}
          className="flex-1 resize-none rounded-lg border border-gray-300 px-3 py-2 focus:outline-none focus:ring-2 focus:ring-ai-primary focus:border-transparent disabled:opacity-50 disabled:cursor-not-allowed"
          rows={1}
          style={{ minHeight: '40px', maxHeight: '120px' }}
          onInput={(e) => {
            const target = e.target as HTMLTextAreaElement
            target.style.height = 'auto'
            target.style.height = Math.min(target.scrollHeight, 120) + 'px'
          }}
        />
        <button
          onClick={handleSend}
          disabled={!message.trim() || disabled}
          className="bg-ai-primary hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed text-white px-4 py-2 rounded-lg transition-colors duration-200 focus:outline-none focus:ring-2 focus:ring-ai-primary focus:ring-offset-2"
        >
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
          </svg>
        </button>
      </div>
    </div>
  )
}
```

#### Chat Container Component

```typescript
// components/chat/ChatContainer.tsx
import { useState, useRef, useEffect } from 'react'
import MessageBubble from './MessageBubble'
import TypingIndicator from './TypingIndicator'
import MessageInput from './MessageInput'
import { ChatMessage } from '@/types/ai'

interface ChatContainerProps {
  messages: ChatMessage[]
  onSendMessage: (message: string) => Promise<void>
  isTyping?: boolean
  disabled?: boolean
}

export default function ChatContainer({
  messages,
  onSendMessage,
  isTyping = false,
  disabled = false
}: ChatContainerProps) {
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const [inputDisabled, setInputDisabled] = useState(disabled)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages, isTyping])

  useEffect(() => {
    setInputDisabled(disabled || isTyping)
  }, [disabled, isTyping])

  const handleSendMessage = async (message: string) => {
    setInputDisabled(true)
    try {
      await onSendMessage(message)
    } finally {
      setInputDisabled(false)
    }
  }

  return (
    <div className="flex flex-col h-full bg-gray-50">
      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.length === 0 ? (
          <div className="flex items-center justify-center h-full text-gray-500">
            <div className="text-center">
              <svg className="w-12 h-12 mx-auto mb-4 text-gray-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
              </svg>
              <p className="text-lg font-medium">Start a conversation</p>
              <p className="text-sm">Send a message to begin chatting with AI</p>
            </div>
          </div>
        ) : (
          <>
            {messages.map((message) => (
              <MessageBubble
                key={message.id}
                message={message}
                isUser={message.role === 'user'}
              />
            ))}
            <TypingIndicator isTyping={isTyping} />
            <div ref={messagesEndRef} />
          </>
        )}
      </div>

      {/* Input Area */}
      <MessageInput
        onSendMessage={handleSendMessage}
        disabled={inputDisabled}
        placeholder={isTyping ? "AI is responding..." : "Type your message..."}
      />
    </div>
  )
}
```

### Step 2: Create AI Chat Hook

```typescript
// hooks/useAIChat.ts
import { useState, useCallback, useRef } from 'react'
import { ChatMessage } from '@/types/ai'
import { aiPerformanceMonitor } from '@/utils/performanceMonitor'

export function useAIChat() {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [isTyping, setIsTyping] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const abortControllerRef = useRef<AbortController | null>(null)

  const sendMessage = useCallback(async (content: string) => {
    if (!content.trim()) return

    // Cancel any ongoing request
    if (abortControllerRef.current) {
      abortControllerRef.current.abort()
    }

    const userMessage: ChatMessage = {
      id: `user-${Date.now()}-${Math.random()}`,
      role: 'user',
      content: content.trim(),
      timestamp: new Date()
    }

    // Add user message
    setMessages(prev => [...prev, userMessage])
    setIsTyping(true)
    setError(null)

    // Create new abort controller
    abortControllerRef.current = new AbortController()

    try {
      const endTiming = aiPerformanceMonitor.startTiming('ai_chat_request')

      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          messages: [...messages, userMessage].map(msg => ({
            role: msg.role,
            content: msg.content
          }))
        }),
        signal: abortControllerRef.current.signal
      })

      if (!response.ok) {
        throw new Error(`API request failed: ${response.status}`)
      }

      const data = await response.json()
      endTiming()

      const aiMessage: ChatMessage = {
        id: `ai-${Date.now()}-${Math.random()}`,
        role: 'assistant',
        content: data.response,
        timestamp: new Date(),
        metadata: {
          model: data.model,
          tokens: data.tokens,
          processing_time: data.processing_time
        }
      }

      setMessages(prev => [...prev, aiMessage])

    } catch (err) {
      if (err instanceof Error && err.name === 'AbortError') {
        // Request was cancelled, ignore
        return
      }

      const errorMessage = err instanceof Error ? err.message : 'Unknown error occurred'
      setError(errorMessage)

      // Add error message to chat
      const errorChatMessage: ChatMessage = {
        id: `error-${Date.now()}-${Math.random()}`,
        role: 'assistant',
        content: `Sorry, I encountered an error: ${errorMessage}`,
        timestamp: new Date(),
        metadata: { type: 'error' }
      }

      setMessages(prev => [...prev, errorChatMessage])

    } finally {
      setIsTyping(false)
      abortControllerRef.current = null
    }
  }, [messages])

  const clearChat = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort()
    }
    setMessages([])
    setError(null)
    setIsTyping(false)
  }, [])

  const retryLastMessage = useCallback(() => {
    const lastUserMessage = [...messages].reverse().find(msg => msg.role === 'user')
    if (lastUserMessage) {
      sendMessage(lastUserMessage.content)
    }
  }, [messages, sendMessage])

  return {
    messages,
    isTyping,
    error,
    sendMessage,
    clearChat,
    retryLastMessage,
    canRetry: messages.some(msg => msg.role === 'user')
  }
}
```

### Step 3: Create API Route for Chat

```typescript
// pages/api/chat.ts
import type { NextApiRequest, NextApiResponse } from 'next'
import OpenAI from 'openai'

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY
})

interface ChatRequest {
  messages: Array<{
    role: 'user' | 'assistant' | 'system'
    content: string
  }>
  model?: string
  temperature?: number
  max_tokens?: number
}

interface ChatResponse {
  response: string
  model: string
  tokens: number
  processing_time: number
}

export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse<ChatResponse | { error: string }>
) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' })
  }

  try {
    const { messages, model = 'gpt-3.5-turbo', temperature = 0.7, max_tokens = 1000 }: ChatRequest = req.body

    if (!messages || !Array.isArray(messages)) {
      return res.status(400).json({ error: 'Messages array is required' })
    }

    if (!process.env.OPENAI_API_KEY) {
      return res.status(500).json({ error: 'OpenAI API key not configured' })
    }

    const startTime = Date.now()

    // Call OpenAI API
    const response = await openai.chat.completions.create({
      model,
      messages,
      temperature,
      max_tokens,
      stream: false // We'll implement streaming later
    })

    const processingTime = Date.now() - startTime

    const aiResponse = response.choices[0].message.content || 'No response generated'

    res.status(200).json({
      response: aiResponse,
      model,
      tokens: response.usage?.total_tokens || 0,
      processing_time: processingTime
    })

  } catch (error) {
    console.error('Chat API error:', error)

    if (error instanceof Error) {
      // Handle specific OpenAI errors
      if (error.message.includes('insufficient_quota')) {
        return res.status(429).json({ error: 'API quota exceeded' })
      }

      if (error.message.includes('invalid_api_key')) {
        return res.status(401).json({ error: 'Invalid API key' })
      }
    }

    res.status(500).json({
      error: 'Internal server error'
    })
  }
}
```

### Step 4: Create Main Chat Page

```typescript
// pages/chat.tsx
import { useState, useEffect } from 'react'
import Layout from '@/components/layout/Layout'
import ChatContainer from '@/components/chat/ChatContainer'
import AILoading from '@/components/AILoading'
import { useAIChat } from '@/hooks/useAIChat'

export default function ChatPage() {
  const { messages, isTyping, error, sendMessage, clearChat, retryLastMessage, canRetry } = useAIChat()
  const [isClient, setIsClient] = useState(false)

  useEffect(() => {
    setIsClient(true)
  }, [])

  if (!isClient) {
    return (
      <Layout>
        <AILoading message="Loading chat interface..." />
      </Layout>
    )
  }

  return (
    <Layout title="AI Chat">
      <div className="h-screen flex flex-col">
        {/* Header */}
        <div className="bg-white border-b px-4 py-3 flex items-center justify-between">
          <div>
            <h1 className="text-xl font-semibold text-gray-900">AI Chat Assistant</h1>
            <p className="text-sm text-gray-500">Powered by OpenAI GPT</p>
          </div>

          <div className="flex items-center space-x-2">
            {canRetry && (
              <button
                onClick={retryLastMessage}
                className="text-sm text-ai-primary hover:text-blue-700 px-3 py-1 rounded-md hover:bg-blue-50 transition-colors"
                disabled={isTyping}
              >
                Retry
              </button>
            )}

            <button
              onClick={clearChat}
              className="text-sm text-gray-600 hover:text-gray-800 px-3 py-1 rounded-md hover:bg-gray-50 transition-colors"
              disabled={isTyping || messages.length === 0}
            >
              Clear Chat
            </button>
          </div>
        </div>

        {/* Error Display */}
        {error && (
          <div className="bg-red-50 border-l-4 border-red-400 p-4 mx-4 mt-4">
            <div className="flex">
              <div className="flex-shrink-0">
                <svg className="h-5 w-5 text-red-400" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                </svg>
              </div>
              <div className="ml-3">
                <p className="text-sm text-red-700">{error}</p>
              </div>
            </div>
          </div>
        )}

        {/* Chat Interface */}
        <div className="flex-1 min-h-0">
          <ChatContainer
            messages={messages}
            onSendMessage={sendMessage}
            isTyping={isTyping}
            disabled={false}
          />
        </div>
      </div>
    </Layout>
  )
}
```

### Step 5: Add Streaming Support (Advanced)

```typescript
// hooks/useAIChatStreaming.ts
import { useState, useCallback, useRef } from 'react'
import { ChatMessage } from '@/types/ai'

export function useAIChatStreaming() {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [isTyping, setIsTyping] = useState(false)
  const [currentStreamingMessage, setCurrentStreamingMessage] = useState<string>('')
  const abortControllerRef = useRef<AbortController | null>(null)

  const sendMessage = useCallback(async (content: string) => {
    if (!content.trim()) return

    // Cancel any ongoing request
    if (abortControllerRef.current) {
      abortControllerRef.current.abort()
    }

    const userMessage: ChatMessage = {
      id: `user-${Date.now()}-${Math.random()}`,
      role: 'user',
      content: content.trim(),
      timestamp: new Date()
    }

    // Add user message
    setMessages(prev => [...prev, userMessage])
    setIsTyping(true)
    setCurrentStreamingMessage('')

    // Create new abort controller
    abortControllerRef.current = new AbortController()

    try {
      const response = await fetch('/api/chat/stream', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          messages: [...messages, userMessage].map(msg => ({
            role: msg.role,
            content: msg.content
          }))
        }),
        signal: abortControllerRef.current.signal
      })

      if (!response.ok) {
        throw new Error(`API request failed: ${response.status}`)
      }

      const reader = response.body?.getReader()
      if (!reader) throw new Error('No response body')

      let accumulatedContent = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        const chunk = new TextDecoder().decode(value)
        const lines = chunk.split('\n')

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = line.slice(6)
            if (data === '[DONE]') continue

            try {
              const parsed = JSON.parse(data)
              if (parsed.choices && parsed.choices[0].delta?.content) {
                accumulatedContent += parsed.choices[0].delta.content
                setCurrentStreamingMessage(accumulatedContent)
              }
            } catch (e) {
              // Ignore parsing errors for incomplete chunks
            }
          }
        }
      }

      // Add completed AI message
      const aiMessage: ChatMessage = {
        id: `ai-${Date.now()}-${Math.random()}`,
        role: 'assistant',
        content: accumulatedContent,
        timestamp: new Date()
      }

      setMessages(prev => [...prev, aiMessage])
      setCurrentStreamingMessage('')

    } catch (err) {
      if (err instanceof Error && err.name === 'AbortError') {
        return
      }

      const errorMessage = err instanceof Error ? err.message : 'Unknown error occurred'
      const errorChatMessage: ChatMessage = {
        id: `error-${Date.now()}-${Math.random()}`,
        role: 'assistant',
        content: `Sorry, I encountered an error: ${errorMessage}`,
        timestamp: new Date(),
        metadata: { type: 'error' }
      }

      setMessages(prev => [...prev, errorChatMessage])
      setCurrentStreamingMessage('')

    } finally {
      setIsTyping(false)
      abortControllerRef.current = null
    }
  }, [messages])

  const streamingMessage: ChatMessage | null = currentStreamingMessage ? {
    id: 'streaming',
    role: 'assistant',
    content: currentStreamingMessage,
    timestamp: new Date(),
    metadata: { streaming: true }
  } : null

  const displayMessages = streamingMessage
    ? [...messages, streamingMessage]
    : messages

  return {
    messages: displayMessages,
    isTyping,
    sendMessage,
    clearChat: () => {
      setMessages([])
      setCurrentStreamingMessage('')
    }
  }
}
```

### Step 6: Create Streaming API Route

```typescript
// pages/api/chat/stream.ts
import type { NextApiRequest, NextApiResponse } from 'next'
import OpenAI from 'openai'

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY
})

export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse
) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' })
  }

  try {
    const { messages, model = 'gpt-3.5-turbo', temperature = 0.7 } = req.body

    if (!messages || !Array.isArray(messages)) {
      return res.status(400).json({ error: 'Messages array is required' })
    }

    // Set up SSE headers
    res.writeHead(200, {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      'Connection': 'keep-alive',
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Headers': 'Cache-Control',
    })

    // Create streaming response
    const stream = await openai.chat.completions.create({
      model,
      messages,
      temperature,
      stream: true
    })

    for await (const chunk of stream) {
      const content = chunk.choices[0]?.delta?.content || ''
      if (content) {
        res.write(`data: ${JSON.stringify(chunk)}\n\n`)
      }
    }

    res.write('data: [DONE]\n\n')
    res.end()

  } catch (error) {
    console.error('Streaming chat API error:', error)
    res.write(`data: ${JSON.stringify({ error: 'Internal server error' })}\n\n`)
    res.end()
  }
}

export const config = {
  api: {
    bodyParser: {
      sizeLimit: '10mb',
    },
  },
}
```

### Step 7: Add Chat History and Persistence

```typescript
// utils/chatStorage.ts
import { ChatMessage } from '@/types/ai'

const CHAT_STORAGE_KEY = 'ai_chat_history'
const MAX_STORED_CHATS = 10
const MAX_MESSAGES_PER_CHAT = 100

export class ChatStorage {
  static saveChat(sessionId: string, messages: ChatMessage[]) {
    try {
      const existingChats = this.getAllChats()
      const chatData = {
        sessionId,
        messages: messages.slice(-MAX_MESSAGES_PER_CHAT), // Keep only recent messages
        lastUpdated: new Date().toISOString(),
        messageCount: messages.length
      }

      // Update or add chat
      const existingIndex = existingChats.findIndex(chat => chat.sessionId === sessionId)
      if (existingIndex >= 0) {
        existingChats[existingIndex] = chatData
      } else {
        existingChats.unshift(chatData) // Add to beginning
      }

      // Keep only recent chats
      const trimmedChats = existingChats.slice(0, MAX_STORED_CHATS)

      localStorage.setItem(CHAT_STORAGE_KEY, JSON.stringify(trimmedChats))
    } catch (error) {
      console.warn('Failed to save chat:', error)
    }
  }

  static getChat(sessionId: string): ChatMessage[] | null {
    try {
      const chats = this.getAllChats()
      const chat = chats.find(c => c.sessionId === sessionId)
      return chat ? chat.messages : null
    } catch (error) {
      console.warn('Failed to load chat:', error)
      return null
    }
  }

  static getAllChats(): Array<{
    sessionId: string
    messages: ChatMessage[]
    lastUpdated: string
    messageCount: number
  }> {
    try {
      const stored = localStorage.getItem(CHAT_STORAGE_KEY)
      return stored ? JSON.parse(stored) : []
    } catch (error) {
      console.warn('Failed to load chats:', error)
      return []
    }
  }

  static deleteChat(sessionId: string) {
    try {
      const chats = this.getAllChats().filter(chat => chat.sessionId !== sessionId)
      localStorage.setItem(CHAT_STORAGE_KEY, JSON.stringify(chats))
    } catch (error) {
      console.warn('Failed to delete chat:', error)
    }
  }

  static clearAllChats() {
    try {
      localStorage.removeItem(CHAT_STORAGE_KEY)
    } catch (error) {
      console.warn('Failed to clear chats:', error)
    }
  }
}
```

## Testing the Implementation

### Step 1: Create Test Components

```typescript
// components/chat/__tests__/MessageBubble.test.tsx
import { render, screen } from '@testing-library/react'
import MessageBubble from '../MessageBubble'
import { ChatMessage } from '@/types/ai'

const mockMessage: ChatMessage = {
  id: '1',
  role: 'user',
  content: 'Hello AI!',
  timestamp: new Date('2024-01-15T10:00:00')
}

describe('MessageBubble', () => {
  it('renders user message correctly', () => {
    render(<MessageBubble message={mockMessage} isUser={true} />)

    expect(screen.getByText('Hello AI!')).toBeInTheDocument()
    expect(screen.getByText('10:00:00 AM')).toBeInTheDocument()
  })

  it('renders AI message correctly', () => {
    const aiMessage = { ...mockMessage, role: 'assistant' }
    render(<MessageBubble message={aiMessage} isUser={false} />)

    expect(screen.getByText('Hello AI!')).toBeInTheDocument()
  })
})
```

### Step 2: Integration Testing

```typescript
// __tests__/integration/chat.integration.test.ts
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import ChatPage from '@/pages/chat'

// Mock the API
global.fetch = jest.fn()

describe('Chat Integration', () => {
  beforeEach(() => {
    (global.fetch as jest.Mock).mockClear()
  })

  it('sends message and receives response', async () => {
    const user = userEvent.setup()

    // Mock API response
    ;(global.fetch as jest.Mock).mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        response: 'Hello! How can I help you?',
        model: 'gpt-3.5-turbo',
        tokens: 10,
        processing_time: 500
      })
    })

    render(<ChatPage />)

    // Type message
    const input = screen.getByPlaceholderText('Type your message...')
    await user.type(input, 'Hello AI')

    // Send message
    const sendButton = screen.getByRole('button')
    await user.click(sendButton)

    // Check user message appears
    expect(screen.getByText('Hello AI')).toBeInTheDocument()

    // Wait for AI response
    await waitFor(() => {
      expect(screen.getByText('Hello! How can I help you?')).toBeInTheDocument()
    })
  })

  it('handles API errors gracefully', async () => {
    const user = userEvent.setup()

    // Mock API error
    ;(global.fetch as jest.Mock).mockRejectedValueOnce(new Error('Network error'))

    render(<ChatPage />)

    const input = screen.getByPlaceholderText('Type your message...')
    await user.type(input, 'Test message')

    const sendButton = screen.getByRole('button')
    await user.click(sendButton)

    // Check error message appears
    await waitFor(() => {
      expect(screen.getByText(/Sorry, I encountered an error/)).toBeInTheDocument()
    })
  })
})
```

## Deployment Considerations

### Step 1: Environment Variables

```bash
# .env.production
OPENAI_API_KEY=your_production_api_key
NEXT_PUBLIC_APP_ENV=production
NEXT_PUBLIC_API_URL=https://your-api-domain.com
```

### Step 2: API Rate Limiting

```typescript
// middleware/rateLimit.ts
import { NextRequest, NextResponse } from 'next/server'

const RATE_LIMIT_WINDOW = 60 * 1000 // 1 minute
const MAX_REQUESTS = 10 // requests per window

const requests = new Map<string, { count: number; resetTime: number }>()

export function rateLimit(request: NextRequest): NextResponse | null {
  const ip = request.ip || 'unknown'
  const now = Date.now()

  const userRequests = requests.get(ip)

  if (!userRequests || now > userRequests.resetTime) {
    requests.set(ip, { count: 1, resetTime: now + RATE_LIMIT_WINDOW })
    return null
  }

  if (userRequests.count >= MAX_REQUESTS) {
    return NextResponse.json(
      { error: 'Rate limit exceeded' },
      { status: 429 }
    )
  }

  userRequests.count++
  return null
}
```

## Summary

You've built a complete AI chat interface with:

1. **Real-time messaging** with typing indicators
2. **Streaming responses** for better user experience
3. **Error handling** and retry mechanisms
4. **Message persistence** using local storage
5. **Responsive design** that works on all devices
6. **API integration** with OpenAI
7. **Comprehensive testing** for reliability

## Key Features Implemented

- ✅ Real-time chat interface
- ✅ AI API integration
- ✅ Message streaming
- ✅ Error handling
- ✅ Responsive design
- ✅ Message persistence
- ✅ Loading states
- ✅ TypeScript support
- ✅ Unit and integration tests

## Next Steps

- [Tutorial 02: Natural Language Processing Frontend](../tutorials/02-nlp-frontend.md)
- [Workshop 02: Image Analysis Web App](../workshops/workshop-02-image-analysis.md)

## Resources

- [OpenAI API Documentation](https://platform.openai.com/docs/api-reference/chat)
- [Next.js API Routes](https://nextjs.org/docs/api-routes/introduction)
- [Server-Sent Events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events)
- [React Testing Library](https://testing-library.com/docs/react-testing-library/intro/)
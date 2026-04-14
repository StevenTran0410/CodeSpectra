import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import type { QAResponse, DeepResearchResponse } from '../types/electron'

export interface QAChatMessage {
  role: 'user' | 'assistant'
  content: string
  response?: QAResponse
  deepResearchResponse?: DeepResearchResponse
  error?: string
}

export interface QAConversation {
  id: string
  name: string
  snapshotId: string
  repoId: string
  messages: QAChatMessage[]
  createdAt: string
}

interface QAStore {
  conversations: QAConversation[]
  activeId: string | null
  deepResearchDisabledIds: string[]

  createConversation: (snapshotId: string, repoId: string) => string
  deleteConversation: (id: string) => void
  renameConversation: (id: string, name: string) => void
  updateConversationSnapshot: (id: string, snapshotId: string, repoId: string) => void
  addMessage: (id: string, msg: QAChatMessage) => void
  truncateMessages: (id: string, fromIndex: number) => void
  setActive: (id: string) => void
  disableDeepResearch: (conversationId: string) => void
  isDeepResearchDisabled: (conversationId: string) => boolean
}

const generateId = () => Math.random().toString(36).substring(2, 11)

export const useQAStore = create<QAStore>()(
  persist(
    (set, get) => ({
      conversations: [],
      activeId: null,
      deepResearchDisabledIds: [],

      createConversation: (snapshotId, repoId) => {
        const id = generateId()
        set((state) => ({
          conversations: [
            ...state.conversations,
            {
              id,
              name: 'New conversation',
              snapshotId,
              repoId,
              messages: [],
              createdAt: new Date().toISOString(),
            },
          ],
          activeId: id,
        }))
        return id
      },

      deleteConversation: (id) => {
        set((state) => {
          const conversations = state.conversations.filter((c) => c.id !== id)
          const activeId =
            state.activeId === id
              ? conversations.length > 0
                ? conversations[conversations.length - 1].id
                : null
              : state.activeId
          return { conversations, activeId }
        })
      },

      renameConversation: (id, name) => {
        set((state) => ({
          conversations: state.conversations.map((c) =>
            c.id === id ? { ...c, name } : c
          ),
        }))
      },

      updateConversationSnapshot: (id, snapshotId, repoId) => {
        set((state) => ({
          conversations: state.conversations.map((c) =>
            c.id === id ? { ...c, snapshotId, repoId } : c
          ),
        }))
      },

      addMessage: (id, msg) => {
        set((state) => {
          const conversations = state.conversations.map((c) => {
            if (c.id !== id) return c
            const updatedMessages = [...c.messages, msg]
            const updatedConvo = { ...c, messages: updatedMessages }

            // Auto-name only if user hasn't customised the name yet
            if (c.messages.length === 0 && msg.role === 'user' && c.name === 'New conversation') {
              const firstLine = msg.content.split('\n')[0]
              updatedConvo.name = firstLine.substring(0, 40)
            }

            return updatedConvo
          })
          return { conversations }
        })
      },

      truncateMessages: (id, fromIndex) => {
        set((state) => ({
          conversations: state.conversations.map((c) =>
            c.id === id ? { ...c, messages: c.messages.slice(0, fromIndex) } : c
          ),
        }))
      },

      setActive: (id) => set({ activeId: id }),

      disableDeepResearch: (conversationId) =>
        set((state) => ({
          deepResearchDisabledIds: state.deepResearchDisabledIds.includes(conversationId)
            ? state.deepResearchDisabledIds
            : [...state.deepResearchDisabledIds, conversationId],
        })),

      isDeepResearchDisabled: (conversationId) =>
        get().deepResearchDisabledIds.includes(conversationId),
    }),
    {
      name: 'codespectra.qa.v1',
    }
  )
)

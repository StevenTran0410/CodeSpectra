import { create } from 'zustand'
import type {
  ProviderConfig,
  CreateProviderRequest,
  UpdateProviderRequest
} from '../types/electron'

interface TestResult {
  ok: boolean
  message: string
}

interface ProviderState {
  providers: ProviderConfig[]
  loading: boolean
  error: string | null
  /** Map of providerId → test result (cleared when config changes) */
  testResults: Record<string, TestResult>
  /** Map of providerId → boolean (is test in-flight) */
  testing: Record<string, boolean>
  /** Map of providerId → string[] (available models from live fetch) */
  modelLists: Record<string, string[]>
  loadingModels: Record<string, boolean>

  load: () => Promise<void>
  create: (req: CreateProviderRequest) => Promise<ProviderConfig>
  update: (id: string, req: UpdateProviderRequest) => Promise<ProviderConfig>
  remove: (id: string) => Promise<void>
  testConnection: (id: string) => Promise<TestResult>
  fetchModels: (id: string) => Promise<string[]>
  clearError: () => void
}

export const useProviderStore = create<ProviderState>((set, get) => ({
  providers: [],
  loading: false,
  error: null,
  testResults: {},
  testing: {},
  modelLists: {},
  loadingModels: {},

  load: async () => {
    set({ loading: true, error: null })
    try {
      const providers = await window.api.provider.list()
      set({ providers, loading: false })
    } catch (err) {
      set({ error: String(err), loading: false })
    }
  },

  create: async (req) => {
    const provider = await window.api.provider.create(req)
    set((s) => ({ providers: [...s.providers, provider] }))
    return provider
  },

  update: async (id, req) => {
    const updated = await window.api.provider.update(id, req)
    set((s) => ({
      providers: s.providers.map((p) => (p.id === id ? updated : p)),
      // clear stale test results when config changes
      testResults: { ...s.testResults, [id]: undefined as unknown as TestResult }
    }))
    return updated
  },

  remove: async (id) => {
    await window.api.provider.delete(id)
    set((s) => ({
      providers: s.providers.filter((p) => p.id !== id),
      testResults: Object.fromEntries(
        Object.entries(s.testResults).filter(([k]) => k !== id)
      ),
      modelLists: Object.fromEntries(
        Object.entries(s.modelLists).filter(([k]) => k !== id)
      )
    }))
  },

  testConnection: async (id) => {
    set((s) => ({ testing: { ...s.testing, [id]: true } }))
    try {
      const result = await window.api.provider.test(id)
      set((s) => ({
        testing: { ...s.testing, [id]: false },
        testResults: { ...s.testResults, [id]: result }
      }))
      return result
    } catch (err) {
      const result: TestResult = { ok: false, message: String(err) }
      set((s) => ({
        testing: { ...s.testing, [id]: false },
        testResults: { ...s.testResults, [id]: result }
      }))
      return result
    }
  },

  fetchModels: async (id) => {
    set((s) => ({ loadingModels: { ...s.loadingModels, [id]: true } }))
    try {
      const { models } = await window.api.provider.models(id)
      set((s) => ({
        loadingModels: { ...s.loadingModels, [id]: false },
        modelLists: { ...s.modelLists, [id]: models }
      }))
      return models
    } catch {
      set((s) => ({ loadingModels: { ...s.loadingModels, [id]: false } }))
      return get().modelLists[id] ?? []
    }
  },

  clearError: () => set({ error: null })
}))

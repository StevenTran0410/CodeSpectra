import { create } from 'zustand'
import type {
  DeviceFlowStart,
  GitHubAccount,
  GitHubRepo,
} from '../types/electron'

// ──────────────────────────────────────────────────────────────────────────────
// State shape
// ──────────────────────────────────────────────────────────────────────────────

interface GitHubState {
  // Account
  account: GitHubAccount | null
  loadingAccount: boolean

  // Device flow (connection wizard)
  deviceFlow: DeviceFlowStart | null
  connecting: boolean
  connectStatus: 'idle' | 'waiting' | 'success' | 'expired' | 'denied' | 'error'
  connectError: string | null

  // Repositories
  repos: GitHubRepo[]
  repoPage: number
  repoHasMore: boolean
  loadingRepos: boolean
  repoQuery: string
  repoError: string | null

  // Actions
  loadAccount: () => Promise<void>
  startConnect: () => Promise<void>
  pollConnect: () => Promise<void>
  cancelConnect: () => void
  disconnect: () => Promise<void>
  loadRepos: (query?: string, page?: number) => Promise<void>
  loadMoreRepos: () => Promise<void>
  clearRepoError: () => void
}

// ──────────────────────────────────────────────────────────────────────────────
// Store
// ──────────────────────────────────────────────────────────────────────────────

export const useGitHubStore = create<GitHubState>((set, get) => ({
  account: null,
  loadingAccount: false,
  deviceFlow: null,
  connecting: false,
  connectStatus: 'idle',
  connectError: null,
  repos: [],
  repoPage: 1,
  repoHasMore: false,
  loadingRepos: false,
  repoQuery: '',
  repoError: null,

  // ── Account ──────────────────────────────────────────────────────────────

  loadAccount: async () => {
    set({ loadingAccount: true })
    try {
      const account = await window.api.github.getAccount()
      set({ account, loadingAccount: false })
    } catch {
      set({ account: null, loadingAccount: false })
    }
  },

  // ── Connection wizard ────────────────────────────────────────────────────

  startConnect: async () => {
    set({ connecting: true, connectStatus: 'waiting', connectError: null, deviceFlow: null })
    try {
      const flow = await window.api.github.startDeviceFlow()
      set({ deviceFlow: flow })
      // Open browser automatically
      await window.api.github.openBrowser(flow.verification_uri)
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err)
      set({ connecting: false, connectStatus: 'error', connectError: msg })
    }
  },

  pollConnect: async () => {
    const { deviceFlow } = get()
    if (!deviceFlow) return
    try {
      const result = await window.api.github.pollDeviceFlow(deviceFlow.device_code)
      if (result.status === 'success' && result.account) {
        set({
          account: result.account,
          connecting: false,
          connectStatus: 'success',
          deviceFlow: null,
        })
        // Auto-load repos on connect
        get().loadRepos()
      } else if (result.status === 'expired') {
        set({ connecting: false, connectStatus: 'expired', deviceFlow: null })
      } else if (result.status === 'denied') {
        set({ connecting: false, connectStatus: 'denied', deviceFlow: null })
      } else if (result.status === 'error') {
        set({ connecting: false, connectStatus: 'error', connectError: 'Authorization failed', deviceFlow: null })
      }
      // "pending" and "slow_down" → caller keeps polling
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err)
      set({ connecting: false, connectStatus: 'error', connectError: msg, deviceFlow: null })
    }
  },

  cancelConnect: () => {
    set({ connecting: false, connectStatus: 'idle', connectError: null, deviceFlow: null })
  },

  disconnect: async () => {
    try {
      await window.api.github.disconnect()
      set({ account: null, repos: [], repoPage: 1, repoHasMore: false, connectStatus: 'idle' })
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err)
      set({ repoError: msg })
    }
  },

  // ── Repositories ─────────────────────────────────────────────────────────

  loadRepos: async (query?: string, page = 1) => {
    set({ loadingRepos: true, repoError: null })
    if (page === 1) set({ repos: [] })
    const effectiveQuery = query ?? get().repoQuery
    if (query !== undefined) set({ repoQuery: query })
    try {
      const result = await window.api.github.listRepos(effectiveQuery || undefined, page)
      set((s) => ({
        repos: page === 1 ? result.repos : [...s.repos, ...result.repos],
        repoPage: result.page,
        repoHasMore: result.has_more,
        loadingRepos: false,
      }))
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err)
      set({ repoError: msg, loadingRepos: false })
    }
  },

  loadMoreRepos: async () => {
    const { repoPage, repoHasMore, loadingRepos } = get()
    if (!repoHasMore || loadingRepos) return
    get().loadRepos(undefined, repoPage + 1)
  },

  clearRepoError: () => set({ repoError: null }),
}))

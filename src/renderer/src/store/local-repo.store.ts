import { create } from 'zustand'
import type { LocalRepo, ValidateFolderResponse } from '../types/electron'

interface LocalRepoState {
  repos: LocalRepo[]
  loading: boolean
  error: string | null
  validating: boolean
  validation: ValidateFolderResponse | null
  adding: boolean
  revalidatingId: string | null
  branchesMap: Record<string, string[]>     // repo_id → branch list
  loadingBranchesId: string | null

  load: () => Promise<void>
  validate: (path: string) => Promise<ValidateFolderResponse | null>
  clearValidation: () => void
  add: (path: string) => Promise<LocalRepo | null>
  remove: (id: string) => Promise<void>
  revalidate: (id: string) => Promise<void>
  loadBranches: (id: string, refresh?: boolean) => Promise<void>
  setBranch: (id: string, branch: string) => Promise<void>
  clearError: () => void
}

export const useLocalRepoStore = create<LocalRepoState>((set, get) => ({
  repos: [],
  loading: false,
  error: null,
  validating: false,
  validation: null,
  adding: false,
  revalidatingId: null,
  branchesMap: {},
  loadingBranchesId: null,

  load: async () => {
    set({ loading: true, error: null })
    try {
      const repos = await window.api.folder.list()
      set({ repos, loading: false })
    } catch (err) {
      set({ loading: false, error: String(err) })
    }
  },

  validate: async (path: string) => {
    set({ validating: true, validation: null, error: null })
    try {
      const result = await window.api.folder.validate(path)
      set({ validating: false, validation: result })
      return result
    } catch (err) {
      set({ validating: false, error: String(err) })
      return null
    }
  },

  clearValidation: () => set({ validation: null }),

  add: async (path: string) => {
    set({ adding: true, error: null })
    try {
      const repo = await window.api.folder.add(path)
      set((s) => ({ repos: [...s.repos, repo], adding: false, validation: null }))
      return repo
    } catch (err) {
      set({ adding: false, error: String(err) })
      return null
    }
  },

  remove: async (id: string) => {
    try {
      await window.api.folder.remove(id)
      set((s) => ({ repos: s.repos.filter((r) => r.id !== id) }))
    } catch (err) {
      set({ error: String(err) })
    }
  },

  revalidate: async (id: string) => {
    set({ revalidatingId: id })
    try {
      const updated = await window.api.folder.revalidate(id)
      set((s) => ({
        repos: s.repos.map((r) => (r.id === id ? updated : r)),
        revalidatingId: null,
      }))
    } catch (err) {
      set({ revalidatingId: null, error: String(err) })
    }
  },

  loadBranches: async (id: string, refresh = false) => {
    set({ loadingBranchesId: id })
    try {
      const branches = await window.api.folder.branches(id, refresh)
      set((s) => ({ branchesMap: { ...s.branchesMap, [id]: branches }, loadingBranchesId: null }))
    } catch (err) {
      set({ loadingBranchesId: null, error: String(err) })
    }
  },

  setBranch: async (id: string, branch: string) => {
    try {
      const updated = await window.api.folder.setBranch(id, branch)
      set((s) => ({ repos: s.repos.map((r) => (r.id === id ? updated : r)) }))
    } catch (err) {
      set({ error: String(err) })
    }
  },

  clearError: () => set({ error: null }),
}))

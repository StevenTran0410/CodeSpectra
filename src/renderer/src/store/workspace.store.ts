import { create } from 'zustand'
import type { Workspace } from '../types/electron'

interface WorkspaceState {
  workspaces: Workspace[]
  activeWorkspaceId: string | null
  isLoading: boolean
  error: string | null

  load: () => Promise<void>
  create: (name: string) => Promise<Workspace>
  rename: (id: string, name: string) => Promise<void>
  remove: (id: string) => Promise<void>
  setActive: (id: string | null) => void
  clearError: () => void
}

export const useWorkspaceStore = create<WorkspaceState>((set, get) => ({
  workspaces: [],
  activeWorkspaceId: null,
  isLoading: false,
  error: null,

  load: async () => {
    set({ isLoading: true, error: null })
    try {
      const workspaces = await window.api.workspace.list()
      const current = get().activeWorkspaceId
      set({
        workspaces,
        isLoading: false,
        activeWorkspaceId:
          current && workspaces.find((w) => w.id === current) ? current : (workspaces[0]?.id ?? null)
      })
    } catch (err) {
      set({ isLoading: false, error: String(err) })
    }
  },

  create: async (name: string) => {
    const workspace = await window.api.workspace.create(name)
    set((s) => ({
      workspaces: [...s.workspaces, workspace],
      activeWorkspaceId: s.activeWorkspaceId ?? workspace.id
    }))
    return workspace
  },

  rename: async (id: string, name: string) => {
    const updated = await window.api.workspace.rename(id, name)
    set((s) => ({
      workspaces: s.workspaces.map((w) => (w.id === id ? updated : w))
    }))
  },

  remove: async (id: string) => {
    await window.api.workspace.delete(id)
    set((s) => {
      const workspaces = s.workspaces.filter((w) => w.id !== id)
      const activeWorkspaceId =
        s.activeWorkspaceId === id ? (workspaces[0]?.id ?? null) : s.activeWorkspaceId
      return { workspaces, activeWorkspaceId }
    })
  },

  setActive: (id) => set({ activeWorkspaceId: id }),
  clearError: () => set({ error: null })
}))

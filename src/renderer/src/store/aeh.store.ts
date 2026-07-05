import { create } from 'zustand'

interface AEHState {
  isActive: boolean
  isLoading: boolean
  port: number | null
  error: string | null

  startAEH: () => Promise<number>
  setActive: (active: boolean) => void
  showView: (bounds: { x: number; y: number; width: number; height: number }) => Promise<void>
  hideView: () => Promise<void>
  resizeView: (bounds: { x: number; y: number; width: number; height: number }) => Promise<void>
}

export const useAEHStore = create<AEHState>((set, get) => ({
  isActive: false,
  isLoading: false,
  port: null,
  error: null,

  startAEH: async () => {
    const existingPort = get().port
    if (existingPort !== null) return existingPort

    set({ isLoading: true, error: null })
    try {
      const port = await window.api.aeh.start()
      set({ port, isLoading: false })
      return port
    } catch (err) {
      set({ isLoading: false, error: String(err) })
      throw err
    }
  },

  setActive: (active) => set({ isActive: active }),

  showView: async (bounds) => {
    await window.api.aeh.showView(bounds)
  },

  hideView: async () => {
    await window.api.aeh.hideView()
  },

  resizeView: async (bounds) => {
    await window.api.aeh.resizeView(bounds)
  }
}))

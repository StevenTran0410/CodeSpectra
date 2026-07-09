import { create } from 'zustand'

type BackendStatus = 'running' | 'restarting' | 'dead'

interface AppBackendState {
  backendStatus: BackendStatus
  restartAttempt: number
  setBackendStatus: (status: BackendStatus) => void
  setRestartAttempt: (attempt: number) => void
  incrementRestartAttempt: () => void
  resetRestartAttempt: () => void
}

export const useAppBackendStore = create<AppBackendState>((set) => ({
  backendStatus: 'running',
  restartAttempt: 0,

  setBackendStatus: (status: BackendStatus) => set({ backendStatus: status }),
  setRestartAttempt: (attempt: number) => set({ restartAttempt: attempt }),

  incrementRestartAttempt: () =>
    set((state) => ({
      backendStatus: 'restarting',
      restartAttempt: state.restartAttempt + 1,
    })),

  resetRestartAttempt: () =>
    set({
      backendStatus: 'running',
      restartAttempt: 0,
    }),
}))

// Initialize IPC event listeners on module load
let listenersAttached = false

export function attachBackendListeners(): void {
  if (listenersAttached || typeof window === 'undefined') return

  try {
    const electronAPI = (window as any)?.electron
    if (!electronAPI?.ipcRenderer) return

    // Listen for backend crash event from main process
    electronAPI.ipcRenderer.on('app:backend-died', (_event: unknown, data: unknown) => {
      const store = useAppBackendStore.getState()
      store.incrementRestartAttempt()
    })

    // Listen for backend recovery event from main process
    electronAPI.ipcRenderer.on('backend:ready', () => {
      const store = useAppBackendStore.getState()
      store.resetRestartAttempt()
    })

    listenersAttached = true
  } catch (error) {
    console.warn('[app-backend.store] Failed to attach IPC listeners:', error)
  }
}

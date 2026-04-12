import { create } from 'zustand'

export interface Toast {
  id: string
  type: 'success' | 'error' | 'info' | 'warning'
  message: string
  duration: number
}

interface ToastStore {
  toasts: Toast[]
  push: (toast: Omit<Toast, 'id'>) => void
  dismiss: (id: string) => void
  success: (message: string, duration?: number) => void
  error: (message: string, duration?: number) => void
  info: (message: string, duration?: number) => void
  warning: (message: string, duration?: number) => void
}

const timers = new Map<string, ReturnType<typeof setTimeout>>()

export const useToastStore = create<ToastStore>((set) => ({
  toasts: [],

  push: (toast) => {
    const id = Date.now().toString() + Math.random().toString(36).slice(2, 9)
    const newToast: Toast = {
      ...toast,
      id
    }

    set((state) => {
      let toasts = [...state.toasts, newToast]
      if (toasts.length > 3) {
        const removed = toasts.shift()
        if (removed) {
          const timer = timers.get(removed.id)
          if (timer) {
            clearTimeout(timer)
            timers.delete(removed.id)
          }
        }
      }
      return { toasts }
    })

    const timer = setTimeout(() => {
      set((state) => ({
        toasts: state.toasts.filter((t) => t.id !== id)
      }))
      timers.delete(id)
    }, toast.duration || 4000)

    timers.set(id, timer)
  },

  dismiss: (id) => {
    const timer = timers.get(id)
    if (timer) {
      clearTimeout(timer)
      timers.delete(id)
    }
    set((state) => ({
      toasts: state.toasts.filter((t) => t.id !== id)
    }))
  },

  success: (message, duration = 4000) => {
    set((state) => state) // trigger push through store
    ;(useToastStore.getState() as any).push({ type: 'success', message, duration })
  },

  error: (message, duration = 4000) => {
    ;(useToastStore.getState() as any).push({ type: 'error', message, duration })
  },

  info: (message, duration = 4000) => {
    ;(useToastStore.getState() as any).push({ type: 'info', message, duration })
  },

  warning: (message, duration = 4000) => {
    ;(useToastStore.getState() as any).push({ type: 'warning', message, duration })
  }
}))

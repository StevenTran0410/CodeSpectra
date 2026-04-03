import { create } from 'zustand'
import type { Job } from '../types/electron'

interface JobState {
  // active job being polled
  activeJob: Job | null
  polling: boolean

  // recent job history
  history: Job[]
  historyLoading: boolean

  // actions
  startPolling: (jobId: string) => void
  stopPolling: () => void
  cancel: (jobId: string) => Promise<void>
  loadHistory: (repoId?: string) => Promise<void>
  clearActive: () => void
}

let _pollInterval: ReturnType<typeof setInterval> | null = null

export const useJobStore = create<JobState>((set, get) => ({
  activeJob: null,
  polling: false,
  history: [],
  historyLoading: false,

  startPolling: (jobId) => {
    if (_pollInterval) clearInterval(_pollInterval)

    const poll = async () => {
      try {
        const job = await window.api.job.get(jobId)
        set({ activeJob: job })
        if (['done', 'failed', 'cancelled'].includes(job.status)) {
          get().stopPolling()
          get().loadHistory()
        }
      } catch {
        get().stopPolling()
      }
    }

    poll()
    _pollInterval = setInterval(poll, 800)
    set({ polling: true })
  },

  stopPolling: () => {
    if (_pollInterval) { clearInterval(_pollInterval); _pollInterval = null }
    set({ polling: false })
  },

  cancel: async (jobId) => {
    await window.api.job.cancel(jobId)
    get().stopPolling()
    const job = await window.api.job.get(jobId)
    set({ activeJob: job })
  },

  loadHistory: async (repoId) => {
    set({ historyLoading: true })
    try {
      const jobs = repoId
        ? await window.api.job.listForRepo(repoId)
        : await window.api.job.listRecent()
      set({ history: jobs })
    } finally {
      set({ historyLoading: false })
    }
  },

  clearActive: () => {
    get().stopPolling()
    set({ activeJob: null })
  },
}))

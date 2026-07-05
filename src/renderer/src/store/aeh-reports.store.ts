import { create } from 'zustand'
import type {
  AEHRunListItem,
  AEHRunDetailResponse,
  AEHEvaluationDetailItem,
  AEHTraceDetailResponse,
  AEHProviderSummary
} from '../types/electron'

interface AEHReportsStore {
  runs: AEHRunListItem[]
  selectedRun: AEHRunDetailResponse | null
  evaluations: AEHEvaluationDetailItem[]
  traceData: AEHTraceDetailResponse | null
  availableProviders: AEHProviderSummary[]
  compareRunA: AEHRunDetailResponse | null
  compareRunB: AEHRunDetailResponse | null
  compareEvalsA: AEHEvaluationDetailItem[]
  compareEvalsB: AEHEvaluationDetailItem[]
  loading: boolean
  error: string | null

  fetchRunsList: () => Promise<void>
  fetchRunDetail: (runId: string) => Promise<void>
  fetchComponentEvaluations: (runId: string, componentId: string) => Promise<void>
  fetchTraceDetail: (traceId: string) => Promise<void>
  fetchProviders: () => Promise<void>
  fetchComparison: (runIdA: string, runIdB: string) => Promise<void>
  triggerRerun: (runId: string, body: { model_overrides: Record<string, string>; active_defects: string[] }) => Promise<{ run_id: string }>
}

export const useAEHReportsStore = create<AEHReportsStore>((set) => ({
  runs: [],
  selectedRun: null,
  evaluations: [],
  traceData: null,
  availableProviders: [],
  compareRunA: null,
  compareRunB: null,
  compareEvalsA: [],
  compareEvalsB: [],
  loading: false,
  error: null,

  fetchRunsList: async () => {
    set({ loading: true, error: null })
    try {
      const data = await window.api.aeh.listRuns()
      set({ runs: data, loading: false })
    } catch (err: any) {
      set({ error: err.message || String(err), loading: false })
    }
  },

  fetchRunDetail: async (runId: string) => {
    set({ loading: true, error: null })
    try {
      const data = await window.api.aeh.runDetail(runId)
      set({ selectedRun: data, loading: false })
    } catch (err: any) {
      set({ error: err.message || String(err), loading: false })
    }
  },

  fetchComponentEvaluations: async (runId: string, componentId: string) => {
    set({ loading: true, error: null })
    try {
      const runMeta = await window.api.aeh.runDetail(runId)
      const evals = await window.api.aeh.componentEvaluations(runId, componentId)
      set({ selectedRun: runMeta, evaluations: evals, loading: false })
    } catch (err: any) {
      set({ error: err.message || String(err), loading: false })
    }
  },

  fetchTraceDetail: async (traceId: string) => {
    set({ loading: true, error: null })
    try {
      const data = await window.api.aeh.traceDetail(traceId)
      set({ traceData: data, loading: false })
    } catch (err: any) {
      set({ error: err.message || String(err), loading: false })
    }
  },

  fetchProviders: async () => {
    try {
      const providers = await window.api.aeh.providers()
      set({ availableProviders: providers })
    } catch (err) {
      console.error('Failed to fetch providers:', err)
    }
  },

  fetchComparison: async (runIdA: string, runIdB: string) => {
    set({ loading: true, error: null })
    try {
      const [dataA, dataB] = await Promise.all([
        window.api.aeh.runDetail(runIdA),
        window.api.aeh.runDetail(runIdB)
      ])

      const compsA = Object.keys(dataA.component_aggregates)
      const evalsPromisesA = compsA.map(cid =>
        window.api.aeh.componentEvaluations(runIdA, cid)
      )
      const evalsPromisesB = compsA.map(cid =>
        window.api.aeh.componentEvaluations(runIdB, cid)
      )

      const evalsA = await Promise.all(evalsPromisesA)
      const evalsB = await Promise.all(evalsPromisesB)

      set({
        compareRunA: dataA,
        compareRunB: dataB,
        compareEvalsA: evalsA.flat(),
        compareEvalsB: evalsB.flat(),
        loading: false
      })
    } catch (err: any) {
      set({ error: err.message || String(err), loading: false })
    }
  },

  triggerRerun: async (runId: string, body) => {
    return await window.api.aeh.rerun(runId, body)
  }
}))

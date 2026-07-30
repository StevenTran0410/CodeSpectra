import { create } from 'zustand'

export interface CaseFlags {
  soft_fail_fields: string[]
  expected_suspect: boolean
  input_starved: boolean
  schema_suspect_fields: string[]
  reasons: string[]
}

export interface AgentVerdictInfo {
  verdict: 'healthy' | 'likely-dataset' | 'likely-agent' | 'mixed'
  d_count: number
  a_count: number
  low_count: number
}

export interface EvalCaseMetric {
  metric_name: string
  metric_class: string
  score: number | null
  details: Record<string, unknown>
}

export interface EvalCaseItem {
  case_id: string | null
  trace_id: string
  latency_ms?: number | null
  input: string | null
  result: string | null
  expected: unknown
  evaluations: EvalCaseMetric[]
  flags?: CaseFlags
}

export interface AgentSummaryInfo {
  insight: string
  case_count: number
  avg_score: number | null
}

export interface EvalRunCasesData {
  run_id: string
  status: string
  agents: Record<string, EvalCaseItem[]>
  agent_summaries: Record<string, AgentSummaryInfo>
  agent_verdicts?: Record<string, AgentVerdictInfo>
}

export interface AEHEvalRunSummary {
  id: string
  target_system_id: string
  system_name: string
  framework: string | null
  started_at: string
  finished_at: string | null
  status: string
  case_count: number
  scored_count: number
  agent_count: number
  avg_semantic_match: number | null
  field_exact_pass_rate: number | null
  avg_precision_recall: number | null
  total_latency_ms: number
  semantic_match_histogram: number[]
  judged_status: 'judged' | 'partially_judged' | 'not_judged'
  cases_data?: EvalRunCasesData
}

export function buildRunSummaryFromCases(
  r: {
    id: string
    target_system_id: string
    started_at: string
    finished_at: string | null
    status: string
  },
  casesData: EvalRunCasesData | null,
  siblingInfo?: { name?: string; framework?: string | null } | null
): AEHEvalRunSummary {
  if (!casesData) {
    return {
      id: r.id,
      target_system_id: r.target_system_id,
      system_name: siblingInfo?.name || r.target_system_id,
      framework: siblingInfo?.framework || null,
      started_at: r.started_at,
      finished_at: r.finished_at,
      status: r.status,
      case_count: 0,
      scored_count: 0,
      agent_count: 0,
      avg_semantic_match: null,
      field_exact_pass_rate: null,
      avg_precision_recall: null,
      total_latency_ms: 0,
      semantic_match_histogram: [0, 0, 0, 0, 0],
      judged_status: 'not_judged'
    }
  }

  const agentIds = Object.keys(casesData.agents || {})
  const allCases: EvalCaseItem[] = []
  for (const aid of agentIds) {
    const list = casesData.agents[aid] || []
    allCases.push(...list)
  }

  const caseCount = allCases.length
  let scoredCount = 0
  const smScores: number[] = []
  const prScores: number[] = []
  const feScores: number[] = []
  let totalLatency = 0

  const histogram = [0, 0, 0, 0, 0]

  for (const c of allCases) {
    if (c.evaluations && c.evaluations.length > 0) {
      scoredCount++
    }
    // Latency is on the trace (total_latency_ms), surfaced per-case by the cases endpoint —
    // not on the judge evaluations, so it must be summed from the case, not from ev.details.
    if (typeof c.latency_ms === 'number') {
      totalLatency += c.latency_ms
    }
    for (const ev of c.evaluations || []) {

      if (ev.metric_name === 'semantic_match' && typeof ev.score === 'number') {
        smScores.push(ev.score)
        const s = ev.score
        if (s < 0.3) histogram[0]++
        else if (s < 0.5) histogram[1]++
        else if (s < 0.7) histogram[2]++
        else if (s < 0.9) histogram[3]++
        else histogram[4]++
      } else if (ev.metric_name.startsWith('precision_recall.') && typeof ev.score === 'number') {
        prScores.push(ev.score)
      } else if (ev.metric_name.startsWith('field_exact.') && typeof ev.score === 'number') {
        feScores.push(ev.score >= 0.999 ? 1.0 : 0.0)
      }
    }
  }

  const avgSemanticMatch =
    smScores.length > 0 ? smScores.reduce((a, b) => a + b, 0) / smScores.length : null
  const avgPrecisionRecall =
    prScores.length > 0 ? prScores.reduce((a, b) => a + b, 0) / prScores.length : null
  const fieldExactPassRate =
    feScores.length > 0 ? feScores.reduce((a, b) => a + b, 0) / feScores.length : null

  let judgedStatus: 'judged' | 'partially_judged' | 'not_judged' = 'not_judged'
  if (scoredCount > 0) {
    judgedStatus = scoredCount >= caseCount ? 'judged' : 'partially_judged'
  }

  return {
    id: r.id,
    target_system_id: r.target_system_id,
    system_name: siblingInfo?.name || r.target_system_id,
    framework: siblingInfo?.framework || null,
    started_at: r.started_at,
    finished_at: r.finished_at,
    status: r.status,
    case_count: caseCount,
    scored_count: scoredCount,
    agent_count: agentIds.length,
    avg_semantic_match: avgSemanticMatch,
    field_exact_pass_rate: fieldExactPassRate,
    avg_precision_recall: avgPrecisionRecall,
    total_latency_ms: totalLatency,
    semantic_match_histogram: histogram,
    judged_status: judgedStatus,
    cases_data: casesData
  }
}

interface AEHReportsStore {
  runs: AEHEvalRunSummary[]
  selectedRunItem: AEHEvalRunSummary | null
  selectedRunCases: EvalRunCasesData | null
  traceData: { trace?: Record<string, unknown> | null; spans: any[] } | null
  compareRunItemA: AEHEvalRunSummary | null
  compareRunItemB: AEHEvalRunSummary | null
  compareCasesA: EvalRunCasesData | null
  compareCasesB: EvalRunCasesData | null
  loading: boolean
  error: string | null

  fetchRunsList: () => Promise<void>
  fetchRunCases: (runId: string) => Promise<void>
  fetchTraceDetail: (traceId: string) => Promise<void>
  fetchComparison: (runIdA: string, runIdB: string) => Promise<void>
  judgeAgentCases: (runId: string, agentId: string) => Promise<void>
  summarizeAgent: (runId: string, agentId: string) => Promise<void>
}

const siblingCache: Record<string, any[]> = {}

async function getCachedSiblingSystems(sessionId: string): Promise<any[]> {
  if (siblingCache[sessionId]) {
    return siblingCache[sessionId]
  }
  try {
    const res = await window.api.aeh.listSiblingSystems(sessionId)
    if (res && Array.isArray(res)) {
      siblingCache[sessionId] = res
    }
    return res || []
  } catch {
    return []
  }
}

export const useAEHReportsStore = create<AEHReportsStore>((set, get) => ({
  runs: [],
  selectedRunItem: null,
  selectedRunCases: null,
  traceData: null,
  compareRunItemA: null,
  compareRunItemB: null,
  compareCasesA: null,
  compareCasesB: null,
  loading: false,
  error: null,

  fetchRunsList: async () => {
    set({ loading: true, error: null })
    try {
      const rawRuns = await window.api.aeh.listRuns()
      const summaries: AEHEvalRunSummary[] = await Promise.all(
        (rawRuns || []).map(async (r) => {
          let casesData: EvalRunCasesData | null = null
          let siblingInfo: { name?: string; framework?: string | null } | null = null

          try {
            casesData = (await window.api.aeh.getEvalRunCases(r.id)) as EvalRunCasesData
          } catch {
            // Un-ingested or un-run
          }

          if (r.target_system_id) {
            try {
              const siblings = await getCachedSiblingSystems(r.target_system_id)
              if (siblings && siblings.length > 0) {
                const match = siblings.find((s) => s.session_id === r.target_system_id) || siblings[0]
                siblingInfo = { name: match.name, framework: match.framework }
              }
            } catch {
              // Ignore
            }
          }

          return buildRunSummaryFromCases(r, casesData, siblingInfo)
        })
      )

      set({ runs: summaries, loading: false })
    } catch (err: any) {
      set({ error: err.message || String(err), loading: false })
    }
  },

  fetchRunCases: async (runId: string) => {
    // Immediately clear previous run state so system-switch clears old content instantly!
    set({
      loading: true,
      error: null,
      selectedRunCases: null,
      selectedRunItem: null
    })
    try {
      const casesData = (await window.api.aeh.getEvalRunCases(runId)) as EvalRunCasesData
      const existingSummary = get().runs.find((r) => r.id === runId)

      let siblingInfo: { name?: string; framework?: string | null } | null = null
      const targetSys = existingSummary?.target_system_id || casesData.run_id
      if (targetSys) {
        try {
          const siblings = await getCachedSiblingSystems(targetSys)
          if (siblings && siblings.length > 0) {
            const match = siblings.find((s) => s.session_id === targetSys) || siblings[0]
            siblingInfo = { name: match.name, framework: match.framework }
          }
        } catch {
          // Ignore
        }
      }

      const summaryItem = buildRunSummaryFromCases(
        {
          id: casesData.run_id,
          target_system_id: existingSummary?.target_system_id || casesData.run_id,
          started_at: existingSummary?.started_at || new Date().toISOString(),
          finished_at: existingSummary?.finished_at || null,
          status: casesData.status
        },
        casesData,
        siblingInfo
      )

      set({
        selectedRunCases: casesData,
        selectedRunItem: summaryItem,
        loading: false
      })
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

  fetchComparison: async (runIdA: string, runIdB: string) => {
    set({ loading: true, error: null })
    try {
      const [casesA, casesB] = await Promise.all([
        window.api.aeh.getEvalRunCases(runIdA) as Promise<EvalRunCasesData>,
        window.api.aeh.getEvalRunCases(runIdB) as Promise<EvalRunCasesData>
      ])

      const itemA = buildRunSummaryFromCases(
        { id: runIdA, target_system_id: casesA.run_id, started_at: '', finished_at: null, status: casesA.status },
        casesA
      )
      const itemB = buildRunSummaryFromCases(
        { id: runIdB, target_system_id: casesB.run_id, started_at: '', finished_at: null, status: casesB.status },
        casesB
      )

      set({
        compareCasesA: casesA,
        compareCasesB: casesB,
        compareRunItemA: itemA,
        compareRunItemB: itemB,
        loading: false
      })
    } catch (err: any) {
      set({ error: err.message || String(err), loading: false })
    }
  },

  judgeAgentCases: async (runId: string, agentId: string) => {
    set({ loading: true, error: null })
    try {
      await window.api.aeh.judgeEvalRunCases(runId, { agent_id: agentId })
      await get().fetchRunCases(runId)
      await get().fetchRunsList()
    } catch (err: any) {
      set({ error: err.message || String(err), loading: false })
    }
  },

  summarizeAgent: async (runId: string, agentId: string) => {
    set({ loading: true, error: null })
    try {
      await window.api.aeh.summarizeEvalRunAgent(runId, { agent_id: agentId })
      await get().fetchRunCases(runId)
    } catch (err: any) {
      set({ error: err.message || String(err), loading: false })
    }
  }
}))

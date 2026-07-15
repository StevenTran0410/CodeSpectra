import React, { useEffect, useState } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'
import {
  Loader2,
  AlertCircle,
  ArrowLeft,
  CheckCircle2,
  Sparkles,
  BarChart3,
} from 'lucide-react'
import { Button, useToastStore } from '../../components/ui'
import { useProviderStore } from '../../store/provider.store'
import LLMConfigModal, { LLMModelButton } from './LLMConfigModal'

type CaseEvaluation = {
  metric_name: string
  metric_class: string
  score: number | null
  details: Record<string, unknown>
}

type CaseRow = {
  case_id: string | null
  trace_id: string
  input: string | null
  result: string | null
  expected: unknown
  evaluations: CaseEvaluation[]
}

function _scoreColor(score: number | null): string {
  if (score === null) return 'text-slate-400'
  if (score >= 0.7) return 'text-emerald-400'
  if (score >= 0.4) return 'text-amber-400'
  return 'text-rose-400'
}

type AgentSummary = {
  agentId: string
  scoredCount: number
  totalCount: number
  avgScore: number | null
  avgPrecision: number | null
  avgRecall: number | null
  worst: { row: CaseRow; score: number; notes: string } | null
  best: { row: CaseRow; score: number; notes: string } | null
}

function _computeAgentSummaries(agentCases: Record<string, CaseRow[]>): AgentSummary[] {
  return Object.keys(agentCases).sort().map((agentId) => {
    const rows = agentCases[agentId]
    const scored = rows
      .map((row) => ({ row, sm: row.evaluations.find((e) => e.metric_name === 'semantic_match') }))
      .filter((x): x is { row: CaseRow; sm: CaseEvaluation } => !!x.sm && x.sm.score !== null)
    const avgScore = scored.length > 0
      ? scored.reduce((sum, x) => sum + (x.sm.score ?? 0), 0) / scored.length
      : null
    const prAll = rows.flatMap((row) => row.evaluations.filter((e) => e.metric_name.startsWith('precision_recall.')))
    const avgPrecision = prAll.length > 0
      ? prAll.reduce((sum, e) => sum + ((e.details.precision as number) ?? 0), 0) / prAll.length
      : null
    const avgRecall = prAll.length > 0
      ? prAll.reduce((sum, e) => sum + ((e.details.recall as number) ?? 0), 0) / prAll.length
      : null
    const sorted = [...scored].sort((a, b) => (a.sm.score ?? 0) - (b.sm.score ?? 0))
    const toEntry = (x: typeof sorted[number]) => ({
      row: x.row, score: x.sm.score ?? 0, notes: String(x.sm.details?.notes ?? ''),
    })
    return {
      agentId,
      scoredCount: scored.length,
      totalCount: rows.length,
      avgScore,
      avgPrecision,
      avgRecall,
      worst: sorted.length > 0 ? toEntry(sorted[0]) : null,
      best: sorted.length > 0 ? toEntry(sorted[sorted.length - 1]) : null,
    }
  })
}

function _pretty(value: unknown): string {
  if (value === null || value === undefined) return '—'
  if (typeof value === 'string') {
    try {
      return JSON.stringify(JSON.parse(value), null, 2)
    } catch {
      return value
    }
  }
  return JSON.stringify(value, null, 2)
}

export default function Stage5Screen(): React.ReactElement {
  const [searchParams] = useSearchParams()
  const navigate = useNavigate()
  const toast = useToastStore()
  const sessionId = searchParams.get('sessionId') || ''

  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<{ run_id: string; status: string } | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [checkingExisting, setCheckingExisting] = useState(true)

  const [casesLoading, setCasesLoading] = useState(false)
  const [agentCases, setAgentCases] = useState<Record<string, CaseRow[]> | null>(null)
  const [viewingAgentId, setViewingAgentId] = useState<string | null>(null)
  const [showSummary, setShowSummary] = useState(false)

  const { providers, load: loadProviders } = useProviderStore()
  const [providerId, setProviderId] = useState('')
  const [modelId, setModelId] = useState('')
  const [reasoningEffort, setReasoningEffort] = useState<string | null>(null)
  const [thinkingBudget, setThinkingBudget] = useState<number | null>(null)
  const [llmConfigOpen, setLlmConfigOpen] = useState(false)
  const [judgeConfirmOpen, setJudgeConfirmOpen] = useState(false)
  const [judging, setJudging] = useState(false)
  const [judgeAllConfirmOpen, setJudgeAllConfirmOpen] = useState(false)
  const [judgingAll, setJudgingAll] = useState(false)
  const [judgeAllProgress, setJudgeAllProgress] = useState<{ done: number; total: number; agentId: string } | null>(null)

  const [agentSummaries, setAgentSummaries] = useState<Record<string, { insight: string; case_count: number; avg_score?: number | null }>>({})
  const [insightsConfirmOpen, setInsightsConfirmOpen] = useState(false)
  const [generatingInsights, setGeneratingInsights] = useState(false)
  const [insightsProgress, setInsightsProgress] = useState<{ done: number; total: number; agentId: string } | null>(null)

  useEffect(() => {
    loadProviders()
  }, [loadProviders])

  useEffect(() => {
    if (providers.length > 0 && !providerId) {
      setProviderId(providers[0].id)
      setModelId(providers[0].model_id ?? '')
    }
  }, [providers, providerId])

  // Recall an existing run from a previous visit instead of forcing a fresh "Load Results" click.
  useEffect(() => {
    if (!sessionId) {
      setCheckingExisting(false)
      return
    }
    let cancelled = false
    ;(async () => {
      try {
        const sess = await window.api.aeh.getExpansionSession(sessionId)
        if (cancelled || !sess.eval_run_id) return
        const data = await window.api.aeh.getEvalRunCases(sess.eval_run_id)
        if (cancelled) return
        setResult({ run_id: data.run_id, status: data.status })
        setAgentCases(data.agents)
        setAgentSummaries(data.agent_summaries ?? {})
      } catch {
        // No existing run, or it's no longer resolvable — fall through to the normal
        // "Load Results" button, same as a session that's never been loaded before.
      } finally {
        if (!cancelled) setCheckingExisting(false)
      }
    })()
    return () => {
      cancelled = true
    }
  }, [sessionId])

  async function handleLoadResults() {
    setLoading(true)
    setError(null)
    try {
      const data = await window.api.aeh.loadEvalResults(sessionId)
      setResult(data)
      toast.success('Eval results loaded successfully')
    } catch (err: any) {
      if (err?.message?.includes('manifest not found')) {
        setError(
          'No results yet. Start your CodeSpectra backend on the eval branch, ' +
          'run POST /aeh/run-eval from /docs, then click Load Results again.'
        )
        return
      }
      setError(err?.message ?? 'Failed to load eval results')
    } finally {
      setLoading(false)
    }
  }

  async function handleViewResults() {
    if (!result) return
    setCasesLoading(true)
    setError(null)
    try {
      const data = await window.api.aeh.getEvalRunCases(result.run_id)
      setAgentCases(data.agents)
      setAgentSummaries(data.agent_summaries ?? {})
    } catch (err: any) {
      setError(err?.message ?? 'Failed to load per-agent results')
    } finally {
      setCasesLoading(false)
    }
  }

  async function handleJudgeAgent() {
    if (!result || !viewingAgentId) return
    setJudgeConfirmOpen(false)
    setJudging(true)
    setError(null)
    try {
      const outcome = await window.api.aeh.judgeEvalRunCases(result.run_id, {
        agent_id: viewingAgentId,
        provider_id: providerId || null,
        model_id: modelId || null,
        reasoning_effort: reasoningEffort,
        thinking_budget: thinkingBudget,
      })
      toast.success(`Scored ${outcome.scored} case(s)${outcome.skipped ? `, ${outcome.skipped} already scored` : ''}`)
      const data = await window.api.aeh.getEvalRunCases(result.run_id)
      setAgentCases(data.agents)
      setAgentSummaries(data.agent_summaries ?? {})
    } catch (err: any) {
      setError(err?.message ?? 'Failed to score cases')
    } finally {
      setJudging(false)
    }
  }

  async function handleJudgeAll() {
    if (!result || !agentCases) return
    setJudgeAllConfirmOpen(false)
    setJudgingAll(true)
    setError(null)
    const agentIds = Object.keys(agentCases).sort()
    let totalScored = 0
    let totalSkipped = 0
    try {
      for (let i = 0; i < agentIds.length; i++) {
        const agentId = agentIds[i]
        setJudgeAllProgress({ done: i, total: agentIds.length, agentId })
        const outcome = await window.api.aeh.judgeEvalRunCases(result.run_id, {
          agent_id: agentId,
          provider_id: providerId || null,
          model_id: modelId || null,
          reasoning_effort: reasoningEffort,
          thinking_budget: thinkingBudget,
        })
        totalScored += outcome.scored
        totalSkipped += outcome.skipped
      }
      toast.success(`Scored ${totalScored} case(s) across ${agentIds.length} agents${totalSkipped ? `, ${totalSkipped} already scored` : ''}`)
      const data = await window.api.aeh.getEvalRunCases(result.run_id)
      setAgentCases(data.agents)
      setAgentSummaries(data.agent_summaries ?? {})
    } catch (err: any) {
      setError(err?.message ?? 'Failed to score all agents')
    } finally {
      setJudgingAll(false)
      setJudgeAllProgress(null)
    }
  }

  async function handleRunSummary() {
    if (!result || !agentCases) return
    setInsightsConfirmOpen(false)
    const eligible = _computeAgentSummaries(agentCases)
      .filter((s) => s.scoredCount > 0 && !agentSummaries[s.agentId])
      .map((s) => s.agentId)
    if (eligible.length === 0) return
    setGeneratingInsights(true)
    setError(null)
    try {
      for (let i = 0; i < eligible.length; i++) {
        const agentId = eligible[i]
        setInsightsProgress({ done: i, total: eligible.length, agentId })
        const outcome = await window.api.aeh.summarizeEvalRunAgent(result.run_id, {
          agent_id: agentId,
          provider_id: providerId || null,
          model_id: modelId || null,
          reasoning_effort: reasoningEffort,
          thinking_budget: thinkingBudget,
        })
        setAgentSummaries((prev) => ({
          ...prev,
          [agentId]: { insight: outcome.insight, case_count: outcome.case_count },
        }))
      }
      toast.success(`Generated insight for ${eligible.length} agent(s)`)
    } catch (err: any) {
      setError(err?.message ?? 'Failed to run summary')
    } finally {
      setGeneratingInsights(false)
      setInsightsProgress(null)
    }
  }

  if (!sessionId) {
    return (
      <div className="flex flex-col h-full bg-[#090d16] text-slate-100 p-6">
        <div className="flex items-center gap-2 mb-6">
          <Button
            variant="ghost"
            className="w-8 h-8 p-0 rounded-full flex items-center justify-center border border-slate-800 bg-slate-900/40 text-slate-400 hover:text-slate-200"
            onClick={() => navigate('/aeh/analysis')}
          >
            <ArrowLeft size={16} />
          </Button>
          <h1 className="text-lg font-semibold">Stage 5: Load Results</h1>
        </div>
        <div className="text-sm text-amber-400">No session ID provided.</div>
      </div>
    )
  }

  if (checkingExisting) {
    return (
      <div className="flex flex-col h-full bg-[#090d16] text-slate-100 items-center justify-center gap-2">
        <Loader2 size={20} className="animate-spin text-slate-500" />
        <p className="text-xs text-slate-500">Checking for existing results...</p>
      </div>
    )
  }

  // Per-agent case detail view — same "internal view switch" pattern as Stage 3's agent grid.
  if (agentCases && viewingAgentId) {
    const rows = agentCases[viewingAgentId] ?? []
    const unscoredCount = rows.filter((r) => !r.evaluations.some((e) => e.metric_name === 'semantic_match')).length
    return (
      <div className="flex flex-col h-full bg-[#090d16] text-slate-100">
        <div className="screen-header shrink-0 flex items-center justify-between px-6 py-4 border-b border-slate-850">
          <div className="flex items-center gap-3">
            <Button
              variant="ghost"
              className="w-8 h-8 p-0 rounded-full flex items-center justify-center border border-slate-800 bg-slate-900/40 text-slate-400 hover:text-slate-200"
              onClick={() => setViewingAgentId(null)}
            >
              <ArrowLeft size={16} />
            </Button>
            <div>
              <h1 className="screen-title">{viewingAgentId}</h1>
              <p className="screen-subtitle">{rows.length} case{rows.length === 1 ? '' : 's'}</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <LLMModelButton
              providerId={providerId}
              modelId={modelId}
              providers={providers}
              onClick={() => setLlmConfigOpen(true)}
              labelPrefix="Judge model"
            />
            <Button
              variant="primary"
              onClick={() => setJudgeConfirmOpen(true)}
              loading={judging}
              disabled={judging || !providerId || unscoredCount === 0}
              className="text-xs h-8 px-3 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white font-medium flex items-center gap-1.5"
            >
              <Sparkles size={13} />
              {judging ? 'Scoring...' : unscoredCount === 0 ? 'All Scored' : `Score with LLM (${unscoredCount})`}
            </Button>
          </div>
        </div>
        <div className="flex-1 overflow-y-auto p-6 space-y-4">
          {rows.map((row) => {
            const semanticMatch = row.evaluations.find((e) => e.metric_name === 'semantic_match')
            const precisionRecall = row.evaluations.filter((e) => e.metric_name.startsWith('precision_recall.'))
            return (
            <div key={row.trace_id} className="border border-slate-850 rounded-xl bg-slate-950/40 overflow-hidden">
              {(semanticMatch || precisionRecall.length > 0) && (
                <div className="px-3 pt-3 flex flex-wrap items-center gap-2 text-[10px]">
                  {semanticMatch && (
                    <span className={`font-mono font-semibold ${_scoreColor(semanticMatch.score)}`}>
                      match: {semanticMatch.score?.toFixed(2)}
                    </span>
                  )}
                  {semanticMatch?.details?.notes ? (
                    <span className="text-slate-500 italic truncate">{String(semanticMatch.details.notes)}</span>
                  ) : null}
                  {precisionRecall.map((pr) => (
                    <span key={pr.metric_name} className="font-mono text-slate-500">
                      {pr.metric_name.replace('precision_recall.', '')}: P={(pr.details.precision as number)?.toFixed(2)} R={(pr.details.recall as number)?.toFixed(2)}
                    </span>
                  ))}
                </div>
              )}
              <div className="grid grid-cols-1 lg:grid-cols-3 divide-y lg:divide-y-0 lg:divide-x divide-slate-850">
                <div className="p-3">
                  <div className="text-[10px] font-semibold text-slate-500 uppercase mb-2">Input</div>
                  <pre className="text-[10px] text-slate-300 font-mono whitespace-pre-wrap break-words max-h-64 overflow-y-auto">
                    {_pretty(row.input)}
                  </pre>
                </div>
                <div className="p-3">
                  <div className="text-[10px] font-semibold text-slate-500 uppercase mb-2">Result</div>
                  <pre className="text-[10px] text-emerald-300/90 font-mono whitespace-pre-wrap break-words max-h-64 overflow-y-auto">
                    {_pretty(row.result)}
                  </pre>
                </div>
                <div className="p-3">
                  <div className="text-[10px] font-semibold text-slate-500 uppercase mb-2">Expected</div>
                  <pre className="text-[10px] text-indigo-300/90 font-mono whitespace-pre-wrap break-words max-h-64 overflow-y-auto">
                    {_pretty(row.expected)}
                  </pre>
                </div>
              </div>
            </div>
            )
          })}
        </div>

        {judgeConfirmOpen && (
          <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/80 backdrop-blur-sm p-4">
            <div className="glass rounded-2xl w-full max-w-sm border-slate-850 shadow-2xl p-5 space-y-4 text-slate-100">
              <div className="flex items-center gap-3 text-amber-400">
                <AlertCircle className="shrink-0 w-6 h-6" />
                <h3 className="text-sm font-semibold text-slate-200">Score with LLM?</h3>
              </div>
              <p className="text-[11px] text-slate-450 leading-relaxed">
                This will call {providerId}:{modelId || '(no model)'} once per unscored case —
                {' '}{unscoredCount} case{unscoredCount === 1 ? '' : 's'}, {unscoredCount} LLM call{unscoredCount === 1 ? '' : 's'}.
                Already-scored cases are skipped automatically.
              </p>
              <div className="flex justify-end gap-2.5 pt-2">
                <Button variant="ghost" onClick={() => setJudgeConfirmOpen(false)} className="text-xs px-4 py-2">
                  Cancel
                </Button>
                <Button variant="primary" onClick={handleJudgeAgent} className="text-xs px-4 py-2 bg-indigo-600 hover:bg-indigo-500">
                  Continue
                </Button>
              </div>
            </div>
          </div>
        )}

        <LLMConfigModal
          isOpen={llmConfigOpen}
          onClose={() => setLlmConfigOpen(false)}
          providerId={providerId}
          modelId={modelId}
          reasoningEffort={reasoningEffort}
          thinkingBudget={thinkingBudget}
          onChange={(p, m, effort, budget) => {
            setProviderId(p)
            setModelId(m)
            setReasoningEffort(effort ?? null)
            setThinkingBudget(budget ?? null)
          }}
          title="Judge Model (Stage 5 scoring)"
        />
      </div>
    )
  }

  // Summary view — aggregates avg score + best/worst case per agent, computed client-side
  // from data already loaded in agentCases (no extra backend call).
  if (agentCases && showSummary) {
    const summaries = _computeAgentSummaries(agentCases)
    const overallScored = summaries.reduce((sum, s) => sum + s.scoredCount, 0)
    const overallTotal = summaries.reduce((sum, s) => sum + s.totalCount, 0)
    const overallAvg = overallScored > 0
      ? summaries.reduce((sum, s) => sum + (s.avgScore ?? 0) * s.scoredCount, 0) / overallScored
      : null
    // Worst-first ordering surfaces problem agents at the top; unscored agents sort last.
    const ordered = [...summaries].sort((a, b) => {
      if (a.avgScore === null && b.avgScore === null) return 0
      if (a.avgScore === null) return 1
      if (b.avgScore === null) return -1
      return a.avgScore - b.avgScore
    })
    const eligibleForInsight = summaries.filter((s) => s.scoredCount > 0 && !agentSummaries[s.agentId])
    return (
      <div className="flex flex-col h-full bg-[#090d16] text-slate-100">
        <div className="screen-header shrink-0 flex items-center justify-between px-6 py-4 border-b border-slate-850">
          <div className="flex items-center gap-3">
            <Button
              variant="ghost"
              className="w-8 h-8 p-0 rounded-full flex items-center justify-center border border-slate-800 bg-slate-900/40 text-slate-400 hover:text-slate-200"
              onClick={() => setShowSummary(false)}
            >
              <ArrowLeft size={16} />
            </Button>
            <div>
              <h1 className="screen-title">Stage 5: Summary</h1>
              <p className="screen-subtitle">
                {generatingInsights && insightsProgress
                  ? `Summarizing ${insightsProgress.agentId}... (${insightsProgress.done}/${insightsProgress.total} agents)`
                  : overallAvg !== null
                    ? `Overall avg ${overallAvg.toFixed(2)} — ${overallScored}/${overallTotal} cases scored`
                    : `${overallScored}/${overallTotal} cases scored — run Judge All to populate`}
              </p>
            </div>
          </div>
          <Button
            variant="primary"
            onClick={() => setInsightsConfirmOpen(true)}
            loading={generatingInsights}
            disabled={generatingInsights || !providerId || eligibleForInsight.length === 0}
            className="text-xs h-8 px-3 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white font-medium flex items-center gap-1.5"
          >
            <Sparkles size={13} />
            {generatingInsights
              ? 'Summarizing...'
              : eligibleForInsight.length === 0
                ? 'All Summarized'
                : `Run Summary (${eligibleForInsight.length})`}
          </Button>
        </div>
        <div className="flex-1 overflow-y-auto p-6 space-y-3">
          {ordered.map((s) => (
            <div key={s.agentId} className="border border-slate-850 rounded-xl bg-slate-950/40 p-4 space-y-3">
              <div className="flex items-center justify-between">
                <button
                  onClick={() => { setViewingAgentId(s.agentId); setShowSummary(false) }}
                  className="text-sm font-semibold text-slate-200 hover:text-indigo-300 transition-colors"
                >
                  {s.agentId}
                </button>
                <div className="flex items-center gap-3 text-[10px] font-mono">
                  {s.avgPrecision !== null && (
                    <span className="text-slate-500">avg P={s.avgPrecision.toFixed(2)} R={s.avgRecall?.toFixed(2)}</span>
                  )}
                  <span className={`font-semibold ${_scoreColor(s.avgScore)}`}>
                    {s.avgScore !== null ? `avg ${s.avgScore.toFixed(2)}` : 'not scored'}
                  </span>
                  <span className="text-slate-600">{s.scoredCount}/{s.totalCount}</span>
                </div>
              </div>
              {agentSummaries[s.agentId] && (
                <div className="rounded-lg border border-indigo-900/40 bg-indigo-950/10 p-3 space-y-1">
                  <div className="text-[9px] font-semibold text-indigo-400 uppercase">
                    AI Insight — {agentSummaries[s.agentId].case_count} case(s)
                  </div>
                  <p className="text-[11px] text-indigo-200/80 leading-relaxed">
                    {agentSummaries[s.agentId].insight}
                  </p>
                </div>
              )}
              {s.worst && s.best && (
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
                  <div className="rounded-lg border border-rose-900/40 bg-rose-950/10 p-3 space-y-1">
                    <div className="text-[9px] font-semibold text-rose-400 uppercase">
                      Worst case — {s.worst.score.toFixed(2)}
                    </div>
                    <p className="text-[11px] text-rose-200/80 leading-relaxed">
                      {s.worst.notes || 'No notes recorded.'}
                    </p>
                  </div>
                  {s.best.row.trace_id !== s.worst.row.trace_id && (
                    <div className="rounded-lg border border-emerald-900/40 bg-emerald-950/10 p-3 space-y-1">
                      <div className="text-[9px] font-semibold text-emerald-400 uppercase">
                        Best case — {s.best.score.toFixed(2)}
                      </div>
                      <p className="text-[11px] text-emerald-200/80 leading-relaxed">
                        {s.best.notes || 'No notes recorded.'}
                      </p>
                    </div>
                  )}
                </div>
              )}
              {!s.worst && (
                <p className="text-[11px] text-slate-500 italic">No cases scored yet for this agent.</p>
              )}
            </div>
          ))}
        </div>

        {insightsConfirmOpen && (
          <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/80 backdrop-blur-sm p-4">
            <div className="glass rounded-2xl w-full max-w-sm border-slate-850 shadow-2xl p-5 space-y-4 text-slate-100">
              <div className="flex items-center gap-3 text-amber-400">
                <AlertCircle className="shrink-0 w-6 h-6" />
                <h3 className="text-sm font-semibold text-slate-200">Run summary?</h3>
              </div>
              <p className="text-[11px] text-slate-450 leading-relaxed">
                This will call {providerId}:{modelId || '(no model)'} once per agent that has scored
                cases and no insight yet — {eligibleForInsight.length} agent{eligibleForInsight.length === 1 ? '' : 's'},
                {' '}{eligibleForInsight.length} LLM call{eligibleForInsight.length === 1 ? '' : 's'} total. Each call reads
                that agent's already-computed scores and notes (not the raw case content) to produce one overall insight.
                Agents already summarized are skipped.
              </p>
              <div className="flex justify-end gap-2.5 pt-2">
                <Button variant="ghost" onClick={() => setInsightsConfirmOpen(false)} className="text-xs px-4 py-2">
                  Cancel
                </Button>
                <Button variant="primary" onClick={handleRunSummary} className="text-xs px-4 py-2 bg-indigo-600 hover:bg-indigo-500">
                  Continue
                </Button>
              </div>
            </div>
          </div>
        )}
      </div>
    )
  }

  // Agent grid view — after results have been fetched.
  if (agentCases) {
    const agentIds = Object.keys(agentCases).sort()
    const unscoredByAgent = agentIds.map((id) => ({
      id,
      count: agentCases[id].filter((r) => !r.evaluations.some((e) => e.metric_name === 'semantic_match')).length,
    }))
    const totalUnscored = unscoredByAgent.reduce((sum, a) => sum + a.count, 0)
    return (
      <div className="flex flex-col h-full bg-[#090d16] text-slate-100">
        <div className="screen-header shrink-0 flex items-center justify-between px-6 py-4 border-b border-slate-850">
          <div className="flex items-center gap-3">
            <Button
              variant="ghost"
              className="w-8 h-8 p-0 rounded-full flex items-center justify-center border border-slate-800 bg-slate-900/40 text-slate-400 hover:text-slate-200"
              onClick={() => navigate('/aeh/analysis')}
            >
              <ArrowLeft size={16} />
            </Button>
            <div>
              <h1 className="screen-title">Stage 5: Results</h1>
              <p className="screen-subtitle">
                {judgeAllProgress
                  ? `Judging ${judgeAllProgress.agentId}... (${judgeAllProgress.done}/${judgeAllProgress.total} agents)`
                  : `Run ${result?.run_id} — click an agent to see cases`}
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Button
              variant="ghost"
              onClick={() => setShowSummary(true)}
              className="text-xs h-8 px-3 border border-slate-800 bg-slate-900/40 text-slate-300 hover:text-slate-100 font-medium flex items-center gap-1.5"
            >
              <BarChart3 size={13} />
              Summary
            </Button>
            <LLMModelButton
              providerId={providerId}
              modelId={modelId}
              providers={providers}
              onClick={() => setLlmConfigOpen(true)}
              labelPrefix="Judge model"
            />
            <Button
              variant="primary"
              onClick={() => setJudgeAllConfirmOpen(true)}
              loading={judgingAll}
              disabled={judgingAll || !providerId || totalUnscored === 0}
              className="text-xs h-8 px-3 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white font-medium flex items-center gap-1.5"
            >
              <Sparkles size={13} />
              {judgingAll ? 'Judging All...' : totalUnscored === 0 ? 'All Scored' : `Judge All (${totalUnscored})`}
            </Button>
          </div>
        </div>
        <div className="flex-1 overflow-y-auto p-6">
          <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-3 gap-3">
            {agentIds.map((agentId) => (
              <button
                key={agentId}
                onClick={() => setViewingAgentId(agentId)}
                className="text-left border border-slate-850 rounded-lg p-3 bg-slate-900/20 hover:border-indigo-500/50 hover:bg-slate-900/40 transition-colors space-y-1.5"
              >
                <div className="text-[11px] font-semibold text-slate-200 truncate">{agentId}</div>
                <div className="text-[9px] text-slate-500 font-mono">
                  {agentCases[agentId].length} case{agentCases[agentId].length === 1 ? '' : 's'}
                </div>
              </button>
            ))}
          </div>
        </div>

        {judgeAllConfirmOpen && (
          <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/80 backdrop-blur-sm p-4">
            <div className="glass rounded-2xl w-full max-w-sm border-slate-850 shadow-2xl p-5 space-y-4 text-slate-100">
              <div className="flex items-center gap-3 text-amber-400">
                <AlertCircle className="shrink-0 w-6 h-6" />
                <h3 className="text-sm font-semibold text-slate-200">Judge all agents?</h3>
              </div>
              <p className="text-[11px] text-slate-450 leading-relaxed">
                This will call {providerId}:{modelId || '(no model)'} once per unscored case across all
                {' '}{agentIds.length} agents — {totalUnscored} case{totalUnscored === 1 ? '' : 's'}, {totalUnscored} LLM call{totalUnscored === 1 ? '' : 's'} total.
                Agents run one at a time; already-scored cases are skipped automatically.
              </p>
              <div className="flex justify-end gap-2.5 pt-2">
                <Button variant="ghost" onClick={() => setJudgeAllConfirmOpen(false)} className="text-xs px-4 py-2">
                  Cancel
                </Button>
                <Button variant="primary" onClick={handleJudgeAll} className="text-xs px-4 py-2 bg-indigo-600 hover:bg-indigo-500">
                  Continue
                </Button>
              </div>
            </div>
          </div>
        )}

        <LLMConfigModal
          isOpen={llmConfigOpen}
          onClose={() => setLlmConfigOpen(false)}
          providerId={providerId}
          modelId={modelId}
          reasoningEffort={reasoningEffort}
          thinkingBudget={thinkingBudget}
          onChange={(p, m, effort, budget) => {
            setProviderId(p)
            setModelId(m)
            setReasoningEffort(effort ?? null)
            setThinkingBudget(budget ?? null)
          }}
          title="Judge Model (Stage 5 scoring)"
        />
      </div>
    )
  }

  return (
    <div className="flex flex-col h-full bg-[#090d16] text-slate-100">
      <div className="screen-header shrink-0 flex items-center justify-between px-6 py-4 border-b border-slate-850">
        <div className="flex items-center gap-3">
          <Button
            variant="ghost"
            className="w-8 h-8 p-0 rounded-full flex items-center justify-center border border-slate-800 bg-slate-900/40 text-slate-400 hover:text-slate-200"
            onClick={() => navigate('/aeh/analysis')}
          >
            <ArrowLeft size={16} />
          </Button>
          <div>
            <h1 className="screen-title">Stage 5: Load Results</h1>
            <p className="screen-subtitle">Ingest evaluation results from the instrumented backend</p>
          </div>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto p-6">
        <div className="space-y-6 max-w-2xl">
          {error && (
            <div className="bg-rose-950/20 border border-rose-800/40 rounded-xl p-4 text-xs text-rose-300 space-y-1">
              <div className="font-semibold flex items-center gap-1.5">
                <AlertCircle size={14} className="text-rose-400" />
                <span>Note</span>
              </div>
              <p className="text-[11px] leading-relaxed text-rose-300/80">{error}</p>
            </div>
          )}

          <div className="bg-slate-950/40 border border-slate-850 rounded-xl p-6 space-y-4">
            <div>
              <h2 className="text-sm font-semibold text-slate-200 mb-3">Load Evaluation Results</h2>
              <p className="text-xs text-slate-400 mb-6">
                Once you have run the eval driver on your instrumented backend (by calling POST /aeh/run-eval),
                click below to ingest the results into AEH.
              </p>

              {result ? (
                <div className="space-y-4">
                  <div className="flex items-start gap-3 p-4 bg-emerald-950/20 border border-emerald-800/40 rounded-lg">
                    <CheckCircle2 size={20} className="text-emerald-400 shrink-0 mt-0.5" />
                    <div>
                      <div className="text-sm font-semibold text-emerald-300 mb-1">Results Loaded</div>
                      <div className="text-xs text-emerald-300/70 space-y-1">
                        <div>Run ID: <span className="font-mono">{result.run_id}</span></div>
                        <div>Status: <span className="font-mono">{result.status}</span></div>
                      </div>
                    </div>
                  </div>

                  <Button
                    variant="primary"
                    onClick={handleViewResults}
                    loading={casesLoading}
                    disabled={casesLoading}
                    className="w-full text-xs h-8 bg-indigo-600 hover:bg-indigo-500 text-white font-medium"
                  >
                    {casesLoading ? 'Loading...' : 'View Results'}
                  </Button>
                </div>
              ) : (
                <Button
                  variant="primary"
                  onClick={handleLoadResults}
                  loading={loading}
                  disabled={loading}
                  className="w-full text-xs h-9 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white font-medium flex items-center justify-center gap-2"
                >
                  {loading && <Loader2 size={14} className="animate-spin" />}
                  {loading ? 'Loading Results...' : 'Load Results'}
                </Button>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

import React, { useEffect } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import {
  ArrowLeft,
  AlertTriangle,
  CheckCircle2,
  ExternalLink,
  Code,
  Check,
  X,
  Info
} from 'lucide-react'
import { useAEHReportsStore, EvalCaseItem } from '../../store/aeh-reports.store'
import { VerdictChip, computeClientAgentVerdict } from './VerdictChip'

function _scoreColor(score: number | null): string {
  if (score === null) return 'text-slate-400'
  if (score >= 0.7) return 'text-emerald-400'
  if (score >= 0.4) return 'text-amber-400'
  return 'text-rose-400'
}

function formatJsonOrString(val: unknown): string {
  if (val === null || val === undefined) return 'N/A'
  if (typeof val === 'string') {
    try {
      const parsed = JSON.parse(val)
      return JSON.stringify(parsed, null, 2)
    } catch {
      return val
    }
  }
  return JSON.stringify(val, null, 2)
}

export default function ComponentDetailScreen(): React.ReactElement {
  const { runId, agentId: routeAgentId, componentId } = useParams<{
    runId: string
    agentId?: string
    componentId?: string
  }>()
  const agentId = routeAgentId || componentId
  const navigate = useNavigate()

  const { selectedRunCases, selectedRunItem, loading, error, fetchRunCases } =
    useAEHReportsStore()

  useEffect(() => {
    if (runId && (!selectedRunCases || selectedRunCases.run_id !== runId)) {
      fetchRunCases(runId).catch(() => {})
    }
  }, [runId, selectedRunCases, fetchRunCases])

  if (loading && !selectedRunCases) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center py-20 gap-4 text-slate-100 bg-slate-950">
        <div className="h-10 w-10 rounded-full border-t-2 border-indigo-500 animate-spin"></div>
        <p className="text-sm text-gray-400">Loading agent cases...</p>
      </div>
    )
  }

  if (error || !selectedRunCases || !agentId) {
    return (
      <div className="flex-1 p-6 text-slate-100 bg-slate-950">
        <button
          onClick={() => navigate(runId ? `/aeh/reports/runs/${runId}` : '/aeh/reports')}
          className="mb-4 px-4 py-2 text-xs font-semibold text-gray-300 hover:text-white bg-slate-900 border border-slate-800 rounded-lg transition-colors flex items-center gap-2 cursor-pointer"
        >
          <ArrowLeft className="h-4 w-4" />
          Back to Run Details
        </button>
        <div className="p-4 rounded-xl border border-rose-900/40 bg-rose-950/20 text-rose-300 flex items-center gap-3 text-sm font-medium">
          <AlertTriangle className="h-5 w-5 flex-shrink-0 text-rose-400" />
          <p>{error || 'Agent cases could not be loaded'}</p>
        </div>
      </div>
    )
  }

  const agentCases: EvalCaseItem[] = selectedRunCases.agents[agentId] || []
  const agentSummary = selectedRunCases.agent_summaries?.[agentId]
  const agentVerdict = computeClientAgentVerdict(agentCases, 0.70)

  return (
    <div className="flex-1 flex flex-col min-h-screen text-slate-100 bg-slate-950 pb-10">
      {/* Navigation Header */}
      <header className="border-b border-slate-800 bg-slate-900 px-6 py-4 flex items-center justify-between sticky top-0 z-50">
        <div className="flex items-center gap-2 text-xs font-semibold text-gray-400 uppercase tracking-wider">
          <span className="hover:text-white cursor-pointer" onClick={() => navigate('/aeh/reports')}>
            Runs
          </span>
          <span>/</span>
          <span
            className="hover:text-white cursor-pointer"
            onClick={() => navigate(`/aeh/reports/runs/${runId}`)}
          >
            {selectedRunItem?.system_name || runId?.slice(0, 8)}
          </span>
          <span>/</span>
          <span className="text-white font-mono">Agent: {agentId}</span>
        </div>
        <button
          onClick={() => navigate(`/aeh/reports/runs/${runId}`)}
          className="px-4 py-2 text-xs font-semibold text-gray-300 hover:text-white bg-slate-800 border border-slate-700 rounded-lg transition-colors flex items-center gap-1.5 cursor-pointer"
        >
          <ArrowLeft className="h-3.5 w-3.5" />
          Back to Heatmap
        </button>
      </header>

      {/* Main Content */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-6 flex flex-col gap-6">
        {/* Agent Overview Summary Card */}
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 flex flex-col gap-4">
          <div className="flex items-center justify-between">
            <div>
              <div className="flex items-center gap-3">
                <span className="text-xs text-indigo-400 font-mono font-semibold uppercase tracking-wider">
                  Level 3 Agent Drill-Down
                </span>
                {/* Unified Shared Verdict Chip */}
                <VerdictChip verdictInfo={agentVerdict} compact={false} />
              </div>
              <h2 className="text-2xl font-bold text-gray-100 font-mono mt-1">{agentId}</h2>
            </div>
            {agentSummary?.avg_score !== undefined && agentSummary.avg_score !== null && (
              <div className="text-right">
                <span className="text-xs text-gray-400 block font-medium">Avg Semantic Score</span>
                <span className={`text-2xl font-extrabold ${_scoreColor(agentSummary.avg_score)}`}>
                  {(agentSummary.avg_score * 100).toFixed(0)}%
                </span>
              </div>
            )}
          </div>

          {agentSummary?.insight && (
            <div className="p-3.5 bg-purple-950/20 border border-purple-900/30 rounded-lg text-xs text-purple-200">
              <span className="font-bold text-purple-300 block mb-1">Synthesized Insight:</span>
              {agentSummary.insight}
            </div>
          )}
        </div>

        {/* Per-Case Evaluation List */}
        <div className="flex flex-col gap-4">
          <h3 className="text-sm font-bold text-gray-100 flex items-center gap-2">
            <Code className="h-4 w-4 text-indigo-400" />
            <span>Evaluation Cases ({agentCases.length})</span>
          </h3>

          {agentCases.length === 0 ? (
            <div className="bg-slate-900 border border-slate-800 rounded-xl p-12 text-center text-gray-500 font-medium text-sm">
              No cases recorded for this agent in this run.
            </div>
          ) : (
            agentCases.map((c, idx) => (
              <CaseCard key={c.case_id || c.trace_id || idx} caseItem={c} runId={runId || ''} agentId={agentId} navigate={navigate} />
            ))
          )}
        </div>
      </main>
    </div>
  )
}

function CaseCard({
  caseItem,
  runId,
  agentId,
  navigate
}: {
  caseItem: EvalCaseItem
  runId: string
  agentId: string
  navigate: (path: string, options?: any) => void
}): React.ReactElement {
  const flags = caseItem.flags
  const semanticEval = caseItem.evaluations?.find((e) => e.metric_name === 'semantic_match')
  const prEvals = caseItem.evaluations?.filter((e) => e.metric_name.startsWith('precision_recall.')) || []
  const feEvals = caseItem.evaluations?.filter((e) => e.metric_name.startsWith('field_exact.')) || []

  const isSuspectGold = flags?.expected_suspect || (flags?.schema_suspect_fields && flags.schema_suspect_fields.length > 0)

  // Render-time label for judge JSON parse failure: show "unscored (judge error)" instead of red 0%
  const isJudgeError =
    semanticEval?.score === 0.0 &&
    typeof semanticEval?.details?.notes === 'string' &&
    /not valid json|json_parse_error|output_repair_failed/i.test(semanticEval.details.notes)

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 flex flex-col gap-4">
      {/* Case Header */}
      <div className="flex items-center justify-between pb-3 border-b border-slate-800">
        <div className="flex items-center gap-3">
          <span className="text-sm font-bold text-gray-100 font-mono">
            Case: {caseItem.case_id || 'unlabeled'}
          </span>
          <span className="text-xs text-gray-500 font-mono">
            Trace: {caseItem.trace_id.slice(0, 8)}...
          </span>
          {flags?.input_starved && (
            <span className="text-xs px-2 py-0.5 rounded bg-amber-950/50 border border-amber-800/60 text-amber-300 font-semibold">
              Input Starved
            </span>
          )}
        </div>

        <button
          onClick={() => navigate(`/aeh/reports/runs/${runId}/traces/${caseItem.trace_id}?componentId=${agentId}`)}
          className="px-3 py-1.5 text-xs font-semibold text-indigo-400 hover:text-indigo-300 bg-slate-950 border border-slate-800 rounded-lg transition-colors flex items-center gap-1.5 cursor-pointer"
        >
          <span>View Trace</span>
          <ExternalLink className="h-3.5 w-3.5" />
        </button>
      </div>

      {/* Heuristic Audit Reasons Banner */}
      {flags?.reasons && flags.reasons.length > 0 && (
        <div className="p-3 bg-slate-950 border border-slate-800 rounded-lg text-xs flex flex-col gap-1">
          <span className="text-xs font-bold text-gray-400 uppercase tracking-wider flex items-center gap-1">
            <Info className="h-3.5 w-3.5 text-indigo-400" />
            Heuristic Audit Flags ({flags.reasons.length})
          </span>
          <ul className="list-disc list-inside text-gray-300 space-y-0.5 font-mono text-xs">
            {flags.reasons.map((r, idx) => (
              <li key={idx}>{r}</li>
            ))}
          </ul>
        </div>
      )}

      {/* Metric Badges */}
      <div className="flex flex-wrap items-center gap-2">
        {/* Semantic Match Badge */}
        {semanticEval && (
          <div className="flex items-center gap-2 bg-slate-950 border border-slate-800 rounded-lg px-3 py-1.5 text-xs">
            <span className="font-semibold text-gray-400">Semantic Match:</span>
            {isJudgeError ? (
              <span className="font-bold text-amber-400 bg-amber-950/40 border border-amber-800/50 px-2 py-0.5 rounded text-xs">
                unscored (judge error)
              </span>
            ) : (
              <span className={`font-bold ${_scoreColor(semanticEval.score)}`}>
                {semanticEval.score !== null ? `${(semanticEval.score * 100).toFixed(0)}%` : 'N/A'}
              </span>
            )}
            {semanticEval.details && typeof semanticEval.details.notes === 'string' && (
              <span className="text-gray-400 border-l border-slate-800 pl-2 text-xs">
                {semanticEval.details.notes}
              </span>
            )}
          </div>
        )}

        {/* Precision / Recall Badges */}
        {prEvals.map((ev) => {
          const fieldName = ev.metric_name.replace('precision_recall.', '')
          return (
            <div key={ev.metric_name} className="bg-slate-950 border border-slate-800 rounded-lg px-2.5 py-1 text-xs flex items-center gap-1.5">
              <span className="text-purple-300 font-mono">{fieldName}:</span>
              <span className="font-bold text-gray-200">
                P/R {ev.score !== null ? (ev.score * 100).toFixed(0) + '%' : 'N/A'}
              </span>
            </div>
          )
        })}

        {/* Field Exact Badges */}
        {feEvals.map((ev) => {
          const fieldName = ev.metric_name.replace('field_exact.', '')
          const passed = ev.score !== null && ev.score >= 0.999
          const isSoft = flags?.soft_fail_fields?.includes(fieldName)

          if (passed) {
            return (
              <div
                key={ev.metric_name}
                className="rounded-lg px-2.5 py-1 text-xs font-semibold border flex items-center gap-1 bg-emerald-950/30 text-emerald-300 border-emerald-800/40"
              >
                <Check className="h-3.5 w-3.5" />
                <span>{fieldName}</span>
              </div>
            )
          }

          if (isSoft) {
            return (
              <div
                key={ev.metric_name}
                className="rounded-lg px-2.5 py-1 text-xs font-semibold border flex items-center gap-1 bg-amber-950/30 text-amber-300 border-amber-800/40 cursor-help"
                title="Subjective/estimated value differs — soft under flexibility standard."
              >
                <span className="font-bold">≈</span>
                <span>{fieldName} (soft)</span>
              </div>
            )
          }

          return (
            <div
              key={ev.metric_name}
              className="rounded-lg px-2.5 py-1 text-xs font-semibold border flex items-center gap-1 bg-rose-950/30 text-rose-300 border-rose-800/40"
            >
              <X className="h-3.5 w-3.5" />
              <span>{fieldName}</span>
            </div>
          )
        })}
      </div>

      {/* 3 Columns Grid: Input / Result / Expected (Gold as-is) */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 pt-1">
        {/* Column 1: Input */}
        <div className="flex flex-col gap-1.5">
          <span className="text-xs font-bold text-gray-400 uppercase tracking-wider">Input</span>
          <pre className="bg-slate-950 border border-slate-800 rounded-lg p-3 text-xs text-gray-300 font-mono overflow-x-auto max-h-60 whitespace-pre-wrap">
            {formatJsonOrString(caseItem.input)}
          </pre>
        </div>

        {/* Column 2: Result */}
        <div className="flex flex-col gap-1.5">
          <span className="text-xs font-bold text-gray-400 uppercase tracking-wider">Result (Agent Output)</span>
          <pre className="bg-slate-950 border border-slate-800 rounded-lg p-3 text-xs text-indigo-200 font-mono overflow-x-auto max-h-60 whitespace-pre-wrap">
            {formatJsonOrString(caseItem.result)}
          </pre>
        </div>

        {/* Column 3: Expected (Gold As-Is) */}
        <div className="flex flex-col gap-1.5">
          <span className="text-xs font-bold text-emerald-400 uppercase tracking-wider flex items-center gap-1">
            <CheckCircle2 className="h-3.5 w-3.5" />
            Expected (Gold As-Is)
          </span>

          {/* ⚠ Warning Banner when gold or schema is suspect */}
          {isSuspectGold && (
            <div className="p-2.5 bg-amber-950/30 border border-amber-800/50 rounded-lg text-xs text-amber-300 flex items-start gap-2">
              <AlertTriangle className="h-4 w-4 text-amber-400 flex-shrink-0 mt-0.5" />
              <div className="leading-relaxed">
                <span className="font-bold block">⚠ Suspect Gold / Schema Mismatch</span>
                {flags?.schema_suspect_fields && flags.schema_suspect_fields.length > 0
                  ? `Wrong schema type for field(s): ${flags.schema_suspect_fields.join(', ')}. `
                  : 'Gold contains failure artifact or is near-empty. '}
                <span>Shown as-is; low score may not reflect agent.</span>
              </div>
            </div>
          )}

          <pre
            className={`rounded-lg p-3 text-xs font-mono overflow-x-auto max-h-60 whitespace-pre-wrap ${
              isSuspectGold
                ? 'bg-amber-950/10 border border-amber-800/40 text-amber-200'
                : 'bg-slate-950 border border-emerald-900/30 text-emerald-200'
            }`}
          >
            {formatJsonOrString(caseItem.expected)}
          </pre>
        </div>
      </div>
    </div>
  )
}

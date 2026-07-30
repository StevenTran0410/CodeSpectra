import React, { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  Layers,
  RefreshCw,
  AlertTriangle,
  Boxes,
  Activity,
  ArrowRight,
  TrendingUp,
  Cpu
} from 'lucide-react'
import { useAEHReportsStore, AEHEvalRunSummary } from '../../store/aeh-reports.store'

function formatTime(isoStr: string) {
  if (!isoStr) return 'N/A'
  try {
    const d = new Date(isoStr)
    return d.toLocaleString()
  } catch (e) {
    return isoStr
  }
}

function formatLatency(ms: number) {
  if (!ms || ms <= 0) return 'N/A'
  if (ms < 1000) return `${ms}ms`
  return `${(ms / 1000).toFixed(2)}s`
}

function _scoreColor(score: number | null): string {
  if (score === null) return 'text-slate-400'
  if (score >= 0.7) return 'text-emerald-400'
  if (score >= 0.4) return 'text-amber-400'
  return 'text-rose-400'
}

export default function RunsListScreen(): React.ReactElement {
  const navigate = useNavigate()
  const { runs, loading, error, fetchRunsList } = useAEHReportsStore()

  const [compareRunIdA, setCompareRunIdA] = useState<string>('')
  const [compareRunIdB, setCompareRunIdB] = useState<string>('')

  useEffect(() => {
    fetchRunsList().catch(() => {})
  }, [fetchRunsList])

  const judgedRuns = runs.filter((r) => r.judged_status !== 'not_judged')
  const avgOverallSemantic =
    judgedRuns.length > 0
      ? judgedRuns.reduce((acc, r) => acc + (r.avg_semantic_match ?? 0), 0) / judgedRuns.length
      : null

  return (
    <div className="flex-1 flex flex-col min-h-screen text-slate-100 bg-slate-950 pb-10">
      {/* Navigation Header */}
      <header className="border-b border-slate-800 bg-slate-900 px-6 py-4 flex items-center justify-between sticky top-0 z-50">
        <div className="flex items-center gap-3">
          <div className="h-9 w-9 rounded-lg bg-indigo-600/20 border border-indigo-500/30 flex items-center justify-center">
            <Layers className="h-5 w-5 text-indigo-400" />
          </div>
          <div>
            <h1 className="text-lg font-bold text-gray-100">AEH Evaluation Dashboard</h1>
            <p className="text-xs text-gray-400">Agentic Evaluation Harness · Runs & Systems</p>
          </div>
        </div>

        <button
          onClick={() => fetchRunsList()}
          disabled={loading}
          className="p-2 text-gray-400 hover:text-white bg-slate-800 border border-slate-700 rounded-lg transition-colors cursor-pointer"
          title="Refresh runs"
        >
          <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
        </button>
      </header>

      {/* Main Content */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-6 flex flex-col gap-6">
        {error && (
          <div className="p-4 rounded-xl border border-rose-900/40 bg-rose-950/20 text-rose-300 flex items-center gap-3 text-sm">
            <AlertTriangle className="h-5 w-5 flex-shrink-0 text-rose-400" />
            <p className="font-medium">{error}</p>
          </div>
        )}

        {loading && runs.length === 0 ? (
          <div className="flex-1 flex flex-col items-center justify-center py-20 gap-4">
            <div className="h-10 w-10 rounded-full border-t-2 border-indigo-500 animate-spin"></div>
            <p className="text-sm text-gray-400">Loading evaluation runs...</p>
          </div>
        ) : (
          <>
            {/* Top Overview KPI Banner */}
            <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 grid grid-cols-1 md:grid-cols-4 gap-6 items-center">
              <div>
                <span className="text-xs font-semibold uppercase tracking-wider text-gray-400 block mb-1">
                  Overall Avg Semantic Match
                </span>
                <span className={`text-3xl font-extrabold ${_scoreColor(avgOverallSemantic)}`}>
                  {avgOverallSemantic !== null ? `${(avgOverallSemantic * 100).toFixed(0)}%` : 'N/A'}
                </span>
              </div>

              <div>
                <span className="text-xs font-semibold uppercase tracking-wider text-gray-400 block mb-1">
                  Total Runs
                </span>
                <span className="text-2xl font-bold text-gray-100">{runs.length}</span>
              </div>

              <div>
                <span className="text-xs font-semibold uppercase tracking-wider text-gray-400 block mb-1">
                  Judged Runs
                </span>
                <span className="text-2xl font-bold text-gray-100">{judgedRuns.length}</span>
              </div>

              <div>
                <span className="text-xs font-semibold uppercase tracking-wider text-gray-400 block mb-1">
                  Evaluated Systems
                </span>
                <span className="text-2xl font-bold text-indigo-400">
                  {new Set(runs.map((r) => r.target_system_id)).size}
                </span>
              </div>
            </div>

            {/* Run Comparison Launcher */}
            {runs.length >= 2 && (
              <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 flex flex-wrap items-center justify-between gap-4">
                <div className="flex items-center gap-2 text-sm font-semibold text-gray-200">
                  <TrendingUp className="h-4 w-4 text-indigo-400" />
                  <span>Compare Two Runs</span>
                </div>

                <div className="flex items-center gap-3">
                  <select
                    value={compareRunIdA}
                    onChange={(e) => setCompareRunIdA(e.target.value)}
                    className="bg-slate-950 border border-slate-800 text-xs text-gray-200 rounded-lg px-3 py-2 focus:outline-none focus:border-indigo-500"
                  >
                    <option value="">Select Baseline Run</option>
                    {runs.map((r) => (
                      <option key={r.id} value={r.id}>
                        {r.system_name} ({r.id.slice(0, 8)})
                      </option>
                    ))}
                  </select>

                  <span className="text-xs text-gray-500 font-bold">VS</span>

                  <select
                    value={compareRunIdB}
                    onChange={(e) => setCompareRunIdB(e.target.value)}
                    className="bg-slate-950 border border-slate-800 text-xs text-gray-200 rounded-lg px-3 py-2 focus:outline-none focus:border-indigo-500"
                  >
                    <option value="">Select Target Run</option>
                    {runs.map((r) => (
                      <option key={r.id} value={r.id}>
                        {r.system_name} ({r.id.slice(0, 8)})
                      </option>
                    ))}
                  </select>

                  <button
                    disabled={!compareRunIdA || !compareRunIdB || compareRunIdA === compareRunIdB}
                    onClick={() => navigate(`/aeh/reports/compare/${compareRunIdA}/${compareRunIdB}`)}
                    className="px-4 py-2 text-xs font-semibold text-white bg-indigo-600 hover:bg-indigo-500 disabled:opacity-40 rounded-lg transition-colors cursor-pointer flex items-center gap-1.5"
                  >
                    <span>Compare Runs</span>
                    <ArrowRight className="h-3.5 w-3.5" />
                  </button>
                </div>
              </div>
            )}

            {/* Systems / Runs List */}
            <div className="flex flex-col gap-4">
              <h2 className="text-md font-bold text-gray-200 flex items-center gap-2">
                <Boxes className="h-4 w-4 text-indigo-400" />
                <span>Ingested Runs ({runs.length})</span>
              </h2>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {runs.map((run) => (
                  <RunCard key={run.id} run={run} navigate={navigate} />
                ))}
              </div>
            </div>
          </>
        )}
      </main>
    </div>
  )
}

function RunCard({
  run,
  navigate
}: {
  run: AEHEvalRunSummary
  navigate: (path: string) => void
}): React.ReactElement {
  return (
    <div
      onClick={() => navigate(`/aeh/reports/runs/${run.id}`)}
      className="bg-slate-900 border border-slate-800 hover:border-indigo-500/50 rounded-xl p-5 flex flex-col gap-4 transition-colors cursor-pointer group"
    >
      {/* Card Header */}
      <div className="flex items-start justify-between gap-3">
        <div className="flex flex-col gap-1">
          <div className="flex items-center gap-2">
            <span className="font-bold text-base text-gray-100 group-hover:text-indigo-400 transition-colors">
              {run.system_name}
            </span>
            {run.framework && (
              <span className="text-[11px] font-mono font-semibold px-2 py-0.5 rounded bg-indigo-950/60 text-indigo-300 border border-indigo-800/40 uppercase">
                {run.framework}
              </span>
            )}
          </div>
          <span className="text-xs text-gray-500 font-mono">Run ID: {run.id}</span>
        </div>

        {/* Judged Status Pill */}
        <StatusPill status={run.judged_status} scoredCount={run.scored_count} totalCount={run.case_count} />
      </div>

      {/* Metric Highlights */}
      <div className="grid grid-cols-3 gap-3 border-y border-slate-800/60 py-3">
        <div>
          <span className="text-xs text-gray-400 block font-medium">Semantic Match</span>
          <span className={`text-xl font-bold ${_scoreColor(run.avg_semantic_match)}`}>
            {run.avg_semantic_match !== null ? `${(run.avg_semantic_match * 100).toFixed(0)}%` : 'N/A'}
          </span>
        </div>

        <div>
          <span className="text-xs text-gray-400 block font-medium">Field-Exact Pass</span>
          <span className="text-xl font-bold text-gray-200">
            {run.field_exact_pass_rate !== null ? `${(run.field_exact_pass_rate * 100).toFixed(0)}%` : 'N/A'}
          </span>
        </div>

        <div>
          <span className="text-xs text-gray-400 block font-medium">Precision / Recall</span>
          <span className="text-xl font-bold text-gray-200">
            {run.avg_precision_recall !== null ? run.avg_precision_recall.toFixed(2) : 'N/A'}
          </span>
        </div>
      </div>

      {/* Footer Sparkline & Case Meta */}
      <div className="flex items-center justify-between text-xs text-gray-400">
        <div className="flex items-center gap-3">
          <span className="flex items-center gap-1 font-mono">
            <Cpu className="h-3.5 w-3.5 text-indigo-400" />
            {run.agent_count} Agents
          </span>
          <span>·</span>
          <span className="flex items-center gap-1 font-mono">
            <Activity className="h-3.5 w-3.5 text-purple-400" />
            {run.case_count} Cases
          </span>
          <span>·</span>
          <span>{formatLatency(run.total_latency_ms)}</span>
        </div>

        {/* 5-Bucket Distribution Sparkline */}
        {run.judged_status !== 'not_judged' && (
          <div className="flex items-end gap-1 h-5" title="Semantic Match 5-Bucket Score Distribution">
            {run.semantic_match_histogram.map((count, i) => {
              const max = Math.max(...run.semantic_match_histogram, 1)
              const pct = count > 0 ? Math.max((count / max) * 100, 25) : 10
              return (
                <div
                  key={i}
                  className="w-1.5 bg-indigo-500/70 rounded-t transition-all"
                  style={{ height: `${pct}%` }}
                />
              )
            })}
          </div>
        )}
      </div>
    </div>
  )
}

function StatusPill({
  status,
  scoredCount,
  totalCount
}: {
  status: 'judged' | 'partially_judged' | 'not_judged'
  scoredCount: number
  totalCount: number
}): React.ReactElement {
  if (status === 'judged') {
    return (
      <span className="text-xs font-semibold px-2.5 py-1 rounded-full bg-emerald-950/40 text-emerald-300 border border-emerald-800/50">
        Judged
      </span>
    )
  }

  if (status === 'partially_judged') {
    return (
      <span className="text-xs font-semibold px-2.5 py-1 rounded-full bg-amber-950/40 text-amber-300 border border-amber-800/50">
        Partially Judged ({scoredCount}/{totalCount})
      </span>
    )
  }

  return (
    <span className="text-xs font-semibold px-2.5 py-1 rounded-full bg-slate-800 text-slate-400 border border-slate-700">
      Not judged yet
    </span>
  )
}

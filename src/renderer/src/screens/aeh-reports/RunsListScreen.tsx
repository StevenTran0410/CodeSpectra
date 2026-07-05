import React, { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  Layers,
  RefreshCw,
  AlertTriangle,
  Database,
  CheckCircle2,
  ArrowRight,
  TrendingUp,
  Boxes,
  DollarSign
} from 'lucide-react'
import { useAEHReportsStore } from '../../store/aeh-reports.store'

function formatTime(isoStr: string) {
  try {
    const d = new Date(isoStr)
    return d.toLocaleString()
  } catch (e) {
    return isoStr
  }
}

export default function RunsListScreen(): React.ReactElement {
  const navigate = useNavigate()
  const { runs, loading, error, fetchRunsList } = useAEHReportsStore()

  const [compareRunIdA, setCompareRunIdA] = useState<string>('')
  const [compareRunIdB, setCompareRunIdB] = useState<string>('')

  useEffect(() => {
    fetchRunsList().catch(() => {})
  }, [fetchRunsList])

  return (
    <div className="flex-1 flex flex-col min-h-screen text-slate-100 selection:bg-indigo-500 selection:text-white pb-10">
      {/* Premium Header */}
      <header className="border-b border-surface-border bg-surface-overlay/30 backdrop-blur-md sticky top-0 z-50 px-6 py-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="h-10 w-10 rounded-xl bg-gradient-to-tr from-indigo-500 to-purple-600 flex items-center justify-center shadow-lg shadow-indigo-500/20">
            <Layers className="h-5 w-5 text-white" />
          </div>
          <div>
            <h1 className="text-xl font-bold tracking-tight bg-gradient-to-r from-gray-100 via-gray-200 to-gray-400 bg-clip-text text-transparent">
              AEH Dashboard
            </h1>
            <p className="text-xs text-gray-400 font-medium">Agentic Evaluation Harness</p>
          </div>
        </div>

        <div className="flex items-center gap-4">
          <button
            onClick={() => fetchRunsList()}
            disabled={loading}
            className="p-2 text-gray-400 hover:text-indigo-400 hover:bg-surface-overlay border border-surface-border hover:border-indigo-500/30 rounded-lg transition-all"
            title="Refresh current view"
          >
            <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
          </button>
        </div>
      </header>

      {/* Main Content Area */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-6 flex flex-col gap-6">
        {error && (
          <div className="p-4 rounded-xl border border-red-500/20 bg-red-950/20 text-red-400 flex items-center gap-3">
            <AlertTriangle className="h-5 w-5 flex-shrink-0" />
            <p className="text-sm font-medium">{error}</p>
          </div>
        )}

        {loading && runs.length === 0 ? (
          <div className="flex-1 flex flex-col items-center justify-center py-20 gap-4">
            <div className="relative">
              <div className="h-12 w-12 rounded-full border-t-2 border-indigo-500 animate-spin"></div>
            </div>
            <p className="text-sm text-gray-400 font-medium">Loading evaluation runs...</p>
          </div>
        ) : (
          <>
            {/* Dashboard Stats */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="glass rounded-2xl p-6 relative overflow-hidden">
                <div className="absolute top-0 right-0 h-32 w-32 bg-indigo-500/5 rounded-full blur-3xl -mr-8 -mt-8"></div>
                <p className="text-xs font-semibold uppercase tracking-wider text-gray-400">Total Runs</p>
                <p className="text-4xl font-extrabold text-gray-100 mt-2">{runs.length}</p>
                <div className="flex items-center gap-2 mt-4 text-xs text-indigo-400 font-medium">
                  <Database className="h-4 w-4" />
                  <span>Loaded from SQLite store</span>
                </div>
              </div>

              <div className="glass rounded-2xl p-6 relative overflow-hidden">
                <div className="absolute top-0 right-0 h-32 w-32 bg-emerald-500/5 rounded-full blur-3xl -mr-8 -mt-8"></div>
                <p className="text-xs font-semibold uppercase tracking-wider text-gray-400">Latest Run Pass Rate</p>
                <p className="text-4xl font-extrabold text-emerald-400 mt-2">
                  {runs.length > 0
                    ? `${(runs[0].pass_rate * 100).toFixed(0)}%`
                    : 'N/A'}
                </p>
                <div className="flex items-center gap-2 mt-4 text-xs text-emerald-500 font-medium">
                  <CheckCircle2 className="h-4 w-4" />
                  <span>Metric-level correctness</span>
                </div>
              </div>

              <div className="glass rounded-2xl p-6 relative overflow-hidden">
                <div className="absolute top-0 right-0 h-32 w-32 bg-purple-500/5 rounded-full blur-3xl -mr-8 -mt-8"></div>
                <p className="text-xs font-semibold uppercase tracking-wider text-gray-400">Run Comparison</p>
                <div className="flex items-center gap-2 mt-2">
                  <select
                    value={compareRunIdA}
                    onChange={(e) => setCompareRunIdA(e.target.value)}
                    className="bg-slate-900 border border-slate-800 text-xs rounded-lg p-2 text-gray-300 w-full focus:outline-none focus:border-indigo-500"
                  >
                    <option value="">Baseline Run</option>
                    {runs.map((r) => (
                      <option key={r.id} value={r.id}>
                        {r.target_system_id} ({r.id.slice(0, 8)})
                      </option>
                    ))}
                  </select>
                  <ArrowRight className="h-4 w-4 text-gray-500 flex-shrink-0" />
                  <select
                    value={compareRunIdB}
                    onChange={(e) => setCompareRunIdB(e.target.value)}
                    className="bg-slate-900 border border-slate-800 text-xs rounded-lg p-2 text-gray-300 w-full focus:outline-none focus:border-indigo-500"
                  >
                    <option value="">Comparison Run</option>
                    {runs.map((r) => (
                      <option key={r.id} value={r.id}>
                        {r.target_system_id} ({r.id.slice(0, 8)})
                      </option>
                    ))}
                  </select>
                </div>
                <button
                  onClick={() => navigate(`/aeh/reports/compare/${compareRunIdA}/${compareRunIdB}`)}
                  disabled={!compareRunIdA || !compareRunIdB}
                  className="mt-3 w-full bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-500 hover:to-purple-500 disabled:from-slate-800 disabled:to-slate-800 disabled:text-gray-500 disabled:cursor-not-allowed font-medium text-xs text-white py-2 px-3 rounded-lg shadow-md transition-all flex items-center justify-center gap-2"
                >
                  <TrendingUp className="h-3.5 w-3.5" />
                  <span>Compare Selected Runs</span>
                </button>
              </div>
            </div>

            {/* Runs Table */}
            <div className="glass rounded-2xl overflow-hidden">
              <div className="px-6 py-4 border-b border-surface-border bg-surface-overlay flex items-center justify-between">
                <h3 className="text-md font-bold text-gray-100 flex items-center gap-2">
                  <Boxes className="h-4 w-4 text-indigo-400" />
                  <span>Evaluation Runs</span>
                </h3>
              </div>

              <div className="overflow-x-auto">
                <table className="w-full text-left border-collapse">
                  <thead>
                    <tr className="border-b border-surface-border bg-surface-overlay/50 text-gray-400 text-xs font-semibold uppercase tracking-wider">
                      <th className="py-4 px-6">Target System</th>
                      <th className="py-4 px-6">Run ID / Plan</th>
                      <th className="py-4 px-6">Timestamp</th>
                      <th className="py-4 px-6">Defects Active</th>
                      <th className="py-4 px-6">Pass Rate</th>
                      <th className="py-4 px-6">Cost</th>
                      <th className="py-4 px-6">Status</th>
                      <th className="py-4 px-6 text-right">Actions</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-surface-border/60 text-sm">
                    {runs.length === 0 ? (
                      <tr>
                        <td colSpan={8} className="py-12 text-center text-gray-500 font-medium">
                          No evaluation runs found in database. Run `aeh eval` first to populate.
                        </td>
                      </tr>
                    ) : (
                      runs.map((run) => (
                        <tr key={run.id} className="hover:bg-surface-overlay/20 transition-colors group">
                          <td className="py-4 px-6 font-bold text-gray-100">
                            {run.target_system_id}
                          </td>
                          <td className="py-4 px-6">
                            <span className="font-mono text-xs text-gray-400 block">
                              {run.id.slice(0, 8)}...
                            </span>
                            <span className="text-xs text-gray-500 block truncate max-w-xs">
                              {run.eval_plan_id || 'no suite plan'}
                            </span>
                          </td>
                          <td className="py-4 px-6 text-xs text-gray-400">
                            {formatTime(run.started_at)}
                          </td>
                          <td className="py-4 px-6">
                            <div className="flex flex-wrap gap-1.5 max-w-xs">
                              {run.active_defects.length === 0 ? (
                                <span className="text-xs text-gray-500 italic">None (Clean)</span>
                              ) : (
                                run.active_defects.map((def) => (
                                  <span
                                    key={def}
                                    className="inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-semibold bg-red-950/30 text-red-400 border border-red-900/40"
                                  >
                                    {def.toUpperCase()}
                                  </span>
                                ))
                              )}
                            </div>
                          </td>
                          <td className="py-4 px-6">
                            <div className="flex items-center gap-2">
                              <div className="w-16 bg-slate-800 rounded-full h-1.5 overflow-hidden">
                                <div
                                  className={`h-full rounded-full ${
                                    run.pass_rate === 1.0 ? 'bg-emerald-500' : 'bg-indigo-500'
                                  }`}
                                  style={{ width: `${run.pass_rate * 100}%` }}
                                ></div>
                              </div>
                              <span className="text-xs font-bold text-gray-100">
                                {(run.pass_rate * 100).toFixed(0)}%
                              </span>
                            </div>
                          </td>
                          <td className="py-4 px-6 text-xs text-purple-400 font-bold">
                            {run.judge_cost > 0 ? (
                              <span className="flex items-center">
                                <DollarSign className="h-3 w-3 mr-0.5" />
                                {run.judge_cost} tokens
                              </span>
                            ) : (
                              <span className="text-gray-500">Free</span>
                            )}
                          </td>
                          <td className="py-4 px-6">
                            <span
                              className={`inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium border ${
                                run.status === 'completed'
                                  ? 'bg-emerald-950/20 text-emerald-400 border-emerald-900/30'
                                  : run.status === 'failed'
                                  ? 'bg-red-950/20 text-red-400 border-red-900/30'
                                  : 'bg-indigo-950/20 text-indigo-400 border-indigo-900/30 animate-pulse'
                              }`}
                            >
                              {run.status}
                            </span>
                          </td>
                          <td className="py-4 px-6 text-right">
                            <button
                              onClick={() => navigate(`/aeh/reports/runs/${run.id}`)}
                              className="inline-flex items-center gap-1 text-xs font-bold text-indigo-400 hover:text-indigo-300 group-hover:translate-x-1 transition-all bg-transparent border-0 cursor-pointer"
                            >
                              <span>View Report</span>
                              <ArrowRight className="h-3.5 w-3.5" />
                            </button>
                          </td>
                        </tr>
                      ))
                    )}
                  </tbody>
                </table>
              </div>
            </div>
          </>
        )}
      </main>
    </div>
  )
}

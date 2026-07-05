import React, { useEffect, useState } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import {
  Compass,
  AlertTriangle,
  RefreshCw,
  ArrowRight,
  Database
} from 'lucide-react'
import { useAEHReportsStore } from '../../store/aeh-reports.store'
import RerunModal from './RerunModal'

function formatTime(isoStr: string) {
  try {
    const d = new Date(isoStr)
    return d.toLocaleString()
  } catch (e) {
    return isoStr
  }
}

export default function RunDetailScreen(): React.ReactElement {
  const { runId } = useParams<{ runId: string }>()
  const navigate = useNavigate()

  const {
    selectedRun,
    loading,
    error,
    fetchRunDetail,
    fetchRunsList
  } = useAEHReportsStore()

  const [isRerunModalOpen, setIsRerunModalOpen] = useState(false)

  useEffect(() => {
    if (runId) {
      fetchRunDetail(runId).catch(() => {})
    }
  }, [runId, fetchRunDetail])

  if (loading && !selectedRun) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center py-20 gap-4 text-slate-100">
        <div className="h-12 w-12 rounded-full border-t-2 border-indigo-500 animate-spin"></div>
        <p className="text-sm text-gray-400 font-medium">Loading run details...</p>
      </div>
    )
  }

  if (error || !selectedRun) {
    return (
      <div className="flex-1 p-6 text-slate-100">
        <button
          onClick={() => navigate('/aeh/reports')}
          className="mb-4 px-4 py-2 text-sm font-medium text-gray-400 hover:text-gray-200 bg-zinc-900 border border-zinc-800 rounded-lg transition-all"
        >
          &larr; Back to Runs
        </button>
        <div className="p-4 rounded-xl border border-red-500/20 bg-red-950/20 text-red-400 flex items-center gap-3">
          <AlertTriangle className="h-5 w-5 flex-shrink-0" />
          <p className="text-sm font-medium">{error || 'Run details could not be loaded'}</p>
        </div>
      </div>
    )
  }

  return (
    <div className="flex-1 flex flex-col min-h-screen text-slate-100 pb-10">
      {/* Breadcrumbs & Rerun Header */}
      <header className="border-b border-surface-border bg-surface-overlay/30 backdrop-blur-md sticky top-0 z-50 px-6 py-4 flex items-center justify-between">
        <div className="flex flex-col gap-1">
          <div className="flex items-center gap-2 text-xs font-semibold text-gray-400 uppercase tracking-wider">
            <span className="hover:text-white cursor-pointer" onClick={() => navigate('/aeh/reports')}>
              Runs
            </span>
            <span>/</span>
            <span className="text-white">Run Details ({selectedRun.id.slice(0, 8)})</span>
          </div>
          {selectedRun.parent_run_id && (
            <div className="text-xs text-gray-400">
              Re-run of{' '}
              <button
                onClick={() => navigate(`/aeh/reports/runs/${selectedRun.parent_run_id}`)}
                className="text-indigo-400 hover:text-indigo-300 font-mono font-bold hover:underline bg-transparent border-0 p-0 cursor-pointer"
              >
                {selectedRun.parent_run_id.slice(0, 8)}
              </button>
            </div>
          )}
        </div>
        <div className="flex items-center gap-3">
          <button
            onClick={() => navigate('/aeh/reports')}
            className="px-4 py-2 text-sm font-medium text-gray-400 hover:text-gray-200 bg-zinc-900 border border-zinc-850 hover:border-zinc-700 rounded-lg transition-all cursor-pointer"
          >
            &larr; Back to Runs
          </button>
          {!selectedRun.target || !selectedRun.suite_path ? (
            <button
              disabled
              title="This run predates rerun support (missing target or suite_path) and cannot be re-executed."
              className="inline-flex items-center gap-2 bg-slate-800 text-gray-500 px-4 py-2 rounded-xl text-sm font-semibold cursor-not-allowed border border-slate-700/50"
            >
              <RefreshCw className="h-4 w-4" />
              <span>Rerun (Unavailable)</span>
            </button>
          ) : (
            <button
              onClick={() => setIsRerunModalOpen(true)}
              className="inline-flex items-center gap-2 bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-500 hover:to-purple-500 text-white px-4 py-2 rounded-xl text-sm font-semibold shadow-lg shadow-indigo-500/20 active:scale-[0.98] transition-all border border-indigo-500/30 cursor-pointer"
            >
              <RefreshCw className="h-4 w-4" />
              <span>Rerun Evaluation</span>
            </button>
          )}
        </div>
      </header>

      {/* Main Content Area */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-6 flex flex-col gap-6">
        {/* Meta details */}
        <div className="glass rounded-2xl p-6 grid grid-cols-1 md:grid-cols-4 gap-6">
          <div>
            <p className="text-xs text-gray-400 font-semibold uppercase">Target System</p>
            <p className="text-lg font-bold text-gray-100 mt-1">{selectedRun.target_system_id}</p>
          </div>
          <div>
            <p className="text-xs text-gray-400 font-semibold uppercase">Plan File</p>
            <p className="text-sm font-mono text-gray-300 mt-1 truncate" title={selectedRun.map_path || ''}>
              {selectedRun.eval_plan_id || 'Direct Run (No Plan)'}
            </p>
          </div>
          <div>
            <p className="text-xs text-gray-400 font-semibold uppercase">Timestamp</p>
            <p className="text-sm text-gray-300 mt-1">{formatTime(selectedRun.started_at)}</p>
          </div>
          <div>
            <p className="text-xs text-gray-400 font-semibold uppercase">Status</p>
            <p className="mt-1">
              <span className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-emerald-950/20 text-emerald-400 border border-emerald-900/30">
                {selectedRun.status}
              </span>
            </p>
          </div>
        </div>

        {/* Topology & Components Overview */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left Column: Topology Map */}
          <div className="glass rounded-2xl p-6 lg:col-span-2 flex flex-col gap-4">
            <h3 className="text-md font-bold text-gray-100 flex items-center gap-2">
              <Compass className="h-5 w-5 text-indigo-400" />
              <span>System Topology Graph</span>
            </h3>
            <p className="text-xs text-gray-400">
              Execution flow mapping component dependencies and aggregated evaluation pass rates.
            </p>

            <div className="border border-surface-border rounded-xl bg-surface/50 p-6 min-h-[400px] flex flex-col justify-between">
              {selectedRun.system_map.components.length === 0 ? (
                <div className="flex-1 flex items-center justify-center text-gray-500 text-xs italic">
                  No topology map layout loaded.
                </div>
              ) : (
                <div className="flex flex-col gap-6">
                  {selectedRun.system_map.components.map((comp) => {
                    const score = selectedRun.component_aggregates[comp.id]
                    const pct = score ? (score.passed / score.total) * 100 : null

                    return (
                      <div
                        key={comp.id}
                        onClick={() => navigate(`/aeh/reports/runs/${selectedRun.id}/components/${comp.id}`)}
                        className="glass glass-hover p-4 rounded-xl cursor-pointer flex items-center justify-between border-slate-800 hover:border-indigo-500/40"
                      >
                        <div className="flex items-center gap-3">
                          <div className={`h-2 w-2 rounded-full ${
                            pct === 100 ? 'bg-emerald-500' : pct !== null && pct < 100 ? 'bg-amber-500' : 'bg-slate-600'
                          }`} />
                          <div>
                            <p className="text-sm font-bold text-gray-100">{comp.id}</p>
                            <p className="text-[10px] text-gray-500 font-semibold uppercase">{comp.role}</p>
                          </div>
                        </div>

                        <div className="flex items-center gap-4">
                          {comp.model && (
                            <span className="text-[10px] font-mono bg-slate-900 border border-slate-800 text-gray-400 px-1.5 py-0.5 rounded">
                              {comp.model}
                            </span>
                          )}

                          {score ? (
                            <div className="text-right">
                              <p className="text-xs font-bold text-gray-100">
                                {score.passed}/{score.total} passed
                              </p>
                              <p className={`text-[10px] font-bold ${
                                pct === 100 ? 'text-emerald-400' : 'text-amber-400'
                              }`}>
                                {pct?.toFixed(0)}% Correct
                              </p>
                            </div>
                          ) : (
                            <span className="text-xs text-gray-500 italic">No metrics</span>
                          )}
                        </div>
                      </div>
                    )
                  })}
                </div>
              )}
            </div>
          </div>

          {/* Right Column: Active Defects & Configs */}
          <div className="glass rounded-2xl p-6 flex flex-col gap-4">
            <h3 className="text-md font-bold text-gray-100 flex items-center gap-2">
              <AlertTriangle className="h-5 w-5 text-indigo-400" />
              <span>Planted Defects Status</span>
            </h3>
            <p className="text-xs text-gray-400">
              Currently active switches from the regression gauntlet.
            </p>

            <div className="flex flex-col gap-3">
              {[
                'planner_overpack',
                'guard_leak',
                'wrong_tool',
                'judge_rubber_stamp',
                'writer_hallucinate',
                'no_retry',
              ].map((def) => {
                const isActive = selectedRun.active_defects.includes(def)
                return (
                  <div
                    key={def}
                    className={`p-3 rounded-lg border flex items-center justify-between text-xs ${
                      isActive
                        ? 'bg-red-950/20 border-red-900/30 text-red-400 font-bold'
                        : 'bg-slate-900/40 border-slate-800 text-gray-500'
                    }`}
                  >
                    <span>{def.toUpperCase()}</span>
                    <span className="uppercase text-[10px]">
                      {isActive ? 'Active' : 'Off'}
                    </span>
                  </div>
                )
              })}
            </div>
          </div>
        </div>
      </main>

      <RerunModal
        run={selectedRun}
        isOpen={isRerunModalOpen}
        onClose={() => setIsRerunModalOpen(false)}
        onSuccess={() => {
          fetchRunsList().catch(() => {})
          if (runId) {
            fetchRunDetail(runId).catch(() => {})
          }
        }}
      />
    </div>
  )
}

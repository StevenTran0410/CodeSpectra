import React, { useEffect } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import {
  TrendingUp,
  TrendingDown,
  FileText,
  AlertTriangle,
  CheckCircle2,
  XCircle
} from 'lucide-react'
import { useAEHReportsStore } from '../../store/aeh-reports.store'

export default function RunCompareScreen(): React.ReactElement {
  const { runIdA, runIdB } = useParams<{ runIdA: string; runIdB: string }>()
  const navigate = useNavigate()

  const {
    compareRunA,
    compareRunB,
    compareEvalsA,
    compareEvalsB,
    loading,
    error,
    fetchComparison
  } = useAEHReportsStore()

  useEffect(() => {
    if (runIdA && runIdB) {
      fetchComparison(runIdA, runIdB).catch(() => {})
    }
  }, [runIdA, runIdB, fetchComparison])

  if (loading && (!compareRunA || !compareRunB)) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center py-20 gap-4 text-slate-100">
        <div className="h-12 w-12 rounded-full border-t-2 border-indigo-500 animate-spin"></div>
        <p className="text-sm text-gray-400 font-medium">Loading comparisons...</p>
      </div>
    )
  }

  if (error || !compareRunA || !compareRunB) {
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
          <p className="text-sm font-medium">{error || 'Comparison details could not be loaded'}</p>
        </div>
      </div>
    )
  }

  return (
    <div className="flex-1 flex flex-col min-h-screen text-slate-100 pb-10">
      {/* Breadcrumbs */}
      <header className="border-b border-surface-border bg-surface-overlay/30 backdrop-blur-md sticky top-0 z-50 px-6 py-4 flex items-center justify-between">
        <div className="flex items-center gap-2 text-xs font-semibold text-gray-400 uppercase tracking-wider">
          <span className="hover:text-white cursor-pointer" onClick={() => navigate('/aeh/reports')}>
            Runs
          </span>
          <span>/</span>
          <span className="text-white">Run Comparison picker</span>
        </div>
        <button
          onClick={() => navigate('/aeh/reports')}
          className="px-4 py-2 text-sm font-medium text-gray-400 hover:text-gray-200 bg-zinc-900 border border-zinc-850 hover:border-zinc-700 rounded-lg transition-all cursor-pointer"
        >
          &larr; Back to Runs
        </button>
      </header>

      {/* Main Content Area */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-6 flex flex-col gap-6">
        <div className="glass rounded-2xl p-6 relative overflow-hidden">
          <h2 className="text-xl font-bold text-gray-100 flex items-center gap-2">
            <TrendingUp className="h-5 w-5 text-indigo-400" />
            <span>Run A vs Run B Comparison</span>
          </h2>
          <p className="text-xs text-gray-400 mt-1 uppercase tracking-wider font-semibold">
            Compare metric outcomes between baseline and target runs side-by-side
          </p>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-6">
            <div className="bg-indigo-950/20 border border-indigo-900/30 rounded-xl p-4">
              <p className="text-[10px] text-indigo-400 font-bold uppercase">Baseline (Run A)</p>
              <p className="text-sm font-bold text-gray-100 mt-1">{compareRunA.target_system_id} ({compareRunA.id.slice(0, 8)})</p>
              <p className="text-xs text-gray-400 mt-1">Pass Rate: {(compareRunA.overall_pass_rate * 100).toFixed(0)}%</p>
            </div>
            <div className="bg-purple-950/20 border border-purple-900/30 rounded-xl p-4">
              <p className="text-[10px] text-purple-400 font-bold uppercase">Comparison (Run B)</p>
              <p className="text-sm font-bold text-gray-100 mt-1">{compareRunB.target_system_id} ({compareRunB.id.slice(0, 8)})</p>
              <p className="text-xs text-gray-400 mt-1">Pass Rate: {(compareRunB.overall_pass_rate * 100).toFixed(0)}%</p>
            </div>
          </div>
        </div>

        {/* Diffing metric tables */}
        <div className="glass rounded-2xl overflow-hidden">
          <div className="px-6 py-4 border-b border-surface-border bg-surface-overlay flex items-center justify-between">
            <h3 className="text-md font-bold text-gray-100 flex items-center gap-2">
              <FileText className="h-4 w-4 text-indigo-400" />
              <span>Metric Delta Comparison</span>
            </h3>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse">
              <thead>
                <tr className="border-b border-surface-border bg-surface-overlay/50 text-gray-400 text-xs font-semibold uppercase">
                  <th className="py-3 px-6">Evaluation Metric</th>
                  <th className="py-3 px-6">User Query Context</th>
                  <th className="py-3 px-6 text-center">Run A Outcome</th>
                  <th className="py-3 px-6 text-center">Run B Outcome</th>
                  <th className="py-3 px-6 text-right">Delta status</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-surface-border/60 text-sm">
                {compareEvalsA.map((evA) => {
                  const evB = compareEvalsB.find(
                    (b) => b.metric_name === evA.metric_name && b.trace_id === evA.trace_id
                  )
                  const passesA = evA.passed
                  const passesB = evB ? evB.passed : null

                  return (
                    <tr key={evA.id} className="hover:bg-surface-overlay/10">
                      <td className="py-4 px-6 font-semibold text-gray-100">
                        {evA.metric_name}
                      </td>
                      <td className="py-4 px-6 text-xs text-gray-300 max-w-xs truncate" title={evA.root_input || ''}>
                        {evA.root_input || 'N/A'}
                      </td>
                      <td className="py-4 px-6 text-center">
                        <span className={`inline-flex items-center gap-1 ${passesA ? 'text-emerald-400' : 'text-red-400'}`}>
                          {passesA ? <CheckCircle2 className="h-4 w-4" /> : <XCircle className="h-4 w-4" />}
                          <span>{passesA ? 'PASS' : 'FAIL'}</span>
                        </span>
                      </td>
                      <td className="py-4 px-6 text-center">
                        {evB ? (
                          <span className={`inline-flex items-center gap-1 ${passesB ? 'text-emerald-400' : 'text-red-400'}`}>
                            {passesB ? <CheckCircle2 className="h-4 w-4" /> : <XCircle className="h-4 w-4" />}
                            <span>{passesB ? 'PASS' : 'FAIL'}</span>
                          </span>
                        ) : (
                          <span className="text-gray-500 italic">N/A</span>
                        )}
                      </td>
                      <td className="py-4 px-6 text-right">
                        {passesA && !passesB ? (
                          <span className="inline-flex items-center gap-1 text-red-500 font-bold text-xs">
                            <TrendingDown className="h-4 w-4" />
                            <span>REGRESSION</span>
                          </span>
                        ) : !passesA && passesB ? (
                          <span className="inline-flex items-center gap-1 text-emerald-500 font-bold text-xs">
                            <TrendingUp className="h-4 w-4" />
                            <span>IMPROVEMENT</span>
                          </span>
                        ) : (
                          <span className="text-gray-500 text-xs">No Change</span>
                        )}
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </div>
      </main>
    </div>
  )
}

import React, { useEffect } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import {
  Code,
  FileText,
  GitBranch,
  CheckCircle2,
  XCircle,
  AlertTriangle
} from 'lucide-react'
import { useAEHReportsStore } from '../../store/aeh-reports.store'

export default function ComponentDetailScreen(): React.ReactElement {
  const { runId, componentId } = useParams<{ runId: string; componentId: string }>()
  const navigate = useNavigate()

  const {
    selectedRun,
    evaluations,
    loading,
    error,
    fetchComponentEvaluations
  } = useAEHReportsStore()

  useEffect(() => {
    if (runId && componentId) {
      fetchComponentEvaluations(runId, componentId).catch(() => {})
    }
  }, [runId, componentId, fetchComponentEvaluations])

  if (loading && evaluations.length === 0) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center py-20 gap-4 text-slate-100">
        <div className="h-12 w-12 rounded-full border-t-2 border-indigo-500 animate-spin"></div>
        <p className="text-sm text-gray-400 font-medium">Loading component details...</p>
      </div>
    )
  }

  if (error || !selectedRun) {
    return (
      <div className="flex-1 p-6 text-slate-100">
        <button
          onClick={() => navigate(runId ? `/aeh/reports/runs/${runId}` : '/aeh/reports')}
          className="mb-4 px-4 py-2 text-sm font-medium text-gray-400 hover:text-gray-200 bg-zinc-900 border border-zinc-800 rounded-lg transition-all"
        >
          &larr; Back to Run Details
        </button>
        <div className="p-4 rounded-xl border border-red-500/20 bg-red-950/20 text-red-400 flex items-center gap-3">
          <AlertTriangle className="h-5 w-5 flex-shrink-0" />
          <p className="text-sm font-medium">{error || 'Component details could not be loaded'}</p>
        </div>
      </div>
    )
  }

  const componentConfig = selectedRun.system_map.components.find((c) => c.id === componentId)

  return (
    <div className="flex-1 flex flex-col min-h-screen text-slate-100 pb-10">
      {/* Breadcrumbs */}
      <header className="border-b border-surface-border bg-surface-overlay/30 backdrop-blur-md sticky top-0 z-50 px-6 py-4 flex items-center justify-between">
        <div className="flex items-center gap-2 text-xs font-semibold text-gray-400 uppercase tracking-wider">
          <span className="hover:text-white cursor-pointer" onClick={() => navigate('/aeh/reports')}>
            Runs
          </span>
          <span>/</span>
          <span className="hover:text-white cursor-pointer" onClick={() => navigate(`/aeh/reports/runs/${selectedRun.id}`)}>
            Run Details ({selectedRun.id.slice(0, 8)})
          </span>
          <span>/</span>
          <span className="text-white">Component: {componentId}</span>
        </div>
        <button
          onClick={() => navigate(`/aeh/reports/runs/${selectedRun.id}`)}
          className="px-4 py-2 text-sm font-medium text-gray-400 hover:text-gray-200 bg-zinc-900 border border-zinc-850 hover:border-zinc-700 rounded-lg transition-all cursor-pointer"
        >
          &larr; Back to Run Details
        </button>
      </header>

      {/* Main Content Area */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-6 flex flex-col gap-6">
        {/* Component Detail Summary Card */}
        <div className="glass rounded-2xl p-6 relative overflow-hidden">
          <h2 className="text-xl font-bold text-gray-100">{componentId}</h2>
          <p className="text-xs text-gray-400 mt-1 uppercase tracking-wider font-semibold">
            Component Configuration details from system map
          </p>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-6">
            <div className="bg-surface-overlay/40 border border-surface-border rounded-xl p-4">
              <p className="text-[10px] text-gray-500 font-bold uppercase">Component Role</p>
              <p className="text-sm font-bold text-gray-300 mt-1">
                {componentConfig?.role || 'N/A'}
              </p>
            </div>
            <div className="bg-surface-overlay/40 border border-surface-border rounded-xl p-4">
              <p className="text-[10px] text-gray-500 font-bold uppercase">Implementation Model</p>
              <p className="text-sm font-bold text-gray-300 mt-1 font-mono">
                {componentConfig?.model || 'N/A'}
              </p>
            </div>
            <div className="bg-surface-overlay/40 border border-surface-border rounded-xl p-4">
              <p className="text-[10px] text-gray-500 font-bold uppercase">Entry Point</p>
              <p className="text-sm font-bold text-gray-300 mt-1 font-mono truncate" title={componentConfig?.entry_point || ''}>
                {componentConfig?.entry_point || 'N/A'}
              </p>
            </div>
          </div>
        </div>

        {/* Drill Down Panels grouped by metric_class */}
        <div className="flex flex-col gap-6">
          {/* Class 1: Assertions */}
          {evaluations.filter((ev) => ev.metric_class === 'assertion').length > 0 && (
            <div className="glass rounded-2xl overflow-hidden">
              <div className="px-6 py-4 border-b border-surface-border bg-surface-overlay flex items-center justify-between">
                <h3 className="text-md font-bold text-gray-100 flex items-center gap-2">
                  <Code className="h-4 w-4 text-indigo-400" />
                  <span>Structural Assertions</span>
                </h3>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full text-left border-collapse">
                  <thead>
                    <tr className="border-b border-surface-border bg-surface-overlay/50 text-gray-400 text-xs font-semibold uppercase">
                      <th className="py-3 px-6">Assertion Metric</th>
                      <th className="py-3 px-6">User Query Context</th>
                      <th className="py-3 px-6">Details / Params</th>
                      <th className="py-3 px-6">Outcome</th>
                      <th className="py-3 px-6 text-right">Trace Link</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-surface-border/60 text-sm">
                    {evaluations
                      .filter((ev) => ev.metric_class === 'assertion')
                      .map((ev) => (
                        <tr key={ev.id} className="hover:bg-surface-overlay/10">
                          <td className="py-4 px-6 font-semibold text-gray-100">
                            {ev.metric_name}
                          </td>
                          <td className="py-4 px-6 text-xs text-gray-300 max-w-xs truncate" title={ev.root_input || ''}>
                            {ev.root_input || 'N/A'}
                          </td>
                          <td className="py-4 px-6 text-xs font-mono text-gray-400 max-w-xs truncate" title={JSON.stringify(ev.details)}>
                            {JSON.stringify(ev.details)}
                          </td>
                          <td className="py-4 px-6">
                            <span className={`inline-flex items-center gap-1 font-semibold text-xs ${
                              ev.passed ? 'text-emerald-400' : 'text-red-400'
                            }`}>
                              {ev.passed ? <CheckCircle2 className="h-4 w-4" /> : <XCircle className="h-4 w-4" />}
                              <span>{ev.passed ? 'PASS' : 'FAIL'}</span>
                            </span>
                          </td>
                          <td className="py-4 px-6 text-right">
                            {ev.trace_id && (
                              <button
                                onClick={() => navigate(`/aeh/reports/runs/${runId}/traces/${ev.trace_id}?componentId=${componentId}`)}
                                className="text-xs text-indigo-400 hover:text-indigo-300 font-bold bg-transparent border-0 cursor-pointer p-0"
                              >
                                Open Trace Tree
                              </button>
                            )}
                          </td>
                        </tr>
                      ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Class 2: Classifiers (Confusion Matrix) */}
          {evaluations.filter((ev) => ev.metric_class === 'classifier').length > 0 && (
            <div className="glass rounded-2xl overflow-hidden p-6 flex flex-col gap-4">
              <h3 className="text-md font-bold text-gray-100 flex items-center gap-2 border-b border-surface-border pb-3">
                <GitBranch className="h-4 w-4 text-indigo-400" />
                <span>Classification Accuracy (Confusion Matrix)</span>
              </h3>

              {evaluations
                .filter((ev) => ev.metric_class === 'classifier')
                .map((ev) => {
                  const matrix = ev.details.confusion_matrix || ev.details.matrix || {}
                  return (
                    <div key={ev.id} className="grid grid-cols-1 md:grid-cols-2 gap-6 items-center">
                      <div>
                        <p className="text-sm font-bold text-indigo-400">{ev.metric_name}</p>
                        <p className="text-xs text-gray-400 mt-2 font-medium">
                          Overall Accuracy Score: <span className="font-bold text-gray-100">{ev.score !== null ? `${(ev.score * 100).toFixed(1)}%` : 'N/A'}</span>
                        </p>
                        {ev.details.unmatched_classes && (
                          <p className="text-xs text-amber-500 mt-1 font-semibold">
                            Unmatched prediction tags occurred.
                          </p>
                        )}
                      </div>

                      <div className="border border-surface-border rounded-xl p-4 bg-surface/60 font-mono text-xs text-gray-300">
                        <p className="text-[10px] font-bold text-gray-500 uppercase border-b border-surface-border pb-2 mb-2">
                          Confusion Matrix Grid
                        </p>
                        <pre className="overflow-x-auto">
                          {JSON.stringify(matrix, null, 2)}
                        </pre>
                      </div>
                    </div>
                  )
                })}
            </div>
          )}

          {/* Class 3: LLM Judges */}
          {evaluations.filter((ev) => ev.metric_class === 'llm_judge').length > 0 && (
            <div className="glass rounded-2xl overflow-hidden p-6 flex flex-col gap-6">
              <h3 className="text-md font-bold text-gray-100 flex items-center gap-2 border-b border-surface-border pb-3">
                <FileText className="h-4 w-4 text-indigo-400" />
                <span>LLM Judge Evaluations</span>
              </h3>

              <div className="flex flex-col gap-6">
                {evaluations
                  .filter((ev) => ev.metric_class === 'llm_judge')
                  .map((ev) => (
                    <div key={ev.id} className="border border-surface-border rounded-xl p-5 bg-surface/30 flex flex-col gap-3">
                      <div className="flex items-center justify-between">
                        <div>
                          <p className="text-sm font-bold text-gray-100">{ev.metric_name}</p>
                          <p className="text-[10px] text-gray-500 font-semibold uppercase">Evaluated via: {ev.evaluator || 'LLM'}</p>
                        </div>

                        <div className="text-right">
                          <span className="text-lg font-bold text-indigo-400">
                            {ev.score !== null ? ev.score.toFixed(2) : 'N/A'}
                          </span>
                          {ev.cost_tokens && (
                            <p className="text-[10px] text-purple-400 font-bold">
                              {ev.cost_tokens} tokens
                            </p>
                          )}
                        </div>
                      </div>

                      {ev.details.rubric_text && (
                        <div className="p-3 bg-zinc-950 border border-surface-border rounded-lg text-xs text-gray-400 font-medium">
                          <span className="font-bold text-gray-300 block mb-1">Tailored Rubric:</span>
                          {ev.details.rubric_text}
                        </div>
                      )}

                      {ev.details.reason && (
                        <div className="text-xs text-gray-300 italic pl-3 border-l-2 border-gray-600">
                          &ldquo;{ev.details.reason}&rdquo;
                        </div>
                      )}

                      <div className="flex items-center justify-between text-xs mt-3 border-t border-surface-border/40 pt-3">
                        <span className="text-gray-500">Query ID: {ev.trace_id?.slice(0, 8)}...</span>
                        {ev.trace_id && (
                          <button
                            onClick={() => navigate(`/aeh/reports/runs/${runId}/traces/${ev.trace_id}?componentId=${componentId}`)}
                            className="text-indigo-400 hover:text-indigo-300 font-bold bg-transparent border-0 cursor-pointer p-0"
                          >
                            View Spans In Trace Tree &rarr;
                          </button>
                        )}
                      </div>
                    </div>
                  ))}
              </div>
            </div>
          )}
        </div>
      </main>
    </div>
  )
}

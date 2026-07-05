import React, { useEffect, useState } from 'react'
import { useParams, useNavigate, useSearchParams } from 'react-router-dom'
import {
  GitCommit,
  Code,
  AlertTriangle
} from 'lucide-react'
import { useAEHReportsStore } from '../../store/aeh-reports.store'
import type { AEHTraceSpan } from '../../types/electron'

function formatLatency(ms: number | null) {
  if (ms === null) return 'N/A'
  if (ms < 1000) return `${ms}ms`
  return `${(ms / 1000).toFixed(2)}s`
}

export default function TraceViewerScreen(): React.ReactElement {
  const { runId, traceId } = useParams<{ runId: string; traceId: string }>()
  const navigate = useNavigate()
  const [searchParams] = useSearchParams()
  const componentId = searchParams.get('componentId')

  const {
    traceData,
    loading,
    error,
    fetchTraceDetail
  } = useAEHReportsStore()

  const [selectedSpan, setSelectedSpan] = useState<AEHTraceSpan | null>(null)

  useEffect(() => {
    if (traceId) {
      fetchTraceDetail(traceId).catch(() => {})
    }
  }, [traceId, fetchTraceDetail])

  // Select default span if not set
  useEffect(() => {
    if (traceData && traceData.spans && traceData.spans.length > 0 && !selectedSpan) {
      setSelectedSpan(traceData.spans[0])
    }
  }, [traceData, selectedSpan])

  if (loading && !traceData) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center py-20 gap-4 text-slate-100">
        <div className="h-12 w-12 rounded-full border-t-2 border-indigo-500 animate-spin"></div>
        <p className="text-sm text-gray-400 font-medium">Loading trace spans...</p>
      </div>
    )
  }

  if (error || !traceData) {
    return (
      <div className="flex-1 p-6 text-slate-100">
        <button
          onClick={() => navigate(runId ? `/aeh/reports/runs/${runId}` : '/aeh/reports')}
          className="mb-4 px-4 py-2 text-sm font-medium text-gray-400 hover:text-gray-200 bg-zinc-900 border border-zinc-800 rounded-lg transition-all"
        >
          &larr; Back
        </button>
        <div className="p-4 rounded-xl border border-red-500/20 bg-red-950/20 text-red-400 flex items-center gap-3">
          <AlertTriangle className="h-5 w-5 flex-shrink-0" />
          <p className="text-sm font-medium">{error || 'Trace detail could not be loaded'}</p>
        </div>
      </div>
    )
  }

  const handleBack = () => {
    if (componentId && runId) {
      navigate(`/aeh/reports/runs/${runId}/components/${componentId}`)
    } else if (runId) {
      navigate(`/aeh/reports/runs/${runId}`)
    } else {
      navigate('/aeh/reports')
    }
  }

  const backLabel = componentId ? `Back to ${componentId}` : 'Back to Run Details'

  return (
    <div className="flex-1 flex flex-col min-h-screen text-slate-100 pb-10">
      {/* Breadcrumbs */}
      <header className="border-b border-surface-border bg-surface-overlay/30 backdrop-blur-md sticky top-0 z-50 px-6 py-4 flex items-center justify-between">
        <div className="flex items-center gap-2 text-xs font-semibold text-gray-400 uppercase tracking-wider">
          <span className="hover:text-white cursor-pointer" onClick={() => navigate('/aeh/reports')}>
            Runs
          </span>
          <span>/</span>
          {runId && (
            <>
              <span className="hover:text-white cursor-pointer" onClick={() => navigate(`/aeh/reports/runs/${runId}`)}>
                Run Details ({runId.slice(0, 8)})
              </span>
              <span>/</span>
            </>
          )}
          {componentId && runId && (
            <>
              <span
                className="hover:text-white cursor-pointer"
                onClick={() => navigate(`/aeh/reports/runs/${runId}/components/${componentId}`)}
              >
                Component: {componentId}
              </span>
              <span>/</span>
            </>
          )}
          <span className="text-white">Trace Explorer: {traceId?.slice(0, 8)}...</span>
        </div>
        <button
          onClick={handleBack}
          className="px-4 py-2 text-sm font-medium text-gray-400 hover:text-gray-200 bg-zinc-900 border border-zinc-850 hover:border-zinc-700 rounded-lg transition-all cursor-pointer"
        >
          &larr; {backLabel}
        </button>
      </header>

      {/* Main Content Area */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-6 flex flex-col gap-6">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 flex-1 items-stretch">
          {/* Left Column: Trace Span Tree */}
          <div className="glass rounded-2xl p-6 lg:col-span-1 flex flex-col gap-4">
            <h3 className="text-md font-bold text-gray-100 flex items-center gap-2 border-b border-surface-border pb-3">
              <GitCommit className="h-5 w-5 text-indigo-400" />
              <span>Trace Execution Spans</span>
            </h3>

            <div className="flex flex-col gap-2 flex-1 overflow-y-auto max-h-[500px] pr-2">
              {traceData.spans.map((span) => {
                const isSelected = selectedSpan?.id === span.id
                return (
                  <div
                    key={span.id}
                    onClick={() => setSelectedSpan(span)}
                    className={`p-3 rounded-lg border text-left cursor-pointer transition-all ${
                      isSelected
                        ? 'bg-indigo-950/20 border-indigo-500 text-indigo-300 font-bold'
                        : 'bg-slate-900/40 border-slate-800/80 text-gray-400 hover:border-slate-700'
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-bold truncate max-w-[120px]">{span.component_id || span.span_type}</span>
                      <span className="text-[10px] px-1 bg-slate-800 text-gray-500 rounded font-mono">
                        {span.span_type}
                      </span>
                    </div>
                    <div className="flex items-center justify-between text-[10px] text-gray-500 mt-2">
                      <span>{formatLatency(span.latency_ms)}</span>
                      {span.tokens_in !== null && (
                        <span>{span.tokens_in + (span.tokens_out || 0)} tokens</span>
                      )}
                    </div>
                  </div>
                )
              })}
            </div>
          </div>

          {/* Right Column: Span detail pane */}
          <div className="glass rounded-2xl p-6 lg:col-span-2 flex flex-col gap-4">
            <h3 className="text-md font-bold text-gray-100 flex items-center gap-2 border-b border-surface-border pb-3">
              <Code className="h-5 w-5 text-indigo-400" />
              <span>Span Details: {selectedSpan?.component_id || selectedSpan?.span_type || 'N/A'}</span>
            </h3>

            {selectedSpan ? (
              <div className="flex flex-col gap-4 flex-1">
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-xs">
                  <div className="bg-slate-900/60 p-3 border border-slate-800/80 rounded-xl">
                    <span className="text-[10px] text-gray-500 font-bold uppercase">Span ID</span>
                    <span className="font-mono text-gray-300 block mt-1">{selectedSpan.id.slice(0, 8)}...</span>
                  </div>
                  <div className="bg-slate-900/60 p-3 border border-slate-800/80 rounded-xl">
                    <span className="text-[10px] text-gray-500 font-bold uppercase">Latency</span>
                    <span className="text-gray-300 block mt-1 font-semibold">{formatLatency(selectedSpan.latency_ms)}</span>
                  </div>
                  <div className="bg-slate-900/60 p-3 border border-slate-800/80 rounded-xl">
                    <span className="text-[10px] text-gray-500 font-bold uppercase">Tokens In</span>
                    <span className="text-gray-300 block mt-1 font-semibold">{selectedSpan.tokens_in ?? 'N/A'}</span>
                  </div>
                  <div className="bg-slate-900/60 p-3 border border-slate-800/80 rounded-xl">
                    <span className="text-[10px] text-gray-500 font-bold uppercase">Tokens Out</span>
                    <span className="text-gray-300 block mt-1 font-semibold">{selectedSpan.tokens_out ?? 'N/A'}</span>
                  </div>
                </div>

                {/* Input JSON */}
                <div className="flex flex-col gap-1.5">
                  <p className="text-[10px] text-gray-500 font-bold uppercase">Received Payload / Input</p>
                  <div className="bg-slate-950/80 border border-slate-800 rounded-xl p-4 font-mono text-xs text-gray-300 overflow-x-auto max-h-[160px]">
                    <pre>{JSON.stringify(JSON.parse(selectedSpan.input_json || '{}'), null, 2)}</pre>
                  </div>
                </div>

                {/* Output JSON */}
                <div className="flex flex-col gap-1.5">
                  <p className="text-[10px] text-gray-500 font-bold uppercase">Final Output / Response</p>
                  <div className="bg-slate-950/80 border border-slate-800 rounded-xl p-4 font-mono text-xs text-gray-300 overflow-x-auto max-h-[160px]">
                    <pre>{JSON.stringify(JSON.parse(selectedSpan.output_json || '{}'), null, 2)}</pre>
                  </div>
                </div>
              </div>
            ) : (
              <div className="flex-1 flex items-center justify-center text-gray-500 text-xs italic">
                Select a span from the list to view input/output.
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  )
}

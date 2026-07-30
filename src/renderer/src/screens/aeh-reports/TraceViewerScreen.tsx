import React, { useEffect, useState } from 'react'
import { useParams, useNavigate, useSearchParams } from 'react-router-dom'
import {
  GitCommit,
  Code,
  AlertTriangle,
  ArrowLeft,
  Clock,
  Cpu,
  Layers,
  CheckCircle2,
  Info,
  ChevronDown,
  ChevronRight
} from 'lucide-react'
import { useAEHReportsStore, EvalCaseItem } from '../../store/aeh-reports.store'
import type { AEHTraceSpan } from '../../types/electron'

function formatLatency(ms: number | null) {
  if (ms === null || ms === undefined) return 'N/A'
  if (ms < 1000) return `${ms}ms`
  return `${(ms / 1000).toFixed(2)}s`
}

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

export default function TraceViewerScreen(): React.ReactElement {
  const { runId, traceId } = useParams<{ runId: string; traceId: string }>()
  const navigate = useNavigate()
  const [searchParams] = useSearchParams()
  const componentId = searchParams.get('componentId')

  const {
    traceData,
    selectedRunCases,
    loading,
    error,
    fetchTraceDetail,
    fetchRunCases
  } = useAEHReportsStore()

  const [selectedSpan, setSelectedSpan] = useState<AEHTraceSpan | null>(null)
  const [inputExpanded, setInputExpanded] = useState<boolean>(true)
  const [outputExpanded, setOutputExpanded] = useState<boolean>(true)

  useEffect(() => {
    if (traceId) {
      fetchTraceDetail(traceId).catch(() => {})
    }
    if (runId && !selectedRunCases) {
      fetchRunCases(runId).catch(() => {})
    }
  }, [traceId, runId, fetchTraceDetail, fetchRunCases, selectedRunCases])

  // Select default span if not set
  useEffect(() => {
    if (traceData && traceData.spans && traceData.spans.length > 0 && !selectedSpan) {
      setSelectedSpan(traceData.spans[0])
    }
  }, [traceData, selectedSpan])

  if (loading && !traceData) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center py-20 gap-4 text-slate-100 bg-slate-950">
        <div className="h-10 w-10 rounded-full border-t-2 border-indigo-500 animate-spin"></div>
        <p className="text-sm text-gray-400">Loading trace spans...</p>
      </div>
    )
  }

  if (error || !traceData) {
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
          <p>{error || 'Trace detail could not be loaded'}</p>
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

  const backLabel = componentId ? `Back to Agent: ${componentId}` : 'Back to Run Details'

  // Extract trace metadata (C1)
  const rawTrace = traceData.trace as Record<string, any> | undefined
  const spans = traceData.spans || []
  const isSingleSpan = spans.length <= 1

  // Find matching case item from store if available (C2)
  let matchingCase: EvalCaseItem | undefined
  if (selectedRunCases && selectedRunCases.agents) {
    for (const agentId of Object.keys(selectedRunCases.agents)) {
      const list = selectedRunCases.agents[agentId] || []
      const found = list.find((c) => c.trace_id === traceId || c.case_id === rawTrace?.dataset_case_id)
      if (found) {
        matchingCase = found
        break
      }
    }
  }

  const semanticEval = matchingCase?.evaluations?.find((e) => e.metric_name === 'semantic_match')
  const caseFlags = matchingCase?.flags

  // Token usage display (C1)
  const totalTokens = rawTrace?.total_tokens ?? null
  const tokensDisplay =
    totalTokens !== null
      ? `${totalTokens.toLocaleString()} tokens`
      : 'not captured (single-node agent tracer)'

  return (
    <div className="flex-1 flex flex-col min-h-screen text-slate-100 bg-slate-950 pb-10">
      {/* Navigation Header */}
      <header className="border-b border-slate-800 bg-slate-900 px-6 py-4 flex items-center justify-between sticky top-0 z-50">
        <div className="flex items-center gap-2 text-xs font-semibold text-gray-400 uppercase tracking-wider">
          <span className="hover:text-white cursor-pointer" onClick={() => navigate('/aeh/reports')}>
            Runs
          </span>
          <span>/</span>
          {runId && (
            <>
              <span className="hover:text-white cursor-pointer" onClick={() => navigate(`/aeh/reports/runs/${runId}`)}>
                {runId.slice(0, 8)}
              </span>
              <span>/</span>
            </>
          )}
          {componentId && runId && (
            <>
              <span
                className="hover:text-white cursor-pointer font-mono"
                onClick={() => navigate(`/aeh/reports/runs/${runId}/components/${componentId}`)}
              >
                Agent: {componentId}
              </span>
              <span>/</span>
            </>
          )}
          <span className="text-white font-mono">Trace: {traceId?.slice(0, 8)}...</span>
        </div>

        <button
          onClick={handleBack}
          className="px-4 py-2 text-xs font-semibold text-gray-300 hover:text-white bg-slate-800 border border-slate-700 rounded-lg transition-colors flex items-center gap-1.5 cursor-pointer"
        >
          <ArrowLeft className="h-3.5 w-3.5" />
          <span>{backLabel}</span>
        </button>
      </header>

      {/* Main Content Area */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-6 flex flex-col gap-6">
        {/* Trace Summary Banner (C1 & C2) */}
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 flex flex-col gap-4">
          <div className="flex flex-wrap items-center justify-between gap-4 border-b border-slate-800/80 pb-4">
            <div className="flex flex-col gap-1">
              <div className="flex items-center gap-3">
                <span className="text-xs text-indigo-400 font-mono font-semibold uppercase tracking-wider">
                  Trace Explorer
                </span>
                <span className="text-xs px-2.5 py-0.5 rounded-full font-semibold bg-slate-800 text-slate-300 border border-slate-700 uppercase">
                  {rawTrace?.status || 'COMPLETED'}
                </span>
              </div>
              <h2 className="text-xl font-bold text-gray-100 font-mono">
                {rawTrace?.dataset_case_id || matchingCase?.case_id || traceId}
              </h2>
            </div>

            {/* Evaluation Score Context (C2) */}
            {semanticEval && (
              <div className="flex items-center gap-4 bg-slate-950 border border-slate-800 rounded-xl p-3">
                <div className="flex flex-col text-right">
                  <span className="text-[11px] text-gray-400 font-semibold uppercase">Semantic Score</span>
                  <span className={`text-2xl font-extrabold ${_scoreColor(semanticEval.score)}`}>
                    {semanticEval.score !== null ? `${(semanticEval.score * 100).toFixed(0)}%` : 'N/A'}
                  </span>
                </div>
                {semanticEval.details && typeof semanticEval.details.notes === 'string' && (
                  <div className="border-l border-slate-800 pl-3 max-w-xs text-xs text-gray-300 leading-snug">
                    <span className="font-bold text-gray-400 block mb-0.5">Judge Note:</span>
                    {semanticEval.details.notes}
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Heuristic Audit Reasons Banner (C2) */}
          {caseFlags?.reasons && caseFlags.reasons.length > 0 && (
            <div className="p-3 bg-amber-950/20 border border-amber-800/40 rounded-lg text-xs flex flex-col gap-1">
              <span className="text-xs font-bold text-amber-400 uppercase tracking-wider flex items-center gap-1">
                <AlertTriangle className="h-3.5 w-3.5" />
                Case Heuristic Audit Flags ({caseFlags.reasons.length})
              </span>
              <ul className="list-disc list-inside text-amber-200 space-y-0.5 font-mono text-xs">
                {caseFlags.reasons.map((r, idx) => (
                  <li key={idx}>{r}</li>
                ))}
              </ul>
            </div>
          )}

          {/* Metric Bar: Latency & Tokens (C1) */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs font-medium">
            <div className="flex items-center gap-2 bg-slate-950 border border-slate-800/80 rounded-lg p-3">
              <Clock className="h-4 w-4 text-indigo-400 flex-shrink-0" />
              <span className="text-gray-400">Total Latency:</span>
              <span className="font-mono text-gray-100 font-bold">
                {formatLatency(rawTrace?.total_latency_ms ?? selectedSpan?.latency_ms ?? null)}
              </span>
            </div>

            <div className="flex items-center gap-2 bg-slate-950 border border-slate-800/80 rounded-lg p-3">
              <Cpu className="h-4 w-4 text-purple-400 flex-shrink-0" />
              <span className="text-gray-400">Token Usage:</span>
              <span className="font-mono text-gray-300">{tokensDisplay}</span>
            </div>

            <div className="flex items-center gap-2 bg-slate-950 border border-slate-800/80 rounded-lg p-3">
              <Layers className="h-4 w-4 text-emerald-400 flex-shrink-0" />
              <span className="text-gray-400">Captured Spans:</span>
              <span className="font-mono text-gray-100 font-bold">{spans.length} node(s)</span>
            </div>
          </div>
        </div>

        {/* Span Layout: Full-Width Degradation (C4) vs Multi-Span Tree */}
        {isSingleSpan ? (
          /* Single Span Full-Width Detail Layout (C4) */
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 flex flex-col gap-6">
            <div className="flex items-center justify-between border-b border-slate-800 pb-4">
              <h3 className="text-base font-bold text-gray-100 flex items-center gap-2">
                <Code className="h-4.5 w-4.5 text-indigo-400" />
                <span>Agent Node Span Details: {selectedSpan?.component_id || selectedSpan?.span_type || 'N/A'}</span>
              </h3>
              {selectedSpan?.span_type && (
                <span className="text-xs font-mono font-semibold px-2.5 py-0.5 rounded bg-slate-800 text-indigo-300 border border-slate-700">
                  {selectedSpan.span_type}
                </span>
              )}
            </div>

            {/* Span Error Banner (C3) */}
            {selectedSpan?.error && (
              <div className="p-4 rounded-xl border border-rose-900/50 bg-rose-950/30 text-rose-300 flex items-start gap-3 text-xs">
                <AlertTriangle className="h-5 w-5 text-rose-400 flex-shrink-0 mt-0.5" />
                <div className="flex flex-col gap-1">
                  <span className="font-bold text-sm">Span Error Triggered</span>
                  <pre className="font-mono whitespace-pre-wrap">{selectedSpan.error}</pre>
                </div>
              </div>
            )}

            {/* Payload Collapsible Cards */}
            <div className="flex flex-col gap-4">
              {/* Input JSON */}
              <div className="bg-slate-950 border border-slate-800 rounded-lg overflow-hidden">
                <div
                  onClick={() => setInputExpanded(!inputExpanded)}
                  className="px-4 py-3 bg-slate-900/60 flex items-center justify-between cursor-pointer hover:bg-slate-900 transition-colors"
                >
                  <div className="flex items-center gap-2 text-xs font-bold text-gray-300 uppercase tracking-wider">
                    {inputExpanded ? <ChevronDown className="h-4 w-4 text-gray-400" /> : <ChevronRight className="h-4 w-4 text-gray-400" />}
                    <span>Received Payload / Input</span>
                  </div>
                  <span className="text-xs text-gray-500 font-mono">Click to toggle</span>
                </div>
                {inputExpanded && (
                  <pre className="p-4 text-xs font-mono text-gray-300 overflow-x-auto max-h-96 whitespace-pre-wrap border-t border-slate-800">
                    {formatJsonOrString(selectedSpan?.input_json)}
                  </pre>
                )}
              </div>

              {/* Output JSON */}
              <div className="bg-slate-950 border border-slate-800 rounded-lg overflow-hidden">
                <div
                  onClick={() => setOutputExpanded(!outputExpanded)}
                  className="px-4 py-3 bg-slate-900/60 flex items-center justify-between cursor-pointer hover:bg-slate-900 transition-colors"
                >
                  <div className="flex items-center gap-2 text-xs font-bold text-indigo-300 uppercase tracking-wider">
                    {outputExpanded ? <ChevronDown className="h-4 w-4 text-gray-400" /> : <ChevronRight className="h-4 w-4 text-gray-400" />}
                    <span>Final Output / Response</span>
                  </div>
                  <span className="text-xs text-gray-500 font-mono">Click to toggle</span>
                </div>
                {outputExpanded && (
                  <pre className="p-4 text-xs font-mono text-indigo-200 overflow-x-auto max-h-96 whitespace-pre-wrap border-t border-slate-800">
                    {formatJsonOrString(selectedSpan?.output_json)}
                  </pre>
                )}
              </div>
            </div>
          </div>
        ) : (
          /* Multi-Span Tree + Detail Split Layout */
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 items-start">
            {/* Left Tree Pane */}
            <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 lg:col-span-1 flex flex-col gap-4">
              <h3 className="text-sm font-bold text-gray-100 flex items-center gap-2 border-b border-slate-800 pb-3">
                <GitCommit className="h-4 w-4 text-indigo-400" />
                <span>Execution Spans</span>
              </h3>

              <div className="flex flex-col gap-2 overflow-y-auto max-h-[500px] pr-1">
                {spans.map((span) => {
                  const isSelected = selectedSpan?.id === span.id
                  return (
                    <div
                      key={span.id}
                      onClick={() => setSelectedSpan(span)}
                      className={`p-3 rounded-lg border text-left cursor-pointer transition-colors ${
                        isSelected
                          ? 'bg-indigo-950/40 border-indigo-500/80 text-indigo-200 font-bold'
                          : 'bg-slate-950 border-slate-800 text-gray-400 hover:border-slate-700'
                      }`}
                    >
                      <div className="flex items-center justify-between">
                        <span className="text-xs font-bold truncate">{span.component_id || span.span_type}</span>
                        <span className="text-[10px] px-1.5 py-0.5 bg-slate-800 text-gray-400 rounded font-mono">
                          {span.span_type}
                        </span>
                      </div>
                      <div className="flex items-center justify-between text-[11px] text-gray-500 mt-2 font-mono">
                        <span>{formatLatency(span.latency_ms)}</span>
                        {span.error && <span className="text-rose-400 font-bold">ERROR</span>}
                      </div>
                    </div>
                  )
                })}
              </div>
            </div>

            {/* Right Span Detail Pane */}
            <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 lg:col-span-2 flex flex-col gap-4">
              <h3 className="text-sm font-bold text-gray-100 flex items-center gap-2 border-b border-slate-800 pb-3">
                <Code className="h-4 w-4 text-indigo-400" />
                <span>Span Details: {selectedSpan?.component_id || selectedSpan?.span_type || 'N/A'}</span>
              </h3>

              {selectedSpan ? (
                <div className="flex flex-col gap-4">
                  {selectedSpan.error && (
                    <div className="p-3.5 rounded-lg border border-rose-900/50 bg-rose-950/30 text-rose-300 text-xs font-mono">
                      <span className="font-bold block mb-1">Error:</span>
                      {selectedSpan.error}
                    </div>
                  )}

                  <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-xs">
                    <div className="bg-slate-950 p-3 border border-slate-800 rounded-lg">
                      <span className="text-[10px] text-gray-500 font-bold uppercase">Span ID</span>
                      <span className="font-mono text-gray-300 block mt-1">{selectedSpan.id.slice(0, 8)}...</span>
                    </div>
                    <div className="bg-slate-950 p-3 border border-slate-800 rounded-lg">
                      <span className="text-[10px] text-gray-500 font-bold uppercase">Latency</span>
                      <span className="text-gray-200 block mt-1 font-bold">{formatLatency(selectedSpan.latency_ms)}</span>
                    </div>
                    <div className="bg-slate-950 p-3 border border-slate-800 rounded-lg">
                      <span className="text-[10px] text-gray-500 font-bold uppercase">Tokens In</span>
                      <span className="text-gray-300 block mt-1 font-semibold">{selectedSpan.tokens_in ?? 'N/A'}</span>
                    </div>
                    <div className="bg-slate-950 p-3 border border-slate-800 rounded-lg">
                      <span className="text-[10px] text-gray-500 font-bold uppercase">Tokens Out</span>
                      <span className="text-gray-300 block mt-1 font-semibold">{selectedSpan.tokens_out ?? 'N/A'}</span>
                    </div>
                  </div>

                  <div className="flex flex-col gap-1.5">
                    <span className="text-xs font-bold text-gray-400 uppercase tracking-wider">Input</span>
                    <pre className="bg-slate-950 border border-slate-800 rounded-lg p-3 text-xs text-gray-300 font-mono overflow-x-auto max-h-60 whitespace-pre-wrap">
                      {formatJsonOrString(selectedSpan.input_json)}
                    </pre>
                  </div>

                  <div className="flex flex-col gap-1.5">
                    <span className="text-xs font-bold text-indigo-300 uppercase tracking-wider">Output</span>
                    <pre className="bg-slate-950 border border-slate-800 rounded-lg p-3 text-xs text-indigo-200 font-mono overflow-x-auto max-h-60 whitespace-pre-wrap">
                      {formatJsonOrString(selectedSpan.output_json)}
                    </pre>
                  </div>
                </div>
              ) : (
                <div className="p-8 text-center text-gray-500 text-xs italic">
                  Select a span from the list to view input/output.
                </div>
              )}
            </div>
          </div>
        )}
      </main>
    </div>
  )
}

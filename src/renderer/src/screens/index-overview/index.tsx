import React, { useEffect, useMemo, useState } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'
import { ArrowLeft, Loader2, Save, Search } from 'lucide-react'
import type {
  RetrievalBundle,
  RetrievalCompareResponse,
  RetrievalMode,
  RetrievalSection,
  RepoMapSummary,
  StructuralGraphSummary,
  SymbolRecord,
  TwoStageDebugBundle,
  TwoStageCandidate,
  TwoStageExpansion,
  TwoStageRankedChunk,
} from '../../types/electron'
import { ErrorBanner } from '../../components/ui/ErrorBanner'
import { toErrorMessage } from '../../lib/errors'

function RetrievalResultPanel({
  deduped,
  totalChunks,
  mode,
  usedTokens,
  budgetTokens,
}: {
  deduped: Array<{ rel_path: string; chunk_index: number; reason_codes: string[]; score: number; token_estimate: number; excerpt: string }>
  totalChunks: number
  mode: string
  usedTokens: number
  budgetTokens: number
}): React.ReactElement {
  const [expanded, setExpanded] = React.useState<string | null>(null)
  return (
    <div className="text-[11px] text-zinc-500 border border-zinc-800 rounded-md p-2 space-y-1.5">
      <div className="flex items-center gap-2 flex-wrap">
        <span>mode=<span className="text-zinc-300">{mode}</span></span>
        <span className="text-zinc-700">|</span>
        <span>tokens=<span className="text-zinc-300">{usedTokens}/{budgetTokens}</span></span>
        <span className="text-zinc-700">|</span>
        <span><span className="text-zinc-300">{deduped.length}</span> unique files</span>
        <span className="text-zinc-600">({totalChunks} chunks scored, best chunk per file shown)</span>
      </div>
      <div className="max-h-80 overflow-y-auto space-y-1 pr-1">
        {deduped.map((e) => {
          const key = e.rel_path
          const isOpen = expanded === key
          return (
            <div key={key} className="border border-zinc-800 rounded">
              <button
                className="w-full text-left px-2 py-1 flex items-center gap-1.5 hover:bg-zinc-800/40 transition-colors"
                onClick={() => setExpanded(isOpen ? null : key)}
              >
                <span className={`shrink-0 text-[9px] transition-transform ${isOpen ? 'rotate-90' : ''}`}>▶</span>
                <span className="text-zinc-300 font-mono truncate flex-1">{e.rel_path}</span>
                <span className="shrink-0 text-zinc-600 font-mono">chunk#{e.chunk_index}</span>
                <span className="shrink-0 text-zinc-700 mx-1">|</span>
                <span className="shrink-0 text-zinc-500 font-mono truncate max-w-[160px]">{e.reason_codes.join(',')}</span>
                <span className="shrink-0 text-zinc-700 mx-1">|</span>
                <span className="shrink-0 text-zinc-400 font-mono">{e.token_estimate} tok</span>
              </button>
              {isOpen && (
                <pre className="mx-2 mb-2 p-2 bg-zinc-950 border border-zinc-800 rounded text-[10px] text-zinc-300 font-mono whitespace-pre-wrap break-all max-h-48 overflow-y-auto leading-4">
                  {e.excerpt || '(no excerpt)'}
                </pre>
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}

function Stage1Panel({ candidates }: { candidates: TwoStageCandidate[] }): React.ReactElement {
  const [expanded, setExpanded] = React.useState<string | null>(null)
  return (
    <div className="p-2 space-y-1 max-h-80 overflow-y-auto">
      {candidates.map((c) => {
        const key = `${c.rel_path}#${c.chunk_index}`
        const isOpen = expanded === key
        return (
          <div key={key} className="border border-zinc-800 rounded">
            <button
              className="w-full text-left px-2 py-1 flex items-center gap-1.5 hover:bg-zinc-800/40 transition-colors text-[11px]"
              onClick={() => setExpanded(isOpen ? null : key)}
            >
              <span className={`shrink-0 text-[9px] transition-transform ${isOpen ? 'rotate-90' : ''}`}>▶</span>
              <span className="text-zinc-300 font-mono truncate flex-1">{c.rel_path}</span>
              <span className="shrink-0 text-zinc-600 font-mono">chunk#{c.chunk_index}</span>
              <span className="shrink-0 text-zinc-700 mx-1">|</span>
              <span className="shrink-0 text-zinc-400 font-mono">bm25={c.bm25_score.toFixed(2)}</span>
              <span className="shrink-0 text-zinc-700 mx-1">|</span>
              <span className="shrink-0 text-zinc-500 font-mono">{c.token_estimate} tok</span>
            </button>
            {isOpen && (
              <pre className="mx-2 mb-2 p-2 bg-zinc-950 border border-zinc-800 rounded text-[10px] text-zinc-300 font-mono whitespace-pre-wrap break-all max-h-40 overflow-y-auto leading-4">
                {c.excerpt || '(no excerpt)'}
              </pre>
            )}
          </div>
        )
      })}
    </div>
  )
}

function Stage2Panel({ expansions }: { expansions: TwoStageExpansion[] }): React.ReactElement {
  const [expanded, setExpanded] = React.useState<string | null>(null)
  return (
    <div className="p-2 space-y-1 max-h-80 overflow-y-auto">
      {expansions.map((e) => {
        const isOpen = expanded === e.seed_path
        return (
          <div key={e.seed_path} className="border border-zinc-800 rounded">
            <button
              className="w-full text-left px-2 py-1 flex items-center gap-1.5 hover:bg-zinc-800/40 transition-colors text-[11px]"
              onClick={() => setExpanded(isOpen ? null : e.seed_path)}
            >
              <span className={`shrink-0 text-[9px] transition-transform ${isOpen ? 'rotate-90' : ''}`}>▶</span>
              <span className="text-zinc-300 font-mono truncate flex-1">{e.seed_path}</span>
              <span className="shrink-0 text-zinc-500">
                {e.symbol_refs.length} sym / {e.community_members.length} comm / {e.neighbor_files.length} nbr / +{e.net_new_count} new
              </span>
            </button>
            {isOpen && (
              <div className="mx-2 mb-2 p-2 bg-zinc-950 border border-zinc-800 rounded text-[10px] text-zinc-400 space-y-1">
                {e.symbol_refs.length > 0 && (
                  <div><span className="text-zinc-500">symbol refs:</span> {e.symbol_refs.join(', ')}</div>
                )}
                {e.community_members.length > 0 && (
                  <div><span className="text-zinc-500">community:</span> {e.community_members.join(', ')}</div>
                )}
                {e.neighbor_files.length > 0 && (
                  <div><span className="text-zinc-500">neighbors:</span> {e.neighbor_files.join(', ')}</div>
                )}
              </div>
            )}
          </div>
        )
      })}
    </div>
  )
}

function Stage3Panel({
  ranked,
  usedTokens,
  budgetTokens,
  usedCpp,
}: {
  ranked: TwoStageRankedChunk[]
  usedTokens: number
  budgetTokens: number
  usedCpp: boolean
}): React.ReactElement {
  const [expanded, setExpanded] = React.useState<string | null>(null)
  return (
    <div className="p-2 space-y-1.5">
      <div className="flex items-center gap-2 text-[11px] text-zinc-500">
        <span>tokens=<span className="text-zinc-300">{usedTokens}/{budgetTokens}</span></span>
        <span className="text-zinc-700">|</span>
        <span className={`px-1.5 py-0.5 rounded text-[10px] font-mono ${usedCpp ? 'bg-emerald-900/40 text-emerald-400' : 'bg-zinc-800 text-zinc-500'}`}>
          {usedCpp ? 'C++' : 'Python'}
        </span>
        <span className="text-zinc-600">{ranked.length} chunks</span>
      </div>
      <div className="max-h-80 overflow-y-auto space-y-1">
        {ranked.map((c) => {
          const key = `${c.rel_path}#${c.chunk_index}`
          const isOpen = expanded === key
          return (
            <div key={key} className="border border-zinc-800 rounded">
              <button
                className="w-full text-left px-2 py-1 flex items-center gap-1 hover:bg-zinc-800/40 transition-colors text-[11px]"
                onClick={() => setExpanded(isOpen ? null : key)}
              >
                <span className={`shrink-0 text-[9px] transition-transform ${isOpen ? 'rotate-90' : ''}`}>▶</span>
                <span className="text-zinc-300 font-mono truncate flex-1">{c.rel_path}</span>
                <span className="shrink-0 text-zinc-600 font-mono">#{c.chunk_index}</span>
                <span className="shrink-0 text-zinc-700 mx-1">|</span>
                <span className="shrink-0 text-zinc-400 font-mono">{c.score.toFixed(2)}</span>
                <span className="shrink-0 text-zinc-700 mx-0.5">/</span>
                <span className="shrink-0 text-zinc-600 font-mono text-[10px]">bm25={c.bm25_component.toFixed(2)} sym={c.symbol_bonus.toFixed(1)} mod={c.module_bonus.toFixed(1)} cent={c.centrality_bonus.toFixed(1)}</span>
                <span className="shrink-0 text-zinc-700 mx-1">|</span>
                <span className="shrink-0 text-zinc-500 font-mono">{c.token_estimate}tok</span>
              </button>
              {isOpen && (
                <pre className="mx-2 mb-2 p-2 bg-zinc-950 border border-zinc-800 rounded text-[10px] text-zinc-300 font-mono whitespace-pre-wrap break-all max-h-40 overflow-y-auto leading-4">
                  {c.excerpt || '(no excerpt)'}
                </pre>
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}


export default function IndexOverviewScreen(): React.ReactElement {
  const navigate = useNavigate()
  const [params] = useSearchParams()
  const repoId = params.get('repoId') ?? ''
  const snapshotId = params.get('snapshotId') ?? ''

  const [loading, setLoading] = useState(false)
  const [savingCsv, setSavingCsv] = useState(false)
  const [excludeTestsOnExport, setExcludeTestsOnExport] = useState(true)
  const [summary, setSummary] = useState<RepoMapSummary | null>(null)
  const [graphSummary, setGraphSummary] = useState<StructuralGraphSummary | null>(null)
  const [buildingGraph, setBuildingGraph] = useState(false)
  const [symbols, setSymbols] = useState<SymbolRecord[]>([])
  const [query, setQuery] = useState('')
  const [hits, setHits] = useState<SymbolRecord[]>([])
  const [searching, setSearching] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [success, setSuccess] = useState<string | null>(null)
  const [retrievalQuery, setRetrievalQuery] = useState('')
  const [retrievalSection, setRetrievalSection] = useState<RetrievalSection>('architecture')
  const [retrievalMode, setRetrievalMode] = useState<RetrievalMode>('hybrid')
  const [retrievalBusy, setRetrievalBusy] = useState(false)
  const [retrievalBundle, setRetrievalBundle] = useState<RetrievalBundle | null>(null)
  const [retrievalCompare, setRetrievalCompare] = useState<RetrievalCompareResponse | null>(null)
  const [twoStageBundle, setTwoStageBundle] = React.useState<TwoStageDebugBundle | null>(null)
  const [twoStageBusy, setTwoStageBusy] = React.useState(false)

  useEffect(() => {
    const run = async () => {
      if (!snapshotId) return
      setLoading(true)
      setError(null)
      setSuccess(null)
      try {
        const s = await window.api.repomap.summary(snapshotId)
        setSummary(s)
        const sample = await window.api.repomap.symbols(snapshotId, 200)
        setSymbols(sample.symbols)
        const g = await window.api.graph.summary(snapshotId)
        setGraphSummary(g)
      } catch (err) {
        setError(toErrorMessage(err))
      } finally {
        setLoading(false)
      }
    }
    run()
  }, [snapshotId])

  useEffect(() => {
    const run = async () => {
      if (!snapshotId || !query.trim()) {
        setHits([])
        return
      }
      setSearching(true)
      try {
        const res = await window.api.repomap.search(snapshotId, query.trim(), 120)
        setHits(res.symbols)
      } finally {
        setSearching(false)
      }
    }
    const timer = setTimeout(run, 180)
    return () => clearTimeout(timer)
  }, [snapshotId, query])

  const byKind = useMemo(() => Object.entries(summary?.kind_breakdown ?? {}), [summary])
  const byLang = useMemo(() => Object.entries(summary?.language_breakdown ?? {}), [summary])

  return (
    <>
      <div className="screen-header">
        <h1 className="screen-title">Index Overview</h1>
        <p className="screen-subtitle">Deep index quality, symbol breakdown, and export</p>
      </div>
      <div className="h-[calc(100vh-10rem)] p-4 space-y-3 overflow-y-auto">
        {error && <ErrorBanner message={error} onDismiss={() => setError(null)} />}
        {success && (
          <div className="rounded-lg border border-emerald-500/30 bg-emerald-500/10 text-emerald-300 text-xs px-3 py-2">
            {success}
          </div>
        )}

        <div className="flex items-center justify-between bg-zinc-900/60 border border-zinc-700 rounded-lg px-3 py-2">
          <div className="text-xs text-zinc-400">
            <span className="text-zinc-200">Repo:</span> <span className="font-mono">{repoId || '-'}</span>
            <span className="mx-2 text-zinc-600">|</span>
            <span className="text-zinc-200">Snapshot:</span> <span className="font-mono">{snapshotId || '-'}</span>
          </div>
          <div className="flex items-center gap-2">
            <label className="inline-flex items-center gap-1.5 text-[11px] text-zinc-400">
              <input
                type="checkbox"
                checked={excludeTestsOnExport}
                onChange={(e) => setExcludeTestsOnExport(e.target.checked)}
              />
              Exclude tests in CSV
            </label>
            <button
              onClick={async () => {
                if (!snapshotId) return
                setSavingCsv(true)
                setError(null)
                setSuccess(null)
                try {
                  const out = await window.api.repomap.exportCsv(snapshotId, excludeTestsOnExport)
                  if (out.saved && out.file_path) {
                    setSuccess(`CSV saved: ${out.file_path} (${out.row_count} rows)`)
                  }
                } catch (err) {
                  setError(toErrorMessage(err))
                } finally {
                  setSavingCsv(false)
                }
              }}
              disabled={savingCsv || !summary}
              className="px-2.5 py-1.5 text-xs border border-zinc-700 rounded-md text-zinc-300 hover:border-zinc-600 disabled:opacity-50 inline-flex items-center gap-1"
            >
              {savingCsv ? <Loader2 size={12} className="animate-spin" /> : <Save size={12} />}
              Save index CSV
            </button>
            <button
              onClick={() => navigate(`/snapshot-viewer?repoId=${encodeURIComponent(repoId)}&snapshotId=${encodeURIComponent(snapshotId)}`)}
              className="px-2.5 py-1.5 text-xs border border-zinc-700 rounded-md text-zinc-300 hover:border-zinc-600 inline-flex items-center gap-1"
            >
              <ArrowLeft size={12} />
              Back
            </button>
          </div>
        </div>

        {loading ? (
          <div className="text-xs text-zinc-500 inline-flex items-center gap-2">
            <Loader2 size={12} className="animate-spin" />
            Loading index summary...
          </div>
        ) : summary ? (
          <>
            <div className="grid grid-cols-4 gap-2">
              <div className="bg-zinc-900/60 border border-zinc-700 rounded-md p-3">
                <div className="text-[11px] text-zinc-500">Symbols</div>
                <div className="text-lg font-semibold text-zinc-100">{summary.total_symbols}</div>
              </div>
              <div className="bg-zinc-900/60 border border-zinc-700 rounded-md p-3">
                <div className="text-[11px] text-zinc-500">Files indexed</div>
                <div className="text-lg font-semibold text-zinc-100">{summary.files_indexed}</div>
              </div>
              <div className="bg-zinc-900/60 border border-zinc-700 rounded-md p-3">
                <div className="text-[11px] text-zinc-500">Parse failures</div>
                <div className="text-lg font-semibold text-zinc-100">{summary.parse_failures}</div>
              </div>
              <div className="bg-zinc-900/60 border border-zinc-700 rounded-md p-3">
                <div className="text-[11px] text-zinc-500">Mode</div>
                <div className="text-lg font-semibold text-zinc-100">{summary.extract_mode}</div>
              </div>
            </div>

            <div className="bg-zinc-900/60 border border-zinc-700 rounded-lg p-3 space-y-2">
              <div className="flex items-center justify-between">
                <div className="text-xs font-semibold text-zinc-100">Structural Graph</div>
                <button
                  onClick={async () => {
                    if (!snapshotId) return
                    setBuildingGraph(true)
                    setError(null)
                    try {
                      const out = await window.api.graph.build(snapshotId, true)
                      setGraphSummary(out.summary)
                      setSuccess('Structural graph built successfully')
                    } catch (err) {
                      setError(toErrorMessage(err))
                    } finally {
                      setBuildingGraph(false)
                    }
                  }}
                  disabled={buildingGraph}
                  className="px-2.5 py-1.5 text-xs border border-zinc-700 rounded-md text-zinc-300 hover:border-zinc-600 disabled:opacity-50"
                >
                  {buildingGraph ? 'Building...' : 'Build graph'}
                </button>
              </div>
              {graphSummary ? (
                <>
                  <div className="grid grid-cols-4 gap-2">
                    <div className="bg-zinc-950 border border-zinc-800 rounded-md px-2 py-1.5 text-xs text-zinc-400">
                      Nodes: <span className="text-zinc-200">{graphSummary.total_nodes}</span>
                    </div>
                    <div className="bg-zinc-950 border border-zinc-800 rounded-md px-2 py-1.5 text-xs text-zinc-400">
                      Edges: <span className="text-zinc-200">{graphSummary.total_edges}</span>
                    </div>
                    <div className="bg-zinc-950 border border-zinc-800 rounded-md px-2 py-1.5 text-xs text-zinc-400">
                      External: <span className="text-zinc-200">{graphSummary.external_edges}</span>
                    </div>
                    <div className="bg-zinc-950 border border-zinc-800 rounded-md px-2 py-1.5 text-xs text-zinc-400">
                      Native:{' '}
                      <span className={graphSummary.native_toolchain ? 'text-green-400' : 'text-yellow-400'}>
                        {graphSummary.native_toolchain ?? 'not detected'}
                      </span>
                    </div>
                  </div>
                  <div className="text-[11px] text-zinc-500">
                    Entrypoints: {graphSummary.entrypoints.length > 0 ? graphSummary.entrypoints.join(', ') : '-'}
                  </div>
                  <button
                    onClick={() => navigate(`/graph?snapshotId=${snapshotId}`)}
                    className="px-2.5 py-1.5 text-xs border border-indigo-700 text-indigo-300 rounded-md hover:border-indigo-500 hover:text-indigo-100"
                  >
                    Open Graph
                  </button>
                  <div className="bg-zinc-950 border border-zinc-800 rounded-md p-2">
                    <div className="text-[11px] text-zinc-400 mb-1">Top central files</div>
                    <div className="max-h-40 overflow-auto space-y-1">
                      {graphSummary.top_central_files.slice(0, 20).map((n) => (
                        <div key={n.rel_path} className="text-[11px] font-mono text-zinc-400">
                          {n.rel_path}
                        </div>
                      ))}
                    </div>
                  </div>
                </>
              ) : (
                <div className="text-[11px] text-zinc-500">Graph not built yet.</div>
              )}
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div className="bg-zinc-900/60 border border-zinc-700 rounded-md p-3 space-y-2">
                <div className="text-xs font-semibold text-zinc-100">By symbol kind</div>
                {byKind.map(([k, v]) => (
                  <div key={k} className="flex justify-between text-xs text-zinc-400">
                    <span>{k}</span>
                    <span className="text-zinc-200">{v}</span>
                  </div>
                ))}
              </div>
              <div className="bg-zinc-900/60 border border-zinc-700 rounded-md p-3 space-y-2">
                <div className="text-xs font-semibold text-zinc-100">By language</div>
                {byLang.map(([k, v]) => (
                  <div key={k} className="flex justify-between text-xs text-zinc-400">
                    <span>{k}</span>
                    <span className="text-zinc-200">{v}</span>
                  </div>
                ))}
              </div>
            </div>

            <div className="bg-zinc-900/60 border border-zinc-700 rounded-lg p-3 space-y-2">
              <div className="text-xs font-semibold text-zinc-100">Symbol Search</div>
              <div className="flex items-center gap-2">
                <div className="flex-1 relative">
                  <Search size={12} className="absolute left-2 top-2.5 text-zinc-500" />
                  <input
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    placeholder="Search symbol name or file path..."
                    className="w-full bg-zinc-950 border border-zinc-700 rounded-md pl-7 pr-2 py-1.5 text-xs text-zinc-200"
                  />
                </div>
                {searching && <Loader2 size={13} className="animate-spin text-zinc-500" />}
              </div>
              <div className="max-h-56 overflow-y-auto pr-1">
                {(hits.length > 0 ? hits : symbols).slice(0, 120).map((s) => (
                  <div key={s.id} className="text-[11px] text-zinc-400 font-mono border-b border-zinc-800 py-1">
                    <span className="text-zinc-200">{s.name}</span>
                    <span className="mx-2 text-zinc-600">·</span>
                    <span>{s.kind}/{s.extract_source}</span>
                    <span className="mx-2 text-zinc-600">·</span>
                    <span>{s.rel_path}:{s.line_start}</span>
                  </div>
                ))}
              </div>
            </div>

            <div className="bg-zinc-900/60 border border-zinc-700 rounded-lg p-3 space-y-2">
              <div className="text-xs font-semibold text-zinc-100">Retrieval Debug</div>
              <div className="grid grid-cols-1 md:grid-cols-4 gap-2">
                <input
                  value={retrievalQuery}
                  onChange={(e) => setRetrievalQuery(e.target.value)}
                  placeholder="Query to feed retrieval..."
                  className="md:col-span-2 bg-zinc-950 border border-zinc-700 rounded-md px-2 py-1.5 text-xs text-zinc-200"
                />
                <select
                  value={retrievalSection}
                  onChange={(e) => setRetrievalSection(e.target.value as RetrievalSection)}
                  className="bg-zinc-950 border border-zinc-700 rounded-md px-2 py-1.5 text-xs text-zinc-200"
                >
                  <option value="architecture">architecture</option>
                  <option value="conventions">conventions</option>
                  <option value="feature_map">feature_map</option>
                  <option value="important_files">important_files</option>
                  <option value="glossary">glossary</option>
                </select>
                <select
                  value={retrievalMode}
                  onChange={(e) => setRetrievalMode(e.target.value as RetrievalMode)}
                  className="bg-zinc-950 border border-zinc-700 rounded-md px-2 py-1.5 text-xs text-zinc-200"
                >
                  <option value="hybrid">hybrid</option>
                  <option value="vectorless">vectorless</option>
                </select>
              </div>
              <div className="flex items-center gap-2">
                <button
                  onClick={async () => {
                    if (!snapshotId) return
                    setRetrievalBusy(true)
                    setError(null)
                    setRetrievalBundle(null)
                    setRetrievalCompare(null)
                    try {
                      await window.api.retrieval.buildIndex(snapshotId, false)
                      const out = await window.api.retrieval.retrieve({
                        snapshot_id: snapshotId,
                        query: retrievalQuery.trim(),
                        section: retrievalSection,
                        mode: retrievalMode,
                        max_results: 20,
                      })
                      setRetrievalBundle(out)
                    } catch (err) {
                      setError(toErrorMessage(err))
                    } finally {
                      setRetrievalBusy(false)
                    }
                  }}
                  disabled={retrievalBusy || !retrievalQuery.trim()}
                  className="px-2.5 py-1.5 text-xs border border-zinc-700 rounded-md text-zinc-300 hover:border-zinc-600 disabled:opacity-50"
                >
                  {retrievalBusy ? 'Running...' : 'Run retrieval'}
                </button>
                <button
                  onClick={async () => {
                    if (!snapshotId) return
                    setRetrievalBusy(true)
                    setError(null)
                    setRetrievalBundle(null)
                    try {
                      await window.api.retrieval.buildIndex(snapshotId, false)
                      const out = await window.api.retrieval.compare({
                        snapshot_id: snapshotId,
                        query: retrievalQuery.trim(),
                        section: retrievalSection,
                        max_results: 20,
                      })
                      setRetrievalCompare(out)
                    } catch (err) {
                      setError(toErrorMessage(err))
                    } finally {
                      setRetrievalBusy(false)
                    }
                  }}
                  disabled={retrievalBusy || !retrievalQuery.trim()}
                  className="px-2.5 py-1.5 text-xs border border-zinc-700 rounded-md text-zinc-300 hover:border-zinc-600 disabled:opacity-50"
                >
                  {retrievalBusy ? 'Comparing...' : 'A/B compare'}
                </button>
              </div>
              {retrievalBundle && (() => {
                // Dedupe: keep best-scoring chunk per file
                const seen = new Map<string, typeof retrievalBundle.evidences[0]>()
                for (const e of retrievalBundle.evidences) {
                  const prev = seen.get(e.rel_path)
                  if (!prev || e.score > prev.score) seen.set(e.rel_path, e)
                }
                const deduped = Array.from(seen.values()).sort((a, b) => b.score - a.score)
                return (
                  <RetrievalResultPanel
                    deduped={deduped}
                    totalChunks={retrievalBundle.evidences.length}
                    mode={retrievalBundle.mode}
                    usedTokens={retrievalBundle.used_tokens}
                    budgetTokens={retrievalBundle.budget_tokens}
                  />
                )
              })()}
              {retrievalCompare && (
                <div className="text-[11px] text-zinc-500 border border-zinc-800 rounded-md p-2">
                  delta p@5: <span className="text-zinc-300">{retrievalCompare.precision_at_5_delta.toFixed(3)}</span>
                  <span className="mx-2 text-zinc-700">|</span>
                  delta hit-rate: <span className="text-zinc-300">{retrievalCompare.evidence_hit_rate_delta.toFixed(3)}</span>
                  <span className="mx-2 text-zinc-700">|</span>
                  delta tokens: <span className="text-zinc-300">{retrievalCompare.token_cost_delta}</span>
                </div>
              )}
            </div>
            <div className="bg-zinc-900/60 border border-zinc-700 rounded-lg p-3 space-y-2">
              <div className="text-xs font-semibold text-zinc-100">2-Stage Retrieval Debug</div>
              <div className="flex items-center gap-2">
                <button
                  onClick={async () => {
                    if (!snapshotId) return
                    setTwoStageBusy(true)
                    setError(null)
                    setTwoStageBundle(null)
                    try {
                      await window.api.retrieval.buildIndex(snapshotId, false)
                      const out = await window.api.retrieval.retrieveTwoStage({
                        snapshot_id: snapshotId,
                        query: retrievalQuery.trim(),
                        section: retrievalSection,
                      })
                      setTwoStageBundle(out)
                    } catch (err) {
                      setError(toErrorMessage(err))
                    } finally {
                      setTwoStageBusy(false)
                    }
                  }}
                  disabled={twoStageBusy || !retrievalQuery.trim()}
                  className="px-2.5 py-1.5 text-xs border border-zinc-700 rounded-md text-zinc-300 hover:border-zinc-600 disabled:opacity-50"
                >
                  {twoStageBusy ? 'Running...' : 'Run 2-stage retrieval'}
                </button>
                <span className="text-[11px] text-zinc-600">Uses same query field as Retrieval Debug</span>
              </div>
              {twoStageBundle && (
                <div className="space-y-1.5">
                  <details className="border border-zinc-800 rounded">
                    <summary className="px-2 py-1 text-[11px] text-zinc-400 cursor-pointer hover:text-zinc-200">
                      Stage 1 — BM25 top {twoStageBundle.stage1.candidates.length} candidates
                    </summary>
                    <Stage1Panel candidates={twoStageBundle.stage1.candidates} />
                  </details>
                  <details className="border border-zinc-800 rounded">
                    <summary className="px-2 py-1 text-[11px] text-zinc-400 cursor-pointer hover:text-zinc-200">
                      Stage 2 — Graph expansion (top 20 seeds)
                    </summary>
                    <Stage2Panel expansions={twoStageBundle.stage2.expansions} />
                  </details>
                  <details open className="border border-zinc-800 rounded">
                    <summary className="px-2 py-1 text-[11px] text-zinc-400 cursor-pointer hover:text-zinc-200">
                      Stage 3 — Re-ranked ({twoStageBundle.stage3.ranked.length} chunks)
                    </summary>
                    <Stage3Panel
                      ranked={twoStageBundle.stage3.ranked}
                      usedTokens={twoStageBundle.stage3.used_tokens}
                      budgetTokens={twoStageBundle.stage3.budget_tokens}
                      usedCpp={twoStageBundle.stage3.used_cpp_ranker}
                    />
                  </details>
                </div>
              )}
            </div>
          </>
        ) : null}
      </div>
    </>
  )
}

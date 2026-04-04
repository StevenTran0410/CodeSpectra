import React, { useEffect, useMemo, useState } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'
import { ArrowLeft, Loader2, Save, Search } from 'lucide-react'
import dagre from '@dagrejs/dagre'
import {
  Background,
  Controls,
  ReactFlow,
  type Edge,
  type Node,
} from '@xyflow/react'
import '@xyflow/react/dist/style.css'
import type {
  GraphEdge,
  RetrievalBundle,
  RetrievalCompareResponse,
  RetrievalMode,
  RetrievalSection,
  RepoMapSummary,
  StructuralGraphSummary,
  SymbolRecord,
} from '../../types/electron'
import { ErrorBanner } from '../../components/ui/ErrorBanner'

export default function IndexOverviewScreen(): React.ReactElement {
  const MAX_RENDER_NODES = 180
  const MAX_RENDER_EDGES = 320
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
  const [seedPath, setSeedPath] = useState('')
  const [neighborHops, setNeighborHops] = useState(1)
  const [loadingNeighbors, setLoadingNeighbors] = useState(false)
  const [neighborNodes, setNeighborNodes] = useState<string[]>([])
  const [neighborEdges, setNeighborEdges] = useState<GraphEdge[]>([])
  const [loadingFullGraph, setLoadingFullGraph] = useState(false)
  const [fullGraphEdges, setFullGraphEdges] = useState<GraphEdge[]>([])
  const [error, setError] = useState<string | null>(null)
  const [success, setSuccess] = useState<string | null>(null)
  const [graphModalOpen, setGraphModalOpen] = useState(false)
  const [graphModalLoading, setGraphModalLoading] = useState(false)
  const [graphModalEdges, setGraphModalEdges] = useState<GraphEdge[]>([])
  const [graphModalNodes, setGraphModalNodes] = useState<string[]>([])
  const [retrievalQuery, setRetrievalQuery] = useState('')
  const [retrievalSection, setRetrievalSection] = useState<RetrievalSection>('architecture')
  const [retrievalMode, setRetrievalMode] = useState<RetrievalMode>('hybrid')
  const [retrievalBusy, setRetrievalBusy] = useState(false)
  const [retrievalBundle, setRetrievalBundle] = useState<RetrievalBundle | null>(null)
  const [retrievalCompare, setRetrievalCompare] = useState<RetrievalCompareResponse | null>(null)

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
        setError(err instanceof Error ? err.message : String(err))
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
  const fullGraphNodes = useMemo(() => {
    const set = new Set<string>()
    for (const e of fullGraphEdges) {
      set.add(e.src_path)
      set.add(e.dst_path)
    }
    return Array.from(set).sort((a, b) => a.localeCompare(b))
  }, [fullGraphEdges])
  const focusedEdges = useMemo(
    () => fullGraphEdges.filter((e) => e.src_path === seedPath || e.dst_path === seedPath),
    [fullGraphEdges, seedPath],
  )
  const focusedMode = seedPath.trim().length > 0
  const graphNodeOptions = useMemo(() => fullGraphNodes.slice(0, 1000), [fullGraphNodes])
  const graphRenderNodes = useMemo(() => {
    const nodes = (graphModalNodes.length > 0 ? graphModalNodes : fullGraphNodes).slice(0, MAX_RENDER_NODES)
    const srcEdges = (graphModalEdges.length > 0 ? graphModalEdges : fullGraphEdges)
      .slice(0, MAX_RENDER_EDGES)
    const nodeSet = new Set(nodes)

    const g = new dagre.graphlib.Graph()
    g.setDefaultEdgeLabel(() => ({}))
    g.setGraph({
      rankdir: focusedMode ? 'LR' : 'TB',
      nodesep: 36,
      ranksep: focusedMode ? 90 : 72,
      marginx: 20,
      marginy: 20,
    })

    const nodeWidth = 210
    const nodeHeight = 40
    for (const n of nodes) {
      g.setNode(n, { width: nodeWidth, height: nodeHeight })
    }
    for (const e of srcEdges) {
      if (!nodeSet.has(e.src_path) || !nodeSet.has(e.dst_path)) continue
      g.setEdge(e.src_path, e.dst_path)
    }
    dagre.layout(g)

    return nodes.map((n, i) => {
      const p = g.node(n)
      const x = p ? p.x - nodeWidth / 2 : (i % 8) * 240
      const y = p ? p.y - nodeHeight / 2 : Math.floor(i / 8) * 90
      const tail = n.split('/').slice(-2).join('/')
      return {
        id: n,
        position: { x, y },
        data: {
          label: (
            <div className="w-[190px]">
              <div className="text-[10px] text-zinc-100 truncate">{tail || n}</div>
              <div className="text-[9px] text-zinc-500 truncate">{n}</div>
            </div>
          ),
        },
        style: {
          background: '#18181b',
          color: '#e4e4e7',
          border: '1px solid #3f3f46',
          borderRadius: 8,
          fontSize: 10,
          padding: 6,
          width: nodeWidth,
          boxShadow: '0 2px 10px rgba(0,0,0,0.25)',
        },
        draggable: false,
        selectable: true,
      } as Node
    })
  }, [graphModalNodes, fullGraphNodes, graphModalEdges, fullGraphEdges, focusedMode])
  const graphRenderEdges = useMemo(() => {
    const nodeSet = new Set(graphRenderNodes.map((n) => n.id))
    const src = graphModalEdges.length > 0 ? graphModalEdges : fullGraphEdges
    const out: Edge[] = []
    for (const e of src) {
      if (!nodeSet.has(e.src_path) || !nodeSet.has(e.dst_path)) continue
      out.push({
        id: `${e.src_path}->${e.dst_path}:${out.length}`,
        source: e.src_path,
        target: e.dst_path,
        type: 'smoothstep',
        animated: false,
        style: { stroke: '#71717a', strokeWidth: 1.2, opacity: 0.72 },
      })
      if (out.length >= MAX_RENDER_EDGES) break
    }
    return out
  }, [graphModalEdges, fullGraphEdges, graphRenderNodes])

  useEffect(() => {
    const run = async () => {
      if (!snapshotId || !graphSummary) return
      setLoadingFullGraph(true)
      try {
        const res = await window.api.graph.edges(snapshotId, 5000)
        setFullGraphEdges(res.edges)
      } catch {
        setFullGraphEdges([])
      } finally {
        setLoadingFullGraph(false)
      }
    }
    run()
  }, [snapshotId, graphSummary])

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
                  setError(err instanceof Error ? err.message : String(err))
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
                <div className="text-xs font-semibold text-zinc-100">Structural Graph (RPA-033)</div>
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
                      setError(err instanceof Error ? err.message : String(err))
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
                      Toolchain: <span className="text-zinc-200">{graphSummary.native_toolchain ?? 'not detected'}</span>
                    </div>
                  </div>
                  <div className="text-[11px] text-zinc-500">
                    Entrypoints: {graphSummary.entrypoints.length > 0 ? graphSummary.entrypoints.join(', ') : '-'}
                  </div>
                  <div className="bg-zinc-950 border border-zinc-800 rounded-md p-2 space-y-2">
                    <div className="text-[11px] text-zinc-400">
                      Graph view ({focusedMode ? 'node focused' : 'full graph'})
                    </div>
                    <div className="flex items-center gap-2">
                      <input
                        value={seedPath}
                        onChange={(e) => setSeedPath(e.target.value)}
                        list="graph-node-options"
                        placeholder="Type node path to focus graph (leave empty for full graph)"
                        className="flex-1 bg-zinc-900 border border-zinc-700 rounded-md px-2 py-1.5 text-xs text-zinc-200"
                      />
                      <datalist id="graph-node-options">
                        {graphNodeOptions.map((n) => (
                          <option key={n} value={n} />
                        ))}
                      </datalist>
                      <select
                        value={neighborHops}
                        onChange={(e) => setNeighborHops(Number(e.target.value))}
                        className="bg-zinc-900 border border-zinc-700 rounded-md px-2 py-1.5 text-xs text-zinc-200"
                      >
                        <option value={1}>1 hop</option>
                        <option value={2}>2 hops</option>
                        <option value={3}>3 hops</option>
                      </select>
                      <button
                        onClick={async () => {
                          if (!snapshotId) return
                          setGraphModalOpen(true)
                          if (!seedPath.trim()) {
                            setNeighborNodes(fullGraphNodes)
                            setNeighborEdges(fullGraphEdges)
                            setGraphModalNodes(fullGraphNodes)
                            setGraphModalEdges(fullGraphEdges)
                            return
                          }
                          setGraphModalLoading(true)
                          setLoadingNeighbors(true)
                          setError(null)
                          try {
                            const res = await window.api.graph.neighbors(
                              snapshotId,
                              seedPath.trim(),
                              neighborHops,
                              250
                            )
                            setNeighborNodes(res.nodes)
                            setNeighborEdges(res.edges)
                            setGraphModalNodes(res.nodes)
                            setGraphModalEdges(res.edges)
                          } catch (err) {
                            setError(err instanceof Error ? err.message : String(err))
                          } finally {
                            setLoadingNeighbors(false)
                            setGraphModalLoading(false)
                          }
                        }}
                        disabled={loadingNeighbors}
                        className="px-2.5 py-1.5 text-xs border border-zinc-700 rounded-md text-zinc-300 hover:border-zinc-600 disabled:opacity-50"
                      >
                        {loadingNeighbors ? 'Loading...' : 'Show graph'}
                      </button>
                    </div>
                    {loadingFullGraph ? (
                      <div className="text-[11px] text-zinc-500 inline-flex items-center gap-1.5">
                        <Loader2 size={11} className="animate-spin" />
                        Loading full graph...
                      </div>
                    ) : (
                      <div className="text-[11px] text-zinc-500">
                        Full graph: <span className="text-zinc-300">{fullGraphNodes.length}</span> nodes
                        <span className="mx-2 text-zinc-700">|</span>
                        <span className="text-zinc-300">{fullGraphEdges.length}</span> edges
                      </div>
                    )}
                    {(neighborNodes.length > 0 || neighborEdges.length > 0) && (
                      <div className="text-[11px] text-zinc-500">
                        Nodes: <span className="text-zinc-300">{neighborNodes.length}</span>
                        <span className="mx-2 text-zinc-700">|</span>
                        Edges: <span className="text-zinc-300">{neighborEdges.length}</span>
                      </div>
                    )}
                    <div className="grid grid-cols-2 gap-2">
                      <div className="bg-zinc-900/60 border border-zinc-800 rounded-md p-2">
                        <div className="text-[11px] text-zinc-400 mb-1">
                          {focusedMode ? 'Focused node links' : 'Full graph links'}
                        </div>
                        <div className="max-h-40 overflow-auto space-y-1">
                          {(focusedMode ? focusedEdges : fullGraphEdges).slice(0, 220).map((e, i) => (
                            <div key={`${e.src_path}-${e.dst_path}-${i}`} className="text-[11px] font-mono text-zinc-400">
                              <span className="text-zinc-300">{e.src_path}</span>
                              <span className="mx-1 text-zinc-600">-&gt;</span>
                              <span>{e.dst_path}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                      <div className="bg-zinc-900/60 border border-zinc-800 rounded-md p-2">
                        <div className="text-[11px] text-zinc-400 mb-1">
                          {focusedMode ? 'Expanded neighborhood' : 'Top nodes'}
                        </div>
                        <div className="max-h-40 overflow-auto space-y-1">
                          {(focusedMode
                            ? neighborNodes
                            : graphSummary.top_central_files.map((n) => n.rel_path)
                          )
                            .slice(0, 220)
                            .map((n) => (
                              <div key={n} className="text-[11px] font-mono text-zinc-400">
                                {n}
                              </div>
                            ))}
                        </div>
                      </div>
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

            <div className="bg-zinc-900/60 border border-zinc-700 rounded-lg p-3 space-y-2">
              <div className="text-xs font-semibold text-zinc-100">Retrieval Debug (RPA-034)</div>
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
                      setError(err instanceof Error ? err.message : String(err))
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
                      setError(err instanceof Error ? err.message : String(err))
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
              {retrievalBundle && (
                <div className="text-[11px] text-zinc-500 border border-zinc-800 rounded-md p-2 space-y-1">
                  <div>
                    mode=<span className="text-zinc-300">{retrievalBundle.mode}</span>
                    <span className="mx-2 text-zinc-700">|</span>
                    tokens=<span className="text-zinc-300">{retrievalBundle.used_tokens}/{retrievalBundle.budget_tokens}</span>
                    <span className="mx-2 text-zinc-700">|</span>
                    evidences=<span className="text-zinc-300">{retrievalBundle.evidences.length}</span>
                  </div>
                  <div className="max-h-36 overflow-auto space-y-1">
                    {retrievalBundle.evidences.slice(0, 40).map((e) => (
                      <div key={e.chunk_id} className="font-mono">
                        <span className="text-zinc-300">{e.rel_path}</span>
                        <span className="mx-2 text-zinc-700">|</span>
                        <span>{e.reason_codes.join(',')}</span>
                        <span className="mx-2 text-zinc-700">|</span>
                        <span>tok:{e.token_estimate}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
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
          </>
        ) : null}
      </div>
      {graphModalOpen && (
        <div className="fixed inset-0 z-50 bg-black/65 flex items-center justify-center p-4">
          <div className="w-[94vw] h-[86vh] rounded-lg border border-zinc-700 bg-zinc-900 overflow-hidden flex flex-col">
            <div className="px-3 py-2 border-b border-zinc-700 flex items-center justify-between">
              <div className="text-xs text-zinc-300">
                Graph Visualization ({seedPath.trim() ? `focused: ${seedPath}` : 'full graph'})
              </div>
              <div className="flex items-center gap-2">
                <div className="text-[11px] text-zinc-500">
                  rendered {graphRenderNodes.length} nodes / {graphRenderEdges.length} edges
                  {(graphRenderNodes.length >= MAX_RENDER_NODES || graphRenderEdges.length >= MAX_RENDER_EDGES) && (
                    <span className="text-amber-400"> (truncated for performance)</span>
                  )}
                </div>
                <button
                  onClick={() => setGraphModalOpen(false)}
                  className="px-2.5 py-1 text-xs border border-zinc-700 rounded text-zinc-300 hover:border-zinc-600"
                >
                  Close
                </button>
              </div>
            </div>
            <div className="flex-1 min-h-0">
              {graphModalLoading ? (
                <div className="h-full flex items-center justify-center text-zinc-500 text-sm">
                  <Loader2 size={16} className="animate-spin mr-2" />
                  Loading graph...
                </div>
              ) : (
                <ReactFlow
                  nodes={graphRenderNodes}
                  edges={graphRenderEdges}
                  fitView
                  fitViewOptions={{ padding: 0.24 }}
                  onNodeClick={(_e, node) => setSeedPath(String(node.id))}
                  minZoom={0.2}
                  maxZoom={1.8}
                >
                  <Controls />
                  <Background color="#27272a" gap={18} size={1} />
                </ReactFlow>
              )}
            </div>
          </div>
        </div>
      )}
    </>
  )
}

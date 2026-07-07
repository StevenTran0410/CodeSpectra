import React, { useState, useEffect, useCallback, useMemo } from 'react'
import { useSearchParams, useNavigate } from 'react-router-dom'
import {
  Loader2,
  AlertCircle,
  Play,
  ArrowLeft,
  ThumbsUp,
  ThumbsDown,
  Info,
  BadgeAlert,
  Sparkles,
  Search,
} from 'lucide-react'
import {
  ReactFlow,
  Background,
  type Node,
  type Edge,
} from '@xyflow/react'
import '@xyflow/react/dist/style.css'
import { getDagreGraphLayout } from './graphLayout'
import { Button, useToastStore } from '../../components/ui'
import { useAEHStore } from '../../store/aeh.store'
import { useProviderStore } from '../../store/provider.store'
import LLMConfigModal, { LLMModelButton } from './LLMConfigModal'
import { useSessionPolling } from './useSessionPolling'
// AEHDiscoveryCandidate/AEHDiscoverySession are global ambient types — no import needed.

type ExportJson = {
  nodes: string[]
  edges: Array<{ src: string; dst: string; external: boolean }>
  communities: Record<string, number>
  community_groups: Record<string, string[]>
  cycles: string[][]
  test_files: string[]
  generated_at: string
}

type CommunitiesResponse = {
  snapshot_id: string
  total_communities: number
  communities: Array<{
    community_id: number
    member_count: number
    hub_paths: string[]
    modularity_contribution: number
    neighbor_community_ids: number[]
    is_singleton: boolean
    llm_summary: string | null
    generated_at: string
  }>
  node_index: Record<string, number>
}

const CANDIDATE_COLORS = [
  '#6366f1', // Indigo
  '#0ea5e9', // Sky
  '#10b981', // Emerald
  '#f59e0b', // Amber
  '#ef4444', // Red
  '#8b5cf6', // Violet
  '#ec4899', // Pink
  '#14b8a6', // Teal
  '#f97316', // Orange
  '#84cc16', // Lime
]





function WiringBlockPanel({
  selectedNode,
  discoveryFileSet,
  candidates,
}: {
  selectedNode: string | null
  discoveryFileSet: Map<string, AEHDiscoveryCandidate>
  candidates: AEHDiscoveryCandidate[]
}): React.ReactElement {
  const candidate = selectedNode ? discoveryFileSet.get(selectedNode) : candidates[0] ?? null

  if (!candidate) {
    return (
      <div className="h-full flex items-center justify-center p-6 text-center text-slate-500 text-xs select-none">
        No candidate selected.
      </div>
    )
  }

  const wb = candidate.wiring_block

  return (
    <div className="p-4 flex flex-col h-full bg-[#090d16]/30 overflow-y-auto">
      <div className="flex items-center justify-between mb-4 border-b border-slate-800 pb-2 shrink-0">
        <h3 className="text-xs font-semibold text-slate-200">Wiring Block Detection</h3>
        {wb && (
          <div className="flex items-center gap-1.5">
            <span className="text-[9px] uppercase tracking-wider font-mono font-bold px-1.5 py-0.5 rounded bg-indigo-950/50 border border-indigo-900/30 text-indigo-300">
              {wb.framework}
            </span>
            {wb.source === 'llm_fallback' && (
              <span className="text-[9px] bg-amber-950/40 text-amber-400 border border-amber-800/40 px-1 py-0.5 rounded font-semibold">
                LLM Inferred (Triage)
              </span>
            )}
          </div>
        )}
      </div>

      {!wb ? (
        <div className="flex-1 flex flex-col items-center justify-center text-center p-6 bg-slate-950/30 border border-dashed border-slate-800/60 rounded-xl">
          <Info className="w-6 h-6 text-slate-600 mb-2" />
          <p className="text-[11px] font-semibold text-slate-400 mb-1">No agentic wiring block detected</p>
          <p className="text-[10px] text-slate-500 max-w-xs leading-relaxed">
            Likely hand-rolled, or not built on a recognized orchestration framework (Haystack, LangGraph, LangChain).
          </p>
        </div>
      ) : (
        <div className="space-y-4 min-h-0 flex-1">
          <div>
            <div className="text-[9px] uppercase tracking-wider font-bold text-slate-500 mb-2">
              Pipeline Components ({wb.nodes.length})
            </div>
            <div className="space-y-2">
              {wb.nodes.map((node, i) => (
                <div
                  key={i}
                  className="bg-slate-950/50 border border-slate-850 rounded-lg p-2.5 flex items-start justify-between gap-3 text-[10px]"
                >
                  <div className="min-w-0">
                    <div className="font-semibold text-slate-200 truncate">{node.alias}</div>
                    <div className="font-mono text-[9px] text-indigo-400 truncate mt-0.5">
                      {node.class_name || 'Component'}
                    </div>
                  </div>
                  <div className="text-[9px] text-slate-500 font-mono truncate max-w-[140px]" title={node.source_hint_file}>
                    {node.source_hint_file.split('/').pop()}
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div>
            <div className="text-[9px] uppercase tracking-wider font-bold text-slate-500 mb-1">
              Connection Count: <span className="font-mono text-slate-300 font-semibold">{wb.edges.length} edges</span>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

function WiringGraphPanel({
  selectedNode,
  discoveryFileSet,
  candidates,
}: {
  selectedNode: string | null
  discoveryFileSet: Map<string, AEHDiscoveryCandidate>
  candidates: AEHDiscoveryCandidate[]
}): React.ReactElement {
  const candidate = selectedNode ? discoveryFileSet.get(selectedNode) : candidates[0] ?? null

  const graphData = useMemo(() => {
    if (!candidate || !candidate.wiring_block) return { layoutNodes: [], layoutEdges: [] }
    const nodes = candidate.wiring_block.nodes.map((n) => ({
      id: n.alias,
      label: n.alias,
    }))
    const edges = candidate.wiring_block.edges.map((e) => ({
      src: e.src,
      dst: e.dst,
    }))
    return getDagreGraphLayout(nodes, edges)
  }, [candidate])

  if (!candidate || !candidate.wiring_block) {
    return (
      <div className="h-full flex items-center justify-center p-6 text-center text-slate-600 text-[10px] select-none bg-[#090d16]/10">
        No graph visualization available.
      </div>
    )
  }

  return (
    <div className="h-full w-full relative bg-[#090d16]/20">
      <ReactFlow
        nodes={graphData.layoutNodes}
        edges={graphData.layoutEdges}
        nodesDraggable={false}
        nodesConnectable={false}
        fitView
        fitViewOptions={{ padding: 0.2 }}
        minZoom={0.2}
        maxZoom={1.5}
        proOptions={{ hideAttribution: true }}
      >
        <Background color="#1e293b" gap={10} size={1} />
      </ReactFlow>
    </div>
  )
}

// Sidebar Board Component
function DiscoveryBoard({
  selectedNode,
  discoveryFileSet,
  candidateColors,
  onVerdict,
  graphExport,
}: {
  selectedNode: string | null
  discoveryFileSet: Map<string, AEHDiscoveryCandidate>
  candidateColors: Record<string, string>
  onVerdict: (candidateId: string, verdict: 'confirmed' | 'rejected') => Promise<void>
  graphExport: ExportJson | null
}): React.ReactElement {
  const [expandedHitIdx, setExpandedHitIdx] = useState<string | null>(null)
  const navigate = useNavigate()
  const [searchParams] = useSearchParams()
  const repoId = searchParams.get('repoId') || ''
  const snapshotId = searchParams.get('snapshotId') || ''

  const candidate = selectedNode ? discoveryFileSet.get(selectedNode) : null
  const candidateColor = candidate ? candidateColors[candidate.id] : '#94a3b8'

  // File content states
  const [fileContent, setFileContent] = useState<{ content: string; truncated: boolean } | null>(null)
  const [fileContentLoading, setFileContentLoading] = useState(false)
  const [fileContentError, setFileContentError] = useState<string | null>(null)

  useEffect(() => {
    if (!selectedNode || !snapshotId) {
      setFileContent(null)
      return
    }
    let cancelled = false
    setFileContentLoading(true)
    setFileContentError(null)
    window.api.manifest.file(snapshotId, selectedNode)
      .then((res) => { if (!cancelled) setFileContent(res) })
      .catch((e) => { if (!cancelled) setFileContentError(e?.message ?? 'Failed to load file content') })
      .finally(() => { if (!cancelled) setFileContentLoading(false) })
    return () => { cancelled = true }
  }, [snapshotId, selectedNode])

  // Connections derivation
  const imports = useMemo(
    () => (selectedNode && graphExport ? graphExport.edges.filter((e) => e.src === selectedNode).map((e) => e.dst) : []),
    [graphExport, selectedNode]
  )
  const importedBy = useMemo(
    () => (selectedNode && graphExport ? graphExport.edges.filter((e) => e.dst === selectedNode).map((e) => e.src) : []),
    [graphExport, selectedNode]
  )

  // Filter evidence to selected node
  const directHits = useMemo(() => {
    if (!candidate || !selectedNode) return []
    return candidate.evidence.filter((h) => h.file === selectedNode)
  }, [candidate, selectedNode])

  // Get other evidence in same cluster
  const otherHits = useMemo(() => {
    if (!candidate || !selectedNode) return []
    return candidate.evidence.filter((h) => h.file !== selectedNode)
  }, [candidate, selectedNode])

  const verdictColor = (verdict: string) => {
    if (verdict === 'confirmed') return 'text-emerald-400 font-semibold'
    if (verdict === 'rejected') return 'text-red-400 line-through opacity-55'
    return 'text-indigo-300'
  }

  const confidenceColor = (confidence: string) => {
    if (confidence === 'high') return 'text-emerald-400 bg-emerald-950/40 border-emerald-800/60'
    if (confidence === 'medium') return 'text-amber-400 bg-amber-950/40 border-amber-800/60'
    return 'text-slate-400 bg-slate-900/40 border-slate-700/60'
  }

  if (!selectedNode) {
    return (
      <div className="w-80 border-r border-slate-800 bg-[#0c1220] flex flex-col items-center justify-center p-6 text-center text-slate-500 text-xs shrink-0 select-none">
        <Search className="w-8 h-8 mb-3 text-slate-700" />
        <p className="font-semibold text-slate-400 mb-1">No file selected</p>
        <p className="text-slate-500 leading-relaxed">
          Select a file from the explorer list on the right to inspect its candidate system details.
        </p>
      </div>
    )
  }

  if (!candidate) {
    return (
      <div className="w-80 border-r border-slate-800 bg-[#0c1220] flex flex-col p-5 shrink-0 overflow-y-auto">
        <div className="border border-slate-800 bg-[#0d1527] rounded-xl p-4 text-center">
          <BadgeAlert className="w-7 h-7 mx-auto mb-2 text-slate-500" />
          <h4 className="text-xs font-semibold text-slate-300 truncate mb-1">{selectedNode.split('/').pop()}</h4>
          <p className="text-[10px] text-slate-500 leading-relaxed">
            This file was not flagged as part of any discovered agentic candidate.
          </p>
        </div>
      </div>
    )
  }

  return (
    <div className="w-80 border-r border-slate-800 bg-[#0c1220] flex flex-col shrink-0 overflow-y-auto">
      {/* Candidate Card Header */}
      <div className="p-4 border-b border-slate-800 space-y-4">
        <div>
          <div className="flex items-center justify-between mb-1.5">
            <span className="text-[9px] uppercase tracking-wider font-bold text-slate-500">Owning Candidate</span>
            <div className="flex items-center gap-1.5">
              <span className={`text-[9px] font-mono px-1.5 py-0.5 rounded border ${confidenceColor(candidate.confidence)}`}>
                {candidate.confidence}
              </span>
              {candidate.needs_human && (
                <span className="text-[9px] bg-amber-950/40 text-amber-400 border border-amber-800/40 px-1 py-0.5 rounded font-semibold">
                  Triage
                </span>
              )}
            </div>
          </div>
          <h3 className={`text-sm font-semibold truncate ${verdictColor(candidate.verdict)}`}>
            {candidate.name}
          </h3>
          <div className="mt-1 flex items-center gap-1">
            <span className="w-2.5 h-2.5 rounded-full" style={{ background: candidateColor }} />
            <span className="text-[10px] font-mono text-slate-500">Community {candidate.community_id}</span>
          </div>
        </div>

        {candidate.frameworks.length > 0 && (
          <div className="flex flex-wrap gap-1">
            {candidate.frameworks.map((fw) => (
              <span key={fw} className="text-[9px] font-mono px-1.5 py-0.5 rounded bg-indigo-950/40 border border-indigo-900/30 text-indigo-300">
                {fw}
              </span>
            ))}
          </div>
        )}

        {/* Verdict Actions */}
        <div className="flex items-center justify-between gap-3 pt-2.5 border-t border-slate-850">
          <span className="text-[10px] text-slate-400">Review Candidate:</span>
          <div className="flex items-center gap-1.5">
            <Button
              title="Confirm agent system"
              onClick={() => onVerdict(candidate.id, 'confirmed')}
              className={`p-1.5 rounded-lg border transition-colors ${
                candidate.verdict === 'confirmed'
                  ? 'bg-emerald-950/50 border-emerald-700/60 text-emerald-400'
                  : 'bg-slate-900/30 border-slate-800 text-slate-500 hover:border-emerald-800 hover:text-emerald-400'
              }`}
            >
              <ThumbsUp className="w-3.5 h-3.5" />
            </Button>
            <Button
              title="Reject candidate"
              onClick={() => onVerdict(candidate.id, 'rejected')}
              className={`p-1.5 rounded-lg border transition-colors ${
                candidate.verdict === 'rejected'
                  ? 'bg-red-950/50 border-red-700/60 text-red-400'
                  : 'bg-slate-900/30 border-slate-800 text-slate-500 hover:border-red-800 hover:text-red-400'
              }`}
            >
              <ThumbsDown className="w-3 h-3 mx-auto" />
            </Button>
          </div>
        </div>

        {candidate.verdict === 'confirmed' && (
          <div className="pt-2">
            <Button
              variant="primary"
              className="w-full text-[10px] py-1.5 flex items-center justify-center gap-1.5 bg-indigo-600 hover:bg-indigo-500 text-white border-none shadow-md hover:shadow-lg transition-all"
              onClick={() => navigate(`/aeh/analysis/stage2?repoId=${repoId}&snapshotId=${snapshotId}&candidateId=${candidate.id}`)}
            >
              <Sparkles className="w-3.5 h-3.5" />
              <span>Expand to System Map</span>
            </Button>
          </div>
        )}
      </div>

      <div className="p-4 space-y-4 flex-1">
        {/* Selected File Section */}
        <div>
          <span className="text-[9px] uppercase tracking-wider font-bold text-slate-500 block mb-1">Selected Node File</span>
          <div className="bg-slate-950/60 border border-slate-850 rounded-lg p-2.5 text-[10px] font-mono text-slate-300 break-all">
            {selectedNode}
          </div>
        </div>

        {/* File Content */}
        <div>
          <span className="text-[9px] uppercase tracking-wider font-bold text-slate-500 block mb-1">File Content</span>
          {fileContentLoading ? (
            <div className="flex items-center justify-center py-6 bg-slate-950/60 border border-slate-850 rounded-lg text-slate-500 text-[10px]">
              <Loader2 className="w-3.5 h-3.5 animate-spin mr-1.5 text-indigo-400" />
              <span>Loading file content...</span>
            </div>
          ) : fileContentError ? (
            <div className="flex items-center gap-2 bg-red-950/20 border border-red-900/30 rounded-lg p-3 text-[10px] text-red-400 leading-relaxed">
              <Info className="w-3.5 h-3.5 shrink-0 text-red-400" />
              <span>{fileContentError}</span>
            </div>
          ) : fileContent ? (
            <div className="space-y-1">
              <pre className="p-2.5 bg-slate-950 border border-slate-850 rounded-lg text-[9px] text-slate-400 font-mono max-h-96 overflow-y-auto whitespace-pre-wrap break-all select-text">
                {fileContent.content}
              </pre>
              {fileContent.truncated && (
                <div className="text-[8px] text-slate-500 italic mt-0.5">
                  Content truncated due to size.
                </div>
              )}
            </div>
          ) : null}
        </div>

        {/* Connections */}
        <div>
          <span className="text-[9px] uppercase tracking-wider font-bold text-slate-500 block mb-2">Connections</span>
          {imports.length === 0 && importedBy.length === 0 ? (
            <div className="flex items-center gap-2 bg-slate-900/40 border border-slate-850 rounded-lg p-3 text-[10px] text-slate-500 leading-relaxed">
              <Info className="w-3.5 h-3.5 shrink-0" />
              <span>No import connections found for this file.</span>
            </div>
          ) : (
            <div className="space-y-3">
              {imports.length > 0 && (
                <div>
                  <div className="text-[9px] text-slate-400 font-semibold mb-1">Imports ({imports.length})</div>
                  <div className="space-y-1 max-h-36 overflow-y-auto">
                    {imports.map((impPath, idx) => (
                      <div
                        key={idx}
                        title={impPath}
                        className="px-2.5 py-1 bg-slate-950/40 border border-slate-850 rounded text-[9px] text-slate-300 font-mono truncate"
                      >
                        {impPath.split('/').pop()}
                      </div>
                    ))}
                  </div>
                </div>
              )}
              {importedBy.length > 0 && (
                <div>
                  <div className="text-[9px] text-slate-400 font-semibold mb-1">Imported By ({importedBy.length})</div>
                  <div className="space-y-1 max-h-36 overflow-y-auto">
                    {importedBy.map((impPath, idx) => (
                      <div
                        key={idx}
                        title={impPath}
                        className="px-2.5 py-1 bg-slate-950/40 border border-slate-850 rounded text-[9px] text-slate-300 font-mono truncate"
                      >
                        {impPath.split('/').pop()}
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Direct Fingerprint Signals Section */}
        <div>
          <span className="text-[9px] uppercase tracking-wider font-bold text-slate-500 block mb-2">Direct Matches on File</span>
          {directHits.length === 0 ? (
            <div className="flex items-center gap-2 bg-slate-900/40 border border-slate-850 rounded-lg p-3 text-[10px] text-slate-500 leading-relaxed">
              <Info className="w-3.5 h-3.5 shrink-0" />
              <span>This file has no direct fingerprint signals. It was clustered as a same-community graph neighbor.</span>
            </div>
          ) : (
            <div className="space-y-1.5">
              {directHits.map((hit, idx) => {
                const key = `direct-${idx}`
                const isOpen = expandedHitIdx === key
                const tokens = hit.token_estimate ?? (hit.snippet ? Math.floor(hit.snippet.length / 4) : 0)
                return (
                  <div key={key} className="border border-slate-850 rounded-lg bg-slate-950/40 overflow-hidden">
                    <button
                      className="w-full text-left px-2.5 py-1.5 flex items-center gap-1.5 hover:bg-slate-900/40 transition-colors text-[10px]"
                      onClick={() => setExpandedHitIdx(isOpen ? null : key)}
                    >
                      <span className={`shrink-0 text-[8px] text-slate-500 transition-transform ${isOpen ? 'rotate-90' : ''}`}>▶</span>
                      <span className="text-indigo-400 font-mono flex-1 truncate">{hit.fingerprint_id}</span>
                      <span className="shrink-0 text-[9px] text-slate-500 font-mono bg-slate-900 border border-slate-800 px-1 rounded">
                        {tokens} tokens
                      </span>
                    </button>
                    {isOpen && hit.snippet && (
                      <pre className="m-2 p-2 bg-slate-950 border border-slate-900 rounded text-[9px] text-slate-400 font-mono overflow-x-auto whitespace-pre-wrap break-all select-text">
                        {hit.snippet}
                      </pre>
                    )}
                  </div>
                )
              })}
            </div>
          )}
        </div>

        {/* Other Cluster Hits (Collapsible) */}
        {otherHits.length > 0 && (
          <details className="border border-slate-850 rounded-lg bg-slate-950/20 overflow-hidden">
            <summary className="px-3 py-2 text-[10px] font-bold uppercase tracking-wider text-slate-400 cursor-pointer hover:text-slate-200">
              Other cluster hits ({otherHits.length})
            </summary>
            <div className="p-3 space-y-1.5 max-h-56 overflow-y-auto border-t border-slate-850">
              {otherHits.map((hit, idx) => {
                const key = `other-${idx}`
                const isOpen = expandedHitIdx === key
                const tokens = hit.token_estimate ?? (hit.snippet ? Math.floor(hit.snippet.length / 4) : 0)
                return (
                  <div key={key} className="border border-slate-850 rounded-lg bg-slate-950/50">
                    <button
                      className="w-full text-left px-2.5 py-1.5 flex items-center gap-1.5 hover:bg-slate-900/40 transition-colors text-[9px]"
                      onClick={() => setExpandedHitIdx(isOpen ? null : key)}
                    >
                      <span className={`shrink-0 text-[8px] text-slate-500 transition-transform ${isOpen ? 'rotate-90' : ''}`}>▶</span>
                      <div className="flex-1 min-w-0">
                        <div className="text-slate-400 font-mono truncate">{hit.file.split('/').pop()}</div>
                        <div className="text-indigo-400 font-mono text-[8px] truncate">{hit.fingerprint_id}</div>
                      </div>
                      <span className="shrink-0 text-[8px] text-slate-500 font-mono bg-slate-900 border border-slate-800 px-1 rounded">
                        {tokens} t
                      </span>
                    </button>
                    {isOpen && hit.snippet && (
                      <pre className="m-2 p-2 bg-slate-950 border border-slate-900 rounded text-[8px] text-slate-400 font-mono overflow-x-auto whitespace-pre-wrap break-all select-text">
                        {hit.snippet}
                      </pre>
                    )}
                  </div>
                )
              })}
            </div>
          </details>
        )}
      </div>
    </div>
  )
}



function SeedFileListPanel({
  discoveryFileSet,
  candidates,
  candidateColors,
  selectedNode,
  onSelectFile,
  markedWrongIds,
  onToggleWrong,
  onExcludeFile,
}: {
  discoveryFileSet: Map<string, AEHDiscoveryCandidate>
  candidates: AEHDiscoveryCandidate[]
  candidateColors: Record<string, string>
  selectedNode: string | null
  onSelectFile: (path: string) => void
  markedWrongIds: Set<string>
  onToggleWrong: (candidateId: string) => void
  onExcludeFile: (candidateId: string, path: string, excluded: boolean) => void
}): React.ReactElement {
  const [filterText, setFilterText] = useState('')

  const grouped = useMemo(() => {
    return candidates
      .map((cand) => ({
        candidate: cand,
        files: cand.cluster_files.filter((f) =>
          filterText.trim() === '' || f.toLowerCase().includes(filterText.toLowerCase())
        ),
      }))
      .filter((g) => g.files.length > 0)
  }, [candidates, filterText])

  if (candidates.length === 0) {
    return (
      <div className="h-full w-full flex flex-col items-center justify-center p-6 text-center text-slate-500 text-xs select-none">
        <Sparkles className="w-10 h-10 mb-3 text-indigo-500/30 animate-pulse" />
        <p className="font-semibold text-slate-400 mb-1">No discovery candidates generated</p>
        <p className="text-slate-500 max-w-sm leading-relaxed">
          Click the **Run Discovery** button above to scan files and populate the interactive candidate list.
        </p>
      </div>
    )
  }

  return (
    <div className="flex flex-col h-full bg-[#090d16]">
      <div className="p-3 border-b border-slate-800 shrink-0">
        <input
          value={filterText}
          onChange={(e) => setFilterText(e.target.value)}
          placeholder="Filter files by path..."
          className="w-full text-[11px] bg-slate-950 border border-slate-800 rounded-md px-2.5 py-1.5 text-slate-300 placeholder:text-slate-600 focus:outline-none focus:border-indigo-500 transition-colors"
        />
      </div>
      <div className="flex-1 overflow-y-auto divide-y divide-slate-850">
        {grouped.map(({ candidate: cand, files }) => (
          <div key={cand.id} className="pb-2">
            <div className="sticky top-0 px-3 py-2 bg-[#0c1220]/95 backdrop-blur-sm border-b border-slate-850 flex items-center gap-2 text-[10px] font-semibold text-slate-300 z-10">
              {cand.verdict === 'proposed' ? (
                <input
                  type="checkbox"
                  checked={markedWrongIds.has(cand.id)}
                  onChange={() => onToggleWrong(cand.id)}
                  title="Mark this candidate as wrong (will be rejected on Confirm All Others)"
                  className="shrink-0 accent-red-500"
                />
              ) : (
                <span className="w-2.5 h-2.5 rounded-full shrink-0" style={{ background: candidateColors[cand.id] }} />
              )}
              <span className="truncate flex-1">{cand.name}</span>
              {cand.verdict === 'confirmed' && (
                <span className="text-[8px] text-emerald-400 font-semibold shrink-0">confirmed</span>
              )}
              {cand.verdict === 'rejected' && (
                <span className="text-[8px] text-red-400 font-semibold shrink-0 line-through">rejected</span>
              )}
              <span className="text-slate-500 font-mono text-[9px]">({files.length})</span>
            </div>
            <div className="divide-y divide-slate-900/40">
              {files.map((path) => {
                const isHit = cand.evidence.some((h) => h.file === path)
                const isSelected = path === selectedNode
                const isExcluded = (cand.excluded_files || []).includes(path)
                return (
                  <button
                    key={path}
                    onClick={() => onSelectFile(path)}
                    className={`w-full text-left px-3 py-2.5 hover:bg-slate-900/30 transition-colors flex flex-col gap-0.5 border-l-2 ${
                      isSelected
                        ? 'bg-indigo-950/20 border-l-amber-400'
                        : 'border-l-transparent'
                    } ${isExcluded ? 'opacity-50' : ''}`}
                  >
                    <div className="flex items-center gap-1.5 w-full">
                      <div className={`text-[11px] font-medium text-slate-200 truncate flex-1 ${isExcluded ? 'line-through' : ''}`}>
                        {path.split('/').pop()}
                      </div>
                      <button
                        type="button"
                        title={isExcluded ? 'Un-exclude this file' : 'Mark this file as not part of this candidate'}
                        onClick={(e) => {
                          e.stopPropagation()
                          onExcludeFile(cand.id, path, !isExcluded)
                        }}
                        className="shrink-0 p-0.5 rounded hover:bg-slate-800"
                      >
                        <ThumbsDown className={`w-3 h-3 ${isExcluded ? 'text-red-400' : 'text-slate-600'}`} />
                      </button>
                    </div>
                    <div className="flex items-center gap-2 w-full">
                      <span className="text-[9px] text-slate-500 font-mono truncate flex-1" title={path}>
                        {path}
                      </span>
                      {isHit && (
                        <span className="text-[8px] shrink-0 px-1 rounded bg-indigo-950/50 border border-indigo-900/30 text-indigo-300 font-mono">
                          hit
                        </span>
                      )}
                      {isExcluded && (
                        <span className="text-[8px] shrink-0 px-1 rounded bg-red-950/40 border border-red-900/30 text-red-400 font-mono">
                          excluded
                        </span>
                      )}
                    </div>
                  </button>
                )
              })}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

// Main Stage1GraphScreen
export default function Stage1GraphScreen(): React.ReactElement {
  const [searchParams] = useSearchParams()
  const navigate = useNavigate()
  const toast = useToastStore()
  const { startAEH } = useAEHStore()
  const { providers, load: loadProviders } = useProviderStore()

  const repoId = searchParams.get('repoId') || ''
  const snapshotId = searchParams.get('snapshotId') || ''

  const [graphExport, setGraphExport] = useState<ExportJson | null>(null)
  const [communityData, setCommunityData] = useState<CommunitiesResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [loadTimedOut, setLoadTimedOut] = useState(false)
  const [rebuilding, setRebuilding] = useState(false)

  const [selectedNode, setSelectedNode] = useState<string | null>(null)

  // Discovery's Pass A requires a built retrieval index to scan fingerprints.
  const [retrievalStatus, setRetrievalStatus] = useState<'loading' | 'built' | 'missing'>('loading')

  useEffect(() => {
    if (!snapshotId) {
      setRetrievalStatus('missing')
      return
    }
    let cancelled = false
    window.api.retrieval.summary(snapshotId)
      .then((rSummary) => {
        if (cancelled) return
        setRetrievalStatus(rSummary && rSummary.built ? 'built' : 'missing')
      })
      .catch(() => {
        if (!cancelled) setRetrievalStatus('missing')
      })
    return () => { cancelled = true }
  }, [snapshotId])

  // Discovery states
  const [discoverySession, setDiscoverySession] = useState<AEHDiscoverySession | null>(null)
  const [candidates, setCandidates] = useState<AEHDiscoveryCandidate[]>([])
  const [runningDiscovery, setRunningDiscovery] = useState(false)
  const [discoveryError, setDiscoveryError] = useState<string | null>(null)
  const [pauseInfo, setPauseInfo] = useState<AEHDiscoverySession['pause_info']>(null)
  const [resuming, setResuming] = useState(false)

  // Picker options are scoped to this stage and kept in a local config modal.
  const [discoveryProviderId, setDiscoveryProviderId] = useState('')
  const [discoveryModelId, setDiscoveryModelId] = useState('')
  const [llmConfigOpen, setLlmConfigOpen] = useState(false)

  // Load code data and community layout from Codespectra backend
  const LOAD_TIMEOUT_MS = 60_000

  const loadGraphData = useCallback(async () => {
    if (!snapshotId) return
    setLoading(true)
    setError(null)
    setLoadTimedOut(false)
    let timedOut = false
    let timeoutHandle: ReturnType<typeof setTimeout>
    const dataPromise = Promise.all([
      window.api.graph.exportData(snapshotId),
      window.api.graph.communities(snapshotId),
    ])
    // Swallow the losing race's rejection to avoid an unhandled-rejection console warning.
    dataPromise.catch(() => {})
    const timeout = new Promise<never>((_, reject) => {
      timeoutHandle = setTimeout(() => {
        timedOut = true
        reject(new Error('Loading timed out after 60s — the structural graph may need a rebuild.'))
      }, LOAD_TIMEOUT_MS)
    })
    try {
      const [exported, communities] = await Promise.race([dataPromise, timeout])
      clearTimeout(timeoutHandle!)
      setGraphExport(exported)
      setCommunityData(communities)
    } catch (e: any) {
      clearTimeout(timeoutHandle!)
      setLoadTimedOut(timedOut)
      setError(e?.message ?? 'Failed to load graph export data.')
    } finally {
      setLoading(false)
    }
  }, [snapshotId])

  const handleRebuildGraph = async () => {
    if (!snapshotId || rebuilding) return
    setRebuilding(true)
    setError(null)
    try {
      await window.api.graph.build(snapshotId, true)
      await loadGraphData()
    } catch (e: any) {
      setError(e?.message ?? 'Failed to rebuild graph.')
    } finally {
      setRebuilding(false)
    }
  }

  useEffect(() => {
    if (snapshotId) {
      loadGraphData()
    }
  }, [snapshotId, loadGraphData])

  // Load previous session & candidates
  const loadDiscoverySession = useCallback(async () => {
    if (!snapshotId) return
    try {
      await startAEH()
      const sessions = await window.api.aeh.listDiscoverySessions(undefined, snapshotId)
      const matching = sessions.filter((s) => s.snapshot_id === snapshotId)
      if (matching.length > 0) {
        const latest = matching[0]
        setDiscoverySession(latest)
        if (latest.status === 'completed') {
          const cands = await window.api.aeh.listDiscoveryCandidates(latest.id)
          setCandidates(cands)
        } else if (latest.status === 'paused_rate_limit') {
          setPauseInfo(latest.pause_info)
          const cands = await window.api.aeh.listDiscoveryCandidates(latest.id)
          setCandidates(cands)
        } else if (latest.status === 'running') {
          setRunningDiscovery(true)
          startPolling(latest.id)
        }
      }
    } catch (e) {
      console.error('Failed to load discovery session', e)
    }
  }, [snapshotId, startAEH])

  useEffect(() => {
    if (snapshotId) {
      loadDiscoverySession()
    }
  }, [snapshotId, loadDiscoverySession])

  // Seed sensible defaults so Run Discovery is not blocked initially.
  useEffect(() => {
    loadProviders()
  }, [loadProviders])

  useEffect(() => {
    if (providers.length > 0 && !discoveryProviderId) {
      setDiscoveryProviderId(providers[0].id)
      setDiscoveryModelId(providers[0].model_id ?? '')
    }
  }, [providers, discoveryProviderId])

  // Polling discovery progress
  const startPolling = useSessionPolling<AEHDiscoverySession>({
    fetchSession: (sessionId) => window.api.aeh.getDiscoverySession(sessionId),
    intervalMs: 2500,
    onUpdate: setDiscoverySession,
    onDone: async (session) => {
      setRunningDiscovery(false)
      if (session.status === 'completed') {
        const cands = await window.api.aeh.listDiscoveryCandidates(session.id)
        setCandidates(cands)
        toast.success('Discovery completed successfully.')
      } else if (session.status === 'paused_rate_limit') {
        setPauseInfo(session.pause_info)
        const cands = await window.api.aeh.listDiscoveryCandidates(session.id)
        setCandidates(cands)
      } else if (session.status === 'failed') {
        setDiscoveryError(session.error ?? 'Discovery synthesis failed')
      }
    },
    onError: () => setRunningDiscovery(false),
  })

  const handleResumeDiscovery = async (overrideProviderId?: string, overrideModelId?: string) => {
    if (!discoverySession) return
    setResuming(true)
    try {
      await window.api.aeh.resumeDiscoverySession(discoverySession.id, {
        provider_id: overrideProviderId ?? null,
        model_id: overrideModelId ?? null,
      })
      setPauseInfo(null)
      setRunningDiscovery(true)
      startPolling(discoverySession.id)
    } catch (err: any) {
      toast.error(err?.message ?? 'Failed to resume discovery.')
    } finally {
      setResuming(false)
    }
  }

  // Action: Trigger new discovery
  const handleRunDiscovery = async () => {
    if (!snapshotId || runningDiscovery) return
    if (!discoveryProviderId) {
      toast.error('Please configure and select an LLM provider first.')
      return
    }
    if (retrievalStatus !== 'built') {
      toast.error('Retrieval index is not built for this snapshot yet — build it before running Discovery.')
      return
    }

    setRunningDiscovery(true)
    setDiscoveryError(null)
    setCandidates([])
    setDiscoverySession(null)
    setSelectedNode(null)

    try {
      await startAEH()
      const result = await window.api.aeh.startDiscovery({
        repo_ref: snapshotId,
        snapshot_id: snapshotId,
        provider_id: discoveryProviderId,
        model_id: discoveryModelId || null,
      })
      const session = await window.api.aeh.getDiscoverySession(result.session_id)
      setDiscoverySession(session)
      startPolling(result.session_id)
    } catch (err: any) {
      setRunningDiscovery(false)
      setDiscoveryError(err?.message ?? 'Failed to run discovery.')
      toast.error(err?.message ?? 'Discovery run failed.')
    }
  }

  // Action: Verdict update
  const handleVerdict = async (candidateId: string, verdict: 'confirmed' | 'rejected') => {
    try {
      await window.api.aeh.updateDiscoveryCandidateVerdict(candidateId, verdict)
      setCandidates((prev) =>
        prev.map((c) => c.id === candidateId ? { ...c, verdict } : c)
      )
      toast.success(`Candidate marked as ${verdict}.`)
    } catch (err) {
      toast.error('Failed to update review verdict.')
    }
  }

  const handleExcludeFile = async (candidateId: string, path: string, excluded: boolean) => {
    const cand = candidates.find((c) => c.id === candidateId)
    if (!cand) return
    const nextExcluded = excluded
      ? [...(cand.excluded_files || []), path]
      : (cand.excluded_files || []).filter((p) => p !== path)
    try {
      await window.api.aeh.updateDiscoveryCandidateExcludedFiles(candidateId, nextExcluded)
      setCandidates((prev) =>
        prev.map((c) => (c.id === candidateId ? { ...c, excluded_files: nextExcluded } : c))
      )
      toast.success(excluded ? 'File excluded.' : 'File un-excluded.')
    } catch (err) {
      toast.error('Failed to update file exclusion.')
    }
  }

  // Bulk review: mark the wrong candidates, confirm everything else in one click.
  const [markedWrongIds, setMarkedWrongIds] = useState<Set<string>>(new Set())
  const [confirmingAll, setConfirmingAll] = useState(false)

  const toggleMarkedWrong = (candidateId: string) => {
    setMarkedWrongIds((prev) => {
      const next = new Set(prev)
      if (next.has(candidateId)) next.delete(candidateId)
      else next.add(candidateId)
      return next
    })
  }

  const pendingCandidates = candidates.filter((c) => c.verdict === 'proposed')

  const handleConfirmAllOthers = async () => {
    if (pendingCandidates.length === 0 || confirmingAll) return
    setConfirmingAll(true)
    try {
      await Promise.all(
        pendingCandidates.map((c) =>
          window.api.aeh.updateDiscoveryCandidateVerdict(
            c.id,
            markedWrongIds.has(c.id) ? 'rejected' : 'confirmed'
          )
        )
      )
      setCandidates((prev) =>
        prev.map((c) =>
          c.verdict === 'proposed'
            ? { ...c, verdict: markedWrongIds.has(c.id) ? 'rejected' : 'confirmed' }
            : c
        )
      )
      setMarkedWrongIds(new Set())
      toast.success('Reviewed — confirmed all candidates except the ones marked wrong.')
    } catch (err) {
      toast.error('Failed to bulk-update candidate verdicts.')
    } finally {
      setConfirmingAll(false)
    }
  }

  // Candidate mapping: file path -> candidate
  const discoveryFileSet = useMemo(() => {
    const map = new Map<string, AEHDiscoveryCandidate>()
    for (const cand of candidates) {
      for (const file of cand.cluster_files) {
        map.set(file, cand)
      }
    }
    return map
  }, [candidates])

  // Assign colors to candidate IDs
  const candidateColors = useMemo(() => {
    const map: Record<string, string> = {}
    candidates.forEach((cand, idx) => {
      map[cand.id] = CANDIDATE_COLORS[idx % CANDIDATE_COLORS.length]
    })
    return map
  }, [candidates])



  if (loading) {
    return (
      <div className="h-full flex items-center justify-center bg-[#090d16] text-slate-300 text-xs">
        <Loader2 className="w-5 h-5 animate-spin mr-2 text-indigo-400" />
        <span>Loading discovery graph data...</span>
      </div>
    )
  }

  if (error || !graphExport || !communityData) {
    return (
      <div className="h-full flex items-center justify-center bg-[#090d16] p-6">
        <div className="text-center space-y-4 max-w-sm">
          <AlertCircle size={36} className="text-red-500 mx-auto" />
          <h3 className="text-sm font-semibold text-slate-200">
            {loadTimedOut ? 'Loading timed out' : 'Failed to load graph'}
          </h3>
          <p className="text-xs text-slate-500 leading-relaxed">{error || 'No graph data parsed.'}</p>
          {loadTimedOut && (
            <p className="text-[10px] text-slate-600 leading-relaxed">
              The structural graph for this snapshot may be stale (source files changed since the
              last build). Rebuilding re-scans the repo and can take a while for large changes.
            </p>
          )}
          <div className="flex justify-center gap-3">
            <Button variant="secondary" onClick={() => navigate(-1)}>
              Go Back
            </Button>
            {loadTimedOut && (
              <Button variant="primary" onClick={handleRebuildGraph} loading={rebuilding}>
                Rebuild Graph
              </Button>
            )}
            <Button variant={loadTimedOut ? 'secondary' : 'primary'} onClick={loadGraphData}>
              Retry
            </Button>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="flex h-full bg-[#090d16]">
      {/* Side board details */}
      <DiscoveryBoard
        selectedNode={selectedNode}
        discoveryFileSet={discoveryFileSet}
        candidateColors={candidateColors}
        onVerdict={handleVerdict}
        graphExport={graphExport}
      />

      {/* Graph Area */}
      <div className="flex-1 flex flex-col relative min-w-0 min-h-0">
        {/* Graph Header */}
        <div className="px-4 py-3 border-b border-slate-800 bg-[#0f172a]/60 backdrop-blur-sm flex items-center justify-between shrink-0">
          <div className="flex items-center gap-3 min-w-0">
            <button
              onClick={() => navigate(-1)}
              className="p-1 rounded-lg hover:bg-slate-800 transition-colors text-slate-400 hover:text-slate-200"
            >
              <ArrowLeft className="w-4 h-4" />
            </button>
            <div>
              <h2 className="text-xs font-semibold text-slate-200">Discovery Graph Explorer</h2>
              <p className="text-[10px] text-slate-500 truncate mt-0.5">
                {candidates.length} candidates · {discoveryFileSet.size} cluster nodes · {graphExport ? graphExport.edges.filter(e => discoveryFileSet.has(e.src) && discoveryFileSet.has(e.dst)).length : 0} connections
              </p>
            </div>
          </div>

          <div className="flex items-center gap-3 shrink-0">
            {/* Run Controls inside header */}
            <LLMModelButton
              providerId={discoveryProviderId}
              modelId={discoveryModelId}
              providers={providers}
              disabled={runningDiscovery}
              onClick={() => setLlmConfigOpen(true)}
            />
            <LLMConfigModal
              isOpen={llmConfigOpen}
              onClose={() => setLlmConfigOpen(false)}
              providerId={discoveryProviderId}
              modelId={discoveryModelId}
              onChange={(pid, mid) => {
                setDiscoveryProviderId(pid)
                setDiscoveryModelId(mid)
              }}
              onConfirmResume={
                pauseInfo
                  ? () => {
                      handleResumeDiscovery(discoveryProviderId, discoveryModelId)
                      setLlmConfigOpen(false)
                    }
                  : undefined
              }
            />

            <Button
              variant="primary"
              disabled={runningDiscovery || !discoveryProviderId || retrievalStatus !== 'built'}
              onClick={handleRunDiscovery}
              title={retrievalStatus !== 'built' ? 'Retrieval index not built for this snapshot yet' : undefined}
              className="text-[10px] h-7 px-3 flex items-center gap-1.5"
            >
              {runningDiscovery ? (
                <>
                  <Loader2 className="w-3.5 h-3.5 animate-spin" />
                  <span>Discovering...</span>
                </>
              ) : (
                <>
                  <Play className="w-3 h-3 ml-0.5" />
                  <span>Run Discovery</span>
                </>
              )}
            </Button>

            {pendingCandidates.length > 0 && (
              <>
                <span className="h-4 w-px bg-slate-800" />
                <Button
                  variant="primary"
                  disabled={confirmingAll}
                  onClick={handleConfirmAllOthers}
                  title="Rejects candidates you've checked below, confirms every other pending candidate"
                  className="text-[10px] h-7 px-3 flex items-center gap-1.5 bg-emerald-600 hover:bg-emerald-500"
                >
                  {confirmingAll ? (
                    <Loader2 className="w-3.5 h-3.5 animate-spin" />
                  ) : (
                    <ThumbsUp className="w-3.5 h-3.5" />
                  )}
                  <span>Confirm All Others ({pendingCandidates.length - markedWrongIds.size})</span>
                </Button>
              </>
            )}

          </div>
        </div>

        {retrievalStatus === 'missing' && !runningDiscovery && (
          <div className="mx-4 mt-3 flex items-center gap-2 bg-amber-950/20 border border-amber-800/40 rounded-lg px-3 py-2 text-xs text-amber-300 shrink-0">
            <AlertCircle className="w-3.5 h-3.5 shrink-0" />
            Retrieval index not built for this snapshot — build it (Repositories screen) before running Discovery.
          </div>
        )}

        {/* Discovery progress overlay */}
        {runningDiscovery && (
          <div className="absolute inset-x-0 top-0 z-10 bg-indigo-950/80 border-b border-indigo-500/40 px-4 py-3 backdrop-blur-sm flex items-center justify-between text-xs text-indigo-200">
            <div className="flex items-center gap-3">
              <Loader2 className="w-4 h-4 animate-spin text-indigo-400" />
              <span>
                Running Discovery Sweep: Scanning fingerprint triggers → community clustering → synthesizing candidate names
              </span>
            </div>
            <span className="text-[10px] text-indigo-400 animate-pulse font-mono font-semibold">Pass A/B/C Sweep...</span>
          </div>
        )}

        {/* Discovery error header */}
        {discoveryError && (
          <div className="absolute inset-x-0 top-0 z-10 bg-red-950/90 border-b border-red-500/40 px-4 py-3 backdrop-blur-sm flex items-center gap-3 text-xs text-red-200">
            <AlertCircle className="w-4 h-4 shrink-0 text-red-400" />
            <span className="flex-1 truncate">{discoveryError}</span>
            <button
              onClick={() => setDiscoveryError(null)}
              className="text-[10px] font-semibold underline text-red-400 hover:text-red-300"
            >
              Dismiss
            </button>
          </div>
        )}

        {pauseInfo && (
          <div className="absolute inset-x-0 top-0 z-10 bg-amber-950/90 border-b border-amber-500/40 px-4 py-3 backdrop-blur-sm flex items-center gap-3 text-xs text-amber-200">
            <AlertCircle className="w-4 h-4 shrink-0 text-amber-400" />
            <span className="flex-1">
              Rate limited on provider "{pauseInfo.provider_id}" — paused. Wait a bit and resume with the
              same model, or switch to a different one.
            </span>
            <Button variant="secondary" disabled={resuming} onClick={() => handleResumeDiscovery()} className="text-[10px] h-6 px-2">
              Wait & Resume
            </Button>
            <Button variant="primary" disabled={resuming} onClick={() => setLlmConfigOpen(true)} className="text-[10px] h-6 px-2">
              Switch Model
            </Button>
          </div>
        )}

        <div className="flex-1 flex flex-col min-h-0 overflow-hidden divide-y divide-slate-800">
          {/* Top row: Wiring Block details + Small Graph */}
          <div className="flex h-1/2 min-h-0 divide-x divide-slate-800">
            <div className="flex-1 min-h-0 overflow-y-auto">
              <WiringBlockPanel
                selectedNode={selectedNode}
                discoveryFileSet={discoveryFileSet}
                candidates={candidates}
              />
            </div>
            <div className="flex-1 min-h-0 relative">
              <WiringGraphPanel
                selectedNode={selectedNode}
                discoveryFileSet={discoveryFileSet}
                candidates={candidates}
              />
            </div>
          </div>

          {/* Bottom row: Seed files list */}
          <div className="flex-1 min-h-0 overflow-hidden">
            <SeedFileListPanel
              discoveryFileSet={discoveryFileSet}
              candidates={candidates}
              candidateColors={candidateColors}
              selectedNode={selectedNode}
              onSelectFile={setSelectedNode}
              markedWrongIds={markedWrongIds}
              onToggleWrong={toggleMarkedWrong}
              onExcludeFile={handleExcludeFile}
            />
          </div>
        </div>
      </div>
    </div>
  )
}

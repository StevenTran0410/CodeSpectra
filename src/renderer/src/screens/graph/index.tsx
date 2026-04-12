import React, { useState, useEffect, useCallback, useMemo } from 'react'
import { useSearchParams, useNavigate } from 'react-router-dom'
import {
  ReactFlow,
  Background,
  Controls,
  Panel,
  MarkerType,
  type Node,
  type Edge,
} from '@xyflow/react'
import '@xyflow/react/dist/style.css'
import dagre from '@dagrejs/dagre'
import { Loader2, AlertCircle, Save } from 'lucide-react'

type ExportJson = {
  nodes: string[]
  edges: Array<{ src: string; dst: string; external: boolean }>
  communities: Record<string, number>       // node_path -> community_id
  community_groups: Record<string, string[]> // community_id -> [node_paths]
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

type NeighborResult = {
  snapshot_id: string
  seed_path: string
  hops: number
  nodes: string[]
  edges: Array<{
    src_path: string
    dst_path: string
    edge_type: string
    is_external: boolean
  }>
}

const COMMUNITY_COLORS = [
  '#6366f1', '#0ea5e9', '#10b981', '#f59e0b',
  '#ef4444', '#8b5cf6', '#ec4899', '#14b8a6',
  '#f97316', '#84cc16',
]

const MAX_NODES_DISPLAY = 500

function buildFlowGraph(
  data: ExportJson,
  nodeIndex: Record<string, number>,
  selectedNode: string | null,
  neighborNodes: Set<string>,
  cycleNodes: Set<string>,
): { nodes: Node[]; edges: Edge[] } {
  const nodes: Node[] = data.nodes.map((path) => {
    const communityId = nodeIndex[path] ?? -1
    const color = communityId >= 0
      ? COMMUNITY_COLORS[communityId % COMMUNITY_COLORS.length]
      : '#52525b'
    const isCycle = cycleNodes.has(path)
    const isSelected = path === selectedNode
    const isNeighbor = neighborNodes.has(path)

    const filename = path.split('/').pop() ?? path
    return {
      id: path,
      data: { label: filename, fullPath: path, communityId },
      position: { x: 0, y: 0 },
      style: {
        background: isSelected ? '#fbbf24' : isCycle ? '#ef4444' : color,
        color: '#fff',
        fontSize: 10,
        padding: '4px 8px',
        borderRadius: 6,
        border: isNeighbor ? '2px solid #fbbf24' : '1px solid rgba(255,255,255,0.2)',
        opacity: selectedNode && !isSelected && !isNeighbor ? 0.35 : 1,
        maxWidth: 160,
        overflow: 'hidden',
        textOverflow: 'ellipsis',
        whiteSpace: 'nowrap',
        cursor: 'pointer',
      },
    } as Node
  })

  const edges: Edge[] = data.edges
    .filter((e) => !e.external)
    .map((e) => ({
      id: `${e.src}->${e.dst}`,
      source: e.src,
      target: e.dst,
      style: { stroke: '#52525b', strokeWidth: 1 },
      markerEnd: { type: MarkerType.ArrowClosed, color: '#52525b' },
    }))

  return { nodes, edges }
}

function applyDagreLayout(nodes: Node[], edges: Edge[]): Node[] {
  const g = new dagre.graphlib.Graph()
  g.setGraph({ rankdir: 'LR', nodesep: 40, ranksep: 80 })
  g.setDefaultEdgeLabel(() => ({}))

  nodes.forEach((n) => g.setNode(n.id, { width: 160, height: 36 }))
  edges.forEach((e) => g.setEdge(e.source, e.target))

  dagre.layout(g)

  return nodes.map((n) => {
    const pos = g.node(n.id)
    return { ...n, position: { x: pos.x - 80, y: pos.y - 18 } }
  })
}

interface LeftPanelProps {
  selectedNode: string | null
  graphData: ExportJson | null
  communityData: CommunitiesResponse | null
  neighborData: NeighborResult | null
  neighborLoading: boolean
}

function LeftPanel({
  selectedNode,
  graphData,
  communityData,
  neighborData,
  neighborLoading,
}: LeftPanelProps) {
  if (!selectedNode || !graphData || !communityData) {
    return (
      <div className="w-80 border-r border-zinc-700 bg-zinc-900 p-4 text-zinc-400 text-sm flex items-center justify-center">
        Click a node to see details
      </div>
    )
  }

  const communityId = communityData.node_index[selectedNode] ?? -1
  const community = communityData.communities.find((c) => c.community_id === communityId)

  // "Files that import this" = incoming edges (dst === selectedNode)
  const incomingEdges = graphData.edges.filter((e) => e.dst === selectedNode && !e.external)
  const incomingFiles = incomingEdges.map((e) => e.src).slice(0, 10)
  const incomingMore = incomingEdges.length > 10 ? incomingEdges.length - 10 : 0

  // "This imports" = outgoing edges (src === selectedNode)
  const outgoingEdges = graphData.edges.filter((e) => e.src === selectedNode && !e.external)
  const outgoingFiles = outgoingEdges.map((e) => e.dst).slice(0, 10)
  const outgoingMore = outgoingEdges.length > 10 ? outgoingEdges.length - 10 : 0

  const blastRadiusFiles = neighborData?.nodes.filter((n) => n !== selectedNode) ?? []

  return (
    <div className="w-80 border-r border-zinc-700 bg-zinc-900 p-4 overflow-y-auto text-xs space-y-4">
      {/* Node info */}
      <div>
        <div className="text-zinc-300 font-mono text-[10px] break-words">{selectedNode}</div>
      </div>

      {/* Community */}
      {community && (
        <div>
          <div className="text-zinc-400 font-semibold mb-2">Community</div>
          <div className="flex items-center gap-2 text-zinc-300">
            <div
              className="w-3 h-3 rounded-full"
              style={{
                backgroundColor:
                  COMMUNITY_COLORS[community.community_id % COMMUNITY_COLORS.length],
              }}
            />
            <span>Community {community.community_id}</span>
          </div>
          {community.hub_paths.length > 0 && (
            <div className="mt-2 text-zinc-400">
              <div className="text-[10px] mb-1">Hub paths:</div>
              <div className="space-y-1">
                {community.hub_paths.slice(0, 5).map((hp) => (
                  <div key={hp} className="text-[10px] text-zinc-500 truncate">
                    {hp.split('/').pop()}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Files that import this */}
      {incomingFiles.length > 0 && (
        <div>
          <div className="text-zinc-400 font-semibold mb-2">Files that import this</div>
          <div className="space-y-1">
            {incomingFiles.map((f) => (
              <div key={f} className="text-[10px] text-zinc-400 truncate">
                {f.split('/').pop()}
              </div>
            ))}
            {incomingMore > 0 && (
              <div className="text-[10px] text-zinc-500 italic">and {incomingMore} more...</div>
            )}
          </div>
        </div>
      )}

      {/* This imports */}
      {outgoingFiles.length > 0 && (
        <div>
          <div className="text-zinc-400 font-semibold mb-2">This imports</div>
          <div className="space-y-1">
            {outgoingFiles.map((f) => (
              <div key={f} className="text-[10px] text-zinc-400 truncate">
                {f.split('/').pop()}
              </div>
            ))}
            {outgoingMore > 0 && (
              <div className="text-[10px] text-zinc-500 italic">and {outgoingMore} more...</div>
            )}
          </div>
        </div>
      )}

      {/* Blast radius */}
      {neighborData && !neighborLoading && (
        <div>
          <div className="text-zinc-400 font-semibold mb-2">Blast radius (2-hop)</div>
          <div className="space-y-1">
            {blastRadiusFiles.slice(0, 10).map((f) => (
              <div key={f} className="text-[10px] text-zinc-400 truncate">
                {f.split('/').pop()}
              </div>
            ))}
            {blastRadiusFiles.length > 10 && (
              <div className="text-[10px] text-zinc-500 italic">
                and {blastRadiusFiles.length - 10} more...
              </div>
            )}
          </div>
        </div>
      )}

      {neighborLoading && (
        <div className="flex items-center gap-2 text-zinc-500">
          <Loader2 size={12} className="animate-spin" />
          <span>Loading neighbors...</span>
        </div>
      )}
    </div>
  )
}

interface LegendProps {
  communityCount: number
}

function Legend({ communityCount }: LegendProps) {
  const shown = Math.min(10, communityCount)
  const extra = communityCount - shown
  return (
    <Panel position="top-right">
    <div className="bg-zinc-800/90 border border-zinc-700 rounded p-3 text-xs space-y-1 pointer-events-none">
      <div className="font-semibold text-zinc-300 mb-1.5">Legend</div>
      {Array.from({ length: shown }).map((_, i) => (
        <div key={i} className="flex items-center gap-2">
          <div
            className="w-2 h-2 rounded-full shrink-0"
            style={{ backgroundColor: COMMUNITY_COLORS[i % COMMUNITY_COLORS.length] }}
          />
          <span className="text-zinc-400">Community {i}</span>
        </div>
      ))}
      {extra > 0 && (
        <div className="text-zinc-500 italic">+{extra} more</div>
      )}
      <div className="border-t border-zinc-700 pt-1 mt-1" />
      <div className="flex items-center gap-2">
        <div className="w-2 h-2 rounded-full bg-red-500 shrink-0" />
        <span className="text-zinc-400">Circular import</span>
      </div>
      <div className="flex items-center gap-2">
        <div className="w-2 h-2 rounded-full bg-amber-400 shrink-0" />
        <span className="text-zinc-400">Selected</span>
      </div>
      <div className="flex items-center gap-2">
        <div className="w-2 h-2 rounded-full border border-amber-400 shrink-0" />
        <span className="text-zinc-400">Neighbor</span>
      </div>
    </div>
    </Panel>
  )
}

export default function GraphScreen(): React.ReactElement {
  const [searchParams] = useSearchParams()
  const navigate = useNavigate()
  const snapshotId = searchParams.get('snapshotId') ?? ''

  const [graphData, setGraphData] = useState<ExportJson | null>(null)
  const [communityData, setCommunityData] = useState<CommunitiesResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [selectedNode, setSelectedNode] = useState<string | null>(null)
  const [neighborData, setNeighborData] = useState<NeighborResult | null>(null)
  const [neighborLoading, setNeighborLoading] = useState(false)
  const [exporting, setExporting] = useState(false)

  const loadGraph = useCallback(async () => {
    if (!snapshotId) {
      setError('No snapshot ID provided')
      return
    }

    setLoading(true)
    setError(null)
    try {
      const [exported, communities] = await Promise.all([
        window.api.graph.exportData(snapshotId),
        window.api.graph.communities(snapshotId),
      ])
      setGraphData(exported)
      setCommunityData(communities)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load graph data')
    } finally {
      setLoading(false)
    }
  }, [snapshotId])

  useEffect(() => {
    if (snapshotId) {
      loadGraph()
    }
  }, [snapshotId, loadGraph])

  const onNodeClick = useCallback(
    async (_: React.MouseEvent, node: Node) => {
      const path = node.id
      setSelectedNode(path)
      if (!snapshotId) return

      setNeighborLoading(true)
      try {
        const res = await window.api.graph.neighbors(snapshotId, path, 2, 100)
        setNeighborData(res)
      } catch {
        setNeighborData(null)
      } finally {
        setNeighborLoading(false)
      }
    },
    [snapshotId]
  )

  const nodeIndex = graphData?.communities ?? {}
  const communityCount = Object.keys(graphData?.community_groups ?? {}).length

  const { nodes: rawNodes, edges } = useMemo(() => {
    if (!graphData) return { nodes: [], edges: [] }

    const cycleNodes = new Set<string>()
    graphData.cycles.forEach((cycle) => {
      cycle.forEach((path) => cycleNodes.add(path))
    })

    const neighborSet = new Set(neighborData?.nodes ?? [])
    if (selectedNode) neighborSet.add(selectedNode)

    const cappedNodes = graphData.nodes.slice(0, MAX_NODES_DISPLAY)
    const cappedNodeSet = new Set(cappedNodes)
    const cappedData =
      graphData.nodes.length > MAX_NODES_DISPLAY
        ? {
            ...graphData,
            nodes: cappedNodes,
            edges: graphData.edges.filter((e) => cappedNodeSet.has(e.src) && cappedNodeSet.has(e.dst)),
          }
        : graphData

    return buildFlowGraph(cappedData, nodeIndex, selectedNode, neighborSet, cycleNodes)
  }, [graphData, nodeIndex, selectedNode, neighborData])

  const nodes = useMemo(() => {
    if (rawNodes.length === 0) return []
    return applyDagreLayout(rawNodes, edges)
  }, [rawNodes, edges])

  const fitViewOptions = { padding: 0.1 }

  if (!snapshotId) {
    return (
      <div className="h-full flex items-center justify-center bg-zinc-950">
        <div className="text-center space-y-4">
          <AlertCircle size={32} className="text-zinc-500 mx-auto" />
          <div className="text-zinc-400">No snapshot ID provided</div>
          <button
            onClick={() => navigate('/index-overview')}
            className="px-4 py-2 bg-indigo-700 text-white rounded text-sm hover:bg-indigo-600"
          >
            Go back
          </button>
        </div>
      </div>
    )
  }

  if (loading) {
    return (
      <div className="h-full flex items-center justify-center bg-zinc-950">
        <div className="flex items-center gap-3 text-zinc-400">
          <Loader2 size={20} className="animate-spin" />
          <span>Loading graph...</span>
        </div>
      </div>
    )
  }

  if (error || !graphData || !communityData) {
    return (
      <div className="h-full flex items-center justify-center bg-zinc-950">
        <div className="text-center space-y-4">
          <AlertCircle size={32} className="text-red-500 mx-auto" />
          <div className="text-red-400">{error || 'Failed to load graph'}</div>
          <button
            onClick={() => loadGraph()}
            className="px-4 py-2 bg-indigo-700 text-white rounded text-sm hover:bg-indigo-600"
          >
            Retry
          </button>
        </div>
      </div>
    )
  }

  const totalNodes = graphData?.nodes.length ?? 0
  const nodeCountWarning = totalNodes > MAX_NODES_DISPLAY

  return (
    <div className="flex h-full bg-zinc-950">
      {/* Left panel */}
      <LeftPanel
        selectedNode={selectedNode}
        graphData={graphData}
        communityData={communityData}
        neighborData={neighborData}
        neighborLoading={neighborLoading}
      />

      {/* Main canvas area */}
      <div className="flex-1 flex flex-col relative">
        {/* Header */}
        <div className="px-4 py-3 border-b border-zinc-700 bg-zinc-900 flex items-center justify-between shrink-0">
          <div className="text-xs text-zinc-300">
            Graph ({nodes.length} nodes, {edges.length} edges)
            {nodeCountWarning && (
              <span className="ml-2 text-amber-400">
                (showing {MAX_NODES_DISPLAY} of {totalNodes} nodes)
              </span>
            )}
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={async () => {
                setExporting(true)
                try { await window.api.graph.exportJson(snapshotId) }
                finally { setExporting(false) }
              }}
              disabled={exporting}
              className="px-2.5 py-1 text-xs border border-zinc-700 rounded text-zinc-400 hover:border-zinc-600 hover:text-zinc-300 disabled:opacity-40 inline-flex items-center gap-1"
            >
              <Save size={11} />
              {exporting ? 'Exporting…' : 'Export JSON'}
            </button>
            <button
              onClick={() => navigate(-1)}
              className="px-2.5 py-1 text-xs border border-zinc-700 rounded text-zinc-300 hover:border-zinc-600"
            >
              Back
            </button>
          </div>
        </div>

        {/* React Flow canvas */}
        <div className="flex-1 relative" style={{ height: 'calc(100% - 3rem)' }}>
          <ReactFlow
            nodes={nodes}
            edges={edges}
            onNodeClick={onNodeClick}
            fitView
            fitViewOptions={fitViewOptions}
            minZoom={0.05}
            maxZoom={2}
          >
            <Background />
            <Controls />
            <Legend communityCount={communityCount} />
          </ReactFlow>
        </div>
      </div>
    </div>
  )
}

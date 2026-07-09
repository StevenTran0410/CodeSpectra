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
  type NodeChange,
} from '@xyflow/react'
import '@xyflow/react/dist/style.css'
import dagre from '@dagrejs/dagre'
import * as d3force from 'd3-force'
import { Loader2, AlertCircle, Save, Zap, X } from 'lucide-react'
import { Button, PageLoading, useToastStore } from '../../components/ui'

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

type BlastRadiusResponse = {
  changed_files: string[]
  blast_radius: {
    total_affected: number
    by_hop: Record<number, number>
    high_risk_files: string[]
    affected_communities: Array<{
      community_id: number
      member_count: number
      hub_paths: string[]
    }>
    call_chains: unknown[]
  }
  subgraph: {
    nodes: string[]
    edges: unknown[]
    seed_files: string[]
    hop_colors: Record<string, string>
  }
  context_chunks: unknown[]
}

type ImpactState = {
  active: boolean
  seedFiles: string[]
  result: BlastRadiusResponse | null
}

type SymbolEdgeInfo = {
  src_symbol: string
  dst_symbol: string
  edge_type: string
  confidence_score: number
  resolution_method: string
}

type FileSymbolEdgesResponse = {
  snapshot_id: string
  file_path: string
  defined_symbols: string[]
  outgoing: SymbolEdgeInfo[]
  incoming: SymbolEdgeInfo[]
}

/** "file.py::Class.method" -> "file.py" */
function symbolFile(symbol: string): string {
  return symbol.includes('::') ? symbol.split('::')[0] : symbol
}

/** "file.py::Class.method" -> "Class.method" */
function symbolName(symbol: string): string {
  return symbol.includes('::') ? symbol.split('::').slice(1).join('::') : symbol
}

const COMMUNITY_COLORS = [
  '#6366f1', '#0ea5e9', '#10b981', '#f59e0b',
  '#ef4444', '#8b5cf6', '#ec4899', '#14b8a6',
  '#f97316', '#84cc16',
]

const MAX_NODES_DISPLAY = 500

function getNodeStyle(
  path: string,
  communityId: number,
  isSelected: boolean,
  isNeighbor: boolean,
  isCycle: boolean,
  selectedNode: string | null,
  impactData?: BlastRadiusResponse | null,
): React.CSSProperties {
  let background: string
  if (isSelected) {
    background = '#fbbf24'
  } else if (impactData) {
    // Impact mode: color by hop distance
    const hopColor = impactData.subgraph.hop_colors[path]
    background = hopColor ?? '#52525b'
  } else if (isCycle) {
    background = '#ef4444'
  } else {
    background = communityId >= 0
      ? COMMUNITY_COLORS[communityId % COMMUNITY_COLORS.length]
      : '#52525b'
  }

  const dimmed = selectedNode && !isSelected && !isNeighbor && !impactData

  return {
    background,
    color: '#fff',
    fontSize: 10,
    padding: '4px 8px',
    borderRadius: 6,
    border: isNeighbor ? '2px solid #fbbf24' : '1px solid rgba(255,255,255,0.2)',
    opacity: dimmed ? 0.35 : 1,
    maxWidth: 160,
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
    cursor: 'pointer',
  }
}

function buildFlowGraph(
  data: ExportJson,
  nodeIndex: Record<string, number>,
  selectedNode: string | null,
  neighborNodes: Set<string>,
  cycleNodes: Set<string>,
  impactData?: BlastRadiusResponse | null,
): { nodes: Node[]; edges: Edge[] } {
  const nodes: Node[] = data.nodes.map((path) => {
    const communityId = nodeIndex[path] ?? -1
    const isCycle = cycleNodes.has(path)
    const isSelected = path === selectedNode
    const isNeighbor = neighborNodes.has(path)

    const filename = path.split('/').pop() ?? path
    return {
      id: path,
      data: { label: filename, fullPath: path, communityId },
      position: { x: 0, y: 0 },
      style: getNodeStyle(path, communityId, isSelected, isNeighbor, isCycle, selectedNode, impactData),
    } as Node
  })

  const edges: Edge[] = data.edges
    .filter((e) => !e.external)
    .map((e) => ({
      id: `${e.src}->${e.dst}`,
      source: e.src,
      target: e.dst,
      style: { stroke: '#71717a', strokeWidth: 1.25, opacity: 0.7 },
      markerEnd: { type: MarkerType.ArrowClosed, color: '#71717a' },
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

interface ForceNode extends d3force.SimulationNodeDatum {
  id: string
  communityId: number
}

/** Force-directed layout with community clustering: one differential-collision rule gives tight same-cluster packing and wide gaps between clusters, instead of a separate repulsion force fighting for the same job. */
function applyForceClusterLayout(nodes: Node[], edges: Edge[]): Node[] {
  const n = nodes.length
  if (n === 0) return nodes

  // Seed positions on a circle so the simulation doesn't start with every node at the origin.
  const seedRadius = Math.max(300, n * 4)
  const forceNodes: ForceNode[] = nodes.map((node, i) => {
    const angle = (2 * Math.PI * i) / n
    return {
      id: node.id,
      communityId: (node.data?.communityId as number | undefined) ?? -1,
      x: seedRadius * Math.cos(angle),
      y: seedRadius * Math.sin(angle),
    }
  })

  const nodeById = new Map(forceNodes.map((fn) => [fn.id, fn]))
  const forceLinks = edges
    .filter((e) => nodeById.has(e.source) && nodeById.has(e.target))
    .map((e) => ({ source: e.source, target: e.target }))

  // Padding sized for this app's ~160x36 node boxes: same-cluster pairs get just enough
  // room to avoid label overlap, different-cluster pairs get a much wider gap.
  const SAME_CLUSTER_PADDING = 100
  const DIFF_CLUSTER_PADDING = 230

  // Brute-force O(n^2) pairwise collision check — fine at this app's node-count scale;
  // a quadtree-accelerated version only pays off on much larger graphs.
  function differentialCollide(alpha: number): void {
    for (let i = 0; i < forceNodes.length; i++) {
      const a = forceNodes[i]
      for (let j = i + 1; j < forceNodes.length; j++) {
        const b = forceNodes[j]
        const dx = (b.x ?? 0) - (a.x ?? 0)
        const dy = (b.y ?? 0) - (a.y ?? 0)
        const dist = Math.sqrt(dx * dx + dy * dy) || 0.01
        const minDist = a.communityId === b.communityId ? SAME_CLUSTER_PADDING : DIFF_CLUSTER_PADDING
        if (dist >= minDist) continue
        const push = ((minDist - dist) / dist) * alpha * 0.5
        const ox = dx * push
        const oy = dy * push
        a.vx = (a.vx ?? 0) - ox
        a.vy = (a.vy ?? 0) - oy
        b.vx = (b.vx ?? 0) + ox
        b.vy = (b.vy ?? 0) + oy
      }
    }
  }

  function clusterCohesion(alpha: number): void {
    const centroids = new Map<number, { x: number; y: number; count: number }>()
    for (const fn of forceNodes) {
      const c = centroids.get(fn.communityId) ?? { x: 0, y: 0, count: 0 }
      c.x += fn.x ?? 0
      c.y += fn.y ?? 0
      c.count += 1
      centroids.set(fn.communityId, c)
    }
    for (const fn of forceNodes) {
      const c = centroids.get(fn.communityId)
      if (!c || c.count <= 2) continue // singleton/pair communities: nothing to cohere to
      fn.vx = (fn.vx ?? 0) + (c.x / c.count - (fn.x ?? 0)) * alpha * 0.1
      fn.vy = (fn.vy ?? 0) + (c.y / c.count - (fn.y ?? 0)) * alpha * 0.1
    }
  }

  const simulation = d3force
    .forceSimulation(forceNodes)
    // No generic repulsion — differential collision below already does all the separation work.
    .force(
      'link',
      d3force
        .forceLink<ForceNode, { source: string; target: string }>(forceLinks)
        .id((d) => d.id)
        .distance(70)
        .strength(0.3)
    )
    // Very weak per-node gravity — just a safety net against disconnected nodes drifting off.
    .force('gravityX', d3force.forceX(0).strength(0.004))
    .force('gravityY', d3force.forceY(0).strength(0.004))
    .velocityDecay(0.45) // dampens oscillation for a more stable settled layout
    .stop()

  // Run synchronously to a settled state — React Flow only needs final positions. Multiple
  // collision passes per tick are needed to fully resolve overlaps in a dense graph.
  for (let tick = 0; tick < 300; tick++) {
    simulation.tick()
    const alpha = simulation.alpha()
    clusterCohesion(alpha)
    for (let pass = 0; pass < 3; pass++) differentialCollide(alpha)
  }

  const positionById = new Map(forceNodes.map((fn) => [fn.id, { x: fn.x ?? 0, y: fn.y ?? 0 }]))
  return nodes.map((node) => ({ ...node, position: positionById.get(node.id) ?? { x: 0, y: 0 } }))
}

interface LeftPanelProps {
  selectedNode: string | null
  graphData: ExportJson | null
  communityData: CommunitiesResponse | null
  neighborData: NeighborResult | null
  neighborLoading: boolean
  impactState: ImpactState
  impactLoading: boolean
  symbolEdgesData: FileSymbolEdgesResponse | null
  symbolEdgesLoading: boolean
}

function LeftPanel({
  selectedNode,
  graphData,
  communityData,
  neighborData,
  neighborLoading,
  impactState,
  impactLoading,
  symbolEdgesData,
  symbolEdgesLoading,
}: LeftPanelProps) {
  const showImpactPanel = impactState.active && (impactState.result !== null || impactState.seedFiles.length > 0 || impactLoading)

  if (!selectedNode || !graphData || !communityData) {
    if (showImpactPanel) {
      return (
        <ImpactPanel impactState={impactState} impactLoading={impactLoading} />
      )
    }
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

  // Centrality score, computed client-side using the same formula as the backend's
  // _compute_scores_python: indegree*3 + outdegree.
  const indegree = incomingEdges.length
  const outdegree = outgoingEdges.length
  const centralityScore = indegree * 3 + outdegree

  // Cross-file function calls only — same-file self-references are noise here.
  const outgoingCalls = (symbolEdgesData?.outgoing ?? []).filter(
    (e) => symbolFile(e.dst_symbol) !== selectedNode
  )
  const incomingCalls = (symbolEdgesData?.incoming ?? []).filter(
    (e) => symbolFile(e.src_symbol) !== selectedNode
  )

  return (
    <div className="w-80 border-r border-zinc-700 bg-zinc-900 p-4 overflow-y-auto text-xs space-y-4">
      {/* Node info */}
      <div>
        <div className="text-zinc-300 font-mono text-[10px] break-words">{selectedNode}</div>
      </div>

      {/* Centrality score breakdown */}
      <div>
        <div className="text-zinc-400 font-semibold mb-2">Centrality score</div>
        <div className="text-zinc-200 text-base font-semibold">{centralityScore}</div>
        <div className="text-[10px] text-zinc-500 mt-1">
          = indegree ({indegree}) × 3 + outdegree ({outdegree})
        </div>
        <div className="text-[10px] text-zinc-500">
          Higher indegree (more files depend on this) weighs 3× more than outdegree
          (this file depending on others).
        </div>
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

      {/* Function-level drill-down, backed by symbol_graph_edges */}
      {symbolEdgesData && !symbolEdgesLoading && (
        <div>
          <div className="text-zinc-400 font-semibold mb-2">
            Functions in this file ({symbolEdgesData.defined_symbols.length})
          </div>
          {symbolEdgesData.defined_symbols.length === 0 ? (
            <div className="text-[10px] text-zinc-500 italic">
              No function-level call data yet — only Python/TypeScript files are
              analyzed at this level, and the graph may need rebuilding.
            </div>
          ) : (
            <div className="space-y-1 mb-3">
              {symbolEdgesData.defined_symbols.slice(0, 10).map((s) => (
                <div key={s} className="text-[10px] text-zinc-400 font-mono truncate">
                  {s}
                </div>
              ))}
              {symbolEdgesData.defined_symbols.length > 10 && (
                <div className="text-[10px] text-zinc-500 italic">
                  and {symbolEdgesData.defined_symbols.length - 10} more...
                </div>
              )}
            </div>
          )}

          {outgoingCalls.length > 0 && (
            <div className="mb-3">
              <div className="text-[10px] text-zinc-500 mb-1">Calls out to other files:</div>
              <div className="space-y-1">
                {outgoingCalls.slice(0, 8).map((e, i) => (
                  <div key={i} className="text-[10px] text-zinc-400">
                    <span className="font-mono">{symbolName(e.src_symbol)}</span>
                    {' → '}
                    <span className="font-mono text-zinc-300">
                      {symbolFile(e.dst_symbol).split('/').pop()}::{symbolName(e.dst_symbol)}
                    </span>
                    <span
                      className={
                        'ml-1 ' + (e.confidence_score >= 0.7 ? 'text-emerald-500' : 'text-amber-500')
                      }
                    >
                      ({e.resolution_method}, {e.confidence_score.toFixed(2)})
                    </span>
                  </div>
                ))}
                {outgoingCalls.length > 8 && (
                  <div className="text-[10px] text-zinc-500 italic">
                    and {outgoingCalls.length - 8} more...
                  </div>
                )}
              </div>
            </div>
          )}

          {incomingCalls.length > 0 && (
            <div>
              <div className="text-[10px] text-zinc-500 mb-1">Called from other files:</div>
              <div className="space-y-1">
                {incomingCalls.slice(0, 8).map((e, i) => (
                  <div key={i} className="text-[10px] text-zinc-400">
                    <span className="font-mono text-zinc-300">
                      {symbolFile(e.src_symbol).split('/').pop()}::{symbolName(e.src_symbol)}
                    </span>
                    {' → '}
                    <span className="font-mono">{symbolName(e.dst_symbol)}</span>
                    <span
                      className={
                        'ml-1 ' + (e.confidence_score >= 0.7 ? 'text-emerald-500' : 'text-amber-500')
                      }
                    >
                      ({e.resolution_method}, {e.confidence_score.toFixed(2)})
                    </span>
                  </div>
                ))}
                {incomingCalls.length > 8 && (
                  <div className="text-[10px] text-zinc-500 italic">
                    and {incomingCalls.length - 8} more...
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      )}

      {symbolEdgesLoading && (
        <div className="flex items-center gap-2 text-zinc-500">
          <Loader2 size={12} className="animate-spin" />
          <span>Loading function-level calls...</span>
        </div>
      )}

      {neighborLoading && (
        <div className="flex items-center gap-2 text-zinc-500">
          <Loader2 size={12} className="animate-spin" />
          <span>Loading neighbors...</span>
        </div>
      )}

      {/* Impact analysis results (shown when impact mode is active) */}
      {showImpactPanel && (
        <ImpactPanel impactState={impactState} impactLoading={impactLoading} inline />
      )}
    </div>
  )
}

interface ImpactPanelProps {
  impactState: ImpactState
  impactLoading: boolean
  inline?: boolean
}

function ImpactPanel({ impactState, impactLoading, inline = false }: ImpactPanelProps) {
  const { result, seedFiles } = impactState

  const wrapper = inline
    ? 'border-t border-zinc-700 pt-3 space-y-3'
    : 'w-80 border-r border-zinc-700 bg-zinc-900 p-4 overflow-y-auto text-xs space-y-4'

  return (
    <div className={wrapper}>
      {!inline && <div className="text-zinc-400 font-semibold">Impact Analysis</div>}

      {seedFiles.length > 0 && (
        <div>
          <div className="text-zinc-400 font-semibold mb-1">Seed files</div>
          <div className="space-y-1">
            {seedFiles.map((f) => (
              <div key={f} className="flex items-center gap-1">
                <div className="w-2 h-2 rounded-full bg-red-500 shrink-0" />
                <div className="text-[10px] text-zinc-300 truncate">{f.split('/').pop()}</div>
              </div>
            ))}
          </div>
        </div>
      )}

      {impactLoading && (
        <div className="flex items-center gap-2 text-zinc-500">
          <Loader2 size={12} className="animate-spin" />
          <span>Analyzing impact...</span>
        </div>
      )}

      {result && !impactLoading && (
        <>
          <div>
            <div className="text-zinc-400 font-semibold mb-1">Blast radius</div>
            <div className="text-zinc-300">
              <span className="font-mono">{result.blast_radius.total_affected}</span> files affected
            </div>
            {Object.entries(result.blast_radius.by_hop).sort(([a], [b]) => Number(a) - Number(b)).map(([hop, count]) => (
              <div key={hop} className="flex items-center gap-2 mt-1">
                <div
                  className="w-2 h-2 rounded-full shrink-0"
                  style={{ backgroundColor: result.subgraph.hop_colors[`__hop_${hop}`] ?? '#6b7280' }}
                />
                <span className="text-[10px] text-zinc-400">Hop {hop}: {count as number} files</span>
              </div>
            ))}
          </div>

          {result.blast_radius.high_risk_files.length > 0 && (
            <div>
              <div className="text-zinc-400 font-semibold mb-1">High-risk files</div>
              <div className="space-y-1">
                {result.blast_radius.high_risk_files.slice(0, 8).map((f) => (
                  <div key={f} className="text-[10px] text-red-400 truncate">{f.split('/').pop()}</div>
                ))}
                {result.blast_radius.high_risk_files.length > 8 && (
                  <div className="text-[10px] text-zinc-500 italic">
                    and {result.blast_radius.high_risk_files.length - 8} more...
                  </div>
                )}
              </div>
            </div>
          )}

          {result.blast_radius.affected_communities.length > 0 && (
            <div>
              <div className="text-zinc-400 font-semibold mb-1">Affected communities</div>
              <div className="space-y-1">
                {result.blast_radius.affected_communities.slice(0, 5).map((c) => (
                  <div key={c.community_id} className="text-[10px] text-zinc-400">
                    Community {c.community_id} ({c.member_count} members)
                  </div>
                ))}
              </div>
            </div>
          )}
        </>
      )}
    </div>
  )
}

interface LegendProps {
  communityCount: number
  impactActive?: boolean
}

function Legend({ communityCount, impactActive }: LegendProps) {
  const shown = Math.min(10, communityCount)
  const extra = communityCount - shown
  return (
    <Panel position="top-right">
    <div className="bg-zinc-800/90 border border-zinc-700 rounded p-3 text-xs space-y-1 pointer-events-none">
      {impactActive ? (
        <>
          <div className="font-semibold text-zinc-300 mb-1.5">Impact Mode</div>
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 rounded-full shrink-0" style={{ backgroundColor: '#ef4444' }} />
            <span className="text-zinc-400">Seed (hop 0)</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 rounded-full shrink-0" style={{ backgroundColor: '#f97316' }} />
            <span className="text-zinc-400">Hop 1</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 rounded-full shrink-0" style={{ backgroundColor: '#eab308' }} />
            <span className="text-zinc-400">Hop 2</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 rounded-full shrink-0" style={{ backgroundColor: '#6b7280' }} />
            <span className="text-zinc-400">Hop 3+</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 rounded-full shrink-0" style={{ backgroundColor: '#52525b' }} />
            <span className="text-zinc-400">Not in cone</span>
          </div>
        </>
      ) : (
        <>
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
        </>
      )}
    </div>
    </Panel>
  )
}

export default function GraphScreen(): React.ReactElement {
  const [searchParams] = useSearchParams()
  const navigate = useNavigate()
  const toast = useToastStore()
  const snapshotId = searchParams.get('snapshotId') ?? ''

  const [graphData, setGraphData] = useState<ExportJson | null>(null)
  const [communityData, setCommunityData] = useState<CommunitiesResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [selectedNode, setSelectedNode] = useState<string | null>(null)
  const [neighborData, setNeighborData] = useState<NeighborResult | null>(null)
  const [neighborLoading, setNeighborLoading] = useState(false)
  const [symbolEdgesData, setSymbolEdgesData] = useState<FileSymbolEdgesResponse | null>(null)
  const [symbolEdgesLoading, setSymbolEdgesLoading] = useState(false)
  const [exporting, setExporting] = useState(false)
  const [layoutMode, setLayoutMode] = useState<'cluster' | 'hierarchical'>('cluster')
  const [impactState, setImpactState] = useState<ImpactState>({ active: false, seedFiles: [], result: null })
  const [impactLoading, setImpactLoading] = useState(false)

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

      if (impactState.active) {
        // Impact mode: add this node as a seed file and re-run blast radius
        const newSeedFiles = impactState.seedFiles.includes(path)
          ? impactState.seedFiles
          : [...impactState.seedFiles, path]
        setImpactState((prev) => ({ ...prev, seedFiles: newSeedFiles }))
        setImpactLoading(true)
        try {
          const result = await window.api.impact.blastRadius({
            snapshot_id: snapshotId,
            changed_files: newSeedFiles,
            max_hops: 3,
          }) as BlastRadiusResponse
          setImpactState((prev) => ({ ...prev, result }))
        } catch {
          toast.error('Impact analysis failed')
        } finally {
          setImpactLoading(false)
        }
        return
      }

      setNeighborLoading(true)
      try {
        const res = await window.api.graph.neighbors(snapshotId, path, 2, 100)
        setNeighborData(res)
      } catch {
        setNeighborData(null)
      } finally {
        setNeighborLoading(false)
      }

      setSymbolEdgesLoading(true)
      try {
        const res = await window.api.graph.symbolEdges(snapshotId, path)
        setSymbolEdgesData(res)
      } catch {
        // Not every file has function-level data — treat as "nothing to show", not an error.
        setSymbolEdgesData(null)
      } finally {
        setSymbolEdgesLoading(false)
      }
    },
    [snapshotId, impactState.active, impactState.seedFiles, toast]
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

    return buildFlowGraph(cappedData, nodeIndex, selectedNode, neighborSet, cycleNodes, impactState.active ? impactState.result : null)
  }, [graphData, nodeIndex, selectedNode, neighborData, impactState.active, impactState.result])

  const layoutNodes = useMemo(() => {
    if (rawNodes.length === 0) return []
    return layoutMode === 'cluster' ? applyForceClusterLayout(rawNodes, edges) : applyDagreLayout(rawNodes, edges)
  }, [rawNodes, edges, layoutMode])

  // User-dragged positions override the auto-layout, keyed by node id; cleared when
  // layoutMode changes so switching layouts doesn't keep stale drags around.
  const [manualPositions, setManualPositions] = useState<Record<string, { x: number; y: number }>>({})
  useEffect(() => {
    setManualPositions({})
  }, [layoutMode])

  const nodes = useMemo(() => {
    if (Object.keys(manualPositions).length === 0) return layoutNodes
    return layoutNodes.map((node) =>
      manualPositions[node.id] ? { ...node, position: manualPositions[node.id] } : node
    )
  }, [layoutNodes, manualPositions])

  const onNodesChange = useCallback((changes: NodeChange[]) => {
    setManualPositions((prev) => {
      let next = prev
      for (const change of changes) {
        if (change.type === 'position' && change.position) {
          if (next === prev) next = { ...prev }
          next[change.id] = change.position
        }
      }
      return next
    })
  }, [])

  const fitViewOptions = { padding: 0.1 }

  if (!snapshotId) {
    return (
      <div className="h-full flex items-center justify-center bg-zinc-950">
        <div className="text-center space-y-4">
          <AlertCircle size={32} className="text-zinc-500 mx-auto" />
          <div className="text-zinc-400">No snapshot ID provided</div>
          <Button variant="primary" onClick={() => navigate('/index-overview')}>
            Go back
          </Button>
        </div>
      </div>
    )
  }

  if (loading) {
    return <PageLoading label="Loading graph..." />
  }

  if (error || !graphData || !communityData) {
    return (
      <div className="h-full flex items-center justify-center bg-zinc-950">
        <div className="text-center space-y-4">
          <AlertCircle size={32} className="text-red-500 mx-auto" />
          <div className="text-red-400">{error || 'Failed to load graph'}</div>
          <Button variant="primary" onClick={() => loadGraph()}>
            Retry
          </Button>
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
        impactState={impactState}
        impactLoading={impactLoading}
        symbolEdgesData={symbolEdgesData}
        symbolEdgesLoading={symbolEdgesLoading}
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
            <Button
              variant="secondary"
              size="sm"
              onClick={() => setLayoutMode((m) => (m === 'cluster' ? 'hierarchical' : 'cluster'))}
            >
              {layoutMode === 'cluster' ? 'Layout: Clustered' : 'Layout: Hierarchical'}
            </Button>
            <Button
              variant={impactState.active ? 'primary' : 'secondary'}
              size="sm"
              onClick={() => {
                if (impactState.active) {
                  setImpactState({ active: false, seedFiles: [], result: null })
                } else {
                  setImpactState((prev) => ({ ...prev, active: true }))
                }
              }}
            >
              {impactState.active ? <X size={11} /> : <Zap size={11} />}
              {impactState.active ? 'Exit Impact Mode' : 'Impact Mode'}
            </Button>
            <Button
              variant="secondary"
              size="sm"
              onClick={async () => {
                setExporting(true)
                try {
                  const result = await window.api.graph.exportJson(snapshotId)
                  if (result.saved) {
                    toast.success('Graph JSON exported')
                  }
                } catch {
                  toast.error('Export failed')
                } finally {
                  setExporting(false)
                }
              }}
              loading={exporting}
            >
              <Save size={11} />
              Export JSON
            </Button>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => navigate(-1)}
            >
              Back
            </Button>
          </div>
        </div>

        {/* React Flow canvas */}
        <div className="flex-1 relative" style={{ height: 'calc(100% - 3rem)' }}>
          <ReactFlow
            nodes={nodes}
            edges={edges}
            onNodeClick={onNodeClick}
            onNodesChange={onNodesChange}
            nodesDraggable
            fitView
            fitViewOptions={fitViewOptions}
            minZoom={0.05}
            maxZoom={2}
          >
            <Background />
            <Controls />
            <Legend communityCount={communityCount} impactActive={impactState.active} />
          </ReactFlow>
        </div>
      </div>
    </div>
  )
}

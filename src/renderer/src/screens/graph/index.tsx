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

/**
 * Force-directed layout with community clustering (Neo4j-Browser style): nodes
 * repel each other and are pulled along edges, PLUS an extra custom force that
 * pulls every node toward its own community's centroid each tick. That cluster
 * force is what actually produces visually distinct blobs — plain repulsion +
 * links alone (no cluster force) just spreads everything out evenly with no
 * separation between communities, which isn't what we want here.
 *
 * Replaces an earlier single-ring layout that was far too literal a reading of
 * "arrange in a circle" — a real force simulation is what actually looks like
 * the reference (organic clusters, not one ordered ring).
 */
function applyForceClusterLayout(nodes: Node[], edges: Edge[]): Node[] {
  const n = nodes.length
  if (n === 0) return nodes

  // Seed positions on a circle so the simulation starts from a reasonable
  // spread instead of every node collapsing onto the origin.
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

  // Community sizes, computed once -- singleton/tiny communities (1-2 nodes)
  // get a much weaker cluster pull below, since "pull toward your own
  // centroid" is meaningless for a lone node and was previously the reason
  // small communities drifted away with nothing bounding them.
  const communitySize = new Map<number, number>()
  for (const fn of forceNodes) {
    communitySize.set(fn.communityId, (communitySize.get(fn.communityId) ?? 0) + 1)
  }

  // Minimum desired distance between two DIFFERENT communities' centroids.
  // This -- not raw node-level repulsion -- is what actually keeps clusters
  // visually distinct. A previous version relied on generic charge + a global
  // gravity force fighting each other, which either flung small clusters away
  // (gravity too weak) or crushed everything into one ball (gravity too
  // strong relative to charge) -- there was never a force that specifically
  // said "different clusters should stay apart," only "things in general
  // repel" vs "things in general are pulled to the center."
  const MIN_CLUSTER_SEPARATION = 260

  function clusterForce(alpha: number): void {
    const centroids = new Map<number, { x: number; y: number; count: number }>()
    for (const fn of forceNodes) {
      const c = centroids.get(fn.communityId) ?? { x: 0, y: 0, count: 0 }
      c.x += fn.x ?? 0
      c.y += fn.y ?? 0
      c.count += 1
      centroids.set(fn.communityId, c)
    }
    const centroidList = Array.from(centroids.entries()).map(([id, c]) => ({
      id,
      x: c.x / c.count,
      y: c.y / c.count,
    }))

    // 1. Cohesion: pull each node toward its own community's centroid.
    for (const fn of forceNodes) {
      const c = centroids.get(fn.communityId)
      if (!c || c.count === 0) continue
      const size = communitySize.get(fn.communityId) ?? 1
      // Bigger communities pull together more strongly (denser, more cohesive
      // blob); singletons/pairs barely self-pull (there's nothing to cohere to).
      const strength = size <= 2 ? 0.02 : 0.12
      fn.vx = (fn.vx ?? 0) + (c.x / c.count - (fn.x ?? 0)) * alpha * strength
      fn.vy = (fn.vy ?? 0) + (c.y / c.count - (fn.y ?? 0)) * alpha * strength
    }

    // 2. Separation: push two communities' centroids apart whenever they're
    // closer than MIN_CLUSTER_SEPARATION, applied to every node in both
    // communities (moves each cluster as a whole, not node-by-node). This is
    // the piece that actually produces visible gaps between distinct blobs.
    for (let i = 0; i < centroidList.length; i++) {
      for (let j = i + 1; j < centroidList.length; j++) {
        const a = centroidList[i]
        const b = centroidList[j]
        const dx = b.x - a.x
        const dy = b.y - a.y
        const dist = Math.sqrt(dx * dx + dy * dy) || 1
        if (dist >= MIN_CLUSTER_SEPARATION) continue
        const push = ((MIN_CLUSTER_SEPARATION - dist) / MIN_CLUSTER_SEPARATION) * alpha * 0.5
        const ux = dx / dist
        const uy = dy / dist
        for (const fn of forceNodes) {
          if (fn.communityId === a.id) {
            fn.vx = (fn.vx ?? 0) - ux * push
            fn.vy = (fn.vy ?? 0) - uy * push
          } else if (fn.communityId === b.id) {
            fn.vx = (fn.vx ?? 0) + ux * push
            fn.vy = (fn.vy ?? 0) + uy * push
          }
        }
      }
    }
  }

  const simulation = d3force
    .forceSimulation(forceNodes)
    .force('charge', d3force.forceManyBody().strength(-60))
    .force(
      'link',
      d3force
        .forceLink<ForceNode, { source: string; target: string }>(forceLinks)
        .id((d) => d.id)
        .distance(70)
        .strength(0.35)
    )
    // Minimum spacing between node centers so labels never overlap within a
    // cluster. Radius approximates the node's 160x36 bounding box (half-
    // diagonal ~82) plus a small gap.
    .force('collide', d3force.forceCollide(95))
    // Very weak per-node gravity -- just enough of a safety net to stop a
    // fully-disconnected node from sailing off to infinity, not strong enough
    // to compete with cluster separation above (that was the overcorrection:
    // 0.025 was strong enough to crush every cluster into the same point).
    .force('gravityX', d3force.forceX(0).strength(0.004))
    .force('gravityY', d3force.forceY(0).strength(0.004))
    .stop()

  // Run synchronously to a settled state (no live animation needed — React Flow
  // just needs final positions), interleaving our custom cluster force each tick.
  for (let tick = 0; tick < 300; tick++) {
    simulation.tick()
    clusterForce(simulation.alpha())
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

  // Centrality score, computed client-side from already-loaded edges using the
  // same formula as the backend's _compute_scores_python: indegree*3 + outdegree.
  const indegree = incomingEdges.length
  const outdegree = outgoingEdges.length
  const centralityScore = indegree * 3 + outdegree

  // Cross-file function calls only (drop same-file self-references, which are
  // noise for "who else does this file talk to").
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

      {/* Function-level drill-down (CS-250, backed by CS-240/CS-249's symbol_graph_edges) */}
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
        // Not every file has function-level data (e.g. non-Python/TS files,
        // or the graph predates CS-240) — treat as "nothing to show", not an error.
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

  const nodes = useMemo(() => {
    if (rawNodes.length === 0) return []
    return layoutMode === 'cluster' ? applyForceClusterLayout(rawNodes, edges) : applyDagreLayout(rawNodes, edges)
  }, [rawNodes, edges, layoutMode])

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

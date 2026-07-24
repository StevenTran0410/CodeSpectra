import { type Node, type Edge } from '@xyflow/react'
import dagre from '@dagrejs/dagre'

export interface GraphLayoutNode {
  id: string
  label: string
  color?: string
}

export interface GraphLayoutEdge {
  src: string
  dst: string
  kind?: 'wiring' | 'symbol' | 'agent-flow'
  /** Force this edge to span >=N ranks, pushing its target into its own column instead of
   * sharing a rank with other same-distance siblings. Default 1 (dagre's normal behavior). */
  minlen?: number
}

const EDGE_STYLE: Record<string, { stroke: string; strokeWidth: number; strokeDasharray?: string }> = {
  wiring: { stroke: '#818cf8', strokeWidth: 2.0 },
  symbol: { stroke: '#475569', strokeWidth: 1.25 },
  'agent-flow': { stroke: '#a78bfa', strokeWidth: 2.25, strokeDasharray: '4 3' },
}

export function getDagreGraphLayout(
  nodes: GraphLayoutNode[],
  edges: GraphLayoutEdge[],
  spacing?: { nodesep?: number; ranksep?: number }
): { layoutNodes: Node[]; layoutEdges: Edge[] } {
  const g = new dagre.graphlib.Graph()
  g.setGraph({ rankdir: 'LR', nodesep: spacing?.nodesep ?? 30, ranksep: spacing?.ranksep ?? 50 })
  g.setDefaultEdgeLabel(() => ({}))

  // Synthesize a node for any edge endpoint missing from `nodes` (e.g. LangGraph's
  // __start__/__end__ sentinels). ReactFlow throws (#008) on an edge whose source/target
  // node doesn't exist, which would blank the whole graph panel — so never emit dangling edges.
  const knownIds = new Set(nodes.map((n) => n.id))
  const sentinelLabel = (id: string): string =>
    id === '__start__' ? 'START' : id === '__end__' ? 'END' : id
  const synthesized: GraphLayoutNode[] = []
  for (const e of edges) {
    for (const endpoint of [e.src, e.dst]) {
      if (!knownIds.has(endpoint)) {
        knownIds.add(endpoint)
        synthesized.push({ id: endpoint, label: sentinelLabel(endpoint) })
      }
    }
  }
  const allNodes = synthesized.length ? [...nodes, ...synthesized] : nodes

  const layoutNodes: Node[] = allNodes.map((n) => {
    const isMultiline = n.label.includes('\n')
    const width = 130
    const height = isMultiline ? 42 : 32
    g.setNode(n.id, { width, height })
    return {
      id: n.id,
      data: { label: n.label },
      position: { x: 0, y: 0 },
      style: {
        background: '#0f172a',
        color: '#f8fafc',
        fontSize: 9,
        fontWeight: '500',
        padding: isMultiline ? '4px 6px' : '6px 8px',
        borderRadius: 6,
        border: '1px solid #334155',
        borderLeft: n.color ? `3px solid ${n.color}` : '1px solid #334155',
        width,
        height,
        boxSizing: 'border-box',
        overflow: 'hidden',
        textOverflow: 'ellipsis',
        whiteSpace: isMultiline ? 'pre-wrap' : 'nowrap',
        textAlign: 'center',
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'center',
        alignItems: 'center',
        lineHeight: 1.2,
      },
    } as Node
  })

  const layoutEdges: Edge[] = edges.map((e, idx) => {
    g.setEdge(e.src, e.dst, { minlen: e.minlen ?? 1 })
    return {
      id: `edge-${idx}`,
      source: e.src,
      target: e.dst,
      style: EDGE_STYLE[e.kind ?? 'symbol'],
    } as Edge
  })

  try {
    dagre.layout(g)
    layoutNodes.forEach((node) => {
      const pos = g.node(node.id)
      if (pos) {
        const w = (node.style?.width as number) || 130
        const h = (node.style?.height as number) || 32
        node.position = { x: pos.x - w / 2, y: pos.y - h / 2 }
      }
    })
  } catch (err) {
    console.error('Dagre layout failed', err)
  }

  return { layoutNodes, layoutEdges }
}

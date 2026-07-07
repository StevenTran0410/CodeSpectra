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
}

export function getDagreGraphLayout(
  nodes: GraphLayoutNode[],
  edges: GraphLayoutEdge[]
): { layoutNodes: Node[]; layoutEdges: Edge[] } {
  const g = new dagre.graphlib.Graph()
  g.setGraph({ rankdir: 'LR', nodesep: 30, ranksep: 50 })
  g.setDefaultEdgeLabel(() => ({}))

  const layoutNodes: Node[] = nodes.map((n) => {
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
    g.setEdge(e.src, e.dst)
    return {
      id: `edge-${idx}`,
      source: e.src,
      target: e.dst,
      style: { stroke: '#475569', strokeWidth: 1.25 },
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

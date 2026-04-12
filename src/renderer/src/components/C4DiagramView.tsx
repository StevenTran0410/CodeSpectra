/**
 * C4DiagramView — renders LLM-generated C4 architecture diagrams using React Flow + ELK.
 *
 * ELK handles layout with ORTHOGONAL edge routing, which means edges never pass through
 * nodes. The diagram data comes from the backend as a typed JSON structure (not Mermaid).
 *
 * Supported node types: person, container, containerDb, component, systemExt, boundary
 * Supported edge types: rel (one-way), birel (bidirectional)
 */
import React, { useCallback, useEffect, useMemo, useState } from 'react'
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  useNodesState,
  useEdgesState,
  MarkerType,
  type Node,
  type Edge,
  type NodeTypes,
} from '@xyflow/react'
import '@xyflow/react/dist/style.css'
import ELK, { type ElkNode } from 'elkjs/lib/elk.bundled.js'
import type { C4DiagramData, C4DiagramNode, C4DiagramEdge } from '../types/analysis'

// Re-export so callers that imported these from here still work
export type { C4DiagramData, C4DiagramNode as C4Node, C4DiagramEdge as C4Edge }

// Local alias for brevity inside this file
type C4Node = C4DiagramNode

// ─── Node colour palette (C4 standard colours) ───────────────────────────────

const NODE_STYLE: Record<string, React.CSSProperties> = {
  person:      { background: '#1168bd', color: '#fff', borderRadius: 8 },
  container:   { background: '#1168bd', color: '#fff', borderRadius: 4 },
  containerDb: { background: '#1168bd', color: '#fff', borderRadius: 4 },
  component:   { background: '#85bbf0', color: '#000', borderRadius: 4 },
  systemExt:   { background: '#999', color: '#fff', borderRadius: 4 },
  boundary:    { background: 'transparent', color: '#444', borderRadius: 8,
                 border: '2px dashed #aaa' },
}

const NODE_W = 160
const NODE_H = 80
const PERSON_W = 120
const PERSON_H = 90

// ─── Custom node renderers ────────────────────────────────────────────────────

function C4NodeBox({ data }: { data: C4Node }) {
  const style = NODE_STYLE[data.type] ?? NODE_STYLE.container
  const isPerson = data.type === 'person'
  const isDb     = data.type === 'containerDb'
  const stereotype =
    data.type === 'person'      ? 'person'       :
    data.type === 'container'   ? 'container'    :
    data.type === 'containerDb' ? 'container db' :
    data.type === 'component'   ? 'component'    :
    data.type === 'systemExt'   ? 'external system' : ''

  return (
    <div
      style={{
        ...style,
        width: isPerson ? PERSON_W : NODE_W,
        minHeight: isPerson ? PERSON_H : NODE_H,
        padding: '8px 10px',
        fontSize: 11,
        boxShadow: '0 2px 6px rgba(0,0,0,0.25)',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        textAlign: 'center',
        gap: 2,
        position: 'relative',
      }}
    >
      {isPerson && (
        <div style={{ fontSize: 20, marginBottom: 2 }}>👤</div>
      )}
      {isDb && (
        <div style={{ fontSize: 14, marginBottom: 2 }}>🗄</div>
      )}
      {stereotype && (
        <div style={{ fontSize: 9, opacity: 0.75, fontStyle: 'italic' }}>
          &lt;&lt;{stereotype}&gt;&gt;
        </div>
      )}
      <div style={{ fontWeight: 700, fontSize: 12, lineHeight: 1.2 }}>
        {data.label}
      </div>
      {data.technology && (
        <div style={{ fontSize: 9, opacity: 0.8, fontStyle: 'italic' }}>
          [{data.technology}]
        </div>
      )}
      {data.description && (
        <div style={{ fontSize: 9, opacity: 0.75, marginTop: 2, lineHeight: 1.3 }}>
          {data.description}
        </div>
      )}
    </div>
  )
}

const nodeTypes: NodeTypes = {
  c4node: C4NodeBox as unknown as NodeTypes[string],
}

// ─── ELK layout ──────────────────────────────────────────────────────────────

const elk = new ELK()

const ELK_OPTIONS = {
  'elk.algorithm': 'layered',
  'elk.direction': 'DOWN',
  'elk.edgeRouting': 'ORTHOGONAL',
  'elk.layered.nodePlacement.strategy': 'BRANDES_KOEPF',
  'elk.layered.crossingMinimization.strategy': 'LAYER_SWEEP',
  'elk.spacing.nodeNode': '60',
  'elk.spacing.edgeNode': '30',
  'elk.spacing.edgeEdge': '20',
  'elk.layered.spacing.nodeNodeBetweenLayers': '80',
  'elk.layered.considerModelOrder.strategy': 'NODES_AND_EDGES',
}

async function runElkLayout(
  nodes: Node[],
  edges: Edge[],
): Promise<{ nodes: Node[]; edges: Edge[] }> {
  const elkGraph: ElkNode = {
    id: 'root',
    layoutOptions: ELK_OPTIONS,
    children: nodes.map((n) => ({
      id: n.id,
      width: (n.data as unknown as C4Node).type === 'person' ? PERSON_W : NODE_W,
      height: (n.data as unknown as C4Node).type === 'person' ? PERSON_H : NODE_H,
    })),
    edges: edges.map((e) => ({
      id: e.id,
      sources: [e.source],
      targets: [e.target],
    })),
  }

  const laid = await elk.layout(elkGraph)

  const positionedNodes = nodes.map((n) => {
    const elkNode = laid.children?.find((c) => c.id === n.id)
    return {
      ...n,
      position: { x: elkNode?.x ?? 0, y: elkNode?.y ?? 0 },
    }
  })

  return { nodes: positionedNodes, edges }
}

// ─── Data → React Flow converter ─────────────────────────────────────────────

function buildFlowElements(data: C4DiagramData): { nodes: Node[]; edges: Edge[] } {
  const nodes: Node[] = []
  const edges: Edge[] = []

  function addNode(n: C4Node) {
    nodes.push({
      id: n.id,
      type: 'c4node',
      position: { x: 0, y: 0 }, // ELK will set real positions
      data: n as unknown as Record<string, unknown>,
      style: { padding: 0, border: 'none', background: 'transparent' },
    })
    n.children?.forEach(addNode)
  }

  data.nodes.forEach(addNode)

  data.edges.forEach((e) => {
    // One-way arrow
    edges.push({
      id: e.id,
      source: e.source,
      target: e.target,
      label: e.label,
      type: 'smoothstep',
      animated: false,
      markerEnd: { type: MarkerType.ArrowClosed, color: '#555' },
      style: { stroke: '#555', strokeWidth: 1.5 },
      labelStyle: { fontSize: 10, fill: '#333' },
      labelBgStyle: { fill: '#fff', fillOpacity: 0.85 },
    })
    // BiRel — add reverse arrow too
    if (e.bidirectional) {
      edges.push({
        id: `${e.id}_rev`,
        source: e.target,
        target: e.source,
        type: 'smoothstep',
        animated: false,
        markerEnd: { type: MarkerType.ArrowClosed, color: '#555' },
        style: { stroke: '#555', strokeWidth: 1.5 },
      })
    }
  })

  return { nodes, edges }
}

// ─── Main component ───────────────────────────────────────────────────────────

interface Props {
  data: C4DiagramData
  className?: string
  style?: React.CSSProperties
}

export default function C4DiagramView({ data, className, style }: Props): React.ReactElement {
  const [nodes, setNodes, onNodesChange] = useNodesState<Node>([])
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const { nodes: rawNodes, edges: rawEdges } = useMemo(
    () => buildFlowElements(data),
    [data],
  )

  useEffect(() => {
    if (rawNodes.length === 0) {
      setLoading(false)
      return
    }
    setLoading(true)
    setError(null)
    runElkLayout(rawNodes, rawEdges)
      .then(({ nodes: laid, edges: laidEdges }) => {
        setNodes(laid)
        setEdges(laidEdges)
        setLoading(false)
      })
      .catch((err) => {
        console.error('[C4DiagramView] ELK layout failed:', err)
        setError('Layout failed')
        setLoading(false)
      })
  }, [rawNodes, rawEdges])

  if (loading) {
    return (
      <div className="flex items-center justify-center py-10 text-xs text-zinc-500">
        Laying out diagram…
      </div>
    )
  }

  if (error || nodes.length === 0) {
    return (
      <div className="rounded border border-zinc-800 bg-zinc-900/40 px-3 py-2 text-xs text-zinc-500">
        Diagram unavailable
      </div>
    )
  }

  return (
    <div className={className ?? 'w-full'} style={{ height: 420, background: '#f8f9fa', borderRadius: 8, ...style }}>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        nodeTypes={nodeTypes}
        fitView
        fitViewOptions={{ padding: 0.15 }}
        minZoom={0.3}
        maxZoom={2}
        proOptions={{ hideAttribution: true }}
      >
        <Background color="#ddd" gap={20} />
        <Controls showInteractive={false} />
        <MiniMap nodeColor="#1168bd" maskColor="rgba(0,0,0,0.05)" />
      </ReactFlow>
    </div>
  )
}

/**
 * SequenceFlowView — renders a per-feature reading-order sequence diagram
 * using React Flow + ELK (same stack as C4DiagramView, zero new dependencies).
 *
 * Each node is a "step" box representing one file in the feature's reading_order.
 * ELK lays them out top-to-bottom with orthogonal edges.
 */
import React, { useEffect, useMemo, useRef, useState } from 'react'
import {
  ReactFlow,
  Background,
  Handle,
  Position,
  useNodesState,
  useEdgesState,
  MarkerType,
  type Node,
  type Edge,
  type NodeTypes,
} from '@xyflow/react'
import ELK, { type ElkNode } from 'elkjs/lib/elk.bundled.js'
import type { C4DiagramData, C4DiagramNode } from '../types/analysis'

// ─── Node dimensions ──────────────────────────────────────────────────────────

const STEP_W = 176
const STEP_H = 54

// ─── Step node renderer ───────────────────────────────────────────────────────

function StepBox({ data }: { data: C4DiagramNode }) {
  const isEntry = data.description === 'entrypoint'
  const hs: React.CSSProperties = { background: 'transparent', border: 'none' }

  return (
    <div
      style={{
        width: STEP_W,
        height: STEP_H,
        background: isEntry ? '#1c3252' : '#27272a',
        border: `1px solid ${isEntry ? '#3b82f6' : '#3f3f46'}`,
        borderRadius: 6,
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        padding: '4px 10px',
        gap: 2,
        boxSizing: 'border-box',
      }}
    >
      <Handle type="target" position={Position.Top} style={hs} />
      {isEntry && (
        <div style={{ fontSize: 8, color: '#60a5fa', textTransform: 'uppercase', letterSpacing: '0.08em' }}>
          entrypoint
        </div>
      )}
      <div
        style={{
          fontWeight: 600,
          fontSize: 11,
          color: '#f4f4f5',
          textAlign: 'center',
          lineHeight: 1.25,
          wordBreak: 'break-all',
        }}
      >
        {data.label}
      </div>
      {data.technology && (
        <div style={{ fontSize: 8, color: '#71717a', fontStyle: 'italic' }}>
          {data.technology}
        </div>
      )}
      <Handle type="source" position={Position.Bottom} style={hs} />
    </div>
  )
}

const nodeTypes: NodeTypes = {
  step: StepBox as unknown as NodeTypes[string],
}

// ─── ELK layout ──────────────────────────────────────────────────────────────

const elk = new ELK()

const SEQ_ELK_OPTIONS = {
  'elk.algorithm': 'layered',
  'elk.direction': 'DOWN',
  'elk.edgeRouting': 'ORTHOGONAL',
  'elk.spacing.nodeNode': '16',
  'elk.layered.spacing.nodeNodeBetweenLayers': '36',
  'elk.layered.nodePlacement.strategy': 'SIMPLE',
}

async function runLayout(nodes: Node[], edges: Edge[]): Promise<{ nodes: Node[]; edges: Edge[] }> {
  const graph: ElkNode = {
    id: 'root',
    layoutOptions: SEQ_ELK_OPTIONS,
    children: nodes.map((n) => ({ id: n.id, width: STEP_W, height: STEP_H })),
    edges: edges.map((e) => ({ id: e.id, sources: [e.source], targets: [e.target] })),
  }
  const laid = await elk.layout(graph)
  return {
    nodes: nodes.map((n) => ({
      ...n,
      position: {
        x: laid.children?.find((c) => c.id === n.id)?.x ?? 0,
        y: laid.children?.find((c) => c.id === n.id)?.y ?? 0,
      },
    })),
    edges,
  }
}

// ─── Data converter ───────────────────────────────────────────────────────────

function buildElements(data: C4DiagramData): { nodes: Node[]; edges: Edge[] } {
  const nodes: Node[] = data.nodes.map((n) => ({
    id: n.id,
    type: 'step',
    position: { x: 0, y: 0 },
    data: n as unknown as Record<string, unknown>,
    style: { padding: 0, border: 'none', background: 'transparent' },
  }))
  const edges: Edge[] = data.edges.map((e) => ({
    id: e.id,
    source: e.source,
    target: e.target,
    type: 'smoothstep',
    markerEnd: { type: MarkerType.ArrowClosed, color: '#52525b' },
    style: { stroke: '#52525b', strokeWidth: 1.5 },
  }))
  return { nodes, edges }
}

// ─── Main component ───────────────────────────────────────────────────────────

interface Props {
  data: C4DiagramData
}

export default function SequenceFlowView({ data }: Props): React.ReactElement | null {
  const containerRef = useRef<HTMLDivElement>(null)
  const [visible, setVisible] = useState(false)
  const [nodes, setNodes, onNodesChange] = useNodesState<Node>([])
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([])
  const [loading, setLoading] = useState(true)

  // Only run ELK when the component scrolls into view — avoids running 6 layout
  // calls simultaneously when all feature cards are collapsed / off-screen.
  useEffect(() => {
    const el = containerRef.current
    if (!el) return
    const obs = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setVisible(true)
          obs.disconnect()
        }
      },
      { threshold: 0.1 },
    )
    obs.observe(el)
    return () => obs.disconnect()
  }, [])

  const { nodes: rawNodes, edges: rawEdges } = useMemo(() => buildElements(data), [data])

  useEffect(() => {
    if (!visible || rawNodes.length === 0) {
      if (!visible) return   // wait for intersection
      setLoading(false)
      return
    }
    setLoading(true)
    runLayout(rawNodes, rawEdges)
      .then(({ nodes: laid, edges: laidEdges }) => {
        setNodes(laid)
        setEdges(laidEdges)
        setLoading(false)
      })
      .catch(() => setLoading(false))
  }, [rawNodes, rawEdges, visible])

  // Height scales with step count, capped at 420px
  const height = Math.min(80 + data.nodes.length * (STEP_H + 36) + 40, 420)

  return (
    <div ref={containerRef} style={{ minHeight: 40 }}>
    {!visible || loading ? (
      <div
        style={{ height: Math.min(height, 120), background: '#18181b', borderRadius: 6, border: '1px solid #27272a' }}
        className="flex items-center justify-center"
      >
        <span className="text-[10px] text-zinc-700">{visible ? 'Laying out…' : ''}</span>
      </div>
    ) : nodes.length === 0 ? null : (
    <div
      style={{
        height,
        background: '#18181b',
        borderRadius: 6,
        border: '1px solid #27272a',
        overflow: 'hidden',
      }}
    >
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        nodeTypes={nodeTypes}
        fitView
        fitViewOptions={{ padding: 0.2 }}
        minZoom={0.3}
        maxZoom={2}
        proOptions={{ hideAttribution: true }}
        nodesDraggable={false}
        nodesConnectable={false}
        elementsSelectable={false}
        panOnDrag
        zoomOnScroll
      >
        <Background color="#2d2d2d" gap={20} />
      </ReactFlow>
    </div>
    )}
    </div>
  )
}

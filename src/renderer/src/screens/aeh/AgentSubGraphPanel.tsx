import React, { useState, useMemo } from 'react'
import { ReactFlow, Background } from '@xyflow/react'
import '@xyflow/react/dist/style.css'
import { getDagreGraphLayout, type GraphLayoutNode, type GraphLayoutEdge } from './graphLayout'
// AEHDiscoveryCandidate/AEHExpansionSession/AEHSystemMap/AEHAgentFlow/AEHAgentFlowMap are global ambient types.

export const ROLE_COLORS: Record<AEHRole, string> = {
  orchestrator: '#6366f1',
  retrieval_agent: '#0ea5e9',
  tool: '#f59e0b',
  writer: '#10b981',
  validator: '#ec4899',
  worker: '#14b8a6',
  'input_guard.rule': '#ef4444',
  'input_guard.llm': '#ef4444',
  unknown: '#475569',
}

/** `accepted` items are {file, role_hint, key_symbols, follow} — extract plain paths. */
export function acceptedFilePaths(accepted: AEHExpansionSession['accepted']): string[] {
  return accepted.map((a) => a.file)
}

export function wiringBlockFileEdges(wiringBlock: AEHDiscoveryCandidate['wiring_block']): { src: string; dst: string }[] {
  if (!wiringBlock) return []
  const aliasToFile = new Map<string, string>()
  for (const n of wiringBlock.nodes) aliasToFile.set(n.alias, n.source_hint_file)
  const edges: { src: string; dst: string }[] = []
  for (const e of wiringBlock.edges) {
    const src = aliasToFile.get(e.src)
    const dst = aliasToFile.get(e.dst)
    if (src && dst && src !== dst) edges.push({ src, dst })
  }
  return edges
}

/** Sub-graph for ONE agent: its own components, its supporting files/tools (pulled from LLM 1's file graph), plus neighbor agents collapsed into dashed boundary nodes you can click to walk to. */
export function AgentSubGraphPanel({
  agent,
  agentFlowMap,
  systemMap,
  expansionSession,
  candidate,
  selectedNode,
  onSelectNode,
  onNavigateToAgent,
  onHoverNode,
}: {
  agent: AEHAgentFlow | null
  agentFlowMap: AEHAgentFlowMap
  systemMap: AEHSystemMap
  expansionSession: AEHExpansionSession
  candidate: AEHDiscoveryCandidate | null
  selectedNode: string | null
  onSelectNode: (id: string | null) => void
  onNavigateToAgent: (agentId: string) => void
  onHoverNode?: (id: string | null) => void
}): React.ReactElement {
  const [hovered, setHovered] = useState<{ id: string; x: number; y: number } | null>(null)

  const agentById = useMemo(() => new Map(agentFlowMap.agents.map((a) => [a.id, a])), [agentFlowMap])
  const componentById = useMemo(() => new Map(systemMap.components.map((c) => [c.id, c])), [systemMap])

  const ownedIds = useMemo(() => new Set(agent?.component_ids ?? []), [agent])

  const ownedFileToComponentId = useMemo(() => {
    const m = new Map<string, string>()
    for (const cid of ownedIds) {
      const comp = componentById.get(cid)
      if (comp?.file) m.set(comp.file, comp.id)
    }
    return m
  }, [ownedIds, componentById])

  // Which agent (if any) owns each file, system-wide — distinguishes an unowned helper file
  // from a neighbor agent's file (already shown as a boundary node).
  const fileToOwningAgent = useMemo(() => {
    const m = new Map<string, string>()
    for (const a of agentFlowMap.agents) {
      for (const cid of a.component_ids) {
        const file = componentById.get(cid)?.file
        if (file) m.set(file, a.id)
      }
    }
    return m
  }, [agentFlowMap, componentById])

  const neighborAgentIds = useMemo(() => {
    if (!agent) return []
    return Array.from(new Set([...agent.upstream_agents, ...agent.downstream_agents]))
  }, [agent])

  // LLM 1's own BFS-discovered file graph — reused as-is, not re-detected.
  const fileEdges = useMemo(() => {
    const acceptedPaths = new Set(acceptedFilePaths(expansionSession.accepted))
    const wiringEdges = wiringBlockFileEdges(candidate?.wiring_block ?? null)
      .filter((e) => acceptedPaths.has(e.src) && acceptedPaths.has(e.dst))
    const wiringEdgeKeys = new Set(wiringEdges.map((e) => `${e.src}|${e.dst}`))
    const symbolEdges = (expansionSession.accepted_edges || [])
      .filter((e) => !wiringEdgeKeys.has(`${e.src}|${e.dst}`) && !wiringEdgeKeys.has(`${e.dst}|${e.src}`))
    return [...wiringEdges, ...symbolEdges]
  }, [expansionSession, candidate])

  const { nodes, edges, boundaryIds } = useMemo(() => {
    const boundarySet = new Set<string>()
    const nodeList: GraphLayoutNode[] = []
    const nodeIds = new Set<string>()
    const addNode = (n: GraphLayoutNode): void => {
      if (nodeIds.has(n.id)) return
      nodeIds.add(n.id)
      nodeList.push(n)
    }

    for (const cid of ownedIds) {
      const comp = componentById.get(cid)
      if (!comp) continue
      addNode({ id: comp.id, label: comp.id, color: ROLE_COLORS[comp.role as AEHRole] ?? ROLE_COLORS.unknown })
    }
    for (const nid of neighborAgentIds) {
      const neighbor = agentById.get(nid)
      if (!neighbor) continue
      const bId = `agent:${nid}`
      boundarySet.add(bId)
      addNode({ id: bId, label: neighbor.label || nid, color: '#64748b' })
    }

    const edgeList: GraphLayoutEdge[] = []
    const edgeKeys = new Set<string>()
    const addEdge = (e: GraphLayoutEdge): void => {
      const key = `${e.src}|${e.dst}|${e.kind}`
      if (edgeKeys.has(key)) return
      edgeKeys.add(key)
      edgeList.push(e)
    }

    // intra-agent edges: real component topology, both endpoints owned by this agent
    for (const cid of ownedIds) {
      const comp = componentById.get(cid)
      if (!comp) continue
      for (const downId of comp.downstream) {
        if (ownedIds.has(downId)) addEdge({ src: comp.id, dst: downId, kind: 'symbol' })
      }
    }

    // supporting files/tools: no component identity of their own, so they can't come from
    // component topology — pulled straight from the file graph LLM 1 already built.
    for (const e of fileEdges) {
      const ownsSrc = ownedFileToComponentId.has(e.src)
      const ownsDst = ownedFileToComponentId.has(e.dst)
      if (ownsSrc === ownsDst) continue // both owned (covered above), or neither touches this agent

      const ownedFile = ownsSrc ? e.src : e.dst
      const otherFile = ownsSrc ? e.dst : e.src
      if (fileToOwningAgent.has(otherFile)) continue // a real agent's file — boundary node covers it

      const ownerComponentId = ownedFileToComponentId.get(ownedFile)!
      addNode({ id: otherFile, label: otherFile.split('/').pop() || otherFile, color: '#334155' })
      addEdge(
        ownsSrc
          ? { src: ownerComponentId, dst: otherFile, kind: 'symbol' }
          : { src: otherFile, dst: ownerComponentId, kind: 'symbol' }
      )
    }

    // agent-level backbone: agent.id IS the head component's id, so this always connects.
    // minlen: 2 pushes boundary agents one column further out than supporting files, so the
    // two groups don't crowd into the same rank.
    if (agent) {
      for (const upId of agent.upstream_agents) {
        if (agentById.has(upId)) {
          addEdge({ src: `agent:${upId}`, dst: agent.id, kind: 'agent-flow', minlen: 2 })
        }
      }
      for (const downId of agent.downstream_agents) {
        if (agentById.has(downId)) {
          addEdge({ src: agent.id, dst: `agent:${downId}`, kind: 'agent-flow', minlen: 2 })
        }
      }
    }

    return { nodes: nodeList, edges: edgeList, boundaryIds: boundarySet }
  }, [
    ownedIds,
    neighborAgentIds,
    agentById,
    componentById,
    agent,
    fileEdges,
    ownedFileToComponentId,
    fileToOwningAgent,
  ])

  const { layoutNodes, layoutEdges } = useMemo(
    () => getDagreGraphLayout(nodes, edges, { nodesep: 55, ranksep: 90 }),
    [nodes, edges]
  )

  const styledNodes = useMemo(() => {
    return layoutNodes.map((n) => {
      const isBoundary = boundaryIds.has(n.id)
      const isSelected = n.id === selectedNode
      return {
        ...n,
        style: {
          ...n.style,
          ...(isBoundary ? { border: '1px dashed #64748b', fontStyle: 'italic' as const } : {}),
          opacity: selectedNode && !isSelected && !isBoundary ? 0.35 : isBoundary ? 0.85 : 1,
        },
      }
    })
  }, [layoutNodes, boundaryIds, selectedNode])

  if (!agent) {
    return (
      <div className="h-full flex items-center justify-center p-6 text-center text-slate-600 text-[10px]">
        Select an agent from the list to view its flow.
      </div>
    )
  }

  if (layoutNodes.length === 0) {
    return (
      <div className="h-full flex items-center justify-center p-6 text-center text-slate-600 text-[10px]">
        No components resolved for this agent.
      </div>
    )
  }

  return (
    <div className="absolute inset-0 bg-[#090d16]/20">
      <ReactFlow
        nodes={styledNodes}
        edges={layoutEdges}
        onNodeClick={(_e, node) => {
          if (boundaryIds.has(node.id)) {
            onNavigateToAgent(node.id.slice('agent:'.length))
            return
          }
          onSelectNode(node.id === selectedNode ? null : node.id)
        }}
        onNodeMouseEnter={(e, node) => {
          setHovered({ id: node.id, x: e.clientX, y: e.clientY })
          onHoverNode?.(node.id)
        }}
        onNodeMouseMove={(e, node) => setHovered({ id: node.id, x: e.clientX, y: e.clientY })}
        onNodeMouseLeave={() => {
          setHovered(null)
          onHoverNode?.(null)
        }}
        nodesDraggable={false}
        nodesConnectable={false}
        fitView
        fitViewOptions={{ padding: 0.25 }}
        minZoom={0.2}
        maxZoom={1.5}
        proOptions={{ hideAttribution: true }}
      >
        <Background color="#1e293b" gap={10} size={1} />
      </ReactFlow>

      {hovered && (() => {
        const isBoundary = boundaryIds.has(hovered.id)
        if (isBoundary) {
          const neighbor = agentById.get(hovered.id.slice('agent:'.length))
          return (
            <div
              className="fixed z-50 pointer-events-none bg-slate-950/95 border border-slate-700 rounded-lg px-2.5 py-1.5 text-[10px] shadow-xl max-w-[220px]"
              style={{ left: hovered.x + 12, top: hovered.y + 12 }}
            >
              <div className="font-semibold text-slate-200 truncate">{neighbor?.label ?? hovered.id}</div>
              <div className="text-slate-500">Neighbor agent — click to open</div>
            </div>
          )
        }
        const comp = componentById.get(hovered.id)
        if (comp) {
          return (
            <div
              className="fixed z-50 pointer-events-none bg-slate-950/95 border border-slate-700 rounded-lg px-2.5 py-1.5 text-[10px] shadow-xl max-w-[220px]"
              style={{ left: hovered.x + 12, top: hovered.y + 12 }}
            >
              <div className="font-semibold text-indigo-300 truncate font-mono">{comp.id}</div>
              <div className="text-slate-400 capitalize">{comp.role}</div>
              {comp.model && <div className="text-slate-500 font-mono truncate">{comp.model}</div>}
              {comp.file && <div className="text-slate-600 font-mono truncate mt-0.5">{comp.file}</div>}
            </div>
          )
        }
        return (
          <div
            className="fixed z-50 pointer-events-none bg-slate-950/95 border border-slate-700 rounded-lg px-2.5 py-1.5 text-[10px] shadow-xl max-w-[220px]"
            style={{ left: hovered.x + 12, top: hovered.y + 12 }}
          >
            <div className="font-semibold text-slate-300 truncate font-mono">{hovered.id.split('/').pop()}</div>
            <div className="text-slate-500">Supporting file — click for details</div>
          </div>
        )
      })()}
    </div>
  )
}

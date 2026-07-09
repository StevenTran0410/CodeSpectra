import React, { useState, useEffect, useCallback, useMemo } from 'react'
import { useSearchParams, useNavigate } from 'react-router-dom'
import {
  Loader2,
  AlertCircle,
  Play,
  ArrowLeft,
  CheckCircle2,
  Workflow,
  ChevronRight,
  RefreshCw,
} from 'lucide-react'
import { Button, Select, Badge, useToastStore } from '../../components/ui'
import {
  ReactFlow,
  Background,
} from '@xyflow/react'
import '@xyflow/react/dist/style.css'
import { getDagreGraphLayout, type GraphLayoutNode, type GraphLayoutEdge } from './graphLayout'
import { ROLE_COLORS, wiringBlockFileEdges, AgentSubGraphPanel } from './AgentSubGraphPanel'
import { useAEHStore } from '../../store/aeh.store'
import { useProviderStore } from '../../store/provider.store'
import LLMConfigModal, { LLMModelButton } from './LLMConfigModal'
import { useSessionPolling } from './useSessionPolling'
// AEHDiscoveryCandidate/AEHExpansionSession/AEHSystemMap/AEHAgentFlow/AEHAgentFlowMap are global ambient types.

export default function Stage2Screen(): React.ReactElement {
  const [searchParams] = useSearchParams()
  const navigate = useNavigate()
  const toast = useToastStore()
  const { startAEH } = useAEHStore()
  const { providers, load: loadProviders } = useProviderStore()

  const repoId = searchParams.get('repoId') || ''
  const snapshotId = searchParams.get('snapshotId') || ''
  const initialCandidateId = searchParams.get('candidateId') || ''

  const [candidates, setCandidates] = useState<AEHDiscoveryCandidate[]>([])
  const [selectedCandidateId, setSelectedCandidateId] = useState(initialCandidateId)
  const [loadingCandidates, setLoadingCandidates] = useState(false)

  // Running & status states
  const [expansionSession, setExpansionSession] = useState<AEHExpansionSession | null>(null)
  const [running, setRunning] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // LLM Config
  const [selectedProviderId, setSelectedProviderId] = useState('')
  const [selectedModelId, setSelectedModelId] = useState('')
  const [llmConfigOpen, setLlmConfigOpen] = useState(false)

  // Flow Analysis LLM Config (LLM 2 — holistic agent-flow separation pass, needs a strong model)
  const [flowProviderId, setFlowProviderId] = useState('')
  const [flowModelId, setFlowModelId] = useState('')
  const [flowLlmConfigOpen, setFlowLlmConfigOpen] = useState(false)

  // System Map State
  const [systemMap, setSystemMap] = useState<AEHSystemMap | null>(null)
  const [selectedGraphNode, setSelectedGraphNode] = useState<string | null>(null)

  // Agent Flow Map State (LLM 2 output — groups the expanded map into per-agent sub-flows)
  const [agentFlowMap, setAgentFlowMap] = useState<AEHAgentFlowMap | null>(null)
  const [generatingFlows, setGeneratingFlows] = useState(false)
  const [selectedAgentId, setSelectedAgentId] = useState<string | null>(null)
  const [graphScope, setGraphScope] = useState<'agent' | 'full'>('agent')

  // Load providers on mount
  useEffect(() => {
    loadProviders()
  }, [loadProviders])

  // Set default provider/model
  useEffect(() => {
    if (providers.length > 0) {
      const defaultProv = providers[0]
      setSelectedProviderId((prev) => prev || defaultProv.id)
      setSelectedModelId((prev) => prev || defaultProv.model_id || '')
    }
  }, [providers])

  // Load confirmed candidates for latest discovery session
  const loadConfirmedCandidates = useCallback(async () => {
    if (!snapshotId) return
    setLoadingCandidates(true)
    try {
      await startAEH()
      const sessions = await window.api.aeh.listDiscoverySessions(snapshotId)
      const matching = sessions.filter((s) => s.snapshot_id === snapshotId)
      if (matching.length > 0) {
        const latest = matching[0]
        const cands = await window.api.aeh.listDiscoveryCandidates(latest.id)
        const confirmed = cands.filter((c) => c.verdict === 'confirmed')
        setCandidates(confirmed)
        if (confirmed.length > 0 && !selectedCandidateId) {
          setSelectedCandidateId(confirmed[0].id)
        }
      }
    } catch (e) {
      console.error('Failed to load confirmed candidates', e)
    } finally {
      setLoadingCandidates(false)
    }
  }, [snapshotId, selectedCandidateId, startAEH])

  useEffect(() => {
    if (snapshotId) {
      loadConfirmedCandidates()
    }
  }, [snapshotId, loadConfirmedCandidates])

  // Find active candidate
  const activeCandidate = useMemo(() => {
    return candidates.find((c) => c.id === selectedCandidateId) || null
  }, [candidates, selectedCandidateId])

  // null (not yet generated) just means "not generated yet" — never triggers a fresh LLM-2 call or an error.
  const tryLoadAgentFlowMap = useCallback(async (sessionId: string) => {
    try {
      const flows = await window.api.aeh.getAgentFlowMap(sessionId)
      setAgentFlowMap(flows)
      setSelectedAgentId(flows ? flows.entry_agent_ids[0] ?? flows.agents[0]?.id ?? null : null)
    } catch {
      setAgentFlowMap(null)
    }
  }, [])

  const generateAgentFlowsFor = useCallback(
    async (sessionId: string) => {
      setGeneratingFlows(true)
      try {
        const flows = await window.api.aeh.generateAgentFlowMap(sessionId, {
          provider_id: flowProviderId || selectedProviderId,
          model_id: (flowProviderId ? flowModelId : selectedModelId) || null,
        })
        setAgentFlowMap(flows)
        setSelectedAgentId(flows.entry_agent_ids[0] ?? flows.agents[0]?.id ?? null)
        setGraphScope('agent')
        toast.success('Agent flows separated.')
      } catch (err: any) {
        toast.error(err?.message ?? 'Failed to separate agent flows.')
      } finally {
        setGeneratingFlows(false)
      }
    },
    [flowProviderId, flowModelId, selectedProviderId, selectedModelId, toast]
  )

  const startPolling = useSessionPolling<AEHExpansionSession>({
    fetchSession: (sessionId) => window.api.aeh.getExpansionSession(sessionId),
    onUpdate: setExpansionSession,
    onDone: async (session) => {
      setRunning(false)
      if (session.status === 'completed') {
        toast.success('Expansion complete. Separating agent flows...')
        const map = await window.api.aeh.getExpansionMap(session.id)
        setSystemMap(map)
        await generateAgentFlowsFor(session.id)
      } else if (session.status === 'failed') {
        setError(session.error ?? 'Expansion failed.')
      }
    },
    onError: () => {
      setRunning(false)
      setError('Error polling expansion status.')
    },
  })

  // Restore the latest session/map for this candidate so navigating back isn't a blank slate.
  useEffect(() => {
    if (!selectedCandidateId) return
    let cancelled = false
    setExpansionSession(null)
    setSystemMap(null)
    setAgentFlowMap(null)
    setSelectedAgentId(null)
    setError(null)
    ;(async () => {
      try {
        const sessions = await window.api.aeh.listExpansionSessions(selectedCandidateId)
        if (cancelled || sessions.length === 0) return
        const latest = sessions[0]
        setExpansionSession(latest)
        if (latest.status === 'completed') {
          const map = await window.api.aeh.getExpansionMap(latest.id)
          if (cancelled) return
          setSystemMap(map)
          await tryLoadAgentFlowMap(latest.id)
        } else if (latest.status === 'running') {
          setRunning(true)
          startPolling(latest.id)
        } else if (latest.status === 'failed') {
          setError(latest.error ?? 'Expansion failed.')
        }
      } catch (e) {
        console.error('Failed to load previous expansion session', e)
      }
    })()
    return () => {
      cancelled = true
    }
  }, [selectedCandidateId, startPolling, tryLoadAgentFlowMap])

  const countKnownFiles = (candidate: AEHDiscoveryCandidate | null): number => {
    if (!candidate) return 0
    return new Set([
      ...candidate.cluster_files,
      ...candidate.hub_paths,
      ...(candidate.matched_files ?? []),
    ]).size
  }

  // Node budget scales with candidate size: <20 files→100, 20-79→200, ≥80→400.
  const budgetForFileCount = (totalFiles: number): number => {
    if (totalFiles >= 80) return 400
    if (totalFiles >= 20) return 200
    return 100
  }

  const computeNodeBudget = (candidate: AEHDiscoveryCandidate | null): number =>
    budgetForFileCount(countKnownFiles(candidate))

  // Chains into agent-flow separation via startPolling's onDone once expansion completes.
  const handleRun = async () => {
    if (!selectedCandidateId || running || generatingFlows) return
    if (!selectedProviderId) {
      toast.error('Please configure and select an LLM provider.')
      return
    }

    setRunning(true)
    setError(null)
    setSystemMap(null)
    setExpansionSession(null)
    setAgentFlowMap(null)
    setSelectedAgentId(null)

    try {
      await startAEH()
      const nodeBudget = computeNodeBudget(activeCandidate)
      const res = await window.api.aeh.startExpansion(selectedCandidateId, {
        provider_id: selectedProviderId,
        model_id: selectedModelId || null,
        node_budget: nodeBudget,
        hop_cap: 3,
      })
      const session = await window.api.aeh.getExpansionSession(res.session_id)
      setExpansionSession(session)
      startPolling(res.session_id)
    } catch (err: any) {
      setRunning(false)
      setError(err?.message ?? 'Failed to trigger candidate expansion.')
      toast.error(err?.message ?? 'Expansion failed.')
    }
  }

  return (
    <div className="flex flex-col h-full bg-[#090d16] text-slate-300">
      {/* Screen Header */}
      <div className="px-4 py-3 border-b border-slate-800 bg-[#0f172a]/60 backdrop-blur-sm flex items-center justify-between shrink-0">
        <div className="flex items-center gap-3">
          <button
            onClick={() => navigate(`/aeh/analysis?repoId=${repoId}&snapshotId=${snapshotId}`)}
            className="p-1 rounded-lg hover:bg-slate-800 transition-colors text-slate-400 hover:text-slate-200"
          >
            <ArrowLeft className="w-4 h-4" />
          </button>
          <div>
            <h1 className="text-xs font-semibold text-slate-200">Stage 2: Iterative Expansion</h1>
            <p className="text-[10px] text-slate-500 mt-0.5">Grow seeds into a validated system map blueprint</p>
          </div>
        </div>

        <div className="flex items-center gap-3">
          <LLMModelButton
            providerId={selectedProviderId}
            modelId={selectedModelId}
            providers={providers}
            disabled={running || generatingFlows}
            onClick={() => setLlmConfigOpen(true)}
            labelPrefix="LLM 1"
          />
          <LLMConfigModal
            isOpen={llmConfigOpen}
            onClose={() => setLlmConfigOpen(false)}
            providerId={selectedProviderId}
            modelId={selectedModelId}
            onChange={(pid, mid) => {
              setSelectedProviderId(pid)
              setSelectedModelId(mid)
            }}
            title="LLM 1 — Expansion & Classification Model (this stage only)"
          />

          <LLMModelButton
            providerId={flowProviderId}
            modelId={flowModelId}
            providers={providers}
            disabled={running || generatingFlows}
            onClick={() => setFlowLlmConfigOpen(true)}
            labelPrefix="LLM 2"
            emptyLabel="LLM 2: Same as LLM 1"
            emptyVariant="neutral"
          />
          <LLMConfigModal
            isOpen={flowLlmConfigOpen}
            onClose={() => setFlowLlmConfigOpen(false)}
            providerId={flowProviderId}
            modelId={flowModelId}
            onChange={(pid, mid) => {
              setFlowProviderId(pid)
              setFlowModelId(mid)
            }}
            title="LLM 2 — Agent-Flow Analysis Model (optional — reasons over the whole map at once, needs a strong model)"
          />

          <Button
            variant="primary"
            disabled={running || generatingFlows || !selectedCandidateId || providers.length === 0}
            onClick={handleRun}
            className="text-[10px] h-7 px-3.5 flex items-center gap-1.5"
          >
            {running ? (
              <>
                <Loader2 className="w-3.5 h-3.5 animate-spin" />
                <span>Expanding...</span>
              </>
            ) : generatingFlows ? (
              <>
                <Loader2 className="w-3.5 h-3.5 animate-spin" />
                <span>Separating Flows...</span>
              </>
            ) : (
              <>
                <Play className="w-3 h-3 ml-0.5" />
                <span>Run</span>
              </>
            )}
          </Button>

          {/* Node budget badge */}
          {(() => {
            const totalFiles = countKnownFiles(activeCandidate)
            const budget = budgetForFileCount(totalFiles)
            const tier = budget === 400 ? 'XL' : budget === 200 ? 'LG' : 'SM'
            const colorClass =
              budget === 400
                ? 'border-rose-800/60 bg-rose-950/40 text-rose-300'
                : budget === 200
                  ? 'border-amber-800/60 bg-amber-950/40 text-amber-300'
                  : 'border-emerald-800/60 bg-emerald-950/40 text-emerald-300'
            return (
              <div
                title={`Node budget: ${budget} nodes (${totalFiles} known files in candidate)`}
                className={`flex items-center gap-1 h-7 px-2.5 rounded-md border font-mono text-[10px] select-none ${colorClass}`}
              >
                <span className="opacity-60">budget</span>
                <span className="font-bold">{budget}</span>
                <span className="opacity-50 text-[8px]">{tier}</span>
              </div>
            )
          })()}
        </div>
      </div>

      {/* Screen Body */}
      <div className="flex-1 flex overflow-hidden min-w-0">
        {/* Left selector and status overview */}
        <div className="w-72 border-r border-slate-800 bg-[#0c1220] flex flex-col p-4 space-y-4 shrink-0 overflow-y-auto">
          {/* Candidate selector */}
          <div className="space-y-1.5">
            <span className="text-[9px] uppercase tracking-wider font-bold text-slate-500 block">Select Confirmed Candidate</span>
            {loadingCandidates ? (
              <div className="flex items-center text-[10px] text-slate-500">
                <Loader2 className="w-3.5 h-3.5 animate-spin mr-1.5" />
                Loading...
              </div>
            ) : candidates.length === 0 ? (
              <div className="text-[10px] bg-indigo-950/20 border border-indigo-900/30 rounded-lg p-3 text-indigo-300">
                No confirmed candidates found for this snapshot. Go back to Stage 1 to confirm components first.
              </div>
            ) : (
              <Select
                value={selectedCandidateId}
                onChange={(e) => setSelectedCandidateId(e.target.value)}
                disabled={running}
                className="text-xs"
              >
                {candidates.map((c) => (
                  <option key={c.id} value={c.id}>
                    {c.name} (community {c.community_id})
                  </option>
                ))}
              </Select>
            )}
          </div>

          {/* Active Candidate Info Card */}
          {activeCandidate && (
            <div className="bg-slate-900/40 border border-slate-850 rounded-xl p-3.5 space-y-2">
              <h3 className="text-xs font-semibold text-slate-200">{activeCandidate.name}</h3>
              <div className="text-[10px] text-slate-400 space-y-1">
                <div>Frameworks: {activeCandidate.frameworks.join(', ') || 'none'}</div>
                <div>Confidence: {activeCandidate.confidence}</div>
                <div>Seed files: {activeCandidate.cluster_files.length}</div>
              </div>
            </div>
          )}

          {/* Expansion Progress Panel */}
          {expansionSession && (
            <div className="border border-indigo-900/40 bg-indigo-950/10 rounded-xl p-3.5 space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-[9px] uppercase tracking-wider font-bold text-indigo-400">Expansion Status</span>
                <Badge variant={expansionSession.status === 'completed' ? 'success' : expansionSession.status === 'failed' ? 'error' : 'neutral'} size="sm">
                  {expansionSession.status}
                </Badge>
              </div>

              {generatingFlows && (
                <div className="flex items-center gap-1.5 text-[10px] text-indigo-300">
                  <Loader2 className="w-3 h-3 animate-spin" />
                  <span>Separating agent flows (LLM 2)...</span>
                </div>
              )}

              <div className="space-y-1.5 text-[10px] font-mono text-slate-400">
                <div className="flex justify-between">
                  <span>Accepted Files:</span>
                  <span className="text-slate-200 font-semibold">{expansionSession.accepted.length}</span>
                </div>
                <div className="flex justify-between">
                  <span>Boundary Nodes:</span>
                  <span className="text-slate-200">{expansionSession.boundary.length}</span>
                </div>
                {expansionSession.stop_reason && (
                  <div className="flex justify-between border-t border-slate-900 pt-1.5">
                    <span>Stop Reason:</span>
                    <span className="text-amber-400">{expansionSession.stop_reason}</span>
                  </div>
                )}
              </div>

              {expansionSession.accepted.length > 0 && (
                <div className="space-y-1 border-t border-slate-900 pt-2">
                  <span className="text-[9px] uppercase tracking-wider font-bold text-slate-500">Accepted</span>
                  <div className="max-h-40 overflow-y-auto space-y-0.5 pr-1">
                    {expansionSession.accepted.map((path) => (
                      <div key={path} className="text-[9px] font-mono text-slate-300 truncate" title={path}>
                        {path}
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {expansionSession.boundary.length > 0 && (
                <div className="space-y-1 border-t border-slate-900 pt-2">
                  <span className="text-[9px] uppercase tracking-wider font-bold text-slate-500">Boundary (stopped here)</span>
                  <div className="max-h-32 overflow-y-auto space-y-0.5 pr-1">
                    {expansionSession.boundary.map((path) => (
                      <div key={path} className="text-[9px] font-mono text-slate-600 truncate" title={path}>
                        {path}
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {error && (
            <div className="bg-red-950/20 border border-red-900/30 rounded-lg p-3 text-[10px] text-red-400 flex items-start gap-2">
              <AlertCircle className="w-4 h-4 shrink-0" />
              <span>{error}</span>
            </div>
          )}
        </div>

        {/* Right System Map Editor */}
        <div className="flex-1 bg-[#090d16] flex flex-col min-w-0">
          {!systemMap ? (
            <div className="flex-1 flex flex-col items-center justify-center p-6 text-center text-slate-500 text-xs">
              {running ? (
                <>
                  <Loader2 className="w-10 h-10 mb-4 animate-spin text-indigo-500" />
                  <p className="font-semibold text-slate-400 mb-1">Iterative loop executing...</p>
                  <p className="text-slate-500 max-w-xs leading-relaxed">
                    Classifying neighbor files over graph hops. This may take a minute depending on the codebase size.
                  </p>
                </>
              ) : (
                <>
                  <Workflow className="w-12 h-12 mb-3 text-slate-700" />
                  <p className="font-semibold text-slate-400 mb-1">No System Map Generated</p>
                  <p className="text-slate-500 max-w-xs leading-relaxed">
                    Select a confirmed candidate and model configurations on the left, then click **Run Expansion** to start.
                  </p>
                </>
              )}
            </div>
          ) : (
            <div className="flex-1 flex flex-col overflow-hidden min-h-0">
              {/* Map Header details */}
              <div className="px-6 py-4 border-b border-slate-850 flex items-center justify-between shrink-0 bg-slate-900/10">
                <div className="space-y-1">
                  <div className="flex items-center gap-2">
                    <h2 className="text-sm font-semibold text-slate-200">Draft Map: {systemMap.target_system_id}</h2>
                    <Badge variant="success" size="sm">
                      <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400 mr-1" />
                      Draft Blueprint
                    </Badge>
                  </div>
                  <p className="text-[10px] text-slate-500">
                    {agentFlowMap
                      ? 'Browse the system as per-agent flows, or switch to the full component graph.'
                      : generatingFlows
                        ? 'Separating agent flows...'
                        : 'Browsing the full component graph.'}
                  </p>
                </div>

                <div className="flex items-center gap-2 shrink-0">
                  {/* Agent / Full graph scope toggle — only meaningful once agent flows exist */}
                  {agentFlowMap && (
                    <div className="flex items-center bg-slate-950 border border-slate-800 rounded-lg p-0.5 shrink-0">
                      <button
                        onClick={() => setGraphScope('agent')}
                        disabled={!selectedAgentId}
                        className={`text-[10px] px-2.5 py-1 rounded-md transition-colors disabled:opacity-40 disabled:cursor-not-allowed ${
                          graphScope === 'agent'
                            ? 'bg-slate-800 text-slate-200 font-semibold'
                            : 'text-slate-400 hover:text-slate-200'
                        }`}
                      >
                        Agent View
                      </button>
                      <button
                        onClick={() => setGraphScope('full')}
                        className={`text-[10px] px-2.5 py-1 rounded-md transition-colors ${
                          graphScope === 'full'
                            ? 'bg-slate-800 text-slate-200 font-semibold'
                            : 'text-slate-400 hover:text-slate-200'
                        }`}
                      >
                        Full Graph
                      </button>
                    </div>
                  )}

                  {/* Small, secondary — retry JUST the LLM-2 pass without re-running expansion. */}
                  {expansionSession?.status === 'completed' && (
                    <button
                      onClick={() => generateAgentFlowsFor(expansionSession.id)}
                      disabled={running || generatingFlows}
                      title="Retry agent-flow separation only (LLM 2), keeping the existing map"
                      className="flex items-center gap-1 h-[26px] px-2 rounded-md border border-slate-800 text-[9px] text-slate-500 hover:text-slate-300 hover:border-slate-700 disabled:opacity-40 disabled:cursor-not-allowed transition-colors shrink-0"
                    >
                      <RefreshCw className={`w-3 h-3 ${generatingFlows ? 'animate-spin' : ''}`} />
                      <span>Retry flows</span>
                    </button>
                  )}
                </div>
              </div>

              {/* Discrepancies Alerts */}
              {systemMap.discrepancies.length > 0 && (
                <div className="px-6 py-3 border-b border-slate-850 bg-amber-950/15 text-xs text-amber-300 flex flex-col gap-1.5 shrink-0">
                  <div className="flex items-center gap-1.5 font-semibold">
                    <AlertCircle className="w-4 h-4 text-amber-400" />
                    <span>Documentation Discrepancies Detected ({systemMap.discrepancies.length})</span>
                  </div>
                  <ul className="list-disc pl-5 space-y-0.5 text-[10px] text-amber-400/80">
                    {systemMap.discrepancies.map((d, i) => (
                      <li key={i}>{d}</li>
                    ))}
                  </ul>
                </div>
              )}

              {/* Agent List + Graph */}
              <div className="flex-1 flex min-h-0">
                {agentFlowMap && (
                  <div className="w-72 border-r border-slate-850 shrink-0 min-h-0">
                    <AgentFlowListPanel
                      agentFlowMap={agentFlowMap}
                      selectedAgentId={selectedAgentId}
                      onSelectAgent={(id) => {
                        setSelectedAgentId(id)
                        setGraphScope('agent')
                        setSelectedGraphNode(null)
                      }}
                    />
                  </div>
                )}

                <div className="flex-1 flex min-h-0 relative">
                  <div className="flex-1 min-h-0 relative">
                    {expansionSession && (
                      graphScope === 'agent' && agentFlowMap && selectedAgentId ? (
                        <AgentSubGraphPanel
                          agent={agentFlowMap.agents.find((a) => a.id === selectedAgentId) ?? null}
                          agentFlowMap={agentFlowMap}
                          systemMap={systemMap}
                          expansionSession={expansionSession}
                          candidate={activeCandidate}
                          selectedNode={selectedGraphNode}
                          onSelectNode={setSelectedGraphNode}
                          onNavigateToAgent={(id) => {
                            setSelectedAgentId(id)
                            setSelectedGraphNode(null)
                          }}
                        />
                      ) : (
                        <SystemMapGraphPanel
                          expansionSession={expansionSession}
                          systemMap={systemMap}
                          candidate={activeCandidate}
                          selectedGraphNode={selectedGraphNode}
                          onSelectGraphNode={setSelectedGraphNode}
                        />
                      )
                    )}
                  </div>

                  {/* selectedGraphNode is a file path (full graph) or component id (agent graph) */}
                  {selectedGraphNode && (() => {
                    const comp = systemMap.components.find(
                      (c) => c.file === selectedGraphNode || c.id === selectedGraphNode
                    )
                    const byId = comp?.id === selectedGraphNode
                    const componentNeighbors = byId ? [...comp!.upstream, ...comp!.downstream] : []
                    const fileNeighbors = new Set<string>()
                    if (!byId && expansionSession) {
                      for (const e of expansionSession.accepted_edges || []) {
                        if (e.src === selectedGraphNode) fileNeighbors.add(e.dst)
                        if (e.dst === selectedGraphNode) fileNeighbors.add(e.src)
                      }
                    }

                    return (
                      <div className="w-80 border-l border-slate-850 bg-slate-950/80 backdrop-blur-sm p-4 overflow-y-auto space-y-4 text-xs shrink-0 flex flex-col justify-between">
                        <div className="space-y-4">
                          <div className="flex items-center justify-between">
                            <span className="text-[9px] uppercase tracking-wider font-bold text-slate-500">
                              {byId ? 'Component Details' : 'File Details'}
                            </span>
                            <button
                              onClick={() => setSelectedGraphNode(null)}
                              className="text-slate-400 hover:text-slate-200 text-sm font-bold px-1"
                            >
                              &times;
                            </button>
                          </div>

                          <div>
                            <div className="text-[10px] text-slate-400 mb-0.5">{byId ? 'Component ID' : 'File Path'}</div>
                            <div className="font-mono text-[10px] text-slate-200 break-all select-all font-semibold bg-slate-900/50 border border-slate-900 rounded-lg p-2">
                              {selectedGraphNode}
                            </div>
                          </div>

                          {comp ? (
                            <div className="space-y-3 border-t border-slate-900 pt-3">
                              {!byId && (
                                <div>
                                  <div className="text-[10px] text-slate-400 mb-0.5">Component ID</div>
                                  <div className="font-semibold text-indigo-400 truncate font-mono">{comp.id}</div>
                                </div>
                              )}
                              <div>
                                <div className="text-[10px] text-slate-400 mb-0.5">Role</div>
                                <div className="font-medium text-slate-200 capitalize">{comp.role}</div>
                              </div>
                              {comp.model && (
                                <div>
                                  <div className="text-[10px] text-slate-400 mb-0.5">Model</div>
                                  <div className="font-mono text-[10px] text-slate-300 bg-slate-900 border border-slate-850 px-1.5 py-0.5 rounded inline-block">
                                    {comp.model}
                                  </div>
                                </div>
                              )}
                              {comp.entry_point && (
                                <div>
                                  <div className="text-[10px] text-slate-400 mb-0.5">Entry Point</div>
                                  <div className="font-mono text-[9px] text-slate-400 break-all">{comp.entry_point}</div>
                                </div>
                              )}
                              {byId && comp.file && (
                                <div>
                                  <div className="text-[10px] text-slate-400 mb-0.5">File</div>
                                  <div className="font-mono text-[9px] text-slate-400 break-all">{comp.file}</div>
                                </div>
                              )}
                              {comp.constraints.length > 0 && (
                                <div>
                                  <div className="text-[10px] text-slate-400 mb-1">Constraints ({comp.constraints.length})</div>
                                  <div className="space-y-1.5 font-mono text-[9px]">
                                    {comp.constraints.map((cons, i) => (
                                      <div key={i} className="bg-slate-900/55 border border-slate-900 p-1.5 rounded text-slate-400">
                                        <div className="text-slate-300 font-semibold">{cons.name}: {String(cons.value)}</div>
                                        <div className="text-[8px] text-slate-500 truncate mt-0.5" title={cons.source}>{cons.source.split('/').pop()}</div>
                                      </div>
                                    ))}
                                  </div>
                                </div>
                              )}
                            </div>
                          ) : (
                            <div className="border-t border-slate-900 pt-3 text-[10px] text-slate-500 italic">
                              Not classified as a recognized framework component.
                            </div>
                          )}

                          {byId
                            ? componentNeighbors.length > 0 && (
                                <div className="border-t border-slate-900 pt-3">
                                  <div className="text-[10px] text-slate-400 mb-1.5">Direct Neighbors ({componentNeighbors.length})</div>
                                  <div className="space-y-1 max-h-48 overflow-y-auto pr-1">
                                    {componentNeighbors.map((nid) => (
                                      <button
                                        key={nid}
                                        onClick={() => setSelectedGraphNode(nid)}
                                        className="w-full text-left font-mono text-[9px] text-slate-400 hover:text-slate-200 truncate hover:underline"
                                      >
                                        {nid}
                                      </button>
                                    ))}
                                  </div>
                                </div>
                              )
                            : fileNeighbors.size > 0 && (
                                <div className="border-t border-slate-900 pt-3">
                                  <div className="text-[10px] text-slate-400 mb-1.5">Direct Neighbors ({fileNeighbors.size})</div>
                                  <div className="space-y-1 max-h-48 overflow-y-auto pr-1">
                                    {Array.from(fileNeighbors).map((nPath) => (
                                      <button
                                        key={nPath}
                                        onClick={() => setSelectedGraphNode(nPath)}
                                        className="w-full text-left font-mono text-[9px] text-slate-400 hover:text-slate-200 truncate hover:underline"
                                        title={nPath}
                                      >
                                        {nPath.split('/').pop() || nPath}
                                      </button>
                                    ))}
                                  </div>
                                </div>
                              )}
                        </div>
                      </div>
                    )
                  })()}
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

function SystemMapGraphPanel({
  expansionSession,
  systemMap,
  candidate,
  selectedGraphNode,
  onSelectGraphNode,
}: {
  expansionSession: AEHExpansionSession
  systemMap: AEHSystemMap
  candidate: AEHDiscoveryCandidate | null
  selectedGraphNode: string | null
  onSelectGraphNode: (path: string | null) => void
}): React.ReactElement {
  const roleByFile = useMemo(() => {
    const m = new Map<string, string>()
    for (const c of systemMap.components) {
      if (c.file) m.set(c.file, c.role)
    }
    return m
  }, [systemMap])

  const mergedEdges = useMemo(() => {
    const wiringEdges = wiringBlockFileEdges(candidate?.wiring_block ?? null)
      .filter((e) => expansionSession.accepted.includes(e.src) && expansionSession.accepted.includes(e.dst))
    const wiringEdgeKeys = new Set(wiringEdges.map((e) => `${e.src}|${e.dst}`))

    const symbolEdges = (expansionSession.accepted_edges || [])
      .filter((e) => !wiringEdgeKeys.has(`${e.src}|${e.dst}`) && !wiringEdgeKeys.has(`${e.dst}|${e.src}`))

    return [
      ...wiringEdges.map((e) => ({ ...e, kind: 'wiring' as const })),
      ...symbolEdges.map((e) => ({ ...e, kind: 'symbol' as const })),
    ]
  }, [expansionSession, candidate])

  const neighborsOf = useMemo(() => {
    const m = new Map<string, Set<string>>()
    for (const path of expansionSession.accepted) m.set(path, new Set())
    for (const e of mergedEdges) {
      m.get(e.src)?.add(e.dst)
      m.get(e.dst)?.add(e.src)
    }
    return m
  }, [expansionSession, mergedEdges])

  const { layoutNodes, layoutEdges } = useMemo(() => {
    const nodes = expansionSession.accepted.map((path) => ({
      id: path,
      label: path.split('/').pop() || path,
      color: ROLE_COLORS[roleByFile.get(path) ?? 'unknown'] ?? ROLE_COLORS.unknown,
    }))
    return getDagreGraphLayout(nodes, mergedEdges)
  }, [expansionSession, roleByFile, mergedEdges])

  // Dim everything except the selected node + its direct neighbors.
  const styledNodes = useMemo(() => {
    if (!selectedGraphNode) return layoutNodes
    const highlighted = new Set([selectedGraphNode, ...(neighborsOf.get(selectedGraphNode) ?? [])])
    return layoutNodes.map((n) => ({
      ...n,
      style: { ...n.style, opacity: highlighted.has(n.id) ? 1 : 0.2 },
    }))
  }, [layoutNodes, selectedGraphNode, neighborsOf])

  const styledEdges = useMemo(() => {
    if (!selectedGraphNode) return layoutEdges
    return layoutEdges.map((e) => ({
      ...e,
      style: {
        ...e.style,
        opacity: e.source === selectedGraphNode || e.target === selectedGraphNode ? 1 : 0.1,
      },
    }))
  }, [layoutEdges, selectedGraphNode])

  if (layoutNodes.length === 0) {
    return (
      <div className="h-full flex items-center justify-center p-6 text-center text-slate-600 text-[10px]">
        No accepted files to visualize.
      </div>
    )
  }

  return (
    <div className="absolute inset-0 bg-[#090d16]/20">
      <ReactFlow
        nodes={styledNodes}
        edges={styledEdges}
        onNodeClick={(_e, node) => onSelectGraphNode(node.id === selectedGraphNode ? null : node.id)}
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

/** Hierarchical agent list — roots = agents with no valid parent, children under parent_agent. */
function AgentFlowListPanel({
  agentFlowMap,
  selectedAgentId,
  onSelectAgent,
}: {
  agentFlowMap: AEHAgentFlowMap
  selectedAgentId: string | null
  onSelectAgent: (agentId: string) => void
}): React.ReactElement {
  const agentById = useMemo(() => new Map(agentFlowMap.agents.map((a) => [a.id, a])), [agentFlowMap])

  // '__root__' bucket = agents with no valid parent, i.e. the tree's root set.
  const childrenByParent = useMemo(() => {
    const m = new Map<string, AEHAgentFlow[]>()
    for (const a of agentFlowMap.agents) {
      const key = a.parent_agent && agentById.has(a.parent_agent) ? a.parent_agent : '__root__'
      if (!m.has(key)) m.set(key, [])
      m.get(key)!.push(a)
    }
    return m
  }, [agentFlowMap, agentById])

  if (agentFlowMap.agents.length === 0) {
    return (
      <div className="h-full flex items-center justify-center p-6 text-center text-slate-500 text-[10px] select-none">
        No agents identified — every component landed in unassigned_component_ids.
      </div>
    )
  }

  const renderRow = (agent: AEHAgentFlow, depth: number): React.ReactNode => {
    const children = childrenByParent.get(agent.id) ?? []
    const isSelected = agent.id === selectedAgentId
    return (
      <div key={agent.id}>
        <button
          onClick={() => onSelectAgent(agent.id)}
          style={{ paddingLeft: `${10 + depth * 14}px` }}
          className={`w-full text-left pr-3 py-2.5 hover:bg-slate-900/30 transition-colors flex flex-col gap-1 border-l-2 ${
            isSelected ? 'bg-indigo-950/25 border-l-indigo-400' : 'border-l-transparent'
          }`}
        >
          <div className="flex items-center gap-1.5 min-w-0">
            {depth > 0 && <ChevronRight className="w-3 h-3 text-slate-700 shrink-0" />}
            <span
              className="w-2 h-2 rounded-full shrink-0"
              style={{ background: ROLE_COLORS[agent.role] ?? ROLE_COLORS.unknown }}
              title={agent.role}
            />
            <span className="text-[11px] font-semibold text-slate-200 truncate flex-1">
              {agent.label || agent.id}
            </span>
            <span className="text-[8px] text-slate-500 font-mono shrink-0" title="components owned">
              {agent.component_ids.length}
            </span>
          </div>
          {agent.summary && (
            <p className="text-[9px] text-slate-500 leading-snug line-clamp-2">{agent.summary}</p>
          )}
        </button>
        {children.map((child) => renderRow(child, depth + 1))}
      </div>
    )
  }

  return (
    <div className="flex flex-col h-full bg-[#090d16] overflow-y-auto">
      <div className="px-3 py-2 border-b border-slate-800 shrink-0">
        <span className="text-[9px] uppercase tracking-wider font-bold text-slate-500">
          Agents ({agentFlowMap.agents.length})
        </span>
      </div>
      <div className="divide-y divide-slate-900/40">
        {(childrenByParent.get('__root__') ?? []).map((agent) => renderRow(agent, 0))}
      </div>
      {agentFlowMap.unassigned_component_ids.length > 0 && (
        <div className="p-2.5 border-t border-slate-850 text-[9px] text-slate-600">
          {agentFlowMap.unassigned_component_ids.length} component(s) unassigned
        </div>
      )}
    </div>
  )
}


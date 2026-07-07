import React, { useState, useEffect, useCallback, useMemo, useRef } from 'react'
import { useSearchParams, useNavigate } from 'react-router-dom'
import {
  Loader2,
  AlertCircle,
  Play,
  ArrowLeft,
  Sparkles,
  Save,
  CheckCircle2,
  FileText,
  Workflow,
  HelpCircle,
  Shield,
  Layers,
} from 'lucide-react'
import { Button, Select, Badge, useToastStore } from '../../components/ui'
import {
  ReactFlow,
  Background,
} from '@xyflow/react'
import '@xyflow/react/dist/style.css'
import { getDagreGraphLayout } from './graphLayout'
import { useAEHStore } from '../../store/aeh.store'
import { useProviderStore } from '../../store/provider.store'
import LLMConfigModal from './LLMConfigModal'
// AEHDiscoveryCandidate/AEHExpansionSession/AEHSystemMap/AEHSystemMapComponent are global ambient types.

const VALID_ROLES = [
  'unknown',
  'input_guard.rule',
  'input_guard.llm',
  'orchestrator',
  'retrieval_agent',
  'tool',
  'validator',
  'writer',
]

const ROLE_COLORS: Record<string, string> = {
  orchestrator: '#6366f1',
  retrieval_agent: '#0ea5e9',
  tool: '#f59e0b',
  writer: '#10b981',
  validator: '#ec4899',
  'input_guard.rule': '#ef4444',
  'input_guard.llm': '#ef4444',
  unknown: '#475569',
}

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
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  // LLM Config
  const [selectedProviderId, setSelectedProviderId] = useState('')
  const [selectedModelId, setSelectedModelId] = useState('')
  const [llmConfigOpen, setLlmConfigOpen] = useState(false)

  // Classify LLM Config (optional)
  const [classifyProviderId, setClassifyProviderId] = useState('')
  const [classifyModelId, setClassifyModelId] = useState('')
  const [classifyLlmConfigOpen, setClassifyLlmConfigOpen] = useState(false)

  // System Map State
  const [systemMap, setSystemMap] = useState<AEHSystemMap | null>(null)
  const [mapView, setMapView] = useState<'table' | 'graph'>('table')
  const [savingMap, setSavingMap] = useState(false)

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

  // Start polling
  const startPolling = useCallback((sessionId: string) => {
    if (pollRef.current) clearInterval(pollRef.current)
    pollRef.current = setInterval(async () => {
      try {
        const session = await window.api.aeh.getExpansionSession(sessionId)
        setExpansionSession(session)
        if (session.status !== 'running') {
          if (pollRef.current) clearInterval(pollRef.current)
          setRunning(false)
          if (session.status === 'completed') {
            toast.success('Expansion complete. Loading system map...')
            const map = await window.api.aeh.getExpansionMap(sessionId)
            setSystemMap(map)
          } else if (session.status === 'failed') {
            setError(session.error ?? 'Expansion failed.')
          }
        }
      } catch (err) {
        if (pollRef.current) clearInterval(pollRef.current)
        setRunning(false)
        setError('Error polling expansion status.')
      }
    }, 2000)
  }, [toast])

  useEffect(() => () => { if (pollRef.current) clearInterval(pollRef.current) }, [])

  // Load the most recent expansion session/map for the selected candidate — without this,
  // navigating away and back always starts from a blank slate even after a completed run.
  useEffect(() => {
    if (!selectedCandidateId) return
    let cancelled = false
    setExpansionSession(null)
    setSystemMap(null)
    setError(null)
    ;(async () => {
      try {
        const sessions = await window.api.aeh.listExpansionSessions(selectedCandidateId)
        if (cancelled || sessions.length === 0) return
        const latest = sessions[0]
        setExpansionSession(latest)
        if (latest.status === 'completed') {
          const map = await window.api.aeh.getExpansionMap(latest.id)
          if (!cancelled) setSystemMap(map)
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
  }, [selectedCandidateId, startPolling])

  // Trigger Expansion
  const handleRunExpansion = async () => {
    if (!selectedCandidateId || running) return
    if (!selectedProviderId) {
      toast.error('Please configure and select an LLM provider.')
      return
    }

    setRunning(true)
    setError(null)
    setSystemMap(null)
    setExpansionSession(null)

    try {
      await startAEH()
      const res = await window.api.aeh.startExpansion(selectedCandidateId, {
        provider_id: selectedProviderId,
        model_id: selectedModelId || null,
        node_budget: 40,
        hop_cap: 3,
        classify_provider_id: classifyProviderId || null,
        classify_model_id: classifyModelId || null,
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

  // Update component role locally
  const handleRoleChange = (componentId: string, newRole: string) => {
    if (!systemMap) return
    const updatedComponents = systemMap.components.map((c) =>
      c.id === componentId ? { ...c, role: newRole } : c
    )
    setSystemMap({ ...systemMap, components: updatedComponents })
  }

  // Save Map back to the YAML file
  const handleSaveMap = async () => {
    if (!expansionSession || !systemMap) return
    setSavingMap(true)
    try {
      await window.api.aeh.updateExpansionMap(expansionSession.id, systemMap)
      toast.success('System map saved successfully.')
    } catch (err: any) {
      toast.error(err?.message ?? 'Failed to save system map.')
    } finally {
      setSavingMap(false)
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
          <button
            onClick={() => setLlmConfigOpen(true)}
            disabled={running}
            className={`text-[10px] h-7 px-2.5 rounded-md border font-mono transition-colors ${
              selectedProviderId
                ? 'border-slate-700 bg-slate-950 text-slate-300 hover:border-slate-500'
                : 'border-red-900/30 bg-red-950/40 text-red-400 hover:border-red-700/50'
            }`}
          >
            {selectedProviderId
              ? `Model: ${selectedModelId || providers.find((p) => p.id === selectedProviderId)?.display_name || '?'}`
              : 'No LLM Configured'}
          </button>
          <LLMConfigModal
            isOpen={llmConfigOpen}
            onClose={() => setLlmConfigOpen(false)}
            providerId={selectedProviderId}
            modelId={selectedModelId}
            onChange={(pid, mid) => {
              setSelectedProviderId(pid)
              setSelectedModelId(mid)
            }}
            title="Expansion Model (this stage only)"
          />

          <button
            onClick={() => setClassifyLlmConfigOpen(true)}
            disabled={running}
            className={`text-[10px] h-7 px-2.5 rounded-md border font-mono transition-colors ${
              classifyProviderId
                ? 'border-slate-700 bg-slate-950 text-slate-300 hover:border-slate-500'
                : 'border-slate-800 bg-slate-950/30 text-slate-500 hover:border-slate-700'
            }`}
          >
            {classifyProviderId
              ? `Classifier: ${classifyModelId || providers.find((p) => p.id === classifyProviderId)?.display_name || '?'}`
              : 'Classifier: Default'}
          </button>
          <LLMConfigModal
            isOpen={classifyLlmConfigOpen}
            onClose={() => setClassifyLlmConfigOpen(false)}
            providerId={classifyProviderId}
            modelId={classifyModelId}
            onChange={(pid, mid) => {
              setClassifyProviderId(pid)
              setClassifyModelId(mid)
            }}
            title="Chunk Classifier Model (optional — defaults to main model, pick something fast/cheap)"
          />

          <Button
            variant="primary"
            disabled={running || !selectedCandidateId || providers.length === 0}
            onClick={handleRunExpansion}
            className="text-[10px] h-7 px-3.5 flex items-center gap-1.5"
          >
            {running ? (
              <>
                <Loader2 className="w-3.5 h-3.5 animate-spin" />
                <span>Expanding...</span>
              </>
            ) : (
              <>
                <Play className="w-3 h-3 ml-0.5" />
                <span>Run Expansion</span>
              </>
            )}
          </Button>
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
                    Review and override roles dynamically before planning evaluation suites.
                  </p>
                </div>

                <div className="flex items-center gap-3">
                  {/* View Toggle */}
                  <div className="flex items-center bg-slate-950 border border-slate-800 rounded-lg p-0.5 shrink-0">
                    <button
                      onClick={() => setMapView('table')}
                      className={`text-[10px] px-2.5 py-1 rounded-md transition-colors ${
                        mapView === 'table'
                          ? 'bg-slate-800 text-slate-200 font-semibold'
                          : 'text-slate-400 hover:text-slate-200'
                      }`}
                    >
                      Table
                    </button>
                    <button
                      onClick={() => setMapView('graph')}
                      className={`text-[10px] px-2.5 py-1 rounded-md transition-colors ${
                        mapView === 'graph'
                          ? 'bg-slate-800 text-slate-200 font-semibold'
                          : 'text-slate-400 hover:text-slate-200'
                      }`}
                    >
                      Graph
                    </button>
                  </div>

                  <Button
                    variant="primary"
                    onClick={handleSaveMap}
                    loading={savingMap}
                    className="text-[10px] h-8 px-4 flex items-center gap-1.5 bg-emerald-600 hover:bg-emerald-500 text-white"
                  >
                    <Save className="w-3.5 h-3.5" />
                    <span>Save Blueprint Map</span>
                  </Button>
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

              {/* Components List Table OR Graph View */}
              {mapView === 'table' ? (
                <div className="flex-1 overflow-y-auto px-6 py-4">
                  <div className="border border-slate-850 rounded-xl overflow-hidden bg-slate-950/20">
                    <table className="w-full text-left border-collapse text-xs">
                      <thead>
                        <tr className="border-b border-slate-850 bg-slate-900/30 text-slate-400 font-medium">
                          <th className="p-3">Component / Class ID</th>
                          <th className="p-3">Role</th>
                          <th className="p-3">Model</th>
                          <th className="p-3">Entry Point</th>
                          <th className="p-3">Topology</th>
                          <th className="p-3 text-right">Constraints</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-850">
                        {systemMap.components.map((comp) => {
                          const isUnknown = comp.role === 'unknown'
                          return (
                            <tr
                              key={comp.id}
                              className={`hover:bg-slate-900/25 transition-colors ${
                                isUnknown ? 'bg-amber-950/5' : ''
                              }`}
                            >
                              <td className="p-3 font-mono font-medium text-slate-200">
                                <div className="flex items-center gap-2">
                                  <span className="truncate max-w-[160px]" title={comp.id}>
                                    {comp.id}
                                  </span>
                                  {isUnknown && (
                                    <Badge variant="error" size="sm" className="bg-amber-900/30 border border-amber-800/40 text-amber-400 animate-pulse">
                                      Triage
                                    </Badge>
                                  )}
                                </div>
                              </td>
                              <td className="p-3">
                                <Select
                                  value={comp.role}
                                  onChange={(e) => handleRoleChange(comp.id, e.target.value)}
                                  className={`text-[11px] h-7 px-2 py-0.5 ${
                                    isUnknown ? 'border-amber-700 bg-amber-950/20 text-amber-300' : ''
                                  }`}
                                >
                                  {VALID_ROLES.map((r) => (
                                    <option key={r} value={r}>
                                      {r}
                                    </option>
                                  ))}
                                </Select>
                              </td>
                              <td className="p-3">
                                <span className="text-[10px] px-2 py-0.5 rounded font-mono bg-slate-900 border border-slate-800 text-slate-400">
                                  {comp.model || 'None'}
                                </span>
                              </td>
                              <td className="p-3 font-mono text-[10px] text-slate-400 break-all select-text max-w-[200px]" title={comp.entry_point || ''}>
                                {comp.entry_point || '-'}
                              </td>
                              <td className="p-3 text-slate-400">
                                <div className="flex items-center gap-2">
                                  <span title={`Upstream: ${comp.upstream.join(', ') || 'none'}`}>
                                    In: {comp.upstream.length}
                                  </span>
                                  <span className="text-slate-650">•</span>
                                  <span title={`Downstream: ${comp.downstream.join(', ') || 'none'}`}>
                                    Out: {comp.downstream.length}
                                  </span>
                                </div>
                              </td>
                              <td className="p-3 text-right text-slate-400 font-mono font-medium">
                                {comp.constraints.length > 0 ? (
                                  <span
                                    className="underline cursor-help text-indigo-400"
                                    title={comp.constraints.map((c) => `${c.name}: ${c.value} (${c.source})`).join('\n')}
                                  >
                                    {comp.constraints.length} citation{comp.constraints.length !== 1 ? 's' : ''}
                                  </span>
                                ) : (
                                  '0'
                                )}
                              </td>
                            </tr>
                          )
                        })}
                      </tbody>
                    </table>
                  </div>
                </div>
              ) : (
                <div className="flex-1 min-h-0 relative">
                  <SystemMapGraphPanel systemMap={systemMap} />
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

function SystemMapGraphPanel({ systemMap }: { systemMap: AEHSystemMap }): React.ReactElement {
  const { layoutNodes, layoutEdges } = useMemo(() => {
    if (!systemMap) return { layoutNodes: [], layoutEdges: [] }
    const nodes = systemMap.components.map((c) => ({
      id: c.id,
      label: `${c.id}\n${c.role}`,
      color: ROLE_COLORS[c.role] ?? ROLE_COLORS.unknown,
    }))
    const edges = systemMap.components.flatMap((c) =>
      c.downstream.map((d) => ({ src: c.id, dst: d }))
    )
    return getDagreGraphLayout(nodes, edges)
  }, [systemMap])

  if (layoutNodes.length === 0) {
    return (
      <div className="h-full flex items-center justify-center p-6 text-center text-slate-600 text-[10px] select-none bg-[#090d16]/10">
        No components to visualize.
      </div>
    )
  }

  return (
    <div className="absolute inset-0 bg-[#090d16]/20">
      <ReactFlow
        nodes={layoutNodes}
        edges={layoutEdges}
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

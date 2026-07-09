import React, { useState, useEffect, useCallback, useMemo } from 'react'
import { useSearchParams, useNavigate } from 'react-router-dom'
import {
  Loader2,
  AlertCircle,
  Sparkles,
  Save,
  ArrowLeft,
  Settings,
  Plus,
  Trash2,
  Layers,
  Workflow,
  Database,
} from 'lucide-react'
import { Button, Select, Badge, useToastStore } from '../../components/ui'
import { useProviderStore } from '../../store/provider.store'
import LLMConfigModal from './LLMConfigModal'
import { ROLE_COLORS, AgentSubGraphPanel } from './AgentSubGraphPanel'

const METRIC_CLASS_STYLES: Record<string, string> = {
  assertion: 'border-sky-800/60 bg-sky-950/30 text-sky-300',
  classifier: 'border-amber-800/60 bg-amber-950/30 text-amber-300',
  llm_judge: 'border-violet-800/60 bg-violet-950/30 text-violet-300',
}

const TOOLKIT_LABELS: Record<string, string> = {
  assertion: 'assertion',
  classifier: 'classifier',
  ragas: 'ragas',
  deepeval: 'deepeval (G-Eval)',
}

export default function Stage3Screen(): React.ReactElement {
  const [searchParams] = useSearchParams()
  const navigate = useNavigate()
  const toast = useToastStore()
  const { providers, load: loadProviders } = useProviderStore()

  const repoId = searchParams.get('repoId') || ''
  const snapshotId = searchParams.get('snapshotId') || ''
  const initialCandidateId = searchParams.get('candidateId') || ''

  const [candidates, setCandidates] = useState<AEHDiscoveryCandidate[]>([])
  const [selectedCandidateId, setSelectedCandidateId] = useState(initialCandidateId)
  const [loadingCandidates, setLoadingCandidates] = useState(false)

  // Session & Map context
  const [expansionSession, setExpansionSession] = useState<AEHExpansionSession | null>(null)
  const [systemMap, setSystemMap] = useState<AEHSystemMap | null>(null)
  const [loadingSession, setLoadingSession] = useState(false)

  // Stage 2's agent-flow map — Stage 3's hard prerequisite (the planning unit is the agent).
  const [agentFlowMap, setAgentFlowMap] = useState<AEHAgentFlowMap | null>(null)
  const [loadingAgentFlow, setLoadingAgentFlow] = useState(false)

  // null = overview (agent grid). Set = the agent-detail "page". No routing involved —
  // a plain view switch, so Back is instant and doesn't touch the URL/candidate selector.
  const [viewingAgentId, setViewingAgentId] = useState<string | null>(null)

  // The DAG orchestrator's audit report (data-flow profile + gates per agent).
  const [planReport, setPlanReport] = useState<AEHEvaluationPlanReport | null>(null)

  // Plan data states
  const [planSuite, setPlanSuite] = useState<AEHPlanSuite | null>(null)
  const [localEntries, setLocalEntries] = useState<AEHPlanEntry[]>([])
  const [originalEntries, setOriginalEntries] = useState<AEHPlanEntry[]>([])
  const [readiness, setReadiness] = useState<Record<string, { status: 'runnable' | 'blocked'; reasons: string[] }>>({})
  const [loadingPlan, setLoadingPlan] = useState(false)
  const [generatingPlan, setGeneratingPlan] = useState(false)
  const [savingPlan, setSavingPlan] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Controlled params string state (one per entry index)
  const [paramsStrings, setParamsStrings] = useState<Record<number, string>>({})

  // Invalid JSON param indices to block save
  const [invalidJsonIndices, setInvalidJsonIndices] = useState<Record<number, boolean>>({})

  // LLM Config — Stage 3's OWN slot, independent of Stage 2's LLM 1 / LLM 2.
  const [selectedProviderId, setSelectedProviderId] = useState('')
  const [selectedModelId, setSelectedModelId] = useState('')
  const [llmConfigOpen, setLlmConfigOpen] = useState(false)

  // Regenerate Confirmation Modal
  const [confirmModalOpen, setConfirmModalOpen] = useState(false)

  useEffect(() => {
    loadProviders()
  }, [loadProviders])

  useEffect(() => {
    if (providers.length > 0) {
      const defaultProv = providers[0]
      setSelectedProviderId((prev) => prev || defaultProv.id)
      setSelectedModelId((prev) => prev || defaultProv.model_id || '')
    }
  }, [providers])

  const loadConfirmedCandidates = useCallback(async () => {
    if (!snapshotId) return
    setLoadingCandidates(true)
    try {
      const sessions = await window.api.aeh.listDiscoverySessions(undefined, snapshotId)
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
  }, [snapshotId, selectedCandidateId])

  useEffect(() => {
    if (snapshotId) loadConfirmedCandidates()
  }, [snapshotId, loadConfirmedCandidates])

  const candidate = useMemo(
    () => candidates.find((c) => c.id === selectedCandidateId) ?? null,
    [candidates, selectedCandidateId]
  )

  const applyPlanSuite = (suite: any) => {
    setPlanSuite(suite)
    setLocalEntries(suite.entries || [])
    setOriginalEntries(JSON.parse(JSON.stringify(suite.entries || [])))
    setInvalidJsonIndices({})
    const ps: Record<number, string> = {}
    ;(suite.entries || []).forEach((e: AEHPlanEntry, i: number) => { ps[i] = JSON.stringify(e.params || {}) })
    setParamsStrings(ps)
    setReadiness(suite.readiness || {})
  }

  useEffect(() => {
    if (!selectedCandidateId) return
    let cancelled = false
    setExpansionSession(null)
    setSystemMap(null)
    setPlanSuite(null)
    setLocalEntries([])
    setOriginalEntries([])
    setReadiness({})
    setError(null)
    setInvalidJsonIndices({})
    setParamsStrings({})
    setAgentFlowMap(null)
    setPlanReport(null)
    setViewingAgentId(null)

    const fetchSessionAndData = async () => {
      setLoadingSession(true)
      try {
        const sessions = await window.api.aeh.listExpansionSessions(selectedCandidateId)
        if (cancelled || sessions.length === 0) {
          setLoadingSession(false)
          return
        }
        const latest = sessions.find((s) => s.status === 'completed') || sessions[0]
        if (!latest) {
          setLoadingSession(false)
          return
        }
        setExpansionSession(latest)

        if (latest.status === 'completed') {
          const map = await window.api.aeh.getExpansionMap(latest.id)
          if (!cancelled) setSystemMap(map)

          setLoadingAgentFlow(true)
          try {
            const flowMap = await window.api.aeh.getAgentFlowMap(latest.id)
            if (!cancelled) setAgentFlowMap(flowMap)
          } catch {
            if (!cancelled) setAgentFlowMap(null)
          } finally {
            if (!cancelled) setLoadingAgentFlow(false)
          }

          // getPlan/getPlanReport resolve to null when nothing's generated yet — not an error.
          setLoadingPlan(true)
          try {
            const suite = await window.api.aeh.getPlan(latest.id)
            if (cancelled) return
            if (suite) {
              applyPlanSuite(suite)
              const report = await window.api.aeh.getPlanReport(latest.id).catch(() => null)
              if (!cancelled) setPlanReport(report)
            } else {
              setPlanSuite(null)
              setLocalEntries([])
              setOriginalEntries([])
              setPlanReport(null)
            }
          } catch (planErr: any) {
            if (!cancelled) {
              setPlanSuite(null)
              setLocalEntries([])
              setOriginalEntries([])
              setPlanReport(null)
              setError('Failed to fetch plan suite: ' + (planErr?.message ?? 'unknown error'))
            }
          } finally {
            if (!cancelled) setLoadingPlan(false)
          }
        }
      } catch (err: any) {
        console.error('Failed to load expansion session info', err)
        if (!cancelled) setError(err?.message ?? 'Failed to load expansion session.')
      } finally {
        if (!cancelled) setLoadingSession(false)
      }
    }

    fetchSessionAndData()
    return () => {
      cancelled = true
    }
  }, [selectedCandidateId])

  const hasUnsavedChanges = useMemo(() => {
    return JSON.stringify(localEntries) !== JSON.stringify(originalEntries)
  }, [localEntries, originalEntries])

  const handleGeneratePlan = async () => {
    if (!expansionSession) return
    if (!agentFlowMap) {
      toast.error('Run Stage 2 agent-flow separation first — Stage 3 plans per agent.')
      return
    }
    setGeneratingPlan(true)
    setError(null)
    setConfirmModalOpen(false)
    try {
      const suite = await window.api.aeh.generatePlan(expansionSession.id, {
        provider_id: selectedProviderId || null,
        model_id: selectedModelId || null,
      })
      toast.success('Plan suite generated successfully.')
      applyPlanSuite(suite)
      const report = await window.api.aeh.getPlanReport(expansionSession.id).catch(() => null)
      setPlanReport(report)
    } catch (err: any) {
      setError(err?.message ?? 'Failed to generate plan.')
      toast.error(err?.message ?? 'Plan generation failed.')
    } finally {
      setGeneratingPlan(false)
    }
  }

  const handleGenerateClick = () => {
    if (!agentFlowMap) {
      toast.error('Run Stage 2 agent-flow separation first — Stage 3 plans per agent.')
      return
    }
    if (hasUnsavedChanges) {
      setConfirmModalOpen(true)
    } else {
      handleGeneratePlan()
    }
  }

  const handleSaveEdits = async () => {
    if (!expansionSession) return
    if (Object.values(invalidJsonIndices).some(Boolean)) {
      toast.error('Please fix invalid JSON parameters before saving.')
      return
    }
    setSavingPlan(true)
    try {
      await window.api.aeh.updatePlan(expansionSession.id, { entries: localEntries })
      toast.success('Changes saved successfully.')
      const suite = await window.api.aeh.getPlan(expansionSession.id)
      if (suite) applyPlanSuite(suite)
    } catch (err: any) {
      toast.error(err?.message ?? 'Failed to save changes.')
    } finally {
      setSavingPlan(false)
    }
  }

  const updateEntryField = (index: number, field: keyof AEHPlanEntry, value: any) => {
    setLocalEntries((prev) =>
      prev.map((entry, idx) => (idx === index ? { ...entry, [field]: value } : entry))
    )
  }

  const handleParamsChange = (index: number, rawString: string) => {
    setParamsStrings((prev) => ({ ...prev, [index]: rawString }))
    try {
      const parsed = JSON.parse(rawString)
      setInvalidJsonIndices((prev) => ({ ...prev, [index]: false }))
      updateEntryField(index, 'params', parsed)
    } catch {
      setInvalidJsonIndices((prev) => ({ ...prev, [index]: true }))
    }
  }

  const handleAddEntry = (agentId: string, ownedComponentIds: string[]) => {
    const defaultComponent = ownedComponentIds[0] || 'unknown'
    const newEntry: AEHPlanEntry = {
      id: `custom_${Date.now()}`,
      component: defaultComponent,
      metric: '',
      metric_class: 'assertion',
      rationale: 'User added validation case',
      provenance: 'human_added',
      params: {},
      agent_id: agentId,
    }
    setLocalEntries((prev) => [...prev, newEntry])
  }

  const handleDeleteEntry = (index: number) => {
    setLocalEntries((prev) => prev.filter((_, idx) => idx !== index))
    setInvalidJsonIndices((prev) => {
      const next = { ...prev }
      delete next[index]
      return next
    })
  }

  const roleByComponentId = useMemo(() => {
    const m = new Map<string, string>()
    for (const c of systemMap?.components ?? []) m.set(c.id, c.role)
    return m
  }, [systemMap])

  // component+metric -> the DAG's report gate, for toolkit display (report-only, not persisted on the Suite).
  const gateByKey = useMemo(() => {
    const m = new Map<string, AEHEvaluationGate>()
    for (const agentReport of planReport?.agents ?? []) {
      for (const gate of agentReport.gates) m.set(`${gate.component}::${gate.metric}`, gate)
    }
    return m
  }, [planReport])

  const reportByAgentId = useMemo(
    () => new Map((planReport?.agents ?? []).map((a) => [a.agent_id, a])),
    [planReport]
  )

  const viewingAgent = useMemo(
    () => agentFlowMap?.agents.find((a) => a.id === viewingAgentId) ?? null,
    [agentFlowMap, viewingAgentId]
  )

  const visibleIndices = useMemo(() => {
    if (!viewingAgentId) return []
    return localEntries.map((_, i) => i).filter((i) => localEntries[i].agent_id === viewingAgentId)
  }, [localEntries, viewingAgentId])

  const agentFlowBlocked =
    expansionSession?.status === 'completed' && !loadingAgentFlow && agentFlowMap === null

  return (
    <div className="flex flex-col h-full bg-[#090d16] text-slate-100">
      {/* Screen Header — swaps to the agent's own identity while viewing its detail page,
          instead of stacking a second header row underneath. */}
      <div className="screen-header shrink-0 flex items-center justify-between">
        <div className="flex items-center gap-3 min-w-0">
          <Button
            variant="ghost"
            className="w-8 h-8 p-0 rounded-full flex items-center justify-center border border-slate-800 bg-slate-900/40 text-slate-400 hover:text-slate-200 shrink-0"
            onClick={() =>
              viewingAgent ? setViewingAgentId(null) : navigate(`/aeh/analysis?repoId=${repoId}&snapshotId=${snapshotId}`)
            }
            title={viewingAgent ? 'Back to agent list' : undefined}
          >
            <ArrowLeft size={16} />
          </Button>
          {viewingAgent ? (
            <div className="flex items-center gap-2 min-w-0">
              <span
                className="w-2.5 h-2.5 rounded-full shrink-0"
                style={{ background: ROLE_COLORS[viewingAgent.role] ?? ROLE_COLORS.unknown }}
              />
              <div className="min-w-0">
                <h1 className="screen-title truncate">{viewingAgent.label || viewingAgent.id}</h1>
                <p className="screen-subtitle font-mono">{viewingAgent.role}</p>
              </div>
            </div>
          ) : (
            <div>
              <h1 className="screen-title flex items-center gap-2">
                <span>Stage 3: Build Evaluation Plan</span>
              </h1>
              <p className="screen-subtitle">Per-agent evaluation gates — data-flow analysis, gate design, and reconciliation against deterministic rules</p>
            </div>
          )}
        </div>

        <div className="flex items-center gap-3 shrink-0">
          {viewingAgent ? (
            <Button
              variant="primary"
              onClick={handleSaveEdits}
              loading={savingPlan}
              disabled={!hasUnsavedChanges || Object.values(invalidJsonIndices).some(Boolean)}
              className="text-xs h-9 px-3 flex items-center gap-1.5 bg-emerald-600 hover:bg-emerald-500 disabled:opacity-40 text-white font-medium"
            >
              <Save size={13} />
              <span>Save Edits</span>
            </Button>
          ) : (
            <Select
              value={selectedCandidateId}
              onChange={(e) => setSelectedCandidateId(e.target.value)}
              disabled={loadingCandidates || candidates.length === 0}
              className="text-xs h-9 min-w-[200px]"
            >
              {candidates.length === 0 && <option value="">No confirmed candidates</option>}
              {candidates.map((c) => (
                <option key={c.id} value={c.id}>
                  {c.name}
                </option>
              ))}
            </Select>
          )}

          {!viewingAgent && (
            <Button
              variant="ghost"
              onClick={() =>
                navigate(
                  `/aeh/analysis/datasets?repoId=${repoId}&snapshotId=${snapshotId}&sessionId=${expansionSession?.id ?? ''}`
                )
              }
              disabled={!expansionSession}
              className="text-xs h-9 px-3 flex items-center gap-1.5 border border-slate-800 bg-slate-900/40 hover:bg-slate-900 text-slate-300 disabled:opacity-40"
              title="Review and fulfill datasets referenced by this plan"
            >
              <Database size={13} />
              <span>Datasets</span>
            </Button>
          )}

          <Button
            variant="ghost"
            onClick={() => setLlmConfigOpen(true)}
            className="w-9 h-9 p-0 flex items-center justify-center border border-slate-800 bg-slate-900/40 hover:bg-slate-900 text-slate-300"
            title="Configure planning LLM"
          >
            <Settings size={15} />
          </Button>
        </div>
      </div>

      {/* Body */}
      <div className="flex-1 overflow-hidden">
        {loadingSession ? (
          <div className="h-full flex flex-col items-center justify-center">
            <Loader2 className="w-10 h-10 mb-2 animate-spin text-indigo-500" />
            <p className="text-sm font-semibold text-slate-400">Loading session info...</p>
          </div>
        ) : viewingAgent && agentFlowMap && systemMap && expansionSession ? (
          <AgentDetailView
            key={viewingAgent.id}
            agent={viewingAgent}
            agentReport={reportByAgentId.get(viewingAgent.id) ?? null}
            agentFlowMap={agentFlowMap}
            systemMap={systemMap}
            expansionSession={expansionSession}
            candidate={candidate}
            localEntries={localEntries}
            visibleIndices={visibleIndices}
            paramsStrings={paramsStrings}
            invalidJsonIndices={invalidJsonIndices}
            gateByKey={gateByKey}
            roleByComponentId={roleByComponentId}
            onUpdateEntryField={updateEntryField}
            onParamsChange={handleParamsChange}
            onDeleteEntry={handleDeleteEntry}
            onAddEntry={() => handleAddEntry(viewingAgent.id, viewingAgent.component_ids)}
            onNavigateToAgent={setViewingAgentId}
            readiness={readiness}
          />
        ) : (
          <div className="h-full overflow-y-auto p-5 space-y-4">
            <div className="grid grid-cols-1 lg:grid-cols-[340px_1fr] gap-4">
              <SessionContextCard
                expansionSession={expansionSession}
                agentFlowMap={agentFlowMap}
                loadingAgentFlow={loadingAgentFlow}
                planSuite={planSuite}
                localEntriesCount={localEntries.length}
                generatingPlan={generatingPlan}
                savingPlan={savingPlan}
                hasUnsavedChanges={hasUnsavedChanges}
                invalidJsonIndices={invalidJsonIndices}
                onGenerateClick={handleGenerateClick}
                onSaveEdits={handleSaveEdits}
              />
              <CriticNotesCard planReport={planReport} />
            </div>

            {agentFlowBlocked && (
              <div className="bg-amber-950/20 border border-amber-800/40 rounded-xl p-4 text-xs text-amber-300 flex items-center justify-between gap-3">
                <div className="flex items-center gap-2">
                  <Workflow size={14} className="text-amber-400 shrink-0" />
                  <span>Stage 3 plans evaluation per agent — run agent-flow separation in Stage 2 first.</span>
                </div>
                <Button
                  variant="ghost"
                  onClick={() =>
                    navigate(`/aeh/analysis/stage2?repoId=${repoId}&snapshotId=${snapshotId}&candidateId=${selectedCandidateId}`)
                  }
                  className="text-[10px] h-7 px-2.5 border border-amber-800/50 text-amber-300 hover:bg-amber-900/20 shrink-0"
                >
                  Go to Stage 2
                </Button>
              </div>
            )}

            {error && (
              <div className="bg-rose-950/20 border border-rose-800/40 rounded-xl p-4 text-xs text-rose-300 space-y-1">
                <div className="font-semibold flex items-center gap-1.5">
                  <AlertCircle size={14} className="text-rose-400" />
                  <span>Error Occurred</span>
                </div>
                <p className="text-[11px] leading-relaxed text-rose-300/80">{error}</p>
              </div>
            )}

            {agentFlowMap && agentFlowMap.agents.length > 0 ? (
              <div className="border border-slate-850 rounded-xl bg-slate-950/20 overflow-hidden">
                <div className="px-4 py-3 border-b border-slate-850 flex items-center justify-between shrink-0">
                  <span className="text-[11px] font-bold text-slate-400 uppercase tracking-wide">
                    Agents ({agentFlowMap.agents.length})
                  </span>
                  {!planSuite && (
                    <span className="text-[10px] text-slate-600">No plan generated yet — click an agent to preview, or Generate Plan.</span>
                  )}
                </div>
                <div className="p-3 grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-3 gap-3">
                  {agentFlowMap.agents.map((agent) => (
                    <AgentCard
                      key={agent.id}
                      agent={agent}
                      report={reportByAgentId.get(agent.id) ?? null}
                      onClick={() => setViewingAgentId(agent.id)}
                    />
                  ))}
                </div>
              </div>
            ) : loadingAgentFlow || loadingPlan ? (
              <div className="flex flex-col items-center justify-center p-10 text-center gap-2">
                <Loader2 className="w-8 h-8 animate-spin text-indigo-500" />
                <p className="text-xs text-slate-500">Loading agents...</p>
              </div>
            ) : !agentFlowBlocked ? (
              <div className="flex flex-col items-center justify-center p-10 text-center gap-3">
                <Layers className="w-10 h-10 text-slate-700" />
                <p className="text-xs text-slate-500 max-w-sm leading-relaxed">
                  {expansionSession
                    ? 'No agents found for this candidate.'
                    : 'Select a confirmed candidate to build its evaluation plan.'}
                </p>
              </div>
            ) : null}
          </div>
        )}
      </div>

      <LLMConfigModal
        isOpen={llmConfigOpen}
        onClose={() => setLlmConfigOpen(false)}
        providerId={selectedProviderId}
        modelId={selectedModelId}
        onChange={(prov, model) => {
          setSelectedProviderId(prov)
          setSelectedModelId(model)
        }}
        title="Planning LLM Model"
      />

      {confirmModalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/80 backdrop-blur-sm p-4">
          <div className="glass rounded-2xl w-full max-w-sm border-slate-850 shadow-2xl p-5 space-y-4 text-slate-100">
            <div className="flex items-center gap-3 text-amber-400">
              <AlertCircle className="shrink-0 w-6 h-6" />
              <h3 className="text-sm font-semibold text-slate-200">Regenerate Evaluation Plan?</h3>
            </div>
            <p className="text-[11px] text-slate-450 leading-relaxed">
              This will regenerate from scratch. Your edits will be lost. Continue?
            </p>
            <div className="flex justify-end gap-2.5 pt-2">
              <Button variant="ghost" onClick={() => setConfirmModalOpen(false)} className="text-xs px-4 py-2">
                Cancel
              </Button>
              <Button
                variant="primary"
                onClick={handleGeneratePlan}
                className="text-xs px-4 py-2 bg-indigo-600 hover:bg-indigo-500"
              >
                Continue
              </Button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

function SessionContextCard({
  expansionSession,
  agentFlowMap,
  loadingAgentFlow,
  planSuite,
  localEntriesCount,
  generatingPlan,
  savingPlan,
  hasUnsavedChanges,
  invalidJsonIndices,
  onGenerateClick,
  onSaveEdits,
}: {
  expansionSession: AEHExpansionSession | null
  agentFlowMap: AEHAgentFlowMap | null
  loadingAgentFlow: boolean
  planSuite: AEHPlanSuite | null
  localEntriesCount: number
  generatingPlan: boolean
  savingPlan: boolean
  hasUnsavedChanges: boolean
  invalidJsonIndices: Record<number, boolean>
  onGenerateClick: () => void
  onSaveEdits: () => void
}): React.ReactElement {
  return (
    <div className="bg-slate-950/40 border border-slate-850 rounded-xl p-4 space-y-4 flex flex-col">
      <div>
        <div className="text-[11px] font-bold text-slate-500 uppercase tracking-wider mb-3">Session Context</div>
        {expansionSession ? (
          <div className="space-y-2.5 text-xs">
            <div className="flex items-center justify-between">
              <span className="text-slate-400 font-medium">Status</span>
              <Badge
                variant={
                  expansionSession.status === 'completed'
                    ? 'success'
                    : expansionSession.status === 'failed'
                      ? 'error'
                      : 'neutral'
                }
                size="sm"
              >
                {expansionSession.status}
              </Badge>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-slate-400 font-medium">Session ID</span>
              <span className="font-mono text-slate-300">{expansionSession.id.slice(0, 12)}...</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-slate-400 font-medium">Files in Blueprint</span>
              <span className="text-indigo-400 font-semibold font-mono">{expansionSession.accepted.length}</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-slate-400 font-medium">Agents</span>
              <span className="text-indigo-400 font-semibold font-mono">
                {agentFlowMap ? agentFlowMap.agents.length : loadingAgentFlow ? '…' : '—'}
              </span>
            </div>
            {planSuite && (
              <div className="flex items-center justify-between">
                <span className="text-slate-400 font-medium">Plan Size</span>
                <span className="text-slate-200 font-semibold font-mono">{localEntriesCount} entries</span>
              </div>
            )}
          </div>
        ) : (
          <div className="text-xs text-amber-400 flex items-center gap-1.5">
            <AlertCircle size={14} />
            <span>No completed expansion found.</span>
          </div>
        )}
      </div>

      <div className="space-y-2 pt-1 border-t border-slate-850">
        <Button
          variant="primary"
          onClick={onGenerateClick}
          loading={generatingPlan}
          disabled={!expansionSession || expansionSession.status !== 'completed' || !agentFlowMap}
          className="w-full text-xs h-8 flex items-center justify-center gap-1.5 bg-indigo-600 hover:bg-indigo-500 text-white font-medium"
        >
          <Sparkles size={13} />
          <span>Generate Plan</span>
        </Button>
        <Button
          variant="primary"
          onClick={onSaveEdits}
          loading={savingPlan}
          disabled={!hasUnsavedChanges || Object.values(invalidJsonIndices).some(Boolean)}
          className="w-full text-xs h-8 flex items-center justify-center gap-1.5 bg-emerald-600 hover:bg-emerald-500 disabled:opacity-40 text-white font-medium"
        >
          <Save size={13} />
          <span>Save Plan Edits</span>
        </Button>
        {hasUnsavedChanges && (
          <div className="text-center text-[10px] text-amber-400 animate-pulse">Unsaved changes detected</div>
        )}
      </div>
    </div>
  )
}

function CriticNotesCard({ planReport }: { planReport: AEHEvaluationPlanReport | null }): React.ReactElement {
  return (
    <div className="bg-slate-950/40 border border-slate-850 rounded-xl p-4">
      <div className="text-[11px] font-bold text-slate-500 uppercase tracking-wider mb-3">Critic Notes</div>
      {!planReport ? (
        <p className="text-[11px] text-slate-600 italic">Generate a plan to see the critic's review of the full gate set.</p>
      ) : planReport.advisory_notes.length === 0 ? (
        <p className="text-[11px] text-slate-600 italic">No issues flagged.</p>
      ) : (
        <ul className="space-y-1.5 text-[11px] text-violet-300/90 leading-relaxed list-disc list-inside">
          {planReport.advisory_notes.map((note, i) => (
            <li key={i}>{note}</li>
          ))}
        </ul>
      )}
    </div>
  )
}

function AgentCard({
  agent,
  report,
  onClick,
}: {
  agent: AEHAgentFlow
  report: AEHAgentPlanReport | null
  onClick: () => void
}): React.ReactElement {
  const gateCount = report?.gates.length ?? 0
  const needsHumanCount = report?.needs_human.length ?? 0
  return (
    <button
      onClick={onClick}
      className="text-left border border-slate-850 rounded-lg p-3 bg-slate-900/20 hover:border-indigo-500/50 hover:bg-slate-900/40 transition-colors space-y-1.5"
    >
      <div className="flex items-center gap-1.5 min-w-0">
        <span
          className="w-2 h-2 rounded-full shrink-0"
          style={{ background: ROLE_COLORS[agent.role] ?? ROLE_COLORS.unknown }}
        />
        <span className="text-[11px] font-semibold text-slate-200 truncate flex-1">{agent.label || agent.id}</span>
        {needsHumanCount > 0 && (
          <span title={`${needsHumanCount} gate(s) need review`}>
            <AlertCircle className="w-2.5 h-2.5 text-amber-400 shrink-0" />
          </span>
        )}
      </div>
      {agent.summary && <p className="text-[9px] text-slate-500 leading-snug line-clamp-2">{agent.summary}</p>}
      <div className="flex items-center gap-2 text-[8px] text-slate-600 font-mono pt-0.5">
        <span>{agent.component_ids.length} components</span>
        <span>·</span>
        <span>{gateCount} gates</span>
      </div>
    </button>
  )
}

/** The agent-detail "page": data-flow text + component sub-graph on top, evaluation-gates table below. Hovering/clicking a graph node highlights the matching table row via a shared highlight id. */
function AgentDetailView({
  agent,
  agentReport,
  agentFlowMap,
  systemMap,
  expansionSession,
  candidate,
  localEntries,
  visibleIndices,
  paramsStrings,
  invalidJsonIndices,
  gateByKey,
  roleByComponentId,
  onUpdateEntryField,
  onParamsChange,
  onDeleteEntry,
  onAddEntry,
  onNavigateToAgent,
  readiness,
}: {
  agent: AEHAgentFlow
  agentReport: AEHAgentPlanReport | null
  agentFlowMap: AEHAgentFlowMap
  systemMap: AEHSystemMap
  expansionSession: AEHExpansionSession
  candidate: AEHDiscoveryCandidate | null
  localEntries: AEHPlanEntry[]
  visibleIndices: number[]
  paramsStrings: Record<number, string>
  invalidJsonIndices: Record<number, boolean>
  gateByKey: Map<string, AEHEvaluationGate>
  roleByComponentId: Map<string, string>
  onUpdateEntryField: (index: number, field: keyof AEHPlanEntry, value: any) => void
  onParamsChange: (index: number, rawString: string) => void
  onDeleteEntry: (index: number) => void
  onAddEntry: () => void
  onNavigateToAgent: (agentId: string) => void
  readiness: Record<string, { status: 'runnable' | 'blocked'; reasons: string[] }>
}): React.ReactElement {
  const [selectedGraphNode, setSelectedGraphNode] = useState<string | null>(null)
  const [hoveredGraphNode, setHoveredGraphNode] = useState<string | null>(null)
  const highlightedComponentId = hoveredGraphNode ?? selectedGraphNode

  const profile = agentReport?.data_profile ?? null

  return (
    <div className="h-full overflow-hidden p-5 flex flex-col gap-4">
        {/* Top row: data-flow text (left, 3/5) + component sub-graph (right, 2/5) */}
        <div className="grid grid-cols-1 lg:grid-cols-5 gap-4 shrink-0">
          <div className="lg:col-span-3 border border-slate-850 rounded-xl bg-slate-950/20 p-4 h-[430px] min-h-0 overflow-y-auto">
            {!profile && !agentReport?.contract ? (
              <p className="text-[10px] text-slate-600 italic">No data-flow analysis available — generate a plan to populate this.</p>
            ) : (
              <div className="space-y-3 text-[10px]">
                {profile && (
                  <>
                    <div>
                      <div className="text-slate-500 font-semibold uppercase tracking-wide text-[9px] mb-0.5">Input</div>
                      <p className="text-slate-300 leading-relaxed">{profile.input_data || '—'}</p>
                    </div>
                    <div>
                      <div className="text-slate-500 font-semibold uppercase tracking-wide text-[9px] mb-0.5">Output</div>
                      <p className="text-slate-300 leading-relaxed">{profile.output_data || '—'}</p>
                    </div>
                    {profile.internal_tools.length > 0 && (
                      <div>
                        <div className="text-slate-500 font-semibold uppercase tracking-wide text-[9px] mb-0.5">Internal tools</div>
                        <ul className="text-slate-400 list-disc list-inside space-y-0.5">
                          {profile.internal_tools.map((t, i) => (
                            <li key={i}>{t}</li>
                          ))}
                        </ul>
                      </div>
                    )}
                    {profile.failure_modes.length > 0 && (
                      <div>
                        <div className="text-slate-500 font-semibold uppercase tracking-wide text-[9px] mb-0.5">Failure modes</div>
                        <ul className="text-amber-300/80 list-disc list-inside space-y-0.5">
                          {profile.failure_modes.map((f, i) => (
                            <li key={i}>{f}</li>
                          ))}
                        </ul>
                      </div>
                    )}
                    {profile.consistency_notes.length > 0 && (
                      <div>
                        <div className="text-rose-400 font-semibold uppercase tracking-wide text-[9px] mb-0.5">
                          Consistency notes (code vs. declared flow)
                        </div>
                        <ul className="text-rose-300/80 list-disc list-inside space-y-0.5">
                          {profile.consistency_notes.map((n, i) => (
                            <li key={i}>{n}</li>
                          ))}
                        </ul>
                      </div>
                    )}
                  </>
                )}

                {agentReport?.contract && (
                  <EvaluationContractPanel contract={agentReport.contract} bordered={!!profile} />
                )}
              </div>
            )}
          </div>

          <div className="lg:col-span-2 border border-slate-850 rounded-xl bg-slate-950/20 relative h-[430px] overflow-hidden">
            <AgentSubGraphPanel
              agent={agent}
              agentFlowMap={agentFlowMap}
              systemMap={systemMap}
              expansionSession={expansionSession}
              candidate={candidate}
              selectedNode={selectedGraphNode}
              onSelectNode={setSelectedGraphNode}
              onNavigateToAgent={onNavigateToAgent}
              onHoverNode={setHoveredGraphNode}
            />
          </div>
        </div>

        {/* Bottom: component list + evaluation gates — fills all remaining height. */}
        <div className="flex-1 min-h-0 border border-slate-850 rounded-xl bg-slate-950/20 overflow-hidden flex flex-col">
          <div className="px-4 py-3 border-b border-slate-850 flex items-center justify-between shrink-0">
            <div className="text-xs text-slate-400">
              Showing <span className="font-semibold text-slate-200">{visibleIndices.length}</span> assertions & judges
            </div>
            <Button
              variant="ghost"
              onClick={onAddEntry}
              className="text-[10px] h-7 px-3 flex items-center gap-1 border border-slate-800 bg-slate-950 text-indigo-400 hover:text-indigo-300 hover:border-slate-700"
            >
              <Plus size={12} />
              <span>Add Assertion</span>
            </Button>
          </div>

          <div className="flex-1 min-h-0 overflow-y-auto overscroll-contain">
            <table className="w-full text-left border-collapse text-xs table-fixed">
              <thead className="sticky top-0 z-10">
                <tr className="border-b border-slate-850 bg-slate-900/90 backdrop-blur text-slate-400 font-medium">
                  <th className="p-3 w-1/4">Component Target</th>
                  <th className="p-3 w-1/5">Metric / Rubric</th>
                  <th className="p-3 w-[120px]">Class</th>
                  <th className="p-3 w-1/4">Rationale</th>
                  <th className="p-3 w-1/6">Params (JSON)</th>
                  <th className="p-3 w-[110px]">Readiness</th>
                  <th className="p-3 w-[100px]">Provenance</th>
                  <th className="p-3 w-[60px] text-right">Delete</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-850">
                {visibleIndices.length === 0 && (
                  <tr>
                    <td colSpan={8} className="p-6 text-center text-slate-600 text-[11px]">
                      No gates for this agent yet.
                    </td>
                  </tr>
                )}
                {visibleIndices.map((index) => {
                  const entry = localEntries[index]
                  const isJsonInvalid = invalidJsonIndices[index] || false
                  const gate = gateByKey.get(`${entry.component}::${entry.metric}`)
                  const needsHuman = entry.status === 'needs_human'
                  const isHighlighted = entry.component === highlightedComponentId
                  return (
                    <tr
                      key={entry.id}
                      className={`transition-colors ${isHighlighted ? 'bg-indigo-950/40' : 'hover:bg-slate-900/25'}`}
                    >
                      <td className="p-3 align-top">
                        <div className="flex items-center gap-1.5">
                          <span
                            className="w-2 h-2 rounded-full shrink-0"
                            style={{
                              background: ROLE_COLORS[roleByComponentId.get(entry.component) ?? 'unknown'] ?? ROLE_COLORS.unknown,
                            }}
                            title={roleByComponentId.get(entry.component) ?? 'unknown'}
                          />
                          <Select
                            value={entry.component}
                            onChange={(e) => onUpdateEntryField(index, 'component', e.target.value)}
                            className="text-[11px] h-8 px-2 py-0.5 flex-1"
                          >
                            {agent.component_ids.map((opt) => (
                              <option key={opt} value={opt}>
                                {opt}
                              </option>
                            ))}
                          </Select>
                        </div>
                      </td>

                      <td className="p-3 align-top">
                        <input
                          type="text"
                          value={entry.metric}
                          onChange={(e) => onUpdateEntryField(index, 'metric', e.target.value)}
                          className="w-full text-[11px] bg-slate-950 border border-slate-800 rounded px-2 py-1 text-slate-200 focus:border-slate-600 focus:outline-none"
                          placeholder="Metric name"
                        />
                        {gate && (
                          <span className="block mt-1 text-[9px] text-slate-500 font-mono">
                            {TOOLKIT_LABELS[gate.toolkit] ?? gate.toolkit}
                            {gate.location && <span className="text-slate-600"> · {gate.location}</span>}
                          </span>
                        )}
                      </td>

                      <td className="p-3 align-top">
                        <Select
                          value={entry.metric_class}
                          onChange={(e) => onUpdateEntryField(index, 'metric_class', e.target.value)}
                          className={`text-[11px] h-8 px-2 py-0.5 font-medium ${METRIC_CLASS_STYLES[entry.metric_class] ?? ''}`}
                        >
                          <option value="assertion">Assertion</option>
                          <option value="classifier">Classifier</option>
                          <option value="llm_judge">LLM Judge</option>
                        </Select>
                      </td>

                      <td className="p-3 align-top">
                        <textarea
                          value={entry.rationale}
                          onChange={(e) => onUpdateEntryField(index, 'rationale', e.target.value)}
                          rows={2}
                          className="w-full text-[11px] bg-slate-950 border border-slate-800 rounded px-2 py-1 text-slate-300 focus:border-slate-600 focus:outline-none resize-y"
                          placeholder="Explain validation intent"
                        />
                      </td>

                      <td className="p-3 align-top">
                        <textarea
                          value={paramsStrings[index] ?? JSON.stringify(entry.params || {})}
                          onChange={(e) => onParamsChange(index, e.target.value)}
                          rows={2}
                          className={`w-full font-mono text-[10px] bg-slate-950 border rounded px-2 py-1 text-slate-400 focus:outline-none ${
                            isJsonInvalid ? 'border-red-500/80 focus:border-red-500' : 'border-slate-800 focus:border-slate-600'
                          }`}
                          placeholder="{}"
                        />
                      </td>

                      <td className="p-3 align-top pt-4">
                        {(() => {
                          const read = readiness[entry.id]
                          if (!read) {
                            return (
                              <span title="Save plan to compute readiness">
                                <Badge variant="neutral" size="sm" className="font-normal text-[10px] px-2 py-0.5">
                                  unsaved
                                </Badge>
                              </span>
                            )
                          }
                          const isBlocked = read.status === 'blocked'
                          return (
                            <span title={isBlocked ? `Blocked: ${read.reasons.join(', ')}` : 'Ready to run'} className="cursor-help">
                              <Badge
                                variant={isBlocked ? 'warning' : 'success'}
                                size="sm"
                                className="font-normal text-[10px] px-2 py-0.5"
                              >
                                {read.status}
                              </Badge>
                            </span>
                          )
                        })()}
                      </td>

                      <td className="p-3 align-top pt-4">
                        <div className="flex flex-col items-start gap-1">
                          <Badge
                            variant={
                              entry.provenance === 'human_added'
                                ? 'success'
                                : entry.provenance === 'llm_suggested'
                                ? 'info'
                                : 'neutral'
                            }
                            size="sm"
                            className="capitalize font-normal text-[10px] px-2 py-0.5"
                          >
                            {entry.provenance === 'human_added'
                              ? 'Human added'
                              : entry.provenance === 'llm_suggested'
                              ? 'AI suggested'
                              : 'Static rule'}
                          </Badge>
                          {needsHuman && (
                            <span
                              className="flex items-center gap-1 text-[9px] text-amber-400"
                              title="Metric not recognized by the registry — review before running."
                            >
                              <AlertCircle size={10} />
                              needs review
                            </span>
                          )}
                        </div>
                      </td>

                      <td className="p-3 align-top pt-3 text-right">
                        <button
                          onClick={() => onDeleteEntry(index)}
                          className="text-slate-500 hover:text-rose-400 transition-colors p-1"
                          title="Remove test entry"
                        >
                          <Trash2 size={13} />
                        </button>
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </div>
      </div>
  )
}

/** Statically harvested per-agent contract: what Stage 4 will invoke/consume. */
function EvaluationContractPanel({
  contract,
  bordered,
}: {
  contract: AEHEvaluationContract
  bordered: boolean
}): React.ReactElement {
  const obs = contract.observability
  return (
    <div className={bordered ? 'pt-2.5 mt-1 border-t border-slate-850/60 space-y-2.5' : 'space-y-2.5'}>
      <div className="text-slate-500 font-semibold uppercase tracking-wide text-[9px]">
        Evaluation Contract <span className="text-slate-700 normal-case">(statically harvested, for Stage 4)</span>
      </div>

      {contract.invocation ? (
        <div>
          <div className="text-slate-500 text-[9px] mb-0.5">Invocation</div>
          <p className="text-slate-300 font-mono text-[10px] break-all">
            {contract.invocation.callable}
            <span className="text-slate-500">.{contract.invocation.method}(...)</span>
          </p>
          {contract.invocation.kwargs.length > 0 && (
            <ul className="text-slate-400 mt-1 space-y-0.5">
              {contract.invocation.kwargs.map((k) => (
                <li key={k.name} className="font-mono">
                  {k.name}
                  {k.annotation ? <span className="text-slate-600">: {k.annotation}</span> : null}
                  {!k.required ? (
                    <span className="text-slate-600"> = {k.default_repr ?? '…'} (optional)</span>
                  ) : null}
                </li>
              ))}
            </ul>
          )}
          <span
            title="How Stage 4 would drive this agent — 'unsupported' means it needs live constructor dependencies AEH cannot supply on its own."
            className="cursor-help inline-block"
          >
            <Badge
              variant={contract.invocation.invocation_mode === 'unsupported' ? 'warning' : 'success'}
              size="sm"
              className="mt-1 font-normal text-[9px]"
            >
              {contract.invocation.invocation_mode}
            </Badge>
          </span>
        </div>
      ) : (
        <p className="text-[10px] text-slate-600 italic">Invocation signature not statically harvestable.</p>
      )}

      <div>
        <div className="text-slate-500 text-[9px] mb-0.5">Output</div>
        <p className="text-slate-400 leading-relaxed">
          schema: {contract.output?.json_schema ? 'harvested' : '—'}
          {contract.output?.schema_source ? <span className="text-slate-600 font-mono"> ({contract.output.schema_source})</span> : null}
          {' · '}
          fallback: {contract.output?.fallback_literal ? 'harvested' : '—'}
          {' · '}
          validated in target: {contract.output?.validated_in_target ? 'yes' : 'no'}
        </p>
      </div>

      <div>
        <div className="text-slate-500 text-[9px] mb-0.5">Observability</div>
        <div className="flex flex-wrap gap-1.5">
          <Badge size="sm" variant={obs.has_tools ? 'info' : 'neutral'} className="text-[9px] font-normal">
            has_tools: {String(obs.has_tools)}
          </Badge>
          <Badge size="sm" variant={obs.has_separable_context ? 'info' : 'neutral'} className="text-[9px] font-normal">
            separable_context: {obs.has_separable_context === null ? 'unknown' : String(obs.has_separable_context)}
          </Badge>
          <Badge size="sm" variant="neutral" className="text-[9px] font-normal">
            input_kind: {obs.input_kind}
          </Badge>
          {obs.llm_call_budget !== null && (
            <Badge size="sm" variant="neutral" className="text-[9px] font-normal">
              llm_call_budget: {obs.llm_call_budget}
            </Badge>
          )}
        </div>
        {obs.llm_fields.length > 0 && (
          <p className="text-slate-600 text-[9px] mt-1">LLM-inferred (no static evidence): {obs.llm_fields.join(', ')}</p>
        )}
      </div>

      {contract.needs_human.length > 0 && (
        <div>
          <div className="text-amber-400 font-semibold uppercase tracking-wide text-[9px] mb-0.5">Harvest notes</div>
          <ul className="text-amber-300/80 list-disc list-inside space-y-0.5">
            {contract.needs_human.map((n, i) => (
              <li key={i}>{n}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}


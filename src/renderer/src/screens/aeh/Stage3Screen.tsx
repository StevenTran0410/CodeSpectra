import React, { useState, useEffect, useCallback, useMemo } from 'react'
import { useSearchParams, useNavigate } from 'react-router-dom'
import {
  Loader2,
  AlertCircle,
  Sparkles,
  Save,
  CheckCircle2,
  ArrowLeft,
  Settings,
  Plus,
  Trash2,
  HelpCircle,
  Layers,
} from 'lucide-react'
import { Button, Select, Badge, useToastStore } from '../../components/ui'
import { useProviderStore } from '../../store/provider.store'
import LLMConfigModal from './LLMConfigModal'

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

  // Plan data states
  const [planSuite, setPlanSuite] = useState<AEHPlanSuite | null>(null)
  const [localEntries, setLocalEntries] = useState<AEHPlanEntry[]>([])
  const [originalEntries, setOriginalEntries] = useState<AEHPlanEntry[]>([])
  const [loadingPlan, setLoadingPlan] = useState(false)
  const [generatingPlan, setGeneratingPlan] = useState(false)
  const [savingPlan, setSavingPlan] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Controlled params string state (one per entry index)
  const [paramsStrings, setParamsStrings] = useState<Record<number, string>>({})

  // Invalid JSON param indices to block save
  const [invalidJsonIndices, setInvalidJsonIndices] = useState<Record<number, boolean>>({})

  // LLM Config
  const [selectedProviderId, setSelectedProviderId] = useState('')
  const [selectedModelId, setSelectedModelId] = useState('')
  const [llmConfigOpen, setLlmConfigOpen] = useState(false)

  // Regenerate Confirmation Modal
  const [confirmModalOpen, setConfirmModalOpen] = useState(false)

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
    if (snapshotId) {
      loadConfirmedCandidates()
    }
  }, [snapshotId, loadConfirmedCandidates])

  // Fetch expansion session, system map and plan for candidate
  useEffect(() => {
    if (!selectedCandidateId) return
    let cancelled = false
    setExpansionSession(null)
    setSystemMap(null)
    setPlanSuite(null)
    setLocalEntries([])
    setOriginalEntries([])
    setError(null)
    setInvalidJsonIndices({})
    setParamsStrings({})

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
          // Fetch map
          const map = await window.api.aeh.getExpansionMap(latest.id)
          if (!cancelled) setSystemMap(map)

          // Fetch plan
          setLoadingPlan(true)
          try {
            const suite = await window.api.aeh.getPlan(latest.id)
            if (!cancelled) {
              setPlanSuite(suite)
              setLocalEntries(suite.entries || [])
              setOriginalEntries(JSON.parse(JSON.stringify(suite.entries || [])))
              const ps: Record<number, string> = {}
              ;(suite.entries || []).forEach((e: AEHPlanEntry, i: number) => { ps[i] = JSON.stringify(e.params || {}) })
              setParamsStrings(ps)
            }
          } catch (planErr: any) {
            // Plan not found or error loading
            if (!cancelled) {
              setPlanSuite(null)
              setLocalEntries([])
              setOriginalEntries([])
              if (planErr.message && !planErr.message.includes('404')) {
                setError('Failed to fetch plan suite: ' + planErr.message)
              }
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

  // Trigger Plan Generation
  const handleGeneratePlan = async () => {
    if (!expansionSession) return
    setGeneratingPlan(true)
    setError(null)
    setConfirmModalOpen(false)
    try {
      const suite = await window.api.aeh.generatePlan(expansionSession.id, {
        provider_id: selectedProviderId || null,
        model_id: selectedModelId || null,
      })
      toast.success('Plan suite generated successfully.')
      setPlanSuite(suite)
      setLocalEntries(suite.entries || [])
      setOriginalEntries(JSON.parse(JSON.stringify(suite.entries || [])))
      setInvalidJsonIndices({})
      const ps: Record<number, string> = {}
      ;(suite.entries || []).forEach((e: AEHPlanEntry, i: number) => { ps[i] = JSON.stringify(e.params || {}) })
      setParamsStrings(ps)
    } catch (err: any) {
      setError(err?.message ?? 'Failed to generate plan.')
      toast.error(err?.message ?? 'Plan generation failed.')
    } finally {
      setGeneratingPlan(false)
    }
  }

  const handleGenerateClick = () => {
    if (hasUnsavedChanges) {
      setConfirmModalOpen(true)
    } else {
      handleGeneratePlan()
    }
  }

  // Trigger Save Edits
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

      // Re-fetch plan to refresh provenance values
      const suite = await window.api.aeh.getPlan(expansionSession.id)
      setPlanSuite(suite)
      setLocalEntries(suite.entries || [])
      setOriginalEntries(JSON.parse(JSON.stringify(suite.entries || [])))
      setInvalidJsonIndices({})
      const ps: Record<number, string> = {}
      ;(suite.entries || []).forEach((e: AEHPlanEntry, i: number) => { ps[i] = JSON.stringify(e.params || {}) })
      setParamsStrings(ps)
    } catch (err: any) {
      toast.error(err?.message ?? 'Failed to save changes.')
    } finally {
      setSavingPlan(false)
    }
  }

  // Edit fields inline
  const updateEntryField = (index: number, field: keyof AEHPlanEntry, value: any) => {
    setLocalEntries((prev) =>
      prev.map((entry, idx) => (idx === index ? { ...entry, [field]: value } : entry))
    )
  }

  // JSON params change handling
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

  // Add Custom Entry
  const handleAddEntry = () => {
    const componentOptions = systemMap?.components.map((c) => c.id) || []
    const defaultComponent = componentOptions[0] || 'unknown'
    const newEntry: AEHPlanEntry = {
      id: `custom_${Date.now()}`,
      component: defaultComponent,
      metric: '',
      metric_class: 'assertion',
      rationale: 'User added validation case',
      provenance: 'human_added',
      params: {},
    }
    setLocalEntries((prev) => [...prev, newEntry])
  }

  // Delete Entry
  const handleDeleteEntry = (index: number) => {
    setLocalEntries((prev) => prev.filter((_, idx) => idx !== index))
    setInvalidJsonIndices((prev) => {
      const next = { ...prev }
      delete next[index]
      return next
    })
  }

  const componentOptions = useMemo(() => {
    return systemMap?.components.map((c) => c.id) || []
  }, [systemMap])

  return (
    <div className="flex flex-col h-full bg-[#090d16] text-slate-100">
      {/* Screen Header */}
      <div className="screen-header shrink-0 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Button
            variant="ghost"
            className="w-8 h-8 p-0 rounded-full flex items-center justify-center border border-slate-800 bg-slate-900/40 text-slate-400 hover:text-slate-200"
            onClick={() => navigate(`/aeh/analysis?repoId=${repoId}&snapshotId=${snapshotId}`)}
          >
            <ArrowLeft size={16} />
          </Button>
          <div>
            <h1 className="screen-title flex items-center gap-2">
              <span>Stage 3: Build Evaluation Plan</span>
            </h1>
            <p className="screen-subtitle">Propose, customize, and approve component-level evaluation assertions & judges</p>
          </div>
        </div>

        <div className="flex items-center gap-3">
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

      {/* Main Workspace Area */}
      <div className="flex-1 overflow-hidden flex flex-row">
        {/* Left Side: Planning Controls / State card */}
        <div className="w-80 border-r border-slate-850 p-5 flex flex-col justify-between bg-slate-900/10 shrink-0">
          <div className="space-y-4">
            <div className="bg-slate-950/40 border border-slate-850 rounded-xl p-4 space-y-3">
              <div className="text-[11px] font-bold text-slate-500 uppercase tracking-wider">Session Context</div>
              {loadingSession ? (
                <div className="flex items-center gap-2 text-xs text-slate-400">
                  <Loader2 className="animate-spin w-4 h-4 text-indigo-500" />
                  <span>Loading session info...</span>
                </div>
              ) : expansionSession ? (
                <div className="space-y-2 text-xs">
                  <div>
                    <span className="text-slate-400 font-medium">Session ID: </span>
                    <span className="font-mono text-slate-300">{expansionSession.id.slice(0, 12)}...</span>
                  </div>
                  <div>
                    <span className="text-slate-400 font-medium">Files in Blueprint: </span>
                    <span className="text-indigo-400 font-semibold font-mono">{expansionSession.accepted.length} files</span>
                  </div>
                  <div>
                    <span className="text-slate-400 font-medium">Expansion Status: </span>
                    <span className="text-emerald-400 font-medium capitalize">{expansionSession.status}</span>
                  </div>
                  {planSuite && (
                    <div>
                      <span className="text-slate-400 font-medium">Plan Size: </span>
                      <span className="text-slate-200 font-semibold font-mono">{localEntries.length} entries</span>
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

            {error && (
              <div className="bg-rose-950/20 border border-rose-800/40 rounded-xl p-4 text-xs text-rose-300 space-y-1">
                <div className="font-semibold flex items-center gap-1.5">
                  <AlertCircle size={14} className="text-rose-400" />
                  <span>Error Occurred</span>
                </div>
                <p className="text-[11px] leading-relaxed text-rose-300/80">{error}</p>
              </div>
            )}
          </div>

          <div className="space-y-2.5">
            <Button
              variant="primary"
              onClick={handleGenerateClick}
              loading={generatingPlan}
              disabled={!expansionSession || expansionSession.status !== 'completed'}
              className="w-full text-xs h-9 flex items-center justify-center gap-1.5 bg-indigo-600 hover:bg-indigo-500 text-white font-medium shadow-lg"
            >
              <Sparkles size={14} />
              <span>Generate Plan</span>
            </Button>

            <Button
              variant="primary"
              onClick={handleSaveEdits}
              loading={savingPlan}
              disabled={!hasUnsavedChanges || Object.values(invalidJsonIndices).some(Boolean)}
              className="w-full text-xs h-9 flex items-center justify-center gap-1.5 bg-emerald-600 hover:bg-emerald-500 disabled:opacity-40 disabled:hover:bg-emerald-600/40 text-white font-medium shadow-lg"
            >
              <Save size={14} />
              <span>Save Plan Edits</span>
            </Button>

            {hasUnsavedChanges && (
              <div className="text-center text-[10px] text-amber-400 animate-pulse">
                Unsaved changes detected
              </div>
            )}
          </div>
        </div>

        {/* Right Side: Evaluation Suite Table */}
        <div className="flex-1 overflow-hidden flex flex-col bg-slate-950/15">
          {loadingPlan ? (
            <div className="flex-1 flex flex-col items-center justify-center">
              <Loader2 className="w-10 h-10 mb-2 animate-spin text-indigo-500" />
              <p className="text-sm font-semibold text-slate-400">Loading plan entries...</p>
            </div>
          ) : localEntries.length === 0 ? (
            <div className="flex-1 flex flex-col items-center justify-center p-6 text-center space-y-4">
              <Layers className="w-12 h-12 text-slate-700" />
              <div className="max-w-sm space-y-1">
                <h3 className="font-semibold text-slate-300">No Evaluation Plan Generated</h3>
                <p className="text-xs text-slate-500 leading-relaxed">
                  Generate a blueprint-guided evaluation plan to automatically tailor GEval assertions and classifiers.
                </p>
              </div>
              <Button
                variant="primary"
                onClick={handleGenerateClick}
                disabled={!expansionSession || expansionSession.status !== 'completed'}
                className="text-xs h-8 px-4"
              >
                <Sparkles size={13} className="mr-1.5" />
                <span>Generate Initial Plan</span>
              </Button>
            </div>
          ) : (
            <div className="flex-1 overflow-hidden flex flex-col">
              {/* Table Toolbar */}
              <div className="px-6 py-3 border-b border-slate-850 flex items-center justify-between shrink-0 bg-slate-900/10">
                <div className="text-xs text-slate-400">
                  Showing <span className="font-semibold text-slate-200">{localEntries.length}</span> assertions & judges
                </div>
                <Button
                  variant="ghost"
                  onClick={handleAddEntry}
                  className="text-[10px] h-7 px-3 flex items-center gap-1 border border-slate-800 bg-slate-950 text-indigo-400 hover:text-indigo-300 hover:border-slate-700"
                >
                  <Plus size={12} />
                  <span>Add Assertion</span>
                </Button>
              </div>

              {/* Table Viewport */}
              <div className="flex-1 overflow-y-auto px-6 py-4">
                <div className="border border-slate-850 rounded-xl overflow-hidden bg-slate-950/20">
                  <table className="w-full text-left border-collapse text-xs table-fixed">
                    <thead>
                      <tr className="border-b border-slate-850 bg-slate-900/30 text-slate-400 font-medium">
                        <th className="p-3 w-1/4">Component Target</th>
                        <th className="p-3 w-1/5">Metric / Rubric</th>
                        <th className="p-3 w-[120px]">Class</th>
                        <th className="p-3 w-1/4">Rationale</th>
                        <th className="p-3 w-1/6">Params (JSON)</th>
                        <th className="p-3 w-[100px]">Provenance</th>
                        <th className="p-3 w-[60px] text-right">Delete</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-850">
                      {localEntries.map((entry, index) => {
                        const isJsonInvalid = invalidJsonIndices[index] || false
                        return (
                          <tr key={entry.id} className="hover:bg-slate-900/25 transition-colors">
                            {/* Component Selector */}
                            <td className="p-3 align-top">
                              {componentOptions.length > 0 ? (
                                <Select
                                  value={entry.component}
                                  onChange={(e) => updateEntryField(index, 'component', e.target.value)}
                                  className="text-[11px] h-8 px-2 py-0.5"
                                >
                                  {componentOptions.map((opt) => (
                                    <option key={opt} value={opt}>
                                      {opt}
                                    </option>
                                  ))}
                                </Select>
                              ) : (
                                <span className="font-mono text-slate-400">{entry.component}</span>
                              )}
                            </td>

                            {/* Metric Name */}
                            <td className="p-3 align-top">
                              <input
                                type="text"
                                value={entry.metric}
                                onChange={(e) => updateEntryField(index, 'metric', e.target.value)}
                                className="w-full text-[11px] bg-slate-950 border border-slate-800 rounded px-2 py-1 text-slate-200 focus:border-slate-600 focus:outline-none"
                                placeholder="Metric name"
                              />
                            </td>

                            {/* Metric Class */}
                            <td className="p-3 align-top">
                              <Select
                                value={entry.metric_class}
                                onChange={(e) => updateEntryField(index, 'metric_class', e.target.value)}
                                className="text-[11px] h-8 px-2 py-0.5"
                              >
                                <option value="assertion">Assertion</option>
                                <option value="classifier">Classifier</option>
                                <option value="llm_judge">LLM Judge</option>
                              </Select>
                            </td>

                            {/* Rationale */}
                            <td className="p-3 align-top">
                              <textarea
                                value={entry.rationale}
                                onChange={(e) => updateEntryField(index, 'rationale', e.target.value)}
                                rows={2}
                                className="w-full text-[11px] bg-slate-950 border border-slate-800 rounded px-2 py-1 text-slate-300 focus:border-slate-600 focus:outline-none resize-y"
                                placeholder="Explain validation intent"
                              />
                            </td>

                            {/* Params JSON */}
                            <td className="p-3 align-top">
                              <textarea
                                value={paramsStrings[index] ?? JSON.stringify(entry.params || {})}
                                onChange={(e) => handleParamsChange(index, e.target.value)}
                                rows={2}
                                className={`w-full font-mono text-[10px] bg-slate-950 border rounded px-2 py-1 text-slate-400 focus:outline-none ${
                                  isJsonInvalid ? 'border-red-500/80 focus:border-red-500' : 'border-slate-800 focus:border-slate-600'
                                }`}
                                placeholder="{}"
                              />
                            </td>

                            {/* Provenance */}
                            <td className="p-3 align-top pt-4">
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
                            </td>

                            {/* Delete Action */}
                            <td className="p-3 align-top pt-3 text-right">
                              <button
                                onClick={() => handleDeleteEntry(index)}
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
          )}
        </div>
      </div>

      {/* LLMConfigModal */}
      <LLMConfigModal
        isOpen={llmConfigOpen}
        onClose={() => setLlmConfigOpen(false)}
        providerId={selectedProviderId}
        modelId={selectedModelId}
        onChange={(prov, model) => {
          setSelectedProviderId(prov)
          setSelectedModelId(model)
        }}
        title="Planning LLM Model (CS-273)"
      />

      {/* Confirmation Modal */}
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
              <Button
                variant="ghost"
                onClick={() => setConfirmModalOpen(false)}
                className="text-xs px-4 py-2"
              >
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

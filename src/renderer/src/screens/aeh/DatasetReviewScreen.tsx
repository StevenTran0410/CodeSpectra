import React, { useEffect, useState, useCallback } from 'react'
import { useSearchParams, useNavigate } from 'react-router-dom'
import { Loader2, Sparkles, ArrowLeft, Settings, AlertCircle, RefreshCw } from 'lucide-react'
import { Button, Badge, useToastStore } from '../../components/ui'
import { useProviderStore } from '../../store/provider.store'
import LLMConfigModal, { LLMModelButton } from './LLMConfigModal'
import EmbeddingConfigModal, { EmbeddingModelButton } from './EmbeddingConfigModal'

type EditStrings = { input: string; expected: string; labels: string }

// Deterministic, zero-LLM dataset kinds have nothing for a human to accept/reject, so they're excluded from this review list; fail-open, unknown kinds show.
const NON_REVIEWABLE_KINDS = new Set([
  'snapshot_fixture',
  'field_match_gold',
  'snapshot_regression_baseline',
  'recorded_report_replay'
])

export default function DatasetReviewScreen(): React.ReactElement {
  const navigate = useNavigate()
  const [searchParams] = useSearchParams()
  const sessionId = searchParams.get('sessionId')
  const repoId = searchParams.get('repoId')
  const snapshotId = searchParams.get('snapshotId')
  const agentId = searchParams.get('agentId')
  const toast = useToastStore()

  const { providers, load: loadProviders } = useProviderStore()
  const [selectedProviderId, setSelectedProviderId] = useState('')
  const [selectedModelId, setSelectedModelId] = useState('')
  const [selectedReasoningEffort, setSelectedReasoningEffort] = useState<string | null>(null)
  const [selectedThinkingBudget, setSelectedThinkingBudget] = useState<number | null>(null)
  const [llmConfigOpen, setLlmConfigOpen] = useState(false)

  const [selectedEmbedProviderId, setSelectedEmbedProviderId] = useState('')
  const [selectedEmbedModelId, setSelectedEmbedModelId] = useState('')
  const [selectedUseLocalEmbedding, setSelectedUseLocalEmbedding] = useState(false)
  const [embedConfigOpen, setEmbedConfigOpen] = useState(false)

  useEffect(() => {
    loadProviders()
  }, [loadProviders])

  useEffect(() => {
    if (providers.length > 0 && !selectedProviderId) {
      setSelectedProviderId(providers[0].id)
      setSelectedModelId(providers[0].model_id || '')
    }
  }, [providers, selectedProviderId])

  useEffect(() => {
    if (providers.length === 0 || selectedEmbedProviderId || selectedUseLocalEmbedding) return
    let cancelled = false
    ;(async () => {
      // Default to the first cloud provider whose endpoint ACTUALLY serves embeddings, using a real
      // model it exposes — never a hardcoded name that the endpoint may not have.
      const candidates = providers.filter((p) => p.kind === 'openai' || p.kind === 'gemini')
      for (const p of candidates) {
        try {
          const { models } = await window.api.provider.embeddingModels(p.id)
          if (cancelled) return
          if (models.length > 0) {
            setSelectedEmbedProviderId(p.id)
            setSelectedEmbedModelId(models.find((m) => m === 'text-embedding-3-small') || models[0])
            return
          }
        } catch {
          /* endpoint has no embeddings — try the next provider */
        }
      }
      // No cloud embedding endpoint — fall back to the local GPU model if one is available.
      try {
        const status = await window.api.localEmbedding.status()
        if (!cancelled && status.gpu_available) setSelectedUseLocalEmbedding(true)
      } catch {
        /* leave unset — the modal will show the no-embeddings state */
      }
    })()
    return () => {
      cancelled = true
    }
  }, [providers, selectedEmbedProviderId, selectedUseLocalEmbedding])

  const [datasets, setDatasets] = useState<AEHDatasetSummary[]>([])
  const [loadingDatasets, setLoadingDatasets] = useState(false)
  const [selectedDatasetId, setSelectedDatasetId] = useState<string | null>(null)
  const [cases, setCases] = useState<AEHDatasetCase[]>([])
  const [loadingCases, setLoadingCases] = useState(false)
  const [editStrings, setEditStrings] = useState<Record<string, EditStrings>>({})
  const [fulfilling, setFulfilling] = useState(false)
  const [fulfillmentReport, setFulfillmentReport] = useState<Record<
    string,
    AEHFulfillmentGroupResult
  > | null>(null)

  // Scoped mode (agentId present): the dataset this agent's plan entry already points to, resolved from the plan itself.
  const [scopedDatasetId, setScopedDatasetId] = useState<string | null>(null)
  const [scopedLookupDone, setScopedLookupDone] = useState(false)
  const [agentLabel, setAgentLabel] = useState<string | null>(null)
  const [regenerateModalOpen, setRegenerateModalOpen] = useState(false)

  const loadDatasets = useCallback(async () => {
    setLoadingDatasets(true)
    try {
      setDatasets(await window.api.aeh.listDatasets(sessionId ?? undefined))
    } catch (err: any) {
      toast.error(err?.message ?? 'Failed to load datasets.')
    } finally {
      setLoadingDatasets(false)
    }
  }, [toast, sessionId])

  useEffect(() => {
    loadDatasets()
  }, [loadDatasets])

  const refreshScopedDataset = useCallback(async () => {
    if (!agentId || !sessionId) {
      setScopedLookupDone(true)
      return
    }
    try {
      const suite = await window.api.aeh.getPlan(sessionId)
      const entry = suite?.entries.find(
        (e) => e.agent_id === agentId && e.id.endsWith('.synthetic_agent_io') && !!e.dataset?.ref
      )
      setScopedDatasetId(entry?.dataset?.ref ?? null)
    } catch {
      setScopedDatasetId(null)
    } finally {
      setScopedLookupDone(true)
    }
  }, [agentId, sessionId])

  useEffect(() => {
    refreshScopedDataset()
  }, [refreshScopedDataset])

  useEffect(() => {
    if (!agentId || !sessionId) return
    window.api.aeh
      .getAgentFlowMap(sessionId)
      .then((flowMap) => {
        const agent = flowMap?.agents.find((a) => a.id === agentId)
        setAgentLabel(agent?.label || agent?.id || null)
      })
      .catch(() => {})
  }, [agentId, sessionId])

  useEffect(() => {
    if (agentId && scopedDatasetId && datasets.some((d) => d.dataset_id === scopedDatasetId)) {
      setSelectedDatasetId(scopedDatasetId)
    }
  }, [agentId, scopedDatasetId, datasets])

  const loadCases = useCallback(
    async (datasetId: string) => {
      setLoadingCases(true)
      try {
        const list = await window.api.aeh.getDatasetCases(datasetId)
        setCases(list)
        const strings: Record<string, EditStrings> = {}
        for (const c of list) {
          strings[c.id] = {
            input: c.input_json,
            expected: c.expected_json ?? '',
            labels: c.labels_json ?? ''
          }
        }
        setEditStrings(strings)
      } catch (err: any) {
        toast.error(err?.message ?? 'Failed to load cases.')
      } finally {
        setLoadingCases(false)
      }
    },
    [toast]
  )

  useEffect(() => {
    if (selectedDatasetId) loadCases(selectedDatasetId)
  }, [selectedDatasetId, loadCases])

  const handleVerdict = async (caseId: string, verdict: 'accept' | 'edit' | 'reject'): Promise<void> => {
    const strings = editStrings[caseId]
    const body: {
      verdict: 'accept' | 'edit' | 'reject'
      input_json?: Record<string, any>
      expected_json?: Record<string, any>
      labels_json?: Record<string, any>
    } = { verdict }
    if (verdict === 'edit' && strings) {
      const parseField = (raw: string, label: string, required: boolean): [boolean, any] => {
        if (!required && !raw) return [true, undefined]
        try {
          return [true, JSON.parse(raw)]
        } catch {
          toast.error(`${label} JSON is invalid — fix it before saving.`)
          return [false, undefined]
        }
      }
      const [inputOk, inputVal] = parseField(strings.input, 'Input', true)
      if (!inputOk) return
      body.input_json = inputVal
      const [expectedOk, expectedVal] = parseField(strings.expected, 'Expected', false)
      if (!expectedOk) return
      body.expected_json = expectedVal
      const [labelsOk, labelsVal] = parseField(strings.labels, 'Labels', false)
      if (!labelsOk) return
      body.labels_json = labelsVal
    }

    try {
      const result = await window.api.aeh.caseVerdict(caseId, body)
      if (verdict === 'reject' && result.shortfall) {
        toast.error(`Dataset is now short ${result.shortfall} case(s) below min_cases.`)
      } else {
        toast.success(`Case ${verdict}ed.`)
      }
      if (selectedDatasetId) {
        await loadCases(selectedDatasetId)
        await loadDatasets()
      }
    } catch (err: any) {
      toast.error(err?.message ?? `Failed to ${verdict} case.`)
    }
  }

  const handleBulkAccept = async (): Promise<void> => {
    const synthetic = cases.filter((c) => c.provenance === 'synthetic')
    for (const c of synthetic) {
      await window.api.aeh.caseVerdict(c.id, { verdict: 'accept' })
    }
    if (selectedDatasetId) {
      await loadCases(selectedDatasetId)
      await loadDatasets()
    }
    toast.success(`Accepted ${synthetic.length} remaining case(s).`)
  }

  const handleFulfill = async (force = false): Promise<void> => {
    if (!sessionId) {
      toast.error('No session in context — open this screen from a Stage 3 plan.')
      return
    }
    if (!selectedProviderId) {
      toast.error('No LLM provider configured — set one via the gear icon first.')
      return
    }
    setRegenerateModalOpen(false)
    setFulfilling(true)
    // Poll the dataset list while the long fulfill runs so each agent's dataset shows up live as it lands.
    const pollId = setInterval(() => {
      window.api.aeh.listDatasets(sessionId).then(setDatasets).catch(() => {})
    }, 2000)
    try {
      const report = await window.api.aeh.fulfillDatasets(sessionId, {
        provider_id: selectedProviderId,
        model_id: selectedModelId || null,
        reasoning_effort: selectedReasoningEffort,
        thinking_budget: selectedThinkingBudget,
        embedding_provider_id: selectedEmbedProviderId || null,
        embedding_model_id: selectedEmbedModelId || null,
        use_local_embedding: selectedUseLocalEmbedding,
        agent_ids: agentId ? [agentId] : null,
        force_agent_ids: agentId && force ? [agentId] : null
      })
      setFulfillmentReport(report)
      await loadDatasets()
      if (agentId) {
        const outcome = report[`synthetic_agent_io/${agentId}`]
        if (outcome?.status === 'fulfilled') {
          toast.success(force ? 'Dataset regenerated for this agent.' : 'Dataset fulfilled for this agent.')
        } else if (outcome?.status === 'skipped') {
          toast.error('This agent’s eval toggle is off — turn it on in Stage 3 first.')
        } else if (!outcome && scopedDatasetId) {
          // Already fulfilled before this click — outside this run's unfulfilled scope, not a failure.
          toast.success('Dataset already exists for this agent — nothing to regenerate.')
        } else {
          toast.error(outcome?.reason ?? 'Fulfillment did not produce a dataset for this agent.')
        }
        await refreshScopedDataset()
      } else {
        toast.success('Fulfillment run complete.')
      }
    } catch (err: any) {
      toast.error(err?.message ?? 'Fulfillment failed.')
    } finally {
      clearInterval(pollId)
      setFulfilling(false)
    }
  }

  const handleFulfillButtonClick = (): void => {
    if (agentId && scopedDatasetId) {
      setRegenerateModalOpen(true)
    } else {
      handleFulfill(false)
    }
  }

  const selectedDataset = datasets.find((d) => d.dataset_id === selectedDatasetId) ?? null
  // Sidebar shows only human-reviewable datasets; deterministic fixtures/gold stay in the DB.
  const reviewableDatasets = datasets.filter((d) => !NON_REVIEWABLE_KINDS.has(d.kind ?? ''))

  return (
    <div className="flex flex-col h-full bg-[#090d16] text-slate-100">
      <div className="screen-header shrink-0 flex items-center justify-between">
        <div className="flex items-center gap-3 min-w-0">
          <Button
            variant="ghost"
            className="w-8 h-8 p-0 rounded-full flex items-center justify-center border border-slate-800 bg-slate-900/40 text-slate-400 hover:text-slate-200 shrink-0"
            onClick={() => navigate(`/aeh/analysis/stage3?repoId=${repoId ?? ''}&snapshotId=${snapshotId ?? ''}`)}
            title="Back to Stage 3"
          >
            <ArrowLeft size={16} />
          </Button>
          <div>
            <h1 className="screen-title">{agentId ? `Dataset — ${agentLabel || agentId}` : 'Dataset Review'}</h1>
            <p className="screen-subtitle">
              {agentId
                ? 'This agent’s synthetic_agent_io dataset only — fulfill it below if empty'
                : 'Auto-generated dataset cases — review before they can be referenced by a plan'}
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2 shrink-0">
          <LLMModelButton
            providerId={selectedProviderId}
            modelId={selectedModelId}
            providers={providers}
            onClick={() => setLlmConfigOpen(true)}
            labelPrefix="LLM"
            emptyLabel="No LLM Configured"
          />
          <EmbeddingModelButton
            providerId={selectedEmbedProviderId}
            modelId={selectedEmbedModelId}
            useLocal={selectedUseLocalEmbedding}
            providers={providers}
            onClick={() => setEmbedConfigOpen(true)}
          />
          <Button
            variant="primary"
            onClick={handleFulfillButtonClick}
            loading={fulfilling}
            disabled={!sessionId}
            className="text-xs h-9 px-3 flex items-center gap-1.5 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-40 text-white"
          >
            {agentId && scopedDatasetId ? <RefreshCw size={13} /> : <Sparkles size={13} />}
            <span>{agentId ? (scopedDatasetId ? 'Regenerate' : 'Fulfill Data') : 'Fulfill Datasets'}</span>
          </Button>
        </div>
      </div>

      {regenerateModalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/80 backdrop-blur-sm p-4">
          <div className="glass rounded-2xl w-full max-w-sm border-slate-850 shadow-2xl p-5 space-y-4 text-slate-100">
            <div className="flex items-center gap-3 text-amber-400">
              <AlertCircle className="shrink-0 w-6 h-6" />
              <h3 className="text-sm font-semibold text-slate-200">Regenerate this agent's dataset?</h3>
            </div>
            <p className="text-[11px] text-slate-450 leading-relaxed">
              This permanently deletes the existing dataset (all cases and any review verdicts)
              and generates a fresh one. This cannot be undone.
            </p>
            <div className="flex justify-end gap-2.5 pt-2">
              <Button
                variant="ghost"
                onClick={() => setRegenerateModalOpen(false)}
                disabled={fulfilling}
                className="text-xs px-4 py-2"
              >
                Cancel
              </Button>
              <Button
                variant="primary"
                onClick={() => handleFulfill(true)}
                loading={fulfilling}
                className="text-xs px-4 py-2 bg-amber-600 hover:bg-amber-500"
              >
                Regenerate
              </Button>
            </div>
          </div>
        </div>
      )}

      <LLMConfigModal
        isOpen={llmConfigOpen}
        onClose={() => setLlmConfigOpen(false)}
        providerId={selectedProviderId}
        modelId={selectedModelId}
        reasoningEffort={selectedReasoningEffort}
        thinkingBudget={selectedThinkingBudget}
        onChange={(prov, model, effort, budget) => {
          setSelectedProviderId(prov)
          setSelectedModelId(model)
          setSelectedReasoningEffort(effort ?? null)
          setSelectedThinkingBudget(budget ?? null)
        }}
        title="Fulfillment LLM Model"
      />

      <EmbeddingConfigModal
        isOpen={embedConfigOpen}
        onClose={() => setEmbedConfigOpen(false)}
        providerId={selectedEmbedProviderId}
        modelId={selectedEmbedModelId}
        useLocal={selectedUseLocalEmbedding}
        onChange={(prov, model, local) => {
          setSelectedEmbedProviderId(prov)
          setSelectedEmbedModelId(model)
          setSelectedUseLocalEmbedding(local)
        }}
        title="Fulfillment Embedding Model"
      />

      {fulfillmentReport && (
        <div className="mx-5 mt-3 p-3 border border-slate-800 rounded-lg bg-slate-950/40 text-[11px] space-y-1 max-h-32 overflow-y-auto">
          {Object.entries(fulfillmentReport).map(([key, r]) => (
            <div key={key} className="flex items-center gap-2">
              <Badge
                variant={r.status === 'fulfilled' ? 'success' : r.status === 'failed' ? 'error' : 'warning'}
                size="sm"
              >
                {r.status}
              </Badge>
              <span className="text-slate-400 font-mono">{key}</span>
              {r.reason && <span className="text-slate-600">— {r.reason}</span>}
            </div>
          ))}
        </div>
      )}

      <div className="flex-1 overflow-hidden flex">
        {!agentId && (
          <div className="w-72 shrink-0 border-r border-slate-850 overflow-y-auto p-3 space-y-2">
            {loadingDatasets ? (
              <Loader2 className="animate-spin text-indigo-500 mx-auto mt-4" size={20} />
            ) : reviewableDatasets.length === 0 ? (
              <p className="text-[11px] text-slate-600 italic p-2">No datasets yet.</p>
            ) : (
              reviewableDatasets.map((ds) => (
                <button
                  key={ds.dataset_id}
                  onClick={() => setSelectedDatasetId(ds.dataset_id)}
                  className={`w-full text-left p-2.5 rounded-lg border text-[11px] transition-colors ${
                    selectedDatasetId === ds.dataset_id
                      ? 'border-indigo-700 bg-indigo-950/30'
                      : 'border-slate-850 bg-slate-950/20 hover:border-slate-700'
                  }`}
                >
                  <div className="flex items-center justify-between gap-2">
                    <span className="font-mono text-slate-200 truncate">{ds.dataset_id}</span>
                    <Badge variant={ds.review_complete ? 'success' : 'warning'} size="sm" className="shrink-0">
                      {ds.review_complete ? 'ready' : 'pending'}
                    </Badge>
                  </div>
                  <div className="text-slate-500 mt-1">{ds.kind ?? 'unknown kind'}</div>
                  <div className="text-slate-600 mt-0.5">
                    {ds.total_count} cases · {ds.synthetic_count} unreviewed
                  </div>
                </button>
              ))
            )}
          </div>
        )}

        <div className="flex-1 overflow-y-auto p-4">
          {agentId && !scopedLookupDone ? (
            <Loader2 className="animate-spin text-indigo-500" size={20} />
          ) : !selectedDataset ? (
            <p className="text-[11px] text-slate-600 italic">
              {agentId
                ? 'No data fulfilled for this agent yet — click "Fulfill Data" above to generate it.'
                : 'Select a dataset to review its cases.'}
            </p>
          ) : (
            <>
              <div className="flex items-center justify-between mb-3">
                <div>
                  <h2 className="text-sm font-semibold text-slate-200">{selectedDataset.dataset_id}</h2>
                  <p className="text-[11px] text-slate-500">
                    {selectedDataset.total_count} cases · min_cases {selectedDataset.min_cases}
                  </p>
                </div>
                {cases.some((c) => c.provenance === 'synthetic') && (
                  <Button
                    variant="ghost"
                    onClick={handleBulkAccept}
                    className="text-[11px] h-8 px-3 border border-slate-800 text-emerald-400 hover:bg-slate-900"
                  >
                    Bulk-accept remainder
                  </Button>
                )}
              </div>

              {loadingCases ? (
                <Loader2 className="animate-spin text-indigo-500" size={20} />
              ) : (
                <div className="space-y-3">
                  {cases.map((c) => {
                    let caseLabels: Record<string, any> = {}
                    try { caseLabels = JSON.parse(c.labels_json ?? '{}') } catch {}
                    return (
                    <div key={c.id} className="border border-slate-850 rounded-lg p-3 bg-slate-950/20">
                      <div className="flex items-center justify-between mb-2">
                        <Badge
                          variant={
                            c.provenance === 'synthetic'
                              ? 'warning'
                              : c.provenance === 'handwritten'
                              ? 'info'
                              : 'success'
                          }
                          size="sm"
                        >
                          {c.provenance}
                        </Badge>
                        <div className="flex items-center gap-1.5">
                          <button
                            onClick={() => handleVerdict(c.id, 'accept')}
                            className="text-[10px] px-2 py-1 rounded border border-slate-800 text-emerald-400 hover:bg-slate-900"
                          >
                            Accept
                          </button>
                          <button
                            onClick={() => handleVerdict(c.id, 'edit')}
                            className="text-[10px] px-2 py-1 rounded border border-slate-800 text-indigo-400 hover:bg-slate-900"
                          >
                            Save Edit
                          </button>
                          <button
                            onClick={() => handleVerdict(c.id, 'reject')}
                            className="text-[10px] px-2 py-1 rounded border border-slate-800 text-rose-400 hover:bg-slate-900"
                          >
                            Reject
                          </button>
                        </div>
                      </div>
                      {(caseLabels.schema_source === 'none' || caseLabels.near_duplicate || caseLabels.needs_human || caseLabels.archetype) && (
                        <div className="flex flex-wrap items-center gap-1 mb-2">
                          {caseLabels.schema_source === 'none' && (
                            <span className="text-[10px] px-1.5 py-0.5 rounded bg-rose-900/60 text-rose-300 border border-rose-700 font-medium">
                              no schema — gold unvalidated
                            </span>
                          )}
                          {caseLabels.near_duplicate && (
                            <span className="text-[10px] px-1.5 py-0.5 rounded bg-amber-900/60 text-amber-300 border border-amber-700">
                              near-duplicate
                            </span>
                          )}
                          {caseLabels.needs_human && (
                            <span className="text-[10px] px-1.5 py-0.5 rounded bg-orange-900/60 text-orange-300 border border-orange-700">
                              needs human review
                            </span>
                          )}
                          {caseLabels.archetype && (
                            <span className="text-[10px] px-1.5 py-0.5 rounded bg-slate-800 text-slate-400 border border-slate-700">
                              {caseLabels.archetype}
                            </span>
                          )}
                        </div>
                      )}
                      <div className="grid grid-cols-3 gap-2">
                        {(['input', 'expected', 'labels'] as const).map((field) => (
                          <div key={field}>
                            <div className="text-[9px] text-slate-500 uppercase mb-0.5">{field}</div>
                            <textarea
                              value={editStrings[c.id]?.[field] ?? ''}
                              onChange={(e) =>
                                setEditStrings((prev) => ({
                                  ...prev,
                                  [c.id]: { ...prev[c.id], [field]: e.target.value }
                                }))
                              }
                              rows={4}
                              className="w-full font-mono text-[10px] bg-slate-950 border border-slate-800 rounded px-2 py-1 text-slate-300 focus:border-slate-600 focus:outline-none resize-y"
                            />
                          </div>
                        ))}
                      </div>
                    </div>
                  )
                  })}
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  )
}

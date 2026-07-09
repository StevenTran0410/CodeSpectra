import React, { useEffect, useState, useCallback } from 'react'
import { useSearchParams, useNavigate } from 'react-router-dom'
import { Loader2, Sparkles, ArrowLeft, Settings } from 'lucide-react'
import { Button, Badge, useToastStore } from '../../components/ui'
import { useProviderStore } from '../../store/provider.store'
import LLMConfigModal from './LLMConfigModal'

type EditStrings = { input: string; expected: string; labels: string }

export default function DatasetReviewScreen(): React.ReactElement {
  const navigate = useNavigate()
  const [searchParams] = useSearchParams()
  const sessionId = searchParams.get('sessionId')
  const repoId = searchParams.get('repoId')
  const snapshotId = searchParams.get('snapshotId')
  const toast = useToastStore()

  const { providers, load: loadProviders } = useProviderStore()
  const [selectedProviderId, setSelectedProviderId] = useState('')
  const [selectedModelId, setSelectedModelId] = useState('')
  const [llmConfigOpen, setLlmConfigOpen] = useState(false)

  useEffect(() => {
    loadProviders()
  }, [loadProviders])

  useEffect(() => {
    if (providers.length > 0 && !selectedProviderId) {
      setSelectedProviderId(providers[0].id)
      setSelectedModelId(providers[0].model_id || '')
    }
  }, [providers, selectedProviderId])

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

  const loadDatasets = useCallback(async () => {
    setLoadingDatasets(true)
    try {
      setDatasets(await window.api.aeh.listDatasets())
    } catch (err: any) {
      toast.error(err?.message ?? 'Failed to load datasets.')
    } finally {
      setLoadingDatasets(false)
    }
  }, [toast])

  useEffect(() => {
    loadDatasets()
  }, [loadDatasets])

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
      try {
        body.input_json = JSON.parse(strings.input)
      } catch {
        toast.error('Input JSON is invalid — fix it before saving.')
        return
      }
      try {
        body.expected_json = strings.expected ? JSON.parse(strings.expected) : undefined
      } catch {
        toast.error('Expected JSON is invalid — fix it before saving.')
        return
      }
      try {
        body.labels_json = strings.labels ? JSON.parse(strings.labels) : undefined
      } catch {
        toast.error('Labels JSON is invalid — fix it before saving.')
        return
      }
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

  const handleFulfill = async (): Promise<void> => {
    if (!sessionId) {
      toast.error('No session in context — open this screen from a Stage 3 plan.')
      return
    }
    if (!selectedProviderId) {
      toast.error('No LLM provider configured — set one via the gear icon first.')
      return
    }
    setFulfilling(true)
    try {
      const report = await window.api.aeh.fulfillDatasets(sessionId, {
        provider_id: selectedProviderId,
        model_id: selectedModelId || null
      })
      setFulfillmentReport(report)
      await loadDatasets()
      toast.success('Fulfillment run complete.')
    } catch (err: any) {
      toast.error(err?.message ?? 'Fulfillment failed.')
    } finally {
      setFulfilling(false)
    }
  }

  const selectedDataset = datasets.find((d) => d.dataset_id === selectedDatasetId) ?? null

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
            <h1 className="screen-title">Dataset Review</h1>
            <p className="screen-subtitle">
              Auto-generated dataset cases — review before they can be referenced by a plan
            </p>
          </div>
        </div>
        <div className="flex items-center gap-3 shrink-0">
          <Button
            variant="primary"
            onClick={handleFulfill}
            loading={fulfilling}
            disabled={!sessionId}
            className="text-xs h-9 px-3 flex items-center gap-1.5 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-40 text-white"
          >
            <Sparkles size={13} />
            <span>Fulfill Datasets</span>
          </Button>
          <Button
            variant="ghost"
            onClick={() => setLlmConfigOpen(true)}
            className="w-9 h-9 p-0 flex items-center justify-center border border-slate-800 bg-slate-900/40 hover:bg-slate-900 text-slate-300"
            title="Configure fulfillment LLM"
          >
            <Settings size={15} />
          </Button>
        </div>
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
        title="Fulfillment LLM Model"
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
        <div className="w-72 shrink-0 border-r border-slate-850 overflow-y-auto p-3 space-y-2">
          {loadingDatasets ? (
            <Loader2 className="animate-spin text-indigo-500 mx-auto mt-4" size={20} />
          ) : datasets.length === 0 ? (
            <p className="text-[11px] text-slate-600 italic p-2">No datasets yet.</p>
          ) : (
            datasets.map((ds) => (
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

        <div className="flex-1 overflow-y-auto p-4">
          {!selectedDataset ? (
            <p className="text-[11px] text-slate-600 italic">Select a dataset to review its cases.</p>
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
                  {cases.map((c) => (
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
                  ))}
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  )
}

import React, { useEffect } from 'react'
import { createPortal } from 'react-dom'
import { Loader2 } from 'lucide-react'
import { Button, FormGroup, Select } from '../../components/ui'
import { useProviderStore } from '../../store/provider.store'

// Portals to document.body so it renders correctly regardless of ancestor
// stacking contexts (e.g. ReactFlow's transformed viewport on Stage 1).
export default function LLMConfigModal({
  isOpen,
  onClose,
  providerId,
  modelId,
  onChange,
  title = 'LLM Model (this stage only)',
}: {
  isOpen: boolean
  onClose: () => void
  providerId: string
  modelId: string
  onChange: (providerId: string, modelId: string) => void
  title?: string
}): React.ReactElement | null {
  const { providers, load, fetchModels, modelLists, loadingModels, modelErrors } = useProviderStore()

  useEffect(() => {
    load()
  }, [load])

  useEffect(() => {
    if (providerId) fetchModels(providerId)
  }, [providerId, fetchModels])

  // Early return after hooks — always mounted, `isOpen` just toggles.
  if (!isOpen) return null

  const selectedProvider = providers.find((p) => p.id === providerId) ?? null
  const modelOptions = modelLists[providerId]?.length
    ? modelLists[providerId]
    : selectedProvider?.model_id
      ? [selectedProvider.model_id]
      : []

  return createPortal(
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/80 backdrop-blur-sm p-4">
      <div className="glass rounded-2xl w-full max-w-sm border-slate-800 shadow-2xl p-5 space-y-4 text-slate-100">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-semibold text-slate-200">{title}</h3>
          <button onClick={onClose} className="text-slate-400 hover:text-slate-200 text-lg font-bold px-1">&times;</button>
        </div>
        <p className="text-[11px] text-slate-500">
          Applies only to this run — not saved as a global default.
        </p>
        <FormGroup label="Provider">
          <Select
            value={providerId}
            onChange={(e) => {
              const p = providers.find((x) => x.id === e.target.value)
              onChange(e.target.value, p?.model_id ?? '')
            }}
          >
            {providers.length === 0 && <option value="">No providers configured</option>}
            {providers.map((p) => (
              <option key={p.id} value={p.id}>{p.display_name}</option>
            ))}
          </Select>
        </FormGroup>
        <div>
          <label className="block text-xs text-slate-400 mb-1.5">Model</label>
          <div className="flex gap-1.5 items-center">
            <Select
              value={modelId}
              onChange={(e) => onChange(providerId, e.target.value)}
              className="flex-1"
              disabled={!providerId}
            >
              {(modelOptions.length > 0 ? modelOptions : ['']).map((m) => (
                <option key={m} value={m}>{m || '(no model)'}</option>
              ))}
            </Select>
            <button
              type="button"
              title="Fetch available models from provider"
              disabled={!providerId || !!loadingModels[providerId]}
              onClick={() => providerId && fetchModels(providerId)}
              className="shrink-0 flex items-center justify-center w-8 h-8 rounded-md border border-slate-700 bg-slate-950 text-slate-400 hover:text-slate-100 hover:border-slate-500 disabled:opacity-40 transition-colors"
            >
              <Loader2 size={14} className={loadingModels[providerId] ? 'animate-spin' : ''} />
            </button>
          </div>
          {modelErrors[providerId] && (
            <div className="text-[11px] text-rose-400 mt-1">{modelErrors[providerId]}</div>
          )}
        </div>
        <div className="flex justify-end pt-2">
          <Button variant="primary" onClick={onClose} className="text-xs px-4 py-2">Done</Button>
        </div>
      </div>
    </div>,
    document.body
  )
}

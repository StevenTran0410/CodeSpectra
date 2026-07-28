import React, { useEffect } from 'react'
import { createPortal } from 'react-dom'
import { Loader2 } from 'lucide-react'
import { Badge, Button, FormGroup, Input, Select } from '../../components/ui'
import { useProviderStore } from '../../store/provider.store'
import type { ProviderConfig, ProviderKind, ReasoningStyle } from '../../types/electron'

/** UI hint only — the backend adapter is the source of truth and clamps out-of-range values. */
function thinkingBudgetHint(kind: ProviderKind | undefined, style: ReasoningStyle): string {
  if (style === 'budget_tokens') return 'Anthropic: 1024–32000 tokens. Leave blank to disable thinking.'
  if (kind === 'gemini') return 'Gemini: 0–24576 (Flash, 0 disables) or 128–32768 (Pro, cannot disable). -1 = dynamic.'
  return ''
}

// Portals to document.body to escape ReactFlow's transformed stacking context on Stage 1.
export default function LLMConfigModal({
  isOpen,
  onClose,
  providerId,
  modelId,
  reasoningEffort = null,
  thinkingBudget = null,
  onChange,
  onConfirmResume,
  title = 'LLM Model (this stage only)',
}: {
  isOpen: boolean
  onClose: () => void
  providerId: string
  modelId: string
  reasoningEffort?: string | null
  thinkingBudget?: number | null
  onChange: (
    providerId: string,
    modelId: string,
    reasoningEffort?: string | null,
    thinkingBudget?: number | null
  ) => void
  onConfirmResume?: () => void
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
  const modelInfos = modelLists[providerId]?.length
    ? modelLists[providerId]
    : selectedProvider?.model_id
      ? [{ id: selectedProvider.model_id, reasoning_style: 'none' as ReasoningStyle }]
      : []
  const modelOptions = modelInfos.map((m) => m.id)
  const reasoningStyle: ReasoningStyle =
    modelInfos.find((m) => m.id === modelId)?.reasoning_style ?? 'none'

  const handleModelChange = (newModelId: string): void => {
    const style = modelInfos.find((m) => m.id === newModelId)?.reasoning_style ?? 'none'
    // Drop reasoning fields that don't apply to the newly selected model's style.
    const nextEffort =
      style === 'effort'
        ? (reasoningEffort ?? 'medium')
        : style === 'effort_toggle'
          ? (reasoningEffort ?? 'high')
          : null
    const nextBudget = style === 'budget_tokens' || style === 'thinking_budget' ? thinkingBudget : null
    onChange(providerId, newModelId, nextEffort, nextBudget)
  }

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
              onChange(e.target.value, p?.model_id ?? '', null, null)
            }}
          >
            {providers.length === 0 ? (
              <option value="">No providers configured</option>
            ) : (
              !providerId && <option value="">Select a provider…</option>
            )}
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
              onChange={(e) => handleModelChange(e.target.value)}
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
        {reasoningStyle === 'effort' && (
          <FormGroup label="Reasoning effort">
            <Select
              value={reasoningEffort ?? 'medium'}
              onChange={(e) => onChange(providerId, modelId, e.target.value, null)}
            >
              <option value="minimal">Minimal</option>
              <option value="low">Low</option>
              <option value="medium">Medium</option>
              <option value="high">High</option>
              <option value="xhigh">X-High</option>
            </Select>
          </FormGroup>
        )}
        {reasoningStyle === 'effort_toggle' && (
          <FormGroup label="Thinking">
            <Select
              value={reasoningEffort ?? 'high'}
              onChange={(e) => onChange(providerId, modelId, e.target.value, null)}
            >
              <option value="disable">Disable</option>
              <option value="high">High</option>
              <option value="max">Max</option>
            </Select>
          </FormGroup>
        )}
        {(reasoningStyle === 'budget_tokens' || reasoningStyle === 'thinking_budget') && (
          <FormGroup
            label="Thinking budget (tokens)"
            helperText={thinkingBudgetHint(selectedProvider?.kind, reasoningStyle)}
          >
            <Input
              type="number"
              value={thinkingBudget ?? ''}
              placeholder="Provider default"
              onChange={(e) =>
                onChange(
                  providerId,
                  modelId,
                  null,
                  e.target.value === '' ? null : Number(e.target.value)
                )
              }
            />
          </FormGroup>
        )}
        {reasoningStyle === 'toggle' && (
          <Badge variant="info" size="sm">Reasoning always on for this model — no tunable budget</Badge>
        )}
        <div className="flex justify-end gap-2 pt-2">
          {onConfirmResume && (
            <Button variant="primary" onClick={onConfirmResume} className="text-xs px-4 py-2">Confirm & Resume</Button>
          )}
          <Button variant={onConfirmResume ? 'secondary' : 'primary'} onClick={onClose} className="text-xs px-4 py-2">Done</Button>
        </div>
      </div>
    </div>,
    document.body
  )
}

/** Header button that opens an LLMConfigModal, showing the active provider/model or a fallback label. */
export function LLMModelButton({
  providerId,
  modelId,
  providers,
  disabled,
  onClick,
  labelPrefix = 'Model',
  emptyLabel = 'No LLM Configured',
  emptyVariant = 'error',
}: {
  providerId: string
  modelId: string
  providers: ProviderConfig[]
  disabled?: boolean
  onClick: () => void
  labelPrefix?: string
  emptyLabel?: string
  emptyVariant?: 'error' | 'neutral'
}): React.ReactElement {
  const emptyClass =
    emptyVariant === 'neutral'
      ? 'border-slate-800 bg-slate-950/30 text-slate-500 hover:border-slate-700'
      : 'border-red-900/30 bg-red-950/40 text-red-400 hover:border-red-700/50'

  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className={`text-[10px] h-7 px-2.5 rounded-md border font-mono transition-colors ${
        providerId ? 'border-slate-700 bg-slate-950 text-slate-300 hover:border-slate-500' : emptyClass
      }`}
    >
      {providerId
        ? `${labelPrefix}: ${modelId || providers.find((p) => p.id === providerId)?.display_name || '?'}`
        : emptyLabel}
    </button>
  )
}

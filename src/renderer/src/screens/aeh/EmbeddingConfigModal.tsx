import React, { useEffect, useState } from 'react'
import { createPortal } from 'react-dom'
import { Badge, Button, FormGroup, Select } from '../../components/ui'
import { useProviderStore } from '../../store/provider.store'
import type { LocalEmbeddingStatus } from '../../types/electron'

export default function EmbeddingConfigModal({
  isOpen,
  onClose,
  providerId,
  modelId,
  useLocal = false,
  onChange,
  title = 'Embedding Model (synthesis only)',
}: {
  isOpen: boolean
  onClose: () => void
  providerId: string
  modelId: string
  useLocal?: boolean
  onChange: (providerId: string, modelId: string, useLocal: boolean) => void
  title?: string
}): React.ReactElement | null {
  const { providers, load: loadProviders } = useProviderStore()
  const [localStatus, setLocalStatus] = useState<LocalEmbeddingStatus | null>(null)

  useEffect(() => {
    loadProviders()
    window.api.localEmbedding
      .status()
      .then(setLocalStatus)
      .catch(() => {})
  }, [loadProviders])

  if (!isOpen) return null

  // Filter providers to only OpenAI and Gemini
  const embedProviders = providers.filter((p) => p.kind === 'openai' || p.kind === 'gemini')

  // Find currently selected provider kind (cloud)
  const selectedProvider = providers.find((p) => p.id === providerId) ?? null

  const handleOptionChange = (val: string): void => {
    if (val === 'local') {
      onChange('', '', true)
    } else {
      // Format is providerId:modelId
      const [pid, model] = val.split(':')
      onChange(pid, model, false)
    }
  }

  // Determine selected value for dropdown
  let selectVal = ''
  if (useLocal) {
    selectVal = 'local'
  } else if (providerId && modelId) {
    selectVal = `${providerId}:${modelId}`
  }

  return createPortal(
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/80 backdrop-blur-sm p-4">
      <div className="glass rounded-2xl w-full max-w-sm border-slate-800 shadow-2xl p-5 space-y-4 text-slate-100">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-semibold text-slate-200">{title}</h3>
          <button onClick={onClose} className="text-slate-400 hover:text-slate-200 text-lg font-bold px-1">&times;</button>
        </div>
        <p className="text-[11px] text-slate-500">
          Used to cluster contexts during QA testset generation. No dummy vectors will be used.
        </p>

        <FormGroup label="Select Embedding Model">
          <Select value={selectVal} onChange={(e) => handleOptionChange(e.target.value)}>
            <option value="" disabled>Select an option...</option>
            
            {/* OpenAI options */}
            {embedProviders
              .filter((p) => p.kind === 'openai')
              .flatMap((p) => [
                <option key={`${p.id}:text-embedding-3-small`} value={`${p.id}:text-embedding-3-small`}>
                  OpenAI: text-embedding-3-small (Recommended)
                </option>,
                <option key={`${p.id}:text-embedding-3-large`} value={`${p.id}:text-embedding-3-large`}>
                  OpenAI: text-embedding-3-large
                </option>
              ])}

            {/* Gemini options */}
            {embedProviders
              .filter((p) => p.kind === 'gemini')
              .map((p) => (
                <option key={`${p.id}:gemini-embedding-001`} value={`${p.id}:gemini-embedding-001`}>
                  Gemini: gemini-embedding-001
                </option>
              ))}

            {/* Local option */}
            <option
              value="local"
              disabled={!localStatus?.gpu_available}
            >
              Local: Qwen3-Embedding-0.6B (Local GPU)
              {!localStatus?.gpu_available ? ' (No usable GPU)' : ''}
            </option>
          </Select>
        </FormGroup>

        {useLocal && (
          <Badge variant="success" size="sm" className="w-full justify-center">
            Running locally on GPU ({localStatus?.vram_gb ?? '?'} GB VRAM detected)
          </Badge>
        )}

        {!useLocal && selectedProvider && (
          <Badge variant="info" size="sm" className="w-full justify-center">
            Using cloud provider "{selectedProvider.display_name}" ({modelId})
          </Badge>
        )}

        {embedProviders.length === 0 && !localStatus?.gpu_available && (
          <Badge variant="error" size="sm" className="w-full justify-center">
            No cloud keys configured AND no local GPU available.
          </Badge>
        )}

        <div className="flex justify-end gap-2 pt-2">
          <Button variant="primary" onClick={onClose} className="text-xs px-4 py-2">Done</Button>
        </div>
      </div>
    </div>,
    document.body
  )
}

/** Header button that opens an EmbeddingConfigModal, showing the active embedding model or fallback. */
export function EmbeddingModelButton({
  providerId,
  modelId,
  useLocal,
  providers,
  disabled,
  onClick,
}: {
  providerId: string
  modelId: string
  useLocal: boolean
  providers: any[]
  disabled?: boolean
  onClick: () => void
}): React.ReactElement {
  let label = 'No Embedding Config'
  let isConfigured = false

  if (useLocal) {
    label = 'Embedder: Local (Qwen3)'
    isConfigured = true
  } else if (providerId && modelId) {
    const provName = providers.find((p) => p.id === providerId)?.display_name || 'Cloud'
    label = `Embedder: ${provName} (${modelId})`
    isConfigured = true
  }

  const emptyClass = 'border-red-900/30 bg-red-950/40 text-red-400 hover:border-red-700/50'

  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className={`text-[10px] h-7 px-2.5 rounded-md border font-mono transition-colors ${
        isConfigured ? 'border-slate-700 bg-slate-950 text-slate-300 hover:border-slate-500' : emptyClass
      }`}
    >
      {label}
    </button>
  )
}

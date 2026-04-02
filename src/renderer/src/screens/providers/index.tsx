import { useState, useEffect } from 'react'
import {
  Plus,
  Trash2,
  RefreshCw,
  CheckCircle2,
  XCircle,
  AlertTriangle,
  Loader2,
  ChevronDown,
  ChevronRight,
  Pencil,
  Shield,
  Cloud,
  Wifi,
  BookOpen,
  Eye,
  EyeOff
} from 'lucide-react'
import { useEffect, useRef } from 'react'
import { ConsentBanner } from '../../components/providers/ConsentBanner'
import { useProviderStore, type TestResult } from '../../store/provider.store'
import type { ProviderConfig, CreateProviderRequest, UpdateProviderRequest } from '../../types/electron'

// Cloud provider kinds
const CLOUD_KINDS = new Set(['openai', 'anthropic', 'gemini', 'deepseek'])

// Model presets for cloud providers (shown when Browse is unavailable)
const CLOUD_MODEL_PRESETS: Record<string, string[]> = {
  openai: ['gpt-4o', 'gpt-4o-mini', 'gpt-4-turbo', 'gpt-3.5-turbo', 'o3-mini'],
  anthropic: ['claude-opus-4-5', 'claude-sonnet-4-5', 'claude-3-5-sonnet-20241022', 'claude-3-5-haiku-20241022', 'claude-3-haiku-20240307'],
  gemini: ['gemini-2.0-flash', 'gemini-2.0-flash-lite', 'gemini-1.5-pro', 'gemini-1.5-flash'],
  deepseek: ['deepseek-chat', 'deepseek-reasoner'],
}

// ──────────────────────────────────────────────────────────────────────────────
// Sub-components
// ──────────────────────────────────────────────────────────────────────────────

function LocalBadge() {
  return (
    <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
      <Shield size={10} />
      Strict Local
    </span>
  )
}

function CloudBadge() {
  return (
    <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium bg-amber-500/10 text-amber-400 border border-amber-500/20">
      <Cloud size={10} />
      BYOK Cloud
    </span>
  )
}

function PrivacyBadge({ kind }: { kind: string }) {
  return CLOUD_KINDS.has(kind) ? <CloudBadge /> : <LocalBadge />
}

function KindLabel({ kind }: { kind: string }) {
  const label = kind === 'ollama' ? 'Ollama' : 'LM Studio'
  const color = kind === 'ollama' ? 'text-violet-400 bg-violet-500/10 border-violet-500/20' : 'text-sky-400 bg-sky-500/10 border-sky-500/20'
  return (
    <span className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-mono border ${color}`}>
      {label}
    </span>
  )
}

function TestStatus({ ok, message, warning }: TestResult) {
  if (ok && warning) {
    return (
      <div className="space-y-1">
        <div className="flex items-center gap-2 text-xs rounded px-3 py-2 bg-amber-500/10 text-amber-400">
          <AlertTriangle size={13} className="shrink-0" />
          <span>{message}</span>
        </div>
        <div className="flex items-start gap-2 text-xs rounded px-3 py-2 bg-zinc-800 text-zinc-400">
          <AlertTriangle size={12} className="shrink-0 mt-0.5 text-amber-500/60" />
          <span>{warning}</span>
        </div>
      </div>
    )
  }
  return (
    <div className={`flex items-center gap-2 text-xs rounded px-3 py-2 ${ok ? 'bg-emerald-500/10 text-emerald-400' : 'bg-red-500/10 text-red-400'}`}>
      {ok ? <CheckCircle2 size={13} /> : <XCircle size={13} />}
      <span>{message}</span>
    </div>
  )
}

// ──────────────────────────────────────────────────────────────────────────────
// LM Studio setup guide
// ──────────────────────────────────────────────────────────────────────────────
function LMStudioSetupGuide() {
  const [open, setOpen] = useState(false)
  return (
    <div className="border border-zinc-700/60 rounded-lg overflow-hidden">
      <button
        onClick={() => setOpen((v) => !v)}
        className="w-full flex items-center gap-2 px-4 py-3 text-xs text-zinc-400 hover:text-zinc-300 hover:bg-zinc-800/40 transition-colors"
      >
        <BookOpen size={13} className="text-sky-400" />
        <span className="flex-1 text-left font-medium">How to enable LM Studio Local Server</span>
        {open ? <ChevronDown size={13} /> : <ChevronRight size={13} />}
      </button>
      {open && (
        <div className="px-4 pb-4 space-y-3 bg-zinc-800/20">
          <ol className="space-y-2 text-xs text-zinc-400 list-none">
            <li className="flex gap-3">
              <span className="shrink-0 w-5 h-5 rounded-full bg-sky-500/20 text-sky-400 flex items-center justify-center text-[10px] font-bold">1</span>
              <span>Open LM Studio → click the <span className="text-zinc-200 font-mono bg-zinc-700 px-1 rounded">↔</span> <strong className="text-zinc-300">Local Server</strong> tab in the left sidebar.</span>
            </li>
            <li className="flex gap-3">
              <span className="shrink-0 w-5 h-5 rounded-full bg-sky-500/20 text-sky-400 flex items-center justify-center text-[10px] font-bold">2</span>
              <span>Select a model from the dropdown at the top of the server tab, then click <strong className="text-zinc-300">Start Server</strong>.</span>
            </li>
            <li className="flex gap-3">
              <span className="shrink-0 w-5 h-5 rounded-full bg-sky-500/20 text-sky-400 flex items-center justify-center text-[10px] font-bold">3</span>
              <span>Default port is <span className="font-mono text-sky-400">1234</span>. Leave the base URL as <span className="font-mono text-zinc-300">http://localhost:1234</span> unless you changed it.</span>
            </li>
            <li className="flex gap-3">
              <span className="shrink-0 w-5 h-5 rounded-full bg-sky-500/20 text-sky-400 flex items-center justify-center text-[10px] font-bold">4</span>
              <span>Save the provider here, then click <strong className="text-zinc-300">Test connection</strong> — you should see the loaded model appear in Browse.</span>
            </li>
          </ol>
        </div>
      )}
    </div>
  )
}

// ──────────────────────────────────────────────────────────────────────────────
// Add / Edit form
// ──────────────────────────────────────────────────────────────────────────────
const KIND_DEFAULTS: Record<string, Omit<CreateProviderRequest, 'kind'>> = {
  ollama:    { display_name: 'Ollama (local)',     base_url: 'http://localhost:11434',                     model_id: '' },
  lmstudio:  { display_name: 'LM Studio (local)', base_url: 'http://localhost:1234',                      model_id: '' },
  openai:    { display_name: 'OpenAI',             base_url: 'https://api.openai.com',                    model_id: 'gpt-4o' },
  anthropic: { display_name: 'Anthropic',          base_url: 'https://api.anthropic.com',                 model_id: 'claude-3-5-sonnet-20241022' },
  gemini:    { display_name: 'Google Gemini',      base_url: 'https://generativelanguage.googleapis.com', model_id: 'gemini-2.0-flash' },
  deepseek:  { display_name: 'DeepSeek',           base_url: 'https://api.deepseek.com',                  model_id: 'deepseek-chat' },
}

interface ProviderFormProps {
  kind: 'ollama' | 'lmstudio'
  initial?: ProviderConfig
  onClose: () => void
}

function ProviderForm({ kind, initial, onClose }: ProviderFormProps) {
  const { create, update, testConnection, fetchModels, testing, testResults, modelLists, loadingModels, modelErrors } = useProviderStore()

  const isEdit = !!initial
  const defaults = KIND_DEFAULTS[kind]
  const isCloud = CLOUD_KINDS.has(kind)
  const [name, setName] = useState(initial?.display_name ?? defaults.display_name)
  const [url, setUrl] = useState(initial?.base_url ?? defaults.base_url)
  const [modelId, setModelId] = useState(initial?.model_id ?? defaults.model_id ?? '')
  const [apiKey, setApiKey] = useState('')
  const [showKey, setShowKey] = useState(false)
  const [saving, setSaving] = useState(false)
  const [formError, setFormError] = useState<string | null>(null)
  const [showModelPicker, setShowModelPicker] = useState(false)
  const hasExistingKey = initial?.extra?.has_api_key === true

  const tempId = initial?.id ?? '__new__'
  const testResult = testResults[tempId]
  const isTesting = testing[tempId] ?? false
  const models = modelLists[tempId] ?? []
  const isLoadingModels = loadingModels[tempId] ?? false
  const modelFetchError = modelErrors[tempId] ?? ''

  // For new providers we need a temporary saved ID to test — we skip inline test for new
  // Instead show test button only after save.

  const handleSave = async () => {
    setFormError(null)
    if (!name.trim()) { setFormError('Display name is required'); return }
    if (!url.trim()) { setFormError('Base URL is required'); return }
    if (!modelId.trim()) { setFormError('Model ID is required'); return }
    if (isCloud && !isEdit && !apiKey.trim()) { setFormError('API key is required for cloud providers'); return }

    setSaving(true)
    try {
      if (isEdit && initial) {
        const req: UpdateProviderRequest = {
          display_name: name, base_url: url, model_id: modelId,
          ...(apiKey.trim() ? { api_key: apiKey.trim() } : {})
        }
        await update(initial.id, req)
      } else {
        const req: CreateProviderRequest = {
          kind, display_name: name, base_url: url, model_id: modelId,
          ...(apiKey.trim() ? { api_key: apiKey.trim() } : {})
        }
        await create(req)
      }
      onClose()
    } catch (err) {
      setFormError(String(err))
    } finally {
      setSaving(false)
    }
  }

  const handleTest = async () => {
    if (!initial) return
    await testConnection(initial.id)
  }

  const handleFetchModels = async () => {
    setShowModelPicker(true)
    if (!initial) {
      // For cloud providers before save: show presets directly
      return
    }
    await fetchModels(initial.id)
  }

  // Cloud providers can show presets even before saving
  const cloudPresets = CLOUD_MODEL_PRESETS[kind] ?? []
  const displayModels = models.length > 0 ? models : (isCloud ? cloudPresets : [])

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-2 mb-1">
        <KindLabel kind={kind} />
        <PrivacyBadge kind={kind} />
      </div>

      {/* Display name */}
      <div>
        <label className="block text-xs font-medium text-zinc-400 mb-1">Display name</label>
        <input
          className="w-full bg-zinc-800 border border-zinc-700 rounded-md px-3 py-2 text-sm text-zinc-100 focus:outline-none focus:border-violet-500 transition-colors"
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="e.g. Ollama (local)"
        />
      </div>

      {/* Base URL */}
      <div>
        <label className="block text-xs font-medium text-zinc-400 mb-1">Base URL</label>
        <input
          className="w-full bg-zinc-800 border border-zinc-700 rounded-md px-3 py-2 text-sm text-zinc-100 font-mono focus:outline-none focus:border-violet-500 transition-colors"
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          placeholder="http://localhost:11434"
        />
      </div>

      {/* API Key (cloud only) */}
      {isCloud && (
        <div>
          <label className="block text-xs font-medium text-zinc-400 mb-1">
            API Key {hasExistingKey && <span className="text-zinc-500 font-normal">(key saved — leave blank to keep)</span>}
          </label>
          <div className="relative">
            <input
              type={showKey ? 'text' : 'password'}
              className="w-full bg-zinc-800 border border-zinc-700 rounded-md px-3 py-2 pr-9 text-sm text-zinc-100 font-mono focus:outline-none focus:border-amber-500/60 transition-colors"
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              placeholder={hasExistingKey ? '••••••••••••••••' : 'sk-...'}
              autoComplete="off"
            />
            <button
              type="button"
              onClick={() => setShowKey((v) => !v)}
              className="absolute right-2.5 top-1/2 -translate-y-1/2 text-zinc-500 hover:text-zinc-300 transition-colors"
              tabIndex={-1}
            >
              {showKey ? <EyeOff size={14} /> : <Eye size={14} />}
            </button>
          </div>
          <p className="mt-1 text-xs text-zinc-600">Stored locally. Never sent anywhere except the provider's API.</p>
        </div>
      )}

      {/* Model ID */}
      <div>
        <label className="block text-xs font-medium text-zinc-400 mb-1">Model ID</label>
        <div className="flex gap-2">
          <input
            className="flex-1 bg-zinc-800 border border-zinc-700 rounded-md px-3 py-2 text-sm text-zinc-100 font-mono focus:outline-none focus:border-violet-500 transition-colors"
            value={modelId}
            onChange={(e) => setModelId(e.target.value)}
            placeholder="e.g. llama3.2:latest"
          />
          {isEdit && (
            <button
              onClick={handleFetchModels}
              disabled={isLoadingModels}
              className="flex items-center gap-1.5 px-3 py-2 text-xs bg-zinc-700 hover:bg-zinc-600 text-zinc-300 rounded-md transition-colors disabled:opacity-50"
              title="Browse available models"
            >
              {isLoadingModels ? <Loader2 size={13} className="animate-spin" /> : <ChevronDown size={13} />}
              Browse
            </button>
          )}
        </div>
        {!isEdit && (
          <p className="mt-1 text-xs text-zinc-500">Save first, then browse available models via "Test & Browse".</p>
        )}
      </div>

      {/* Model picker dropdown */}
      {showModelPicker && displayModels.length > 0 && (
        <div className="border border-zinc-700 rounded-md bg-zinc-800 divide-y divide-zinc-700 max-h-48 overflow-y-auto">
          {displayModels.map((m) => {
            // LM Studio model IDs can be very long paths — show filename only as label
            const label = m.includes('/') ? m.split('/').pop()! : m
            const isCurrent = modelId === m
            return (
              <button
                key={m}
                onClick={() => { setModelId(m); setShowModelPicker(false) }}
                className={`w-full text-left px-3 py-2 hover:bg-zinc-700 transition-colors ${isCurrent ? 'bg-zinc-700/50' : ''}`}
              >
                <div className={`text-sm font-mono truncate ${isCurrent ? 'text-violet-400' : 'text-zinc-300'}`}>{label}</div>
                {label !== m && <div className="text-[10px] text-zinc-600 truncate font-mono mt-0.5">{m}</div>}
              </button>
            )
          })}
        </div>
      )}
      {showModelPicker && modelFetchError && (
        <div className="flex items-start gap-2 border border-red-500/20 rounded-md bg-red-500/5 px-3 py-3 text-xs text-red-400">
          <XCircle size={13} className="shrink-0 mt-0.5" />
          <span>{modelFetchError}</span>
        </div>
      )}
      {showModelPicker && displayModels.length === 0 && !isLoadingModels && !modelFetchError && (
        <div className="border border-zinc-700 rounded-md bg-zinc-800 px-3 py-3 text-xs text-zinc-500">
          No models found.{kind === 'ollama' ? ' Run `ollama pull <model>` to download one.' : ' Load a model in LM Studio first.'}
        </div>
      )}

      {/* Test connection (edit mode only) */}
      {isEdit && (
        <div className="space-y-2">
          <button
            onClick={handleTest}
            disabled={isTesting}
            className="flex items-center gap-2 px-3 py-2 text-xs bg-zinc-700 hover:bg-zinc-600 text-zinc-300 rounded-md transition-colors disabled:opacity-50"
          >
            {isTesting ? <Loader2 size={13} className="animate-spin" /> : <Wifi size={13} />}
            Test connection
          </button>
          {testResult && <TestStatus ok={testResult.ok} message={testResult.message} />}
        </div>
      )}

      {formError && (
        <p className="text-xs text-red-400 bg-red-500/10 border border-red-500/20 rounded px-3 py-2">{formError}</p>
      )}

      {/* Actions */}
      <div className="flex gap-2 justify-end pt-2">
        <button
          onClick={onClose}
          className="px-4 py-2 text-sm text-zinc-400 hover:text-zinc-200 transition-colors"
        >
          Cancel
        </button>
        <button
          onClick={handleSave}
          disabled={saving}
          className="flex items-center gap-2 px-4 py-2 text-sm bg-violet-600 hover:bg-violet-500 text-white rounded-md transition-colors disabled:opacity-50"
        >
          {saving && <Loader2 size={14} className="animate-spin" />}
          {isEdit ? 'Save changes' : 'Add provider'}
        </button>
      </div>
    </div>
  )
}

// ──────────────────────────────────────────────────────────────────────────────
// Provider card
// ──────────────────────────────────────────────────────────────────────────────
function ProviderCard({ config }: { config: ProviderConfig }) {
  const { remove, testConnection, testing, testResults } = useProviderStore()
  const [editing, setEditing] = useState(false)
  const [confirmDelete, setConfirmDelete] = useState(false)

  const testResult = testResults[config.id]
  const isTesting = testing[config.id] ?? false

  if (editing) {
    return (
      <div className="bg-zinc-800/60 border border-zinc-700 rounded-xl p-5">
        <ProviderForm kind={config.kind} initial={config} onClose={() => setEditing(false)} />
      </div>
    )
  }

  return (
    <div className="bg-zinc-800/60 border border-zinc-700 rounded-xl p-5 space-y-3 hover:border-zinc-600 transition-colors">
      <div className="flex items-start justify-between gap-3">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 flex-wrap">
            <span className="text-sm font-semibold text-zinc-100 truncate">{config.display_name}</span>
            <KindLabel kind={config.kind} />
            <PrivacyBadge kind={config.kind} />
          </div>
          <p className="mt-1 text-xs text-zinc-500 font-mono truncate">{config.base_url}</p>
        </div>
        <div className="flex items-center gap-1 shrink-0">
          <button
            onClick={() => setEditing(true)}
            className="p-1.5 text-zinc-500 hover:text-zinc-300 hover:bg-zinc-700 rounded transition-colors"
            title="Edit"
          >
            <Pencil size={14} />
          </button>
          <button
            onClick={() => setConfirmDelete(true)}
            className="p-1.5 text-zinc-500 hover:text-red-400 hover:bg-zinc-700 rounded transition-colors"
            title="Delete"
          >
            <Trash2 size={14} />
          </button>
        </div>
      </div>

      <div className="flex items-center gap-3 flex-wrap">
        <span className="text-xs text-zinc-400">
          Model: <span className="font-mono text-zinc-300">{config.model_id || <em className="text-zinc-600">not set</em>}</span>
        </span>
        <span className="text-xs text-zinc-600">·</span>
        <span className="text-xs text-zinc-400">
          Context: <span className="text-zinc-300">{config.capabilities.max_context_tokens.toLocaleString()} tokens</span>
        </span>
      </div>

      <div className="flex items-center gap-2">
        <button
          onClick={() => testConnection(config.id)}
          disabled={isTesting}
          className="flex items-center gap-1.5 px-3 py-1.5 text-xs bg-zinc-700 hover:bg-zinc-600 text-zinc-300 rounded-md transition-colors disabled:opacity-50"
        >
          {isTesting ? <Loader2 size={12} className="animate-spin" /> : <RefreshCw size={12} />}
          Test connection
        </button>
        {testResult && <TestStatus ok={testResult.ok} message={testResult.message} />}
      </div>

      {confirmDelete && (
        <div className="flex items-center gap-3 bg-red-500/5 border border-red-500/20 rounded-lg px-4 py-3">
          <span className="text-xs text-red-400 flex-1">Remove this provider?</span>
          <button
            onClick={() => { remove(config.id); setConfirmDelete(false) }}
            className="text-xs px-3 py-1.5 bg-red-600 hover:bg-red-500 text-white rounded transition-colors"
          >
            Remove
          </button>
          <button
            onClick={() => setConfirmDelete(false)}
            className="text-xs px-3 py-1.5 text-zinc-400 hover:text-zinc-200 transition-colors"
          >
            Cancel
          </button>
        </div>
      )}
    </div>
  )
}

// ──────────────────────────────────────────────────────────────────────────────
// Cloud provider section (shared across OpenAI / Anthropic / Gemini / DeepSeek)
// ──────────────────────────────────────────────────────────────────────────────
interface CloudSectionProps {
  kind: 'openai' | 'anthropic' | 'gemini' | 'deepseek'
  title: string
  description: string
  providers: ProviderConfig[]
  adding: string | null
  onAdd: (kind: string) => void
  onCloseAdd: () => void
}

function CloudSection({ kind, title, description, providers: list, adding, onAdd, onCloseAdd }: CloudSectionProps) {
  const accentColor = {
    openai: 'text-green-400 bg-green-500/10 border-green-500/20',
    anthropic: 'text-orange-400 bg-orange-500/10 border-orange-500/20',
    gemini: 'text-blue-400 bg-blue-500/10 border-blue-500/20',
    deepseek: 'text-indigo-400 bg-indigo-500/10 border-indigo-500/20',
  }[kind]

  return (
    <div>
      <div className="flex items-center justify-between mb-3">
        <div>
          <h3 className="text-sm font-semibold text-zinc-300">{title}</h3>
          <p className="text-xs text-zinc-500 mt-0.5">{description}</p>
        </div>
        {adding !== kind && (
          <button
            onClick={() => onAdd(kind)}
            className="flex items-center gap-1.5 px-3 py-1.5 text-xs bg-zinc-700 hover:bg-zinc-600 text-zinc-300 rounded-md transition-colors"
          >
            <Plus size={13} />
            Add {title}
          </button>
        )}
      </div>
      <div className="space-y-3">
        {list.map((p) => <ProviderCard key={p.id} config={p} />)}
        {adding === kind && (
          <div className={`bg-zinc-800/60 border rounded-xl p-5 ${accentColor.split(' ')[0] === 'text-green-400' ? 'border-green-500/20' : accentColor.split(' ')[0] === 'text-orange-400' ? 'border-orange-500/20' : accentColor.split(' ')[0] === 'text-blue-400' ? 'border-blue-500/20' : 'border-indigo-500/20'}`}>
            <ProviderForm kind={kind} onClose={onCloseAdd} />
          </div>
        )}
        {list.length === 0 && adding !== kind && (
          <div className="border border-dashed border-zinc-700 rounded-xl p-5 text-center">
            <p className="text-sm text-zinc-500">No {title} providers configured.</p>
            <button onClick={() => onAdd(kind)} className="mt-2 text-xs text-amber-400 hover:text-amber-300 underline">
              Add {title}
            </button>
          </div>
        )}
      </div>
    </div>
  )
}

// ──────────────────────────────────────────────────────────────────────────────
// Main screen
// ──────────────────────────────────────────────────────────────────────────────
type AddKind = string | null

export default function ProvidersScreen() {
  const { providers, loading, error, load, clearError } = useProviderStore()
  const [adding, setAdding] = useState<AddKind>(null)
  const [consentGiven, setConsentGiven] = useState<boolean | null>(null)
  const [showConsent, setShowConsent] = useState(false)
  const pendingKindRef = useRef<string | null>(null)

  useEffect(() => { load() }, [load])

  useEffect(() => {
    window.api.consent.checkCloud().then((r) => setConsentGiven(r.given)).catch(() => setConsentGiven(false))
  }, [])

  const ollamaProviders = providers.filter((p) => p.kind === 'ollama')
  const lmStudioProviders = providers.filter((p) => p.kind === 'lmstudio')
  const openaiProviders = providers.filter((p) => p.kind === 'openai')
  const anthropicProviders = providers.filter((p) => p.kind === 'anthropic')
  const geminiProviders = providers.filter((p) => p.kind === 'gemini')
  const deepseekProviders = providers.filter((p) => p.kind === 'deepseek')

  const handleAddCloud = (kind: string) => {
    if (!consentGiven) {
      pendingKindRef.current = kind
      setShowConsent(true)
    } else {
      setAdding(kind)
    }
  }

  const handleConsentAccept = async () => {
    await window.api.consent.giveCloud(true)
    setConsentGiven(true)
    setShowConsent(false)
    if (pendingKindRef.current) {
      setAdding(pendingKindRef.current)
      pendingKindRef.current = null
    }
  }

  return (
    <div className="flex flex-col h-full">
      {showConsent && (
        <ConsentBanner
          onAccept={handleConsentAccept}
          onDismiss={() => { setShowConsent(false); pendingKindRef.current = null }}
        />
      )}
      {/* Header */}
      <div className="shrink-0 px-8 pt-8 pb-4 border-b border-zinc-800">
        <div className="flex items-start justify-between">
          <div>
            <h1 className="text-xl font-semibold text-zinc-100">Model Providers</h1>
            <p className="mt-1 text-sm text-zinc-500">
              Connect local AI models to power analysis. All local providers keep your code on-device.
            </p>
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto px-8 py-6 space-y-8">
        {error && (
          <div className="flex items-center gap-3 bg-red-500/10 border border-red-500/20 text-red-400 text-sm rounded-lg px-4 py-3">
            <XCircle size={16} />
            <span className="flex-1">{error}</span>
            <button onClick={clearError} className="text-xs underline">Dismiss</button>
          </div>
        )}

        {loading && (
          <div className="flex items-center gap-2 text-zinc-500 text-sm">
            <Loader2 size={16} className="animate-spin" />
            Loading providers...
          </div>
        )}

        {/* Ollama section */}
        <section>
          <div className="flex items-center justify-between mb-3">
            <div>
              <h2 className="text-sm font-semibold text-zinc-300">Ollama</h2>
              <p className="text-xs text-zinc-500 mt-0.5">Run open-source models locally via Ollama's REST API</p>
            </div>
            {adding !== 'ollama' && (
              <button
                onClick={() => setAdding('ollama')}
                className="flex items-center gap-1.5 px-3 py-1.5 text-xs bg-zinc-700 hover:bg-zinc-600 text-zinc-300 rounded-md transition-colors"
              >
                <Plus size={13} />
                Add Ollama
              </button>
            )}
          </div>

          <div className="space-y-3">
            {ollamaProviders.map((p) => <ProviderCard key={p.id} config={p} />)}

            {adding === 'ollama' && (
              <div className="bg-zinc-800/60 border border-violet-500/30 rounded-xl p-5">
                <ProviderForm kind="ollama" onClose={() => setAdding(null)} />
              </div>
            )}

            {ollamaProviders.length === 0 && adding !== 'ollama' && (
              <div className="border border-dashed border-zinc-700 rounded-xl p-6 text-center">
                <p className="text-sm text-zinc-500">No Ollama providers configured yet.</p>
                <button
                  onClick={() => setAdding('ollama')}
                  className="mt-3 text-xs text-violet-400 hover:text-violet-300 underline transition-colors"
                >
                  Add your first Ollama provider
                </button>
              </div>
            )}
          </div>
        </section>

        {/* LM Studio section */}
        <section>
          <div className="flex items-center justify-between mb-3">
            <div>
              <h2 className="text-sm font-semibold text-zinc-300">LM Studio</h2>
              <p className="text-xs text-zinc-500 mt-0.5">OpenAI-compatible local server via LM Studio</p>
            </div>
            {adding !== 'lmstudio' && (
              <button
                onClick={() => setAdding('lmstudio')}
                className="flex items-center gap-1.5 px-3 py-1.5 text-xs bg-zinc-700 hover:bg-zinc-600 text-zinc-300 rounded-md transition-colors"
              >
                <Plus size={13} />
                Add LM Studio
              </button>
            )}
          </div>

          <div className="space-y-3">
            <LMStudioSetupGuide />
            {lmStudioProviders.map((p) => <ProviderCard key={p.id} config={p} />)}

            {adding === 'lmstudio' && (
              <div className="bg-zinc-800/60 border border-sky-500/30 rounded-xl p-5">
                <ProviderForm kind="lmstudio" onClose={() => setAdding(null)} />
              </div>
            )}

            {lmStudioProviders.length === 0 && adding !== 'lmstudio' && (
              <div className="border border-dashed border-zinc-700 rounded-xl p-6 text-center">
                <p className="text-sm text-zinc-500">No LM Studio providers configured yet.</p>
                <button
                  onClick={() => setAdding('lmstudio')}
                  className="mt-3 text-xs text-sky-400 hover:text-sky-300 underline transition-colors"
                >
                  Add your first LM Studio provider
                </button>
              </div>
            )}
          </div>
        </section>

        {/* Cloud provider divider */}
        <div className="flex items-center gap-3 pt-2">
          <div className="flex-1 h-px bg-zinc-800" />
          <div className="flex items-center gap-1.5 text-xs text-zinc-500 font-medium">
            <Cloud size={11} />
            Cloud Providers (BYOK)
          </div>
          <div className="flex-1 h-px bg-zinc-800" />
        </div>
        <p className="text-xs text-zinc-600">
          Bring Your Own Key — code may be sent to external servers. One-time consent required.
        </p>

        <CloudSection
          kind="openai" title="OpenAI" description="GPT-4o, o3-mini, and other OpenAI models"
          providers={openaiProviders} adding={adding} onAdd={handleAddCloud} onCloseAdd={() => setAdding(null)}
        />
        <CloudSection
          kind="anthropic" title="Anthropic" description="Claude Opus, Sonnet, Haiku models"
          providers={anthropicProviders} adding={adding} onAdd={handleAddCloud} onCloseAdd={() => setAdding(null)}
        />
        <CloudSection
          kind="gemini" title="Google Gemini" description="Gemini 2.0 Flash, 1.5 Pro, and others"
          providers={geminiProviders} adding={adding} onAdd={handleAddCloud} onCloseAdd={() => setAdding(null)}
        />
        <CloudSection
          kind="deepseek" title="DeepSeek" description="DeepSeek Chat and DeepSeek Reasoner"
          providers={deepseekProviders} adding={adding} onAdd={handleAddCloud} onCloseAdd={() => setAdding(null)}
        />
      </div>
    </div>
  )
}

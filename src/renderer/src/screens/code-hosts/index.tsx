import React, { useEffect, useRef, useState } from 'react'
import {
  FolderOpen,
  GitBranch,
  GitCommit,
  Globe,
  Loader2,
  Plus,
  RefreshCw,
  Shield,
  Trash2,
  AlertTriangle,
  ChevronDown,
  XCircle,
  Link,
  ArrowDownToLine,
  KeyRound,
  Check,
  X,
} from 'lucide-react'
import { useLocalRepoStore } from '../../store/local-repo.store'
import { ErrorBanner } from '../../components/ui/ErrorBanner'
import { LoadingRow } from '../../components/ui/LoadingRow'
import type { LocalRepo, ValidateFolderResponse } from '../../types/electron'

// ──────────────────────────────────────────────────────────────────────────────
// Validation preview card — shown after folder picker before saving
// ──────────────────────────────────────────────────────────────────────────────
function ValidationPreview({
  result,
  onConfirm,
  onCancel,
  adding,
}: {
  result: ValidateFolderResponse
  onConfirm: () => void
  onCancel: () => void
  adding: boolean
}) {
  const ok = result.exists && result.is_directory

  return (
    <div className="bg-zinc-800/60 border border-zinc-700 rounded-xl p-5 space-y-4">
      <div className="flex items-start gap-3">
        <div className={`shrink-0 mt-0.5 w-8 h-8 rounded-lg flex items-center justify-center ${ok ? 'bg-emerald-500/10 border border-emerald-500/20' : 'bg-red-500/10 border border-red-500/20'}`}>
          {ok ? <FolderOpen size={15} className="text-emerald-400" /> : <XCircle size={15} className="text-red-400" />}
        </div>
        <div className="min-w-0 flex-1">
          <p className="text-sm font-semibold text-zinc-100 truncate">{result.name}</p>
          <p className="text-xs text-zinc-500 truncate font-mono mt-0.5">{result.path}</p>
        </div>
      </div>

      {!ok && (
        <div className="flex items-center gap-2 bg-red-500/10 border border-red-500/20 rounded-lg px-3 py-2 text-sm text-red-400">
          <XCircle size={13} />
          {!result.exists ? 'Path does not exist' : 'Path is not a directory'}
        </div>
      )}

      {ok && (
        <div className="space-y-2">
          {result.is_git_repo ? (
            <div className="flex flex-wrap items-center gap-2">
              <span className="inline-flex items-center gap-1.5 px-2 py-1 rounded-md text-xs font-medium bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                <GitBranch size={11} />
                {result.git_branch ?? 'detached HEAD'}
              </span>
              {result.git_head_hash && (
                <span className="inline-flex items-center gap-1.5 px-2 py-1 rounded-md text-xs font-mono bg-zinc-700/60 text-zinc-400 border border-zinc-600">
                  <GitCommit size={11} />
                  {result.git_head_hash}
                </span>
              )}
              {result.git_remote_url && (
                <span className="inline-flex items-center gap-1.5 px-2 py-1 rounded-md text-xs bg-zinc-700/60 text-zinc-500 border border-zinc-600 max-w-xs truncate">
                  <Globe size={11} />
                  {result.git_remote_url}
                </span>
              )}
            </div>
          ) : (
            <div className="flex items-center gap-2 bg-amber-500/10 border border-amber-500/20 rounded-lg px-3 py-2 text-xs text-amber-400">
              <AlertTriangle size={13} />
              No .git folder found — folder will be indexed without commit history
            </div>
          )}

          {result.has_size_warning && (
            <div className="flex items-start gap-2 bg-amber-500/5 border border-amber-500/15 rounded-lg px-3 py-2 text-xs text-amber-400/80">
              <AlertTriangle size={12} className="shrink-0 mt-0.5" />
              <span><strong>Large folder:</strong> {result.size_warning_reason}. Initial scan may take longer.</span>
            </div>
          )}

          <div className="flex items-center gap-1.5 text-xs text-emerald-400">
            <Shield size={12} />
            Strict Local — no data leaves this device
          </div>
        </div>
      )}

      <div className="flex items-center gap-2 pt-1">
        <button
          onClick={onCancel}
          className="flex-1 py-2 text-sm text-zinc-400 hover:text-zinc-200 border border-zinc-700 hover:border-zinc-600 rounded-lg transition-colors"
        >
          Cancel
        </button>
        <button
          onClick={onConfirm}
          disabled={!ok || adding}
          className="flex-1 py-2 text-sm font-medium bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 disabled:cursor-not-allowed text-white rounded-lg transition-colors flex items-center justify-center gap-2"
        >
          {adding ? <Loader2 size={14} className="animate-spin" /> : <Plus size={14} />}
          {adding ? 'Adding…' : 'Add folder'}
        </button>
      </div>
    </div>
  )
}

// ──────────────────────────────────────────────────────────────────────────────
// Inline branch picker dropdown
// ──────────────────────────────────────────────────────────────────────────────
function BranchPicker({ repo }: { repo: LocalRepo }) {
  const { branchesMap, loadingBranchesId, loadBranches, setBranch } = useLocalRepoStore()
  const [open, setOpen] = useState(false)
  const ref = useRef<HTMLDivElement>(null)

  const branches = branchesMap[repo.id]
  const activeBranch = repo.selected_branch ?? repo.git_branch ?? 'HEAD'
  const isLoading = loadingBranchesId === repo.id

  const handleOpen = async () => {
    if (!branches) await loadBranches(repo.id)
    setOpen((v) => !v)
  }

  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false)
    }
    if (open) document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [open])

  return (
    <div ref={ref} className="relative">
      <button
        onClick={handleOpen}
        className={`inline-flex items-center gap-1.5 px-2 py-1 rounded-md text-xs font-medium border transition-colors
          ${repo.selected_branch
            ? 'bg-indigo-500/10 text-indigo-400 border-indigo-500/20 hover:bg-indigo-500/20'
            : 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20 hover:bg-emerald-500/20'
          }`}
        title="Click to change analysis branch"
      >
        {isLoading ? <Loader2 size={10} className="animate-spin" /> : <GitBranch size={10} />}
        {activeBranch}
        {repo.selected_branch && repo.selected_branch !== repo.git_branch && (
          <span className="text-indigo-500/70 text-xs">≠ HEAD</span>
        )}
        <ChevronDown size={10} />
      </button>

      {open && branches && (
        <div className="absolute left-0 top-full mt-1 z-30 w-52 bg-zinc-800 border border-zinc-700 rounded-lg shadow-xl overflow-hidden">
          <div className="px-3 py-2 text-xs text-zinc-500 font-medium border-b border-zinc-700">
            Select analysis branch
          </div>
          <div className="max-h-48 overflow-y-auto">
            {branches.map((b) => (
              <button
                key={b}
                onClick={() => { setBranch(repo.id, b); setOpen(false) }}
                className={`w-full text-left px-3 py-2 text-xs flex items-center gap-2 hover:bg-zinc-700 transition-colors
                  ${b === activeBranch ? 'text-zinc-100 font-medium' : 'text-zinc-400'}`}
              >
                <GitBranch size={11} className={b === activeBranch ? 'text-emerald-400' : 'text-zinc-600'} />
                <span className="flex-1 truncate">{b}</span>
                {b === repo.git_branch && (
                  <span className="text-zinc-600 text-xs shrink-0">HEAD</span>
                )}
                {b === activeBranch && b !== repo.git_branch && (
                  <span className="text-indigo-400 text-xs shrink-0">selected</span>
                )}
              </button>
            ))}
          </div>
          {branches.length === 0 && (
            <div className="px-3 py-4 text-xs text-zinc-500 text-center">No local branches found</div>
          )}
        </div>
      )}
    </div>
  )
}

// ──────────────────────────────────────────────────────────────────────────────
// Repo card — shows a saved local folder
// ──────────────────────────────────────────────────────────────────────────────
function LocalRepoCard({
  repo,
  onRemove,
  onRevalidate,
  isRevalidating,
}: {
  repo: LocalRepo
  onRemove: () => void
  onRevalidate: () => void
  isRevalidating: boolean
}) {
  return (
    <div className="bg-zinc-800/40 border border-zinc-700/70 rounded-xl p-4 hover:border-zinc-600 transition-colors">
      <div className="flex items-start gap-3">
        <div className="shrink-0 w-9 h-9 rounded-lg bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-center mt-0.5">
          <FolderOpen size={16} className="text-emerald-400" />
        </div>

        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 flex-wrap">
            <span className="text-sm font-semibold text-zinc-100">{repo.name}</span>

            {repo.is_git_repo ? (
              <BranchPicker repo={repo} />
            ) : (
              <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs bg-amber-500/10 text-amber-400 border border-amber-500/20">
                <AlertTriangle size={10} />
                No git
              </span>
            )}

            <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
              <Shield size={10} />
              Strict Local
            </span>
          </div>

          <p className="text-xs text-zinc-500 font-mono mt-1 truncate">{repo.path}</p>

          <div className="flex items-center gap-3 mt-2 flex-wrap">
            {repo.git_head_hash && (
              <span className="inline-flex items-center gap-1 text-xs text-zinc-500 font-mono">
                <GitCommit size={10} />
                {repo.git_head_hash}
              </span>
            )}
            {repo.git_remote_url && (
              <span className="inline-flex items-center gap-1 text-xs text-zinc-600 truncate max-w-xs">
                <Link size={10} />
                {repo.git_remote_url}
              </span>
            )}
            {repo.selected_branch && repo.selected_branch !== repo.git_branch && (
              <span className="text-xs text-indigo-400/70">
                Analysis: <span className="font-mono">{repo.selected_branch}</span>
              </span>
            )}
          </div>
        </div>

        <div className="flex items-center gap-1 shrink-0">
          <button
            onClick={onRevalidate}
            disabled={isRevalidating}
            title="Refresh git metadata"
            className="p-1.5 text-zinc-500 hover:text-zinc-300 hover:bg-zinc-700 rounded-md transition-colors disabled:opacity-40"
          >
            <RefreshCw size={13} className={isRevalidating ? 'animate-spin' : ''} />
          </button>
          <button
            onClick={onRemove}
            title="Remove folder"
            className="p-1.5 text-zinc-500 hover:text-red-400 hover:bg-red-500/10 rounded-md transition-colors"
          >
            <Trash2 size={13} />
          </button>
        </div>
      </div>
    </div>
  )
}

// ──────────────────────────────────────────────────────────────────────────────
// Add folder panel — folder picker + validation flow
// ──────────────────────────────────────────────────────────────────────────────
function AddFolderPanel({ onClose }: { onClose: () => void }) {
  const { validate, clearValidation, validation, validating, add, adding, error } = useLocalRepoStore()

  const handlePick = async () => {
    clearValidation()
    const picked = await window.api.folder.pick()
    if (picked) await validate(picked)
  }

  const handleConfirm = async () => {
    if (!validation?.path) return
    const repo = await add(validation.path)
    if (repo) onClose()
  }

  return (
    <div className="bg-zinc-800/60 border border-zinc-700 rounded-xl p-5 space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-zinc-200">Open Local Folder</h3>
        <button onClick={onClose} className="text-xs text-zinc-500 hover:text-zinc-300 transition-colors">
          Cancel
        </button>
      </div>

      {!validation && (
        <button
          onClick={handlePick}
          disabled={validating}
          className="w-full flex items-center justify-center gap-2 py-8 border-2 border-dashed border-zinc-600 hover:border-emerald-500/40 hover:bg-emerald-500/5 rounded-xl text-sm text-zinc-400 hover:text-zinc-200 transition-all"
        >
          {validating ? (
            <><Loader2 size={16} className="animate-spin" /> Validating…</>
          ) : (
            <><FolderOpen size={16} /> Browse for folder</>
          )}
        </button>
      )}

      {error && !validation && <ErrorBanner message={error} />}

      {validation && (
        <ValidationPreview
          result={validation}
          onConfirm={handleConfirm}
          onCancel={() => { clearValidation(); onClose() }}
          adding={adding}
        />
      )}
    </div>
  )
}

// ──────────────────────────────────────────────────────────────────────────────
// Clone from URL panel
// ──────────────────────────────────────────────────────────────────────────────
function CloneFromUrlPanel({ onClose, onCloned }: { onClose: () => void; onCloned: () => void }) {
  const [url, setUrl] = useState('')
  const [cloning, setCloning] = useState(false)
  const [cloneError, setCloneError] = useState<string | null>(null)

  const repoName = url.trim()
    ? (url.trim().split('/').pop()?.replace(/\.git$/, '') ?? '')
    : ''

  const handleClone = async () => {
    const trimmed = url.trim()
    if (!trimmed) return
    setCloning(true)
    setCloneError(null)
    try {
      await window.api.folder.cloneFromUrl(trimmed)
      onCloned()
      onClose()
    } catch (err) {
      setCloneError(err instanceof Error ? err.message : String(err))
    } finally {
      setCloning(false)
    }
  }

  return (
    <div className="bg-zinc-800/60 border border-zinc-700 rounded-xl p-5 space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-zinc-200">Clone from URL</h3>
        <button onClick={onClose} className="text-xs text-zinc-500 hover:text-zinc-300 transition-colors">
          Cancel
        </button>
      </div>

      <div className="space-y-2">
        <label className="text-xs text-zinc-500">Git URL (HTTPS or SSH)</label>
        <input
          type="text"
          value={url}
          onChange={(e) => { setUrl(e.target.value); setCloneError(null) }}
          onKeyDown={(e) => { if (e.key === 'Enter') handleClone() }}
          placeholder="https://github.com/user/repo.git  or  git@github.com:user/repo.git"
          autoFocus
          disabled={cloning}
          className="w-full px-3 py-2 bg-zinc-900 border border-zinc-700 rounded-lg text-sm text-zinc-200 placeholder-zinc-600 focus:outline-none focus:border-zinc-500 font-mono transition-colors disabled:opacity-50"
        />
      </div>

      {repoName && (
        <p className="text-xs text-zinc-500">
          Will clone into{' '}
          <code className="text-zinc-400 bg-zinc-800 px-1 py-0.5 rounded">
            ~/CodeSpectra/repos/{repoName}
          </code>
        </p>
      )}

      {cloneError && <ErrorBanner message={cloneError} onDismiss={() => setCloneError(null)} />}

      {cloning && (
        <div className="flex items-center gap-2 text-xs text-zinc-400 py-2">
          <Loader2 size={13} className="animate-spin" />
          Cloning… this may take a moment
        </div>
      )}

      <div className="flex items-center gap-2 pt-1">
        <button
          onClick={onClose}
          disabled={cloning}
          className="flex-1 py-2 text-sm text-zinc-400 hover:text-zinc-200 border border-zinc-700 hover:border-zinc-600 rounded-lg transition-colors disabled:opacity-50"
        >
          Cancel
        </button>
        <button
          onClick={handleClone}
          disabled={!url.trim() || cloning}
          className="flex-1 py-2 text-sm font-medium bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 disabled:cursor-not-allowed text-white rounded-lg transition-colors flex items-center justify-center gap-2"
        >
          {cloning ? <Loader2 size={14} className="animate-spin" /> : <ArrowDownToLine size={14} />}
          {cloning ? 'Cloning…' : 'Clone'}
        </button>
      </div>
    </div>
  )
}

// ──────────────────────────────────────────────────────────────────────────────
// SSH key settings — collapsible row at the bottom of the screen
// ──────────────────────────────────────────────────────────────────────────────
function SshKeySettings() {
  const [keyPath, setKeyPath] = useState<string | null>(null)
  const [open, setOpen] = useState(false)
  const [saving, setSaving] = useState(false)
  const [saved, setSaved] = useState(false)
  const [draft, setDraft] = useState<string | null>(null)

  useEffect(() => {
    window.api.git.getConfig().then((cfg) => {
      setKeyPath(cfg.ssh_key_path)
      setDraft(cfg.ssh_key_path)
    })
  }, [])

  const handlePick = async () => {
    const picked = await window.api.git.pickSshKey()
    if (picked) setDraft(picked)
  }

  const handleSave = async () => {
    setSaving(true)
    try {
      const result = await window.api.git.setConfig(draft ?? null)
      setKeyPath(result.ssh_key_path)
      setSaved(true)
      setTimeout(() => setSaved(false), 2000)
    } finally {
      setSaving(false)
    }
  }

  const handleClear = async () => {
    setSaving(true)
    try {
      await window.api.git.setConfig(null)
      setKeyPath(null)
      setDraft(null)
      setSaved(true)
      setTimeout(() => setSaved(false), 2000)
    } finally {
      setSaving(false)
    }
  }

  const isDirty = draft !== keyPath

  return (
    <div className="border border-zinc-800 rounded-xl overflow-hidden">
      <button
        onClick={() => setOpen((v) => !v)}
        className="w-full flex items-center gap-2.5 px-4 py-3 text-left hover:bg-zinc-800/40 transition-colors"
      >
        <KeyRound size={14} className="text-zinc-500 shrink-0" />
        <span className="text-xs font-medium text-zinc-400 flex-1">SSH Key for Git Clone</span>
        {keyPath && (
          <span className="text-xs text-emerald-400 font-mono truncate max-w-[180px]">
            {keyPath.split(/[\\/]/).pop()}
          </span>
        )}
        {!keyPath && (
          <span className="text-xs text-zinc-600">not set — uses system default</span>
        )}
        <ChevronDown
          size={13}
          className={`text-zinc-600 shrink-0 transition-transform ${open ? 'rotate-180' : ''}`}
        />
      </button>

      {open && (
        <div className="px-4 pb-4 space-y-3 border-t border-zinc-800">
          <p className="text-xs text-zinc-500 pt-3 leading-relaxed">
            Used for <code className="text-zinc-400 bg-zinc-800 px-1 rounded">git@</code> and{' '}
            <code className="text-zinc-400 bg-zinc-800 px-1 rounded">ssh://</code> URLs.
            If not set, git uses your system SSH agent or default{' '}
            <code className="text-zinc-400 bg-zinc-800 px-1 rounded">~/.ssh/id_*</code> keys.
          </p>

          <div className="flex items-center gap-2">
            <input
              type="text"
              value={draft ?? ''}
              onChange={(e) => setDraft(e.target.value || null)}
              placeholder="~/.ssh/id_ed25519"
              className="flex-1 px-3 py-2 bg-zinc-900 border border-zinc-700 rounded-lg text-xs text-zinc-300 placeholder-zinc-600 focus:outline-none focus:border-zinc-500 font-mono transition-colors"
            />
            <button
              onClick={handlePick}
              className="px-3 py-2 text-xs text-zinc-400 hover:text-zinc-200 border border-zinc-700 hover:border-zinc-500 rounded-lg transition-colors whitespace-nowrap"
            >
              Browse…
            </button>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={handleSave}
              disabled={!isDirty || saving}
              className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium bg-emerald-600 hover:bg-emerald-500 disabled:opacity-40 disabled:cursor-not-allowed text-white rounded-lg transition-colors"
            >
              {saving ? <Loader2 size={12} className="animate-spin" /> : saved ? <Check size={12} /> : null}
              {saved ? 'Saved' : 'Save'}
            </button>
            {keyPath && (
              <button
                onClick={handleClear}
                disabled={saving}
                className="flex items-center gap-1.5 px-3 py-1.5 text-xs text-zinc-500 hover:text-red-400 border border-zinc-700 hover:border-red-500/30 rounded-lg transition-colors"
              >
                <X size={12} />
                Clear
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  )
}

// ──────────────────────────────────────────────────────────────────────────────
// Main screen
// ──────────────────────────────────────────────────────────────────────────────
export default function CodeHostsSetup(): React.ReactElement {
  const { repos, loading, error, load, remove, revalidate, revalidatingId, clearError } = useLocalRepoStore()
  const [panel, setPanel] = useState<'none' | 'folder' | 'clone'>('none')

  useEffect(() => { load() }, [load])

  return (
    <>
      <div className="screen-header flex items-center justify-between">
        <div>
        <h1 className="screen-title">Code Hosts</h1>
          <p className="screen-subtitle">
            {repos.length > 0
              ? `${repos.length} repositor${repos.length !== 1 ? 'ies' : 'y'}`
              : 'Open a local folder or clone any git repository'}
          </p>
        </div>
        {panel === 'none' && (
          <div className="flex items-center gap-2">
            <button className="btn-secondary text-xs" onClick={() => setPanel('clone')}>
              <ArrowDownToLine className="w-3.5 h-3.5" />
              Clone URL
            </button>
            <button className="btn-primary" onClick={() => setPanel('folder')}>
              <Plus className="w-4 h-4" />
              Open Folder
            </button>
          </div>
        )}
      </div>

      <div className="p-6 space-y-4 overflow-y-auto">
        {error && <ErrorBanner message={error} onDismiss={clearError} />}

        {panel === 'folder' && (
          <AddFolderPanel onClose={() => setPanel('none')} />
        )}

        {panel === 'clone' && (
          <CloneFromUrlPanel
            onClose={() => setPanel('none')}
            onCloned={() => { load(); setPanel('none') }}
          />
        )}

        {loading && <LoadingRow message="Loading repositories…" />}

        {repos.map((repo) => (
          <LocalRepoCard
            key={repo.id}
            repo={repo}
            onRemove={() => remove(repo.id)}
            onRevalidate={() => revalidate(repo.id)}
            isRevalidating={revalidatingId === repo.id}
          />
        ))}

        {!loading && repos.length === 0 && panel === 'none' && (
          <div className="border border-dashed border-zinc-700 rounded-xl p-10 text-center">
            <div className="w-12 h-12 rounded-xl bg-zinc-800 border border-zinc-700 flex items-center justify-center mx-auto mb-4">
              <FolderOpen size={22} className="text-zinc-500" />
            </div>
            <p className="text-sm font-medium text-zinc-400">No repositories added yet</p>
            <p className="text-xs text-zinc-600 mt-1 max-w-xs mx-auto">
              Open any local folder or clone from a git URL. Authentication uses your existing SSH keys or credential manager — no tokens needed.
            </p>
            <div className="flex items-center justify-center gap-2 mt-4">
              <button className="btn-secondary text-xs" onClick={() => setPanel('clone')}>
                <ArrowDownToLine size={13} />
                Clone URL
              </button>
              <button className="btn-primary text-sm" onClick={() => setPanel('folder')}>
                <FolderOpen size={14} />
                Open Folder
              </button>
            </div>
          </div>
        )}

        <SshKeySettings />
      </div>
    </>
  )
}

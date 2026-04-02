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
  Search,
  LogOut,
  Copy,
  Check,
  ExternalLink,
  Lock,
  Unlock,
} from 'lucide-react'
import { useLocalRepoStore } from '../../store/local-repo.store'
import { useGitHubStore } from '../../store/github.store'
import { ErrorBanner } from '../../components/ui/ErrorBanner'
import { LoadingRow } from '../../components/ui/LoadingRow'
import { GitHubIcon } from '../../components/ui/GitHubIcon'
import type { GitHubRepo, LocalRepo, ValidateFolderResponse } from '../../types/electron'

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
      {/* Path */}
      <div className="flex items-start gap-3">
        <div className={`shrink-0 mt-0.5 w-8 h-8 rounded-lg flex items-center justify-center ${ok ? 'bg-emerald-500/10 border border-emerald-500/20' : 'bg-red-500/10 border border-red-500/20'}`}>
          {ok ? <FolderOpen size={15} className="text-emerald-400" /> : <XCircle size={15} className="text-red-400" />}
        </div>
        <div className="min-w-0 flex-1">
          <p className="text-sm font-semibold text-zinc-100 truncate">{result.name}</p>
          <p className="text-xs text-zinc-500 truncate font-mono mt-0.5">{result.path}</p>
        </div>
      </div>

      {/* Not a valid directory */}
      {!ok && (
        <div className="flex items-center gap-2 bg-red-500/10 border border-red-500/20 rounded-lg px-3 py-2 text-sm text-red-400">
          <XCircle size={13} />
          {!result.exists ? 'Path does not exist' : 'Path is not a directory'}
        </div>
      )}

      {/* Git info */}
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

          {/* Size warning */}
          {result.has_size_warning && (
            <div className="flex items-start gap-2 bg-amber-500/5 border border-amber-500/15 rounded-lg px-3 py-2 text-xs text-amber-400/80">
              <AlertTriangle size={12} className="shrink-0 mt-0.5" />
              <span><strong>Large folder:</strong> {result.size_warning_reason}. Initial scan may take longer.</span>
            </div>
          )}

          {/* Privacy badge */}
          <div className="flex items-center gap-1.5 text-xs text-emerald-400">
            <Shield size={12} />
            Strict Local — no data leaves this device
          </div>
        </div>
      )}

      {/* Actions */}
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

  // Close on outside click
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
        {/* Icon */}
        <div className="shrink-0 w-9 h-9 rounded-lg bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-center mt-0.5">
          <FolderOpen size={16} className="text-emerald-400" />
        </div>

        {/* Info */}
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 flex-wrap">
            <span className="text-sm font-semibold text-zinc-100">{repo.name}</span>

            {/* Branch picker (git repo) or no-git badge */}
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

        {/* Actions */}
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
// GitHub — Device Flow Wizard
// ──────────────────────────────────────────────────────────────────────────────
function GitHubConnectWizard({ onCancel }: { onCancel: () => void }) {
  const { deviceFlow, connecting, connectStatus, connectError, startConnect, pollConnect, cancelConnect } = useGitHubStore()
  const [copied, setCopied] = useState(false)
  // Polling interval ref
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const intervalRef = useRef(deviceFlow?.interval ?? 5)

  const stopPolling = () => {
    if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null }
  }

  useEffect(() => {
    if (deviceFlow && connectStatus === 'waiting') {
      intervalRef.current = deviceFlow.interval
      stopPolling()
      pollRef.current = setInterval(async () => {
        await pollConnect()
        const status = useGitHubStore.getState().connectStatus
        if (status === 'slow_down') {
          intervalRef.current = Math.min(intervalRef.current + 5, 30)
          stopPolling()
          pollRef.current = setInterval(() => pollConnect(), intervalRef.current * 1000)
        }
        if (!['waiting', 'slow_down'].includes(status)) stopPolling()
      }, intervalRef.current * 1000)
    }
    return stopPolling
  }, [deviceFlow, connectStatus, pollConnect])

  const handleCancel = () => { stopPolling(); cancelConnect(); onCancel() }

  const handleCopy = () => {
    if (deviceFlow?.user_code) {
      navigator.clipboard.writeText(deviceFlow.user_code)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    }
  }

  if (connectStatus === 'expired') return (
    <div className="bg-zinc-800/60 border border-zinc-700 rounded-xl p-5 space-y-4">
      <p className="text-sm text-amber-400 font-medium">Code expired. Start again.</p>
      <div className="flex gap-2">
        <button onClick={() => { startConnect() }} className="btn-primary text-xs">Try again</button>
        <button onClick={handleCancel} className="btn-secondary text-xs">Cancel</button>
      </div>
    </div>
  )

  if (connectStatus === 'denied') return (
    <div className="bg-zinc-800/60 border border-zinc-700 rounded-xl p-5 space-y-4">
      <p className="text-sm text-red-400 font-medium">Authorization denied.</p>
      <button onClick={handleCancel} className="btn-secondary text-xs">Close</button>
    </div>
  )

  if (connectStatus === 'error') return (
    <div className="bg-zinc-800/60 border border-zinc-700 rounded-xl p-5 space-y-4">
      <p className="text-sm text-red-400 font-medium">Error: {connectError}</p>
      <button onClick={handleCancel} className="btn-secondary text-xs">Close</button>
    </div>
  )

  return (
    <div className="bg-zinc-800/60 border border-zinc-700 rounded-xl p-5 space-y-5">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-zinc-200">Connect GitHub</h3>
        <button onClick={handleCancel} className="text-xs text-zinc-500 hover:text-zinc-300 transition-colors">Cancel</button>
      </div>

      {!deviceFlow && connecting && (
        <div className="flex items-center gap-2 text-sm text-zinc-400 py-4">
          <Loader2 size={14} className="animate-spin" />
          Starting device flow…
        </div>
      )}

      {deviceFlow && (
        <>
          <div className="text-sm text-zinc-400 space-y-1">
            <p>Your browser has been opened at:</p>
            <a
              href="#"
              onClick={(e) => { e.preventDefault(); window.api.github.openBrowser(deviceFlow.verification_uri) }}
              className="text-xs text-blue-400 hover:underline font-mono"
            >
              {deviceFlow.verification_uri}
            </a>
          </div>

          {/* User code — big & prominent */}
          <div className="flex flex-col items-center gap-3 py-4 border border-zinc-700 rounded-xl bg-zinc-900/60">
            <p className="text-xs text-zinc-500">Enter this code on GitHub</p>
            <div className="flex items-center gap-3">
              <span className="text-3xl font-mono font-bold tracking-widest text-zinc-100 select-all">
                {deviceFlow.user_code}
              </span>
              <button
                onClick={handleCopy}
                className="p-2 text-zinc-500 hover:text-zinc-300 hover:bg-zinc-700 rounded-md transition-colors"
                title="Copy code"
              >
                {copied ? <Check size={14} className="text-emerald-400" /> : <Copy size={14} />}
              </button>
            </div>
            <button
              onClick={() => window.api.github.openBrowser(deviceFlow.verification_uri)}
              className="flex items-center gap-1.5 text-xs text-blue-400 hover:text-blue-300 transition-colors"
            >
              <ExternalLink size={12} />
              Open GitHub again
            </button>
          </div>

          <div className="flex items-center gap-2 text-xs text-zinc-500">
            <Loader2 size={12} className="animate-spin" />
            Waiting for authorization…
          </div>
        </>
      )}
    </div>
  )
}

// ──────────────────────────────────────────────────────────────────────────────
// GitHub — Connected account card
// ──────────────────────────────────────────────────────────────────────────────
function GitHubAccountCard() {
  const { account, disconnect, loadRepos, repos, repoPage, repoHasMore, loadingRepos, repoError, repoQuery, loadMoreRepos, clearRepoError } = useGitHubStore()
  const [query, setQuery] = useState(repoQuery)
  const searchTimeout = useRef<ReturnType<typeof setTimeout> | null>(null)

  useEffect(() => { loadRepos() }, [loadRepos])

  const handleSearch = (val: string) => {
    setQuery(val)
    if (searchTimeout.current) clearTimeout(searchTimeout.current)
    searchTimeout.current = setTimeout(() => loadRepos(val, 1), 400)
  }

  if (!account) return null

  return (
    <div className="space-y-4">
      {/* Account card */}
      <div className="flex items-center gap-3 bg-zinc-800/40 border border-zinc-700/70 rounded-xl px-4 py-3">
        {account.avatar_url ? (
          <img src={account.avatar_url} alt={account.login} className="w-9 h-9 rounded-full border border-zinc-600" />
        ) : (
          <div className="w-9 h-9 rounded-full bg-zinc-700 border border-zinc-600 flex items-center justify-center text-zinc-400 font-semibold text-sm">
            {account.login[0].toUpperCase()}
          </div>
        )}
        <div className="flex-1 min-w-0">
          <p className="text-sm font-semibold text-zinc-100">{account.display_name ?? account.login}</p>
          <p className="text-xs text-zinc-500">@{account.login}</p>
        </div>
        <button
          onClick={() => disconnect()}
          title="Disconnect GitHub"
          className="flex items-center gap-1.5 px-2.5 py-1.5 text-xs text-zinc-500 hover:text-red-400 hover:bg-red-500/10 border border-zinc-700 hover:border-red-500/30 rounded-lg transition-colors"
        >
          <LogOut size={12} />
          Disconnect
        </button>
      </div>

      {/* Repo search */}
      <div className="relative">
        <Search size={13} className="absolute left-3 top-1/2 -translate-y-1/2 text-zinc-500 pointer-events-none" />
        <input
          type="text"
          value={query}
          onChange={(e) => handleSearch(e.target.value)}
          placeholder="Search repositories…"
          className="w-full pl-9 pr-3 py-2 bg-zinc-800 border border-zinc-700 rounded-lg text-sm text-zinc-200 placeholder-zinc-600 focus:outline-none focus:border-zinc-500 transition-colors"
        />
      </div>

      {repoError && <ErrorBanner message={repoError} onDismiss={clearRepoError} />}

      {/* Repo list */}
      <div className="space-y-2">
        {loadingRepos && repos.length === 0 && (
          <LoadingRow message="Loading repositories…" />
        )}

        {repos.map((repo) => (
          <GitHubRepoCard key={repo.id} repo={repo} />
        ))}

        {!loadingRepos && repos.length === 0 && !repoError && (
          <p className="text-sm text-zinc-600 py-4 text-center">No repositories found</p>
        )}
      </div>

      {/* Load more */}
      {repoHasMore && (
        <button
          onClick={loadMoreRepos}
          disabled={loadingRepos}
          className="w-full py-2 text-xs text-zinc-500 hover:text-zinc-300 border border-zinc-700 hover:border-zinc-600 rounded-lg transition-colors flex items-center justify-center gap-1.5"
        >
          {loadingRepos ? <Loader2 size={12} className="animate-spin" /> : null}
          {loadingRepos ? 'Loading…' : `Load more (page ${repoPage + 1})`}
        </button>
      )}
    </div>
  )
}

function GitHubRepoCard({ repo }: { repo: GitHubRepo }) {
  const timeAgo = (iso: string) => {
    const diff = Date.now() - new Date(iso).getTime()
    const days = Math.floor(diff / 86400000)
    if (days === 0) return 'today'
    if (days === 1) return 'yesterday'
    if (days < 30) return `${days}d ago`
    const months = Math.floor(days / 30)
    return months === 1 ? '1mo ago' : `${months}mo ago`
  }

  return (
    <div className="flex items-start gap-3 bg-zinc-800/40 border border-zinc-700/70 rounded-xl p-3 hover:border-zinc-600 transition-colors">
      <div className="shrink-0 w-8 h-8 rounded-lg bg-zinc-700/60 border border-zinc-600 flex items-center justify-center mt-0.5">
        {repo.is_private ? <Lock size={13} className="text-amber-400" /> : <Unlock size={13} className="text-zinc-400" />}
      </div>
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 flex-wrap">
          <span className="text-sm font-semibold text-zinc-100 truncate">{repo.name}</span>
          {repo.is_private && (
            <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-xs bg-amber-500/10 text-amber-400 border border-amber-500/20">
              Private
            </span>
          )}
          <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-xs bg-zinc-700/60 text-zinc-500 border border-zinc-600">
            <GitBranch size={10} />
            {repo.default_branch}
          </span>
        </div>
        {repo.description && (
          <p className="text-xs text-zinc-500 mt-0.5 line-clamp-2">{repo.description}</p>
        )}
        <div className="flex items-center gap-3 mt-1.5">
          <span className="text-xs text-zinc-600">{repo.owner_login}/{repo.name}</span>
          <span className="text-xs text-zinc-600">Updated {timeAgo(repo.updated_at)}</span>
        </div>
      </div>
    </div>
  )
}

// ──────────────────────────────────────────────────────────────────────────────
// GitHub — full section (not connected / connecting / connected)
// ──────────────────────────────────────────────────────────────────────────────
function GitHubSection() {
  const { account, loadingAccount, connecting, connectStatus, connectError, startConnect, loadAccount, cancelConnect } = useGitHubStore()
  const [showWizard, setShowWizard] = useState(false)

  useEffect(() => { loadAccount() }, [loadAccount])

  const handleConnect = async () => {
    setShowWizard(true)
    await startConnect()
  }

  // Auto-close wizard on success
  useEffect(() => {
    if (connectStatus === 'success') setShowWizard(false)
  }, [connectStatus])

  // "Not configured" = client ID env var not set
  const notConfigured = connectError?.toLowerCase().includes('not configured') ||
    connectError?.toLowerCase().includes('codespectra_github_client_id')

  return (
    <section>
      <div className="flex items-center gap-2 mb-4">
        <div className="flex items-center gap-2">
          <GitHubIcon className="text-zinc-300" size={16} />
          <h2 className="text-sm font-semibold text-zinc-300">GitHub</h2>
        </div>
        {!account && !showWizard && !loadingAccount && !notConfigured && (
          <button
            onClick={handleConnect}
            disabled={connecting}
            className="ml-auto btn-primary text-xs py-1.5 px-3"
          >
            {connecting ? <Loader2 size={12} className="animate-spin" /> : <Plus size={12} />}
            Connect
          </button>
        )}
      </div>

      {/* Setup required card */}
      {notConfigured && (
        <div className="bg-amber-500/5 border border-amber-500/20 rounded-xl p-5 space-y-3">
          <div className="flex items-start gap-3">
            <AlertTriangle size={16} className="text-amber-400 shrink-0 mt-0.5" />
            <div>
              <p className="text-sm font-medium text-amber-300">GitHub OAuth App not configured</p>
              <p className="text-xs text-zinc-400 mt-1 leading-relaxed">
                Set the <code className="text-amber-300 bg-zinc-800 px-1 py-0.5 rounded">CODESPECTRA_GITHUB_CLIENT_ID</code> environment variable before starting the app.
              </p>
            </div>
          </div>
          <ol className="text-xs text-zinc-400 space-y-1.5 pl-4 list-decimal">
            <li>Go to <button onClick={() => window.api.github.openBrowser('https://github.com/settings/developers')} className="text-blue-400 hover:underline">github.com/settings/developers</button> → <strong className="text-zinc-300">New OAuth App</strong></li>
            <li>Set Homepage URL to <code className="text-zinc-300">http://localhost</code></li>
            <li>Enable <strong className="text-zinc-300">Device Flow</strong> in app settings</li>
            <li>Copy the <strong className="text-zinc-300">Client ID</strong> (no secret needed)</li>
            <li>Restart app with <code className="text-zinc-300 bg-zinc-800 px-1 py-0.5 rounded">CODESPECTRA_GITHUB_CLIENT_ID=your_id npm run dev</code></li>
          </ol>
          <button
            onClick={() => { cancelConnect(); setShowWizard(false) }}
            className="text-xs text-zinc-500 hover:text-zinc-300 transition-colors"
          >
            Dismiss
          </button>
        </div>
      )}

      {showWizard && !notConfigured && <GitHubConnectWizard onCancel={() => setShowWizard(false)} />}

      {!showWizard && account && <GitHubAccountCard />}

      {!showWizard && !account && !loadingAccount && !notConfigured && (
        <div className="border border-dashed border-zinc-700 rounded-xl p-6 text-center">
          <div className="w-10 h-10 rounded-xl bg-zinc-800 border border-zinc-700 flex items-center justify-center mx-auto mb-3">
            <GitHubIcon className="text-zinc-500" size={20} />
          </div>
          <p className="text-sm font-medium text-zinc-400">Not connected to GitHub</p>
          <p className="text-xs text-zinc-600 mt-1 max-w-xs mx-auto">
            Connect to browse and select repositories using OAuth Device Flow — no password needed.
          </p>
          <button onClick={handleConnect} className="mt-4 btn-primary text-xs">
            <Plus size={12} />
            Connect GitHub
          </button>
        </div>
      )}
    </section>
  )
}

// ──────────────────────────────────────────────────────────────────────────────
// Main screen
// ──────────────────────────────────────────────────────────────────────────────
export default function CodeHostsSetup(): React.ReactElement {
  const { repos, loading, error, load, remove, revalidate, revalidatingId, clearError } = useLocalRepoStore()
  const { account: ghAccount } = useGitHubStore()
  const [showAdd, setShowAdd] = useState(false)

  useEffect(() => { load() }, [load])

  return (
    <>
      <div className="screen-header flex items-center justify-between">
        <div>
          <h1 className="screen-title">Code Hosts</h1>
          <p className="screen-subtitle">
            {[
              ghAccount ? `GitHub: @${ghAccount.login}` : null,
              repos.length > 0 ? `${repos.length} local folder${repos.length !== 1 ? 's' : ''}` : null,
            ].filter(Boolean).join(' · ') || 'Open local folders or connect a code host'}
          </p>
        </div>
        {!showAdd && (
          <button
            className="btn-primary"
            onClick={() => setShowAdd(true)}
          >
            <Plus className="w-4 h-4" />
            Open Folder
          </button>
        )}
      </div>

      <div className="p-6 space-y-6 overflow-y-auto">
        {error && <ErrorBanner message={error} onDismiss={clearError} />}

        {/* Local Folders section */}
        <section>
          <div className="flex items-center gap-2 mb-4">
            <div className="flex items-center gap-2">
              <FolderOpen size={15} className="text-emerald-400" />
              <h2 className="text-sm font-semibold text-zinc-300">Local Folders</h2>
            </div>
            <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
              <Shield size={10} />
              Strict Local
            </span>
          </div>

          <div className="space-y-3">
            {/* Add folder panel */}
            {showAdd && (
              <AddFolderPanel onClose={() => { setShowAdd(false) }} />
            )}

            {loading && <LoadingRow message="Loading folders…" />}

            {/* Repo cards */}
            {repos.map((repo) => (
              <LocalRepoCard
                key={repo.id}
                repo={repo}
                onRemove={() => remove(repo.id)}
                onRevalidate={() => revalidate(repo.id)}
                isRevalidating={revalidatingId === repo.id}
              />
            ))}

            {/* Empty state */}
            {!loading && repos.length === 0 && !showAdd && (
              <div className="border border-dashed border-zinc-700 rounded-xl p-8 text-center">
                <div className="w-12 h-12 rounded-xl bg-zinc-800 border border-zinc-700 flex items-center justify-center mx-auto mb-4">
                  <FolderOpen size={22} className="text-zinc-500" />
                </div>
                <p className="text-sm font-medium text-zinc-400">No folders added yet</p>
                <p className="text-xs text-zinc-600 mt-1 max-w-xs mx-auto">
                  Open any local git repository or plain folder to start analysis without any authentication.
                </p>
                <button
                  className="mt-4 btn-primary"
                  onClick={() => setShowAdd(true)}
                >
                  <FolderOpen size={14} />
                  Open Folder
                </button>
              </div>
            )}
          </div>
        </section>

        {/* GitHub section */}
        <GitHubSection />

        {/* Bitbucket — coming soon */}
        <section>
          <div className="flex items-center gap-2 mb-4">
            <div className="flex items-center gap-2 opacity-40">
              <span className="text-base">⊛</span>
              <h2 className="text-sm font-semibold text-zinc-400">Bitbucket</h2>
              <span className="px-1.5 py-0.5 rounded text-xs bg-zinc-800 text-zinc-600 border border-zinc-700">Coming soon</span>
            </div>
          </div>
          <div className="border border-dashed border-zinc-800 rounded-xl p-5 opacity-40 text-center">
            <p className="text-xs text-zinc-600">Bitbucket OAuth integration (RPA-024)</p>
          </div>
        </section>
      </div>
    </>
  )
}

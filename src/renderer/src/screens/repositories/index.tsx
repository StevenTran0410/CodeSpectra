import React, { useEffect, useMemo, useState } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'
import { AlertTriangle, CheckCircle2, FolderOpen, GitBranch, Loader2, RefreshCw, Trash2 } from 'lucide-react'
import { EmptyState } from '../../components/ui/EmptyState'
import { ErrorBanner } from '../../components/ui/ErrorBanner'
import { LoadingRow } from '../../components/ui/LoadingRow'
import { toErrorMessage } from '../../lib/errors'
import type {
  ClonePolicy,
  EstimateFileCountResponse,
  RepoSnapshot,
} from '../../types/electron'
import { useLocalRepoStore } from '../../store/local-repo.store'
import { useWorkspaceStore } from '../../store/workspace.store'

export default function RepositoriesScreen(): React.ReactElement {
  const navigate = useNavigate()
  const [searchParams] = useSearchParams()
  const activeWorkspaceId = useWorkspaceStore((s) => s.activeWorkspaceId)
  const { repos, loading, error, load, clearError, loadBranches, branchesMap, setBranch } = useLocalRepoStore()
  const [selectedRepoId, setSelectedRepoId] = useState<string | null>(null)
  const [branch, setBranchLocal] = useState<string>('')
  const [syncMode, setSyncMode] = useState<'latest' | 'pinned'>('latest')
  const [pinnedRef, setPinnedRef] = useState('')
  const [detectSubmodules, setDetectSubmodules] = useState(true)
  const [ignoreText, setIgnoreText] = useState('')
  const [clonePolicy, setClonePolicy] = useState<ClonePolicy>('full')
  const [saving, setSaving] = useState(false)
  const [syncing, setSyncing] = useState(false)
  const [syncProgress, setSyncProgress] = useState(0)
  const [snapshots, setSnapshots] = useState<RepoSnapshot[]>([])
  const [selectedSnapshotId, setSelectedSnapshotId] = useState<string | null>(null)
  const [loadingSnapshots, setLoadingSnapshots] = useState(false)
  const [selectingSnapshot, setSelectingSnapshot] = useState(false)
  const [deletingSnapshot, setDeletingSnapshot] = useState(false)
  const [confirmDeleteSnapshotId, setConfirmDeleteSnapshotId] = useState<string | null>(null)
  const [screenError, setScreenError] = useState<string | null>(null)
  const [estimate, setEstimate] = useState<EstimateFileCountResponse | null>(null)
  const [estimating, setEstimating] = useState(false)
  const runtimeHasSyncApi = Boolean((window as Window).api?.sync)

  const selectedRepo = useMemo(
    () => repos.find((r) => r.id === selectedRepoId) ?? null,
    [repos, selectedRepoId],
  )

  useEffect(() => { load(activeWorkspaceId ?? undefined) }, [load, activeWorkspaceId])

  useEffect(() => {
    if (!selectedRepoId && repos.length > 0) setSelectedRepoId(repos[0].id)
  }, [repos, selectedRepoId])

  useEffect(() => {
    const repoIdFromQuery = searchParams.get('repoId')
    if (!repoIdFromQuery) return
    if (repos.some((r) => r.id === repoIdFromQuery)) {
      setSelectedRepoId(repoIdFromQuery)
    }
  }, [repos, searchParams])

  useEffect(() => {
    if (!selectedRepo) return
    setBranchLocal(selectedRepo.selected_branch ?? selectedRepo.git_branch ?? '')
    setSyncMode(selectedRepo.sync_mode)
    setPinnedRef(selectedRepo.pinned_ref ?? '')
    setDetectSubmodules(selectedRepo.detect_submodules)
    setIgnoreText((selectedRepo.ignore_overrides ?? []).join('\n'))
    setEstimate(null)
  }, [selectedRepo])

  useEffect(() => {
    const run = async () => {
      if (!selectedRepoId) return
      if (!window.api.sync?.listForRepo) {
        setScreenError('Sync API is unavailable. Restart the app so preload/main handlers are reloaded.')
        return
      }
      setLoadingSnapshots(true)
      try {
        const rows = await window.api.sync.listForRepo(selectedRepoId)
        setSnapshots(rows)
        const preferred = rows.find((x) => x.id === selectedRepo?.active_snapshot_id)?.id
        setSelectedSnapshotId(preferred ?? rows[0]?.id ?? null)
      } catch (err) {
        setScreenError(toErrorMessage(err))
      } finally {
        setLoadingSnapshots(false)
      }
    }
    run()
  }, [selectedRepoId, selectedRepo?.active_snapshot_id])

  const hasSnapshot = snapshots.length > 0
  const branchChanged = !!selectedRepo && branch && branch !== (selectedRepo.selected_branch ?? selectedRepo.git_branch ?? '')
  const selectedSnapshot = useMemo(
    () => snapshots.find((s) => s.id === selectedSnapshotId) ?? null,
    [snapshots, selectedSnapshotId],
  )

  return (
    <>
      <div className="screen-header">
        <h1 className="screen-title">Repositories</h1>
        <p className="screen-subtitle">Manage repository connections and snapshots</p>
      </div>
      <div className="h-[calc(100vh-10rem)] overflow-y-auto p-6 space-y-4">
        {error && <ErrorBanner message={error} onDismiss={clearError} />}
        {screenError && <ErrorBanner message={screenError} onDismiss={() => setScreenError(null)} />}
        {!runtimeHasSyncApi && (
          <ErrorBanner message="This screen needs the new sync preload bridge. Please restart CodeSpectra (or restart `npm run dev`)." />
        )}

        {!activeWorkspaceId ? (
          <EmptyState
            icon={<FolderOpen className="w-7 h-7" />}
            title="No workspace selected"
            description="Select or create a workspace first, then add repository connections."
          />
        ) : loading ? (
          <LoadingRow message="Loading repositories..." />
        ) : repos.length === 0 ? (
          <EmptyState
            icon={<FolderOpen className="w-7 h-7" />}
            title="No repositories in this workspace"
            description="Add repositories from Code Hosts first, then configure snapshot settings here."
          />
        ) : (
          <div className="grid grid-cols-12 gap-4">
            <div className="col-span-4 space-y-2">
              <h3 className="text-sm font-semibold text-zinc-300">Repositories</h3>
              {repos.map((repo) => (
                <button
                  key={repo.id}
                  onClick={() => setSelectedRepoId(repo.id)}
                  className={`w-full text-left rounded-lg border px-3 py-2 transition-colors ${
                    selectedRepoId === repo.id
                      ? 'border-blue-500/40 bg-blue-500/10'
                      : 'border-zinc-700 hover:border-zinc-600 bg-zinc-800/40'
                  }`}
                >
                  <div className="text-sm text-zinc-100 font-medium truncate">{repo.name}</div>
                  <div className="text-xs text-zinc-500 truncate">{repo.path}</div>
                </button>
              ))}
            </div>

            <div className="col-span-8 space-y-4">
              {selectedRepo && (
                <>
                  <div className="bg-zinc-800/50 border border-zinc-700 rounded-xl p-4 space-y-4">
                    <div className="flex items-center justify-between">
                      <h3 className="text-sm font-semibold text-zinc-200">Repository Setup</h3>
                      <span className="text-xs text-zinc-500">{selectedRepo.name}</span>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                      <div>
                        <label className="text-xs text-zinc-400 mb-1 block">Branch / ref</label>
                        <div className="flex gap-2">
                          <select
                            className="min-w-0 flex-1 bg-zinc-900 border border-zinc-700 rounded-md px-2 py-2 text-sm text-zinc-100"
                            value={branch}
                            onChange={(e) => setBranchLocal(e.target.value)}
                          >
                            {(branchesMap[selectedRepo.id] ?? [selectedRepo.git_branch ?? '']).filter(Boolean).map((b) => (
                              <option key={b} value={b}>{b}</option>
                            ))}
                          </select>
                          <button
                            onClick={() => loadBranches(selectedRepo.id)}
                            className="px-2.5 py-2 text-xs border border-zinc-700 rounded-md text-zinc-400 hover:text-zinc-200 hover:border-zinc-600"
                          >
                            <GitBranch size={13} />
                          </button>
                        </div>
                      </div>

                      <div>
                        <label className="text-xs text-zinc-400 mb-1 block">Clone policy</label>
                        <select
                          className="min-w-0 w-full bg-zinc-900 border border-zinc-700 rounded-md px-2 py-2 text-sm text-zinc-100"
                          value={clonePolicy}
                          onChange={(e) => setClonePolicy(e.target.value as ClonePolicy)}
                        >
                          <option value="full">Full clone</option>
                          <option value="shallow">Shallow (--depth=1)</option>
                          <option value="partial">Partial (--filter=blob:none)</option>
                        </select>
                      </div>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                      <div>
                        <label className="text-xs text-zinc-400 mb-1 block">Sync mode</label>
                        <select
                          className="w-full bg-zinc-900 border border-zinc-700 rounded-md px-2 py-2 text-sm text-zinc-100"
                          value={syncMode}
                          onChange={(e) => setSyncMode(e.target.value as 'latest' | 'pinned')}
                        >
                          <option value="latest">Always pull latest</option>
                          <option value="pinned">Pin to specific ref</option>
                        </select>
                      </div>

                      <div>
                        <label className="text-xs text-zinc-400 mb-1 block">Pinned ref / commit SHA</label>
                        <input
                          className="w-full bg-zinc-900 border border-zinc-700 rounded-md px-2 py-2 text-sm text-zinc-100 font-mono disabled:opacity-50"
                          value={pinnedRef}
                          onChange={(e) => setPinnedRef(e.target.value)}
                          disabled={syncMode !== 'pinned'}
                          placeholder="main or a1b2c3d"
                        />
                      </div>
                    </div>

                    <div>
                      <label className="text-xs text-zinc-400 mb-1 block">Repo-specific ignore overrides (one per line)</label>
                      <textarea
                        rows={4}
                        className="w-full bg-zinc-900 border border-zinc-700 rounded-md px-2 py-2 text-xs text-zinc-200 font-mono"
                        value={ignoreText}
                        onChange={(e) => setIgnoreText(e.target.value)}
                        placeholder="**/*.gen.ts&#10;vendor/**"
                      />
                    </div>

                    <label className="inline-flex items-center gap-2 text-xs text-zinc-300">
                      <input
                        type="checkbox"
                        checked={detectSubmodules}
                        onChange={(e) => setDetectSubmodules(e.target.checked)}
                      />
                      Detect and note submodules (no deep indexing in v1)
                    </label>

                    {hasSnapshot && branchChanged && (
                      <div className="flex items-start gap-2 text-xs text-amber-300 bg-amber-500/10 border border-amber-500/20 rounded-md px-3 py-2">
                        <AlertTriangle size={13} className="shrink-0 mt-0.5" />
                        Changing branch after an existing snapshot will create a new snapshot and invalidate cached indexes.
                      </div>
                    )}

                    {estimate && (
                      <div className="text-xs text-zinc-400 space-y-1">
                        <div>Estimated files after ignore rules: <span className="text-zinc-200 font-semibold">{estimate.estimated_file_count.toLocaleString()}</span></div>
                        <div className="text-zinc-500">Effective ignores: {estimate.effective_ignores.join(', ') || '(none)'}</div>
                      </div>
                    )}

                    <div className="flex items-center gap-2 flex-wrap">
                      <button
                        onClick={async () => {
                          if (!selectedRepo) return
                          setSaving(true)
                          setScreenError(null)
                          try {
                            if (branch) await setBranch(selectedRepo.id, branch)
                            const ignore_overrides = ignoreText
                              .split('\n')
                              .map((x) => x.trim())
                              .filter(Boolean)
                            await window.api.folder.updateSettings(selectedRepo.id, {
                              sync_mode: syncMode,
                              pinned_ref: syncMode === 'pinned' ? (pinnedRef.trim() || null) : null,
                              ignore_overrides,
                              detect_submodules: detectSubmodules,
                            })
                            await load()
                          } catch (err) {
                            setScreenError(toErrorMessage(err))
                          } finally {
                            setSaving(false)
                          }
                        }}
                        disabled={saving}
                        className="px-3 py-2 text-xs bg-blue-600 hover:bg-blue-500 text-white rounded-md flex items-center gap-1.5 disabled:opacity-50"
                      >
                        {saving && <Loader2 size={12} className="animate-spin" />}
                        Save settings
                      </button>
                      <button
                        onClick={async () => {
                          if (!selectedRepo) return
                          setEstimating(true)
                          setScreenError(null)
                          try {
                            const v = await window.api.folder.estimateFileCount(selectedRepo.id)
                            setEstimate(v)
                          } catch (err) {
                            setScreenError(toErrorMessage(err))
                          } finally {
                            setEstimating(false)
                          }
                        }}
                        disabled={estimating}
                        className="px-3 py-2 text-xs border border-zinc-700 text-zinc-300 rounded-md hover:border-zinc-600 disabled:opacity-50"
                      >
                        {estimating ? 'Estimating...' : 'Estimate file count'}
                      </button>
                      <button
                        onClick={async () => {
                          if (!selectedRepo) return
                          if (!window.api.sync?.prepare || !window.api.sync?.listForRepo) {
                            setScreenError('Sync API is unavailable. Restart the app and try again.')
                            return
                          }
                          setSyncing(true)
                          setSyncProgress(8)
                          setScreenError(null)
                          const progressTimer = setInterval(() => {
                            setSyncProgress((p) => (p >= 92 ? p : p + 4))
                          }, 450)
                          try {
                            const created = await window.api.sync.prepare({
                              local_repo_id: selectedRepo.id,
                              branch: syncMode === 'pinned' ? (pinnedRef.trim() || branch || null) : (branch || null),
                              clone_policy: clonePolicy,
                            })
                            setSelectedSnapshotId(created.id)

                            // Poll snapshot until it reaches final status to avoid "frozen spinner" UX.
                            for (let i = 0; i < 300; i += 1) {
                              const rows = await window.api.sync.listForRepo(selectedRepo.id)
                              setSnapshots(rows)
                              const current = rows.find((s) => s.id === created.id)
                              if (!current) break
                              if (current.status === 'pending') setSyncProgress((p) => Math.max(p, 25))
                              if (current.status === 'syncing') setSyncProgress((p) => Math.max(p, 70))
                              if (current.status === 'ready') break
                              if (current.status === 'failed') {
                                throw new Error(current.error ?? 'Snapshot prepare failed')
                              }
                              await new Promise((resolve) => setTimeout(resolve, 700))
                            }
                            setSyncProgress(100)
                          } catch (err) {
                            setScreenError(toErrorMessage(err))
                          } finally {
                            clearInterval(progressTimer)
                            setTimeout(() => setSyncProgress(0), 350)
                            setSyncing(false)
                          }
                        }}
                        disabled={syncing}
                        className="px-3 py-2 text-xs bg-emerald-600 hover:bg-emerald-500 text-white rounded-md flex items-center gap-1.5 disabled:opacity-50"
                      >
                        {syncing ? <RefreshCw size={12} /> : <RefreshCw size={12} />}
                        {syncing ? 'Preparing...' : 'Prepare snapshot'}
                      </button>
                      <div className="w-48 h-2 bg-zinc-800 border border-zinc-700 rounded overflow-hidden">
                        <div
                          className="h-full bg-emerald-500 transition-all duration-300"
                          style={{ width: `${syncProgress}%` }}
                        />
                      </div>
                    </div>
                  </div>

                  <div className="bg-zinc-800/40 border border-zinc-700 rounded-xl p-4">
                    <div className="flex items-center justify-between mb-2">
                      <h4 className="text-sm font-semibold text-zinc-200">Snapshots</h4>
                      {loadingSnapshots && <Loader2 size={13} className="animate-spin text-zinc-500" />}
                    </div>
                    {snapshots.length === 0 ? (
                      <p className="text-xs text-zinc-500">No snapshots yet.</p>
                    ) : (
                      <div className="space-y-2">
                        {snapshots.map((s) => (
                          <button
                            key={s.id}
                            onClick={() => setSelectedSnapshotId(s.id)}
                            className={`w-full text-left flex items-center gap-2 text-xs border rounded-md px-3 py-2 ${
                              selectedSnapshotId === s.id
                                ? 'border-blue-500/40 bg-blue-500/10'
                                : 'border-zinc-700 hover:border-zinc-600'
                            }`}
                          >
                            <CheckCircle2 size={12} className={s.status === 'ready' ? 'text-emerald-400' : 'text-zinc-600'} />
                            <span className="text-zinc-300">{s.branch ?? 'HEAD'}</span>
                            <span className="text-zinc-600">·</span>
                            <span className="font-mono text-zinc-400">{s.commit_hash ?? 'pending'}</span>
                            <span className="text-zinc-600">·</span>
                            <span className="text-zinc-500">{s.clone_policy}</span>
                            {s.id === selectedRepo.active_snapshot_id && (
                              <>
                                <span className="text-zinc-600">·</span>
                                <span className="text-emerald-400">active</span>
                              </>
                            )}
                            <span className="ml-auto text-zinc-600">{new Date(s.synced_at).toLocaleString()}</span>
                          </button>
                        ))}
                      </div>
                    )}
                    {selectedSnapshotId && (
                      <div className="mt-3 flex items-center gap-2">
                        <button
                          onClick={async () => {
                            if (!selectedRepo || !selectedSnapshotId) return
                            if (selectedSnapshot?.status !== 'ready') return
                            setSelectingSnapshot(true)
                            setScreenError(null)
                            try {
                              navigate(`/snapshot-viewer?repoId=${encodeURIComponent(selectedRepo.id)}&snapshotId=${encodeURIComponent(selectedSnapshotId)}`)
                            } catch (err) {
                              setScreenError(toErrorMessage(err))
                            } finally {
                              setSelectingSnapshot(false)
                            }
                          }}
                          disabled={selectingSnapshot || selectedSnapshot?.status !== 'ready'}
                          className="px-3 py-2 text-xs bg-indigo-600 hover:bg-indigo-500 text-white rounded-md flex items-center gap-1.5 disabled:opacity-50"
                        >
                          {selectingSnapshot && <Loader2 size={12} className="animate-spin" />}
                          {selectedSnapshot?.status === 'ready' ? 'Select' : 'Waiting for ready...'}
                        </button>
                        <button
                          onClick={async () => {
                            if (!selectedRepo || !selectedSnapshotId) return
                            setConfirmDeleteSnapshotId(selectedSnapshotId)
                          }}
                          disabled={deletingSnapshot}
                          className="px-3 py-2 text-xs border border-rose-700/60 text-rose-300 rounded-md hover:border-rose-500 disabled:opacity-50 inline-flex items-center gap-1.5"
                        >
                          {deletingSnapshot ? <Loader2 size={12} className="animate-spin" /> : <Trash2 size={12} />}
                          Delete snapshot
                        </button>
                      </div>
                    )}
                  </div>
                </>
              )}
            </div>
          </div>
        )}
      </div>
      {confirmDeleteSnapshotId && (
        <div className="fixed inset-0 z-50 bg-black/55 flex items-center justify-center p-4">
          <div className="w-full max-w-md rounded-lg border border-zinc-700 bg-zinc-900 p-4 space-y-3">
            <div className="text-sm font-semibold text-zinc-100">Delete snapshot?</div>
            <div className="text-xs text-zinc-400">
              This will remove the snapshot and related index artifacts (manifest, symbols, graph).
            </div>
            <div className="text-[11px] text-zinc-500 font-mono break-all">{confirmDeleteSnapshotId}</div>
            <div className="flex items-center justify-end gap-2">
              <button
                onClick={() => setConfirmDeleteSnapshotId(null)}
                disabled={deletingSnapshot}
                className="px-3 py-1.5 text-xs border border-zinc-700 rounded-md text-zinc-300 hover:border-zinc-600 disabled:opacity-50"
              >
                Cancel
              </button>
              <button
                onClick={async () => {
                  if (!selectedRepo || !confirmDeleteSnapshotId) return
                  setDeletingSnapshot(true)
                  setScreenError(null)
                  try {
                    await window.api.sync.deleteSnapshot(confirmDeleteSnapshotId)
                    const rows = await window.api.sync.listForRepo(selectedRepo.id)
                    setSnapshots(rows)
                    setSelectedSnapshotId(rows[0]?.id ?? null)
                    await load()
                    setConfirmDeleteSnapshotId(null)
                  } catch (err) {
                    setScreenError(toErrorMessage(err))
                  } finally {
                    setDeletingSnapshot(false)
                  }
                }}
                disabled={deletingSnapshot}
                className="px-3 py-1.5 text-xs bg-rose-600 hover:bg-rose-500 rounded-md text-white disabled:opacity-50"
              >
                {deletingSnapshot ? 'Deleting...' : 'Delete'}
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  )
}

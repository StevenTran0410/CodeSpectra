import React, { useEffect, useMemo, useState } from 'react'
import { Loader2 } from 'lucide-react'
import { useNavigate } from 'react-router-dom'
import { ErrorBanner } from '../../components/ui/ErrorBanner'
import { JobProgressPanel } from '../../components/ui/JobProgressPanel'
import { useJobStore } from '../../store/job.store'
import { useLocalRepoStore } from '../../store/local-repo.store'
import { useProviderStore } from '../../store/provider.store'

export default function AnalysisRunScreen(): React.ReactElement {
  const navigate = useNavigate()
  const persisted = (() => {
    try {
      return JSON.parse(localStorage.getItem('analysis.runConfig.v1') ?? '{}') as {
        repoId?: string
        snapshotId?: string
        scanMode?: 'quick' | 'full'
        privacyMode?: 'strict_local' | 'byok_cloud'
        providerId?: string
        modelId?: string
      }
    } catch {
      return {}
    }
  })()

  const { repos, load: loadRepos } = useLocalRepoStore()
  const { providers, load: loadProviders, fetchModels, modelLists, loadingModels } = useProviderStore()
  const { activeJob, startPolling, cancel, clearActive, loadHistory } = useJobStore()

  const [repoId, setRepoId] = useState(persisted.repoId ?? '')
  const [snapshotId, setSnapshotId] = useState(persisted.snapshotId ?? '')
  const [scanMode, setScanMode] = useState<'quick' | 'full'>(persisted.scanMode ?? 'quick')
  const [privacyMode, setPrivacyMode] = useState<'strict_local' | 'byok_cloud'>(persisted.privacyMode ?? 'strict_local')
  const [providerId, setProviderId] = useState(persisted.providerId ?? '')
  const [modelId, setModelId] = useState(persisted.modelId ?? '')
  const [cloudConsentGiven, setCloudConsentGiven] = useState(false)
  const [cloudWarningAck, setCloudWarningAck] = useState(false)
  const [dontShowCloudWarning, setDontShowCloudWarning] = useState(
    localStorage.getItem('analysis.cloudWarningHidden') === '1'
  )
  const [snapshots, setSnapshots] = useState<Array<{
    id: string
    branch: string | null
    commit_hash: string | null
    status: 'pending' | 'syncing' | 'ready' | 'failed'
  }>>([])
  const [estimating, setEstimating] = useState(false)
  const [estimate, setEstimate] = useState<{ file_count: number; estimated_tokens: number } | null>(null)
  const [starting, setStarting] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [latestReportId, setLatestReportId] = useState<string | null>(null)

  useEffect(() => {
    loadRepos()
    loadProviders()
    window.api.consent.checkCloud().then((v) => setCloudConsentGiven(v.given)).catch(() => null)
    loadHistory()
  }, [loadRepos, loadProviders, loadHistory])

  useEffect(() => {
    if (!repoId && repos.length > 0) setRepoId(repos[0].id)
  }, [repoId, repos])

  useEffect(() => {
    if (!providerId && providers.length > 0) {
      setProviderId(providers[0].id)
      setModelId(providers[0].model_id)
    }
  }, [providerId, providers])

  useEffect(() => {
    const run = async () => {
      if (!repoId) return
      const rows = await window.api.sync.listForRepo(repoId)
      setSnapshots(rows)
      const ready = rows.find((s) => s.status === 'ready')
      setSnapshotId((prev) => {
        if (prev && rows.some((s) => s.id === prev)) return prev
        return ready?.id || rows[0]?.id || ''
      })
    }
    run().catch((e) => setError(e instanceof Error ? e.message : String(e)))
  }, [repoId])

  useEffect(() => {
    localStorage.setItem(
      'analysis.runConfig.v1',
      JSON.stringify({
        repoId,
        snapshotId,
        scanMode,
        privacyMode,
        providerId,
        modelId,
      }),
    )
  }, [repoId, snapshotId, scanMode, privacyMode, providerId, modelId])

  useEffect(() => {
    const run = async () => {
      if (!repoId || !snapshotId) return
      setEstimating(true)
      try {
        const est = await window.api.analysis.estimate(repoId, snapshotId)
        setEstimate(est)
      } catch {
        setEstimate(null)
      } finally {
        setEstimating(false)
      }
    }
    run()
  }, [repoId, snapshotId])

  const selectedProvider = useMemo(
    () => providers.find((p) => p.id === providerId) ?? null,
    [providers, providerId]
  )
  const modelOptions = useMemo(() => {
    const live = modelLists[providerId] ?? []
    if (live.length > 0) return live
    if (selectedProvider?.model_id) return [selectedProvider.model_id]
    return []
  }, [modelLists, providerId, selectedProvider])

  const canStart =
    !!repoId &&
    !!snapshotId &&
    !!providerId &&
    !!modelId &&
    (privacyMode !== 'byok_cloud' || cloudConsentGiven || cloudWarningAck)

  useEffect(() => {
    const run = async () => {
      if (!activeJob || activeJob.type !== 'analysis' || activeJob.status !== 'done') return
      const out = await window.api.analysis.getReportByJob(activeJob.id)
      setLatestReportId(out.summary.id)
    }
    run().catch(() => null)
  }, [activeJob])

  return (
    <>
      <div className="screen-header">
        <h1 className="screen-title">Analysis</h1>
        <p className="screen-subtitle">Run analysis on a repository and track progress</p>
      </div>
      <div className="h-[calc(100vh-10rem)] overflow-y-auto p-4 space-y-3">
        {error && <ErrorBanner message={error} onDismiss={() => setError(null)} />}

        <div className="bg-zinc-900/60 border border-zinc-700 rounded-xl p-4 space-y-3">
          <div className="text-sm font-semibold text-zinc-200">Run Configuration</div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
            <select
              value={repoId}
              onChange={(e) => setRepoId(e.target.value)}
              className="bg-zinc-950 border border-zinc-700 rounded-md px-2 py-2 text-sm text-zinc-100"
            >
              {repos.map((r) => <option key={r.id} value={r.id}>{r.name}</option>)}
            </select>
            <select
              value={snapshotId}
              onChange={(e) => setSnapshotId(e.target.value)}
              className="bg-zinc-950 border border-zinc-700 rounded-md px-2 py-2 text-sm text-zinc-100"
            >
              {snapshots.map((s) => (
                <option key={s.id} value={s.id}>
                  {(s.branch ?? 'HEAD')} · {(s.commit_hash ?? 'pending').slice(0, 10)} · {s.status}
                </option>
              ))}
            </select>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-2">
            <select
              value={scanMode}
              onChange={(e) => setScanMode(e.target.value as 'quick' | 'full')}
              className="bg-zinc-950 border border-zinc-700 rounded-md px-2 py-2 text-sm text-zinc-100"
            >
              <option value="quick">Quick scan</option>
              <option value="full">Full scan</option>
            </select>
            <select
              value={privacyMode}
              onChange={(e) => setPrivacyMode(e.target.value as 'strict_local' | 'byok_cloud')}
              className="bg-zinc-950 border border-zinc-700 rounded-md px-2 py-2 text-sm text-zinc-100"
            >
              <option value="strict_local">Strict Local</option>
              <option value="byok_cloud">BYOK Cloud</option>
            </select>
            <select
              value={providerId}
              onChange={async (e) => {
                const id = e.target.value
                setProviderId(id)
                const provider = providers.find((p) => p.id === id)
                setModelId(provider?.model_id ?? '')
                await fetchModels(id)
              }}
              className="bg-zinc-950 border border-zinc-700 rounded-md px-2 py-2 text-sm text-zinc-100"
            >
              {providers.map((p) => <option key={p.id} value={p.id}>{p.display_name}</option>)}
            </select>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
            <select
              value={modelId}
              onChange={(e) => setModelId(e.target.value)}
              className="bg-zinc-950 border border-zinc-700 rounded-md px-2 py-2 text-sm text-zinc-100"
            >
              {(modelOptions.length > 0 ? modelOptions : ['']).map((m) => (
                <option key={m} value={m}>{m || '(no model)'}</option>
              ))}
            </select>
            <div className="text-xs text-zinc-500 flex items-center">
              {estimating ? (
                <span className="inline-flex items-center gap-1"><Loader2 size={12} className="animate-spin" /> Estimating...</span>
              ) : estimate ? (
                <span>~ {estimate.file_count.toLocaleString()} files · ~ {estimate.estimated_tokens.toLocaleString()} tokens</span>
              ) : (
                <span>Scope estimate unavailable</span>
              )}
            </div>
          </div>

          {privacyMode === 'byok_cloud' && !dontShowCloudWarning && (
            <div className="rounded-lg border border-amber-500/30 bg-amber-500/10 text-amber-200 text-xs px-3 py-2 space-y-2">
              <div>Cloud mode selected: code context will be sent to your configured cloud provider.</div>
              <label className="inline-flex items-center gap-2">
                <input type="checkbox" checked={cloudWarningAck} onChange={(e) => setCloudWarningAck(e.target.checked)} />
                I understand this run is not strict-local.
              </label>
              <label className="inline-flex items-center gap-2">
                <input
                  type="checkbox"
                  checked={dontShowCloudWarning}
                  onChange={(e) => {
                    const v = e.target.checked
                    setDontShowCloudWarning(v)
                    localStorage.setItem('analysis.cloudWarningHidden', v ? '1' : '0')
                  }}
                />
                Do not show this again
              </label>
            </div>
          )}

          <div className="flex items-center gap-2">
            <button
              onClick={async () => {
                if (!canStart) return
                setStarting(true)
                setError(null)
                setLatestReportId(null)
                try {
                  if (privacyMode === 'byok_cloud' && !cloudConsentGiven) {
                    const c = await window.api.consent.giveCloud(true)
                    setCloudConsentGiven(c.given)
                  }
                  const job = await window.api.analysis.start({
                    repo_id: repoId,
                    snapshot_id: snapshotId,
                    scan_mode: scanMode,
                    privacy_mode: privacyMode,
                    provider_id: providerId,
                    model_id: modelId,
                  })
                  startPolling(job.id)
                } catch (err) {
                  setError(err instanceof Error ? err.message : String(err))
                } finally {
                  setStarting(false)
                }
              }}
              disabled={!canStart || starting}
              className="px-3 py-2 text-xs bg-indigo-600 hover:bg-indigo-500 rounded-md text-white disabled:opacity-50"
            >
              {starting ? 'Starting...' : 'Start analysis'}
            </button>
            {activeJob && (
              <button
                onClick={() => clearActive()}
                className="px-3 py-2 text-xs border border-zinc-700 rounded-md text-zinc-300 hover:border-zinc-600"
              >
                Clear active
              </button>
            )}
            {latestReportId && (
              <button
                onClick={() => navigate(`/reports?reportId=${encodeURIComponent(latestReportId)}`)}
                className="px-3 py-2 text-xs border border-emerald-700 rounded-md text-emerald-300 hover:border-emerald-600"
              >
                View report
              </button>
            )}
          </div>
        </div>

        {activeJob && (
          <JobProgressPanel
            job={activeJob}
            onCancel={activeJob.status === 'running' ? () => cancel(activeJob.id) : undefined}
          />
        )}
      </div>
    </>
  )
}

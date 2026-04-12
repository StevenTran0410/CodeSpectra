import React, { useEffect, useMemo, useRef, useState } from 'react'
import { Loader2 } from 'lucide-react'
import { useNavigate } from 'react-router-dom'
import { ErrorBanner } from '../../components/ui/ErrorBanner'
import { JobProgressPanel } from '../../components/ui/JobProgressPanel'
import { Button, FormGroup, Select, useToastStore } from '../../components/ui'
import type { AnalysisSectionId } from '../../hooks/useAnalysisSectionEvents'
import { useAnalysisSectionEvents } from '../../hooks/useAnalysisSectionEvents'
import { toErrorMessage } from '../../lib/errors'
import { useJobStore } from '../../store/job.store'
import { useLocalRepoStore } from '../../store/local-repo.store'
import { useProviderStore } from '../../store/provider.store'
import { useWorkspaceStore } from '../../store/workspace.store'
import type {
  SectionA,
  SectionB,
  SectionC,
  SectionD,
  SectionE,
  SectionF,
  SectionG,
  SectionH,
  SectionI,
  SectionJ,
  SectionK,
  SectionL,
} from '../../types/analysis'
import SectionCardA from '../reports/components/SectionCardA'
import SectionCardB from '../reports/components/SectionCardB'
import SectionCardC from '../reports/components/SectionCardC'
import SectionCardD from '../reports/components/SectionCardD'
import SectionCardE from '../reports/components/SectionCardE'
import SectionCardF from '../reports/components/SectionCardF'
import SectionCardG from '../reports/components/SectionCardG'
import SectionCardH from '../reports/components/SectionCardH'
import SectionCardI from '../reports/components/SectionCardI'
import SectionCardJ from '../reports/components/SectionCardJ'
import SectionCardK from '../reports/components/SectionCardK'
import SectionCardL from '../reports/components/SectionCardL'

const LIVE_SECTION_ORDER: AnalysisSectionId[] = [
  'A',
  'B',
  'C',
  'D',
  'E',
  'F',
  'G',
  'H',
  'I',
  'J',
  'K',
  'L',
]

function renderLiveSection(letter: AnalysisSectionId, data: unknown): React.ReactElement | null {
  if (data == null || typeof data !== 'object') return null
  switch (letter) {
    case 'A':
      return <SectionCardA data={data as SectionA} />
    case 'B':
      return <SectionCardB data={data as SectionB} />
    case 'C':
      return <SectionCardC data={data as SectionC} />
    case 'D':
      return <SectionCardD data={data as SectionD} />
    case 'E':
      return <SectionCardE data={data as SectionE} />
    case 'F':
      return <SectionCardF data={data as SectionF} />
    case 'G':
      return <SectionCardG data={data as SectionG} />
    case 'H':
      return <SectionCardH data={data as SectionH} />
    case 'I':
      return <SectionCardI data={data as SectionI} />
    case 'J':
      return <SectionCardJ data={data as SectionJ} />
    case 'K':
      return <SectionCardK data={data as SectionK} />
    case 'L':
      return <SectionCardL data={data as SectionL} />
    default:
      return null
  }
}

export default function AnalysisRunScreen(): React.ReactElement {
  const navigate = useNavigate()
  const toast = useToastStore()
  const toastSuccess = useToastStore((s) => s.success)
  const toastedJobRef = useRef<string | null>(null)
  const persisted = (() => {
    try {
      return JSON.parse(localStorage.getItem('analysis.runConfig.v1') ?? '{}') as {
        repoId?: string
        snapshotId?: string
        scanMode?: 'quick' | 'full'
        privacyMode?: 'strict_local' | 'byok_cloud'
        providerId?: string
        modelId?: string
        largecodebaseMode?: boolean
        includeSynthesis?: boolean
      }
    } catch {
      return {}
    }
  })()

  const activeWorkspaceId = useWorkspaceStore((s) => s.activeWorkspaceId)
  const { repos, load: loadRepos } = useLocalRepoStore()
  const { providers, load: loadProviders, fetchModels, modelLists, loadingModels, modelErrors } = useProviderStore()
  const { activeJob, startPolling, cancel, clearActive, loadHistory } = useJobStore()

  const showLiveSections =
    !!activeJob && activeJob.type === 'analysis' && activeJob.status === 'running'
  const { sectionStates, liveSections } = useAnalysisSectionEvents(
    activeJob?.id,
    showLiveSections
  )

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
const [starting, setStarting] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [forceRerun, setForceRerun] = useState(false)
  const [largecodebaseMode, setLargecodebaseMode] = useState(persisted.largecodebaseMode ?? false)
  const [includeSynthesis, setIncludeSynthesis] = useState(persisted.includeSynthesis ?? false)
  const [latestReportId, setLatestReportId] = useState<string | null>(null)
  const [modelWarning, setModelWarning] = useState<{
    code: string
    message: string
    severity: string
  } | null>(null)

  useEffect(() => {
    loadRepos(activeWorkspaceId ?? undefined)
    loadProviders()
    window.api.consent.checkCloud().then((v) => setCloudConsentGiven(v.given)).catch(() => null)
    loadHistory()
  }, [loadRepos, loadProviders, loadHistory, activeWorkspaceId])

  useEffect(() => {
    if (repos.length === 0) {
      if (repoId) setRepoId('')
      if (snapshotId) setSnapshotId('')
      return
    }
    if (!repoId || !repos.some((r) => r.id === repoId)) {
      setRepoId(repos[0].id)
      setSnapshotId('')
    }
  }, [repoId, repos, snapshotId])

  useEffect(() => {
    if (providers.length === 0) return
    // Reset if providerId is empty OR points to a provider that no longer exists (stale localStorage)
    if (!providerId || !providers.some((p) => p.id === providerId)) {
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
    run().catch((e) => setError(toErrorMessage(e)))
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
        largecodebaseMode,
        includeSynthesis,
      }),
    )
  }, [repoId, snapshotId, scanMode, privacyMode, providerId, modelId, largecodebaseMode, includeSynthesis])


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

  useEffect(() => {
    if (providerId) fetchModels(providerId)
  }, [providerId, fetchModels])

  useEffect(() => {
    if (modelOptions.length > 0 && !modelOptions.includes(modelId)) {
      setModelId(modelOptions[0])
    }
  }, [modelOptions, modelId])

  const selectedSnapshot = snapshots.find((s) => s.id === snapshotId) ?? null
  const snapshotReady = selectedSnapshot?.status === 'ready'

  const canStart =
    !!repoId &&
    !!snapshotId &&
    snapshotReady &&
    !!providerId &&
    !!modelId &&
    (privacyMode !== 'byok_cloud' || cloudConsentGiven || cloudWarningAck)

  useEffect(() => {
    if (!activeJob || activeJob.type !== 'analysis' || activeJob.status !== 'done') return
    if (toastedJobRef.current === activeJob.id) return
    toastedJobRef.current = activeJob.id
    window.api.analysis.getReportByJob(activeJob.id)
      .then((out) => {
        setLatestReportId(out.summary.id)
        toastSuccess('Analysis complete — report ready')
      })
      .catch(() => null)
  }, [activeJob?.id, activeJob?.type, activeJob?.status, toastSuccess])

  return (
    <div className="flex flex-col h-full">
      <div className="screen-header">
        <h1 className="screen-title">Analysis</h1>
        <p className="screen-subtitle">Run analysis on a repository and track progress</p>
      </div>
      <div className="flex-1 overflow-y-auto p-4 space-y-3">
        {error && <ErrorBanner message={error} onDismiss={() => setError(null)} />}

        {modelWarning && (
          <div className="rounded-lg border border-amber-400/50 bg-amber-500/15 text-amber-100 text-sm px-3 py-2 flex items-start justify-between gap-3">
            <span>{modelWarning.message}</span>
            <button
              type="button"
              onClick={() => setModelWarning(null)}
              className="shrink-0 text-xs text-amber-200/90 hover:text-amber-50 underline"
            >
              Dismiss
            </button>
          </div>
        )}

        <div className="bg-zinc-900/60 border border-zinc-700 rounded-xl p-4 space-y-3">
          <div className="text-sm font-semibold text-zinc-200">Run Configuration</div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
            <FormGroup label="Repository">
              <Select value={repoId} onChange={(e) => setRepoId(e.target.value)}>
                {repos.map((r) => <option key={r.id} value={r.id}>{r.name}</option>)}
              </Select>
            </FormGroup>
            <FormGroup label="Snapshot">
              <Select value={snapshotId} onChange={(e) => setSnapshotId(e.target.value)}>
                {snapshots.map((s) => (
                  <option key={s.id} value={s.id}>
                    {(s.branch ?? 'HEAD')} · {(s.commit_hash ?? 'pending').slice(0, 10)} · {s.status}
                  </option>
                ))}
              </Select>
            </FormGroup>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-2">
            <FormGroup label="Scan mode">
              <Select value={scanMode} onChange={(e) => setScanMode(e.target.value as 'quick' | 'full')}>
                <option value="quick">Quick scan</option>
                <option value="full">Full scan</option>
              </Select>
            </FormGroup>
            <FormGroup label="Privacy mode">
              <Select value={privacyMode} onChange={(e) => setPrivacyMode(e.target.value as 'strict_local' | 'byok_cloud')}>
                <option value="strict_local">Strict Local</option>
                <option value="byok_cloud">BYOK Cloud</option>
              </Select>
            </FormGroup>
            <FormGroup label="Provider">
              <Select
                value={providerId}
                onChange={async (e) => {
                  const id = e.target.value
                  setProviderId(id)
                  const provider = providers.find((p) => p.id === id)
                  setModelId(provider?.model_id ?? '')
                  await fetchModels(id)
                }}
              >
                {providers.map((p) => <option key={p.id} value={p.id}>{p.display_name}</option>)}
              </Select>
            </FormGroup>
          </div>

          <div>
            <label className="block text-xs text-gray-400 mb-1.5">Model</label>
            <div className="flex gap-1.5 items-center">
              <Select value={modelId} onChange={(e) => setModelId(e.target.value)} className="flex-1">
                {(modelOptions.length > 0 ? modelOptions : ['']).map((m) => (
                  <option key={m} value={m}>{m || '(no model)'}</option>
                ))}
              </Select>
              <button
                type="button"
                title="Fetch available models from provider"
                disabled={!providerId || !!loadingModels[providerId]}
                onClick={() => providerId && fetchModels(providerId)}
                className="shrink-0 flex items-center justify-center w-8 h-8 rounded-md border border-zinc-700 bg-zinc-950 text-zinc-400 hover:text-zinc-100 hover:border-zinc-500 disabled:opacity-40 transition-colors"
              >
                <Loader2
                  size={14}
                  className={loadingModels[providerId] ? 'animate-spin' : ''}
                />
              </button>
            </div>
            {modelErrors[providerId] && (
              <div className="text-[11px] text-rose-400 mt-1">{modelErrors[providerId]}</div>
            )}
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

          <div className="flex items-center gap-3 text-xs text-zinc-400">
            <label className="inline-flex items-center gap-1.5 cursor-pointer select-none">
              <input
                type="checkbox"
                checked={forceRerun}
                onChange={(e) => setForceRerun(e.target.checked)}
                className="accent-indigo-500"
              />
              Force re-run (skip cache)
            </label>
            <label
              className="inline-flex items-center gap-1.5 cursor-pointer select-none"
              title="Recommended for: ~10k+ files, >1.5M LOC, monorepo with many packages/services, or previous runs with missing context"
            >
              <input
                type="checkbox"
                checked={largecodebaseMode}
                onChange={(e) => setLargecodebaseMode(e.target.checked)}
                className="accent-indigo-500"
              />
              Large codebase mode
            </label>
            <label
              className="inline-flex items-center gap-1.5 cursor-pointer select-none"
              title="Agent L synthesises all other sections into a final executive summary. Uncheck to skip it and save time."
            >
              <input
                type="checkbox"
                checked={includeSynthesis}
                onChange={(e) => setIncludeSynthesis(e.target.checked)}
                className="accent-indigo-500"
              />
              Include synthesis agent (L)
            </label>
          </div>

          {snapshotId && !snapshotReady && selectedSnapshot && (
            <div className="rounded-lg border border-amber-500/40 bg-amber-500/10 text-amber-200 text-xs px-3 py-2 space-y-1">
              <div className="font-medium">Snapshot not ready for analysis</div>
              <div className="text-amber-300/80">
                This snapshot has status <span className="font-mono text-amber-200">{selectedSnapshot.status}</span>.
                Go to <span className="font-medium">Repositories</span> → prepare a snapshot and wait for it to complete before running analysis.
              </div>
            </div>
          )}

          <div className="flex items-center gap-2">
            <Button
              variant="primary"
              size="sm"
              disabled={!canStart}
              loading={starting}
              onClick={async () => {
                if (!canStart) return
                setStarting(true)
                setError(null)
                setLatestReportId(null)
                setModelWarning(null)
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
                    force_rerun: forceRerun,
                    large_codebase_mode: largecodebaseMode,
                    skip_synthesis: !includeSynthesis,
                  })
                  setModelWarning((job as unknown as { warning?: { code: string; message: string; severity: string } | null }).warning ?? null)
                  toast.info('Analysis started')
                  startPolling(job.id)
                } catch (err) {
                  setError(toErrorMessage(err))
                } finally {
                  setStarting(false)
                }
              }}
            >
              Start analysis
            </Button>
            {activeJob && (
              <Button
                variant="secondary"
                size="sm"
                onClick={() => clearActive()}
              >
                Clear active
              </Button>
            )}
            {latestReportId && (
              <Button
                variant="secondary"
                size="sm"
                onClick={() => navigate(`/reports?reportId=${encodeURIComponent(latestReportId)}`)}
              >
                View report
              </Button>
            )}
          </div>
        </div>

        {activeJob && (
          <JobProgressPanel
            job={activeJob}
            onCancel={activeJob.status === 'running' ? () => cancel(activeJob.id) : undefined}
          />
        )}

        {showLiveSections && (
          <div className="bg-zinc-900/60 border border-zinc-700 rounded-xl p-4 space-y-3">
            <div className="text-sm font-semibold text-zinc-200">Live report sections</div>
            <p className="text-[11px] text-zinc-500">
              Sections appear here as each agent completes (incremental). Full report is saved when
              the job finishes.
            </p>
            <div className="space-y-2 max-h-[48vh] overflow-y-auto pr-1">
              {LIVE_SECTION_ORDER.map((letter) => {
                const st = sectionStates[letter]
                const live = liveSections[letter]
                if (st === 'done' && live != null) {
                  return <div key={letter}>{renderLiveSection(letter, live)}</div>
                }
                return (
                  <div
                    key={letter}
                    className={`rounded-lg border px-3 py-2 ${
                      st === 'error'
                        ? 'border-rose-800/50 bg-rose-950/20'
                        : 'border-zinc-800 bg-zinc-950/50'
                    }`}
                  >
                    <div className="flex items-center justify-between gap-2">
                      <span className="text-xs font-mono text-zinc-300">{letter}</span>
                      <span className="text-[10px] uppercase text-zinc-500">
                        {st === 'error' ? 'failed' : 'pending'}
                      </span>
                    </div>
                    {st === 'pending' && (
                      <div className="mt-2 h-14 animate-pulse rounded bg-zinc-800/60" />
                    )}
                    {st === 'error' && (
                      <div className="mt-2 text-[11px] text-rose-300/90">Section agent failed.</div>
                    )}
                  </div>
                )
              })}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

import React, { useEffect, useMemo, useState } from 'react'
import { useNavigate, Routes, Route } from 'react-router-dom'
import { useWorkspaceStore } from '../../store/workspace.store'
import { useLocalRepoStore } from '../../store/local-repo.store'
import { useAEHStore } from '../../store/aeh.store'
import {
  Button,
  FormGroup,
  Select,
  Badge,
} from '../../components/ui'
import {
  AlertCircle,
  CheckCircle2,
  FolderOpen,
  FileText,
  Search,
  Network,
  Info,
  ArrowRight,
} from 'lucide-react'
import type { RepoSnapshot, LocalRepo } from '../../types/electron'
// AEHDiscoverySession/AEHDiscoveryCandidate are global ambient types.
import Stage1GraphScreen from './Stage1GraphScreen'
import Stage2Screen from './Stage2Screen'

function AnalysisOverview(): React.ReactElement {
  const navigate = useNavigate()
  const activeWorkspaceId = useWorkspaceStore((s) => s.activeWorkspaceId)
  const { repos, load: loadRepos } = useLocalRepoStore()
  const { startAEH } = useAEHStore()

  const [persistedConfig, setPersistedConfig] = useState<any>(() => {
    try {
      return JSON.parse(localStorage.getItem('analysis.runConfig.v1') ?? '{}')
    } catch {
      return {}
    }
  })

  const [repoId, setRepoId] = useState(persistedConfig.repoId ?? '')
  const [snapshotId, setSnapshotId] = useState<string>(persistedConfig.snapshotId ?? '')
  const [snapshots, setSnapshots] = useState<RepoSnapshot[]>([])
  const [loadingSnapshots, setLoadingSnapshots] = useState(false)
  const [reports, setReports] = useState<any[]>([])
  const [loadingReports, setLoadingReports] = useState(false)

  // Statuses for indices
  const [manifestStatus, setManifestStatus] = useState<'loading' | 'built' | 'missing'>('loading')
  const [graphStatus, setGraphStatus] = useState<'loading' | 'built' | 'missing'>('loading')
  const [retrievalStatus, setRetrievalStatus] = useState<'loading' | 'built' | 'missing'>('loading')

  const [repomapSummary, setRepomapSummary] = useState<any | null>(null)
  const [graphSummary, setGraphSummary] = useState<any | null>(null)

  const [discoverySession, setDiscoverySession] = useState<AEHDiscoverySession | null>(null)
  const [candidates, setCandidates] = useState<AEHDiscoveryCandidate[]>([])

  // Load repos on mount or workspace change
  useEffect(() => {
    loadRepos(activeWorkspaceId ?? undefined, 'aeh')
  }, [loadRepos, activeWorkspaceId])

  // Set default repo if none selected
  useEffect(() => {
    if (repos.length > 0) {
      if (!repoId || !repos.some((r) => r.id === repoId)) {
        const firstRepoId = repos[0].id
        setRepoId(firstRepoId)
        updatePersistedConfig({ repoId: firstRepoId })
      }
    } else {
      setRepoId('')
      setSnapshotId('')
    }
  }, [repos, repoId])

  // Load snapshots when repoId changes
  useEffect(() => {
    if (!repoId) {
      setSnapshots([])
      setSnapshotId('')
      return
    }

    const loadSnapshots = async () => {
      setLoadingSnapshots(true)
      try {
        const list = await window.api.sync.listForRepo(repoId)
        setSnapshots(list)

        // Find best snapshot to select
        const readySnapshot = list.find((s) => s.status === 'ready')
        const defaultSnapshotId = readySnapshot?.id || list[0]?.id || ''

        setSnapshotId((prev) => {
          const currentValid = prev && list.some((s) => s.id === prev)
          const next = currentValid ? prev : defaultSnapshotId
          updatePersistedConfig({ repoId, snapshotId: next })
          return next
        })
      } catch (err) {
        console.error('Failed to list snapshots', err)
      } finally {
        setLoadingSnapshots(false)
      }
    }

    loadSnapshots()
  }, [repoId])

  const [caRepos, setCaRepos] = useState<LocalRepo[]>([])
  const [caSnapshots, setCaSnapshots] = useState<RepoSnapshot[]>([])

  const selectedRepo = useMemo(() => repos.find((r) => r.id === repoId) ?? null, [repos, repoId])
  const selectedSnapshot = useMemo(() => snapshots.find((s) => s.id === snapshotId) ?? null, [snapshots, snapshotId])

  // Load CA repos to find sibling
  useEffect(() => {
    if (activeWorkspaceId) {
      window.api.folder.list(activeWorkspaceId, 'code_analysis')
        .then((list) => setCaRepos(list))
        .catch((err) => console.error('Failed to list CA repos', err))
    }
  }, [activeWorkspaceId])

  const siblingCARepo = useMemo(() => {
    if (!selectedRepo) return null
    return caRepos.find((r) => r.path === selectedRepo.path) ?? null
  }, [caRepos, selectedRepo])

  // Load CA snapshots for the sibling
  useEffect(() => {
    if (!siblingCARepo) {
      setCaSnapshots([])
      return
    }
    window.api.sync.listForRepo(siblingCARepo.id)
      .then((list) => setCaSnapshots(list))
      .catch((err) => console.error('Failed to list CA snapshots', err))
  }, [siblingCARepo])

  // Load CA reports for the sibling
  useEffect(() => {
    if (!siblingCARepo) {
      setReports([])
      return
    }
    setLoadingReports(true)
    window.api.analysis.listReports(siblingCARepo.id)
      .then((list) => setReports(list))
      .catch((err) => console.error('Failed to list CA reports', err))
      .finally(() => setLoadingReports(false))
  }, [siblingCARepo])

  // Check index statuses when snapshotId changes
  useEffect(() => {
    if (!snapshotId) {
      setManifestStatus('missing')
      setGraphStatus('missing')
      setRetrievalStatus('missing')
      setRepomapSummary(null)
      setGraphSummary(null)
      return
    }

    const checkIndexStatuses = async () => {
      setManifestStatus('loading')
      setGraphStatus('loading')
      setRetrievalStatus('loading')

      const selectedSnap = snapshots.find((s) => s.id === snapshotId)
      if (selectedSnap && selectedSnap.status === 'ready') {
        setManifestStatus('built')
      } else {
        setManifestStatus('missing')
      }

      try {
        const rSummary = await window.api.repomap.summary(snapshotId)
        setRepomapSummary(rSummary)
        if (rSummary && rSummary.files_indexed > 0) {
          setRetrievalStatus('built')
        } else {
          setRetrievalStatus('missing')
        }
      } catch (e) {
        setRepomapSummary(null)
        setRetrievalStatus('missing')
      }

      try {
        const gSummary = await window.api.graph.summary(snapshotId)
        setGraphSummary(gSummary)
        if (gSummary && gSummary.total_nodes > 0) {
          setGraphStatus('built')
        } else {
          setGraphStatus('missing')
        }
      } catch (e) {
        setGraphSummary(null)
        setGraphStatus('missing')
      }
    }

    checkIndexStatuses()
  }, [snapshotId, snapshots])

  // Load previous session for this snapshot on mount/change
  useEffect(() => {
    if (!snapshotId) {
      setDiscoverySession(null)
      setCandidates([])
      return
    }
    let cancelled = false
    const repoRef = selectedRepo?.path ?? snapshotId
    startAEH()
      .then(() => window.api.aeh.listDiscoverySessions(repoRef, snapshotId))
      .then((sessions) => {
        if (cancelled) return []
        const matching = sessions.filter((s) => s.snapshot_id === snapshotId)
        if (matching.length > 0) {
          const latest = matching[0]
          setDiscoverySession(latest)
          if (latest.status === 'completed') {
            return window.api.aeh.listDiscoveryCandidates(latest.id)
          }
        }
        return []
      })
      .then((cands) => { if (!cancelled) setCandidates(cands) })
      .catch(() => {})
    return () => { cancelled = true }
  }, [snapshotId, selectedRepo?.path, startAEH])

  const updatePersistedConfig = (updates: Partial<typeof persistedConfig>) => {
    const nextConfig = { ...persistedConfig, ...updates }
    setPersistedConfig(nextConfig)
    localStorage.setItem('analysis.runConfig.v1', JSON.stringify(nextConfig))
  }

  const handleRepoChange = (newRepoId: string) => {
    setRepoId(newRepoId)
    setSnapshotId('')
    setDiscoverySession(null)
    setCandidates([])
  }

  const handleSnapshotChange = (newSnapshotId: string) => {
    setSnapshotId(newSnapshotId)
    updatePersistedConfig({ snapshotId: newSnapshotId })
    setDiscoverySession(null)
    setCandidates([])
  }

  const matchingCASnapshot = useMemo(() => {
    if (!selectedSnapshot || caSnapshots.length === 0) return null
    if (selectedSnapshot.commit_hash) {
      const match = caSnapshots.find((s) => s.commit_hash === selectedSnapshot.commit_hash && s.status === 'ready')
      if (match) return match
    }
    if (selectedSnapshot.branch) {
      const match = caSnapshots.find((s) => s.branch === selectedSnapshot.branch && s.status === 'ready')
      if (match) return match
    }
    if (siblingCARepo?.active_snapshot_id) {
      const match = caSnapshots.find((s) => s.id === siblingCARepo.active_snapshot_id && s.status === 'ready')
      if (match) return match
    }
    return caSnapshots.find((s) => s.status === 'ready') ?? null
  }, [selectedSnapshot, caSnapshots, siblingCARepo])

  const matchingCAReport = useMemo(() => {
    if (!matchingCASnapshot) return null
    return reports.find((r) => r.snapshot_id === matchingCASnapshot.id) ?? null
  }, [reports, matchingCASnapshot])

  const reportExists = matchingCAReport !== null

  const handleStage1Click = () => {
    if (repoId && snapshotId) {
      navigate(`stage1?repoId=${repoId}&snapshotId=${snapshotId}`)
    }
  }

  const renderStatusBadge = (status: 'loading' | 'built' | 'missing') => {
    switch (status) {
      case 'loading':
        return (
          <Badge variant="neutral" size="sm" className="animate-pulse">
            Checking...
          </Badge>
        )
      case 'built':
        return (
          <Badge variant="success" size="sm">
            <CheckCircle2 className="w-3 h-3 text-emerald-400" />
            <span>Built</span>
          </Badge>
        )
      case 'missing':
        return (
          <Badge variant="error" size="sm">
            <AlertCircle className="w-3 h-3 text-red-400" />
            <span>Missing</span>
          </Badge>
        )
    }
  }

  return (
    <div className="flex flex-col h-full bg-[#090d16]">
      <div className="screen-header">
        <h1 className="screen-title flex items-center gap-2">
          <span>AEH Discovery Analysis</span>
        </h1>
        <p className="screen-subtitle">Configure repo context and inspect index status before running discovery</p>
      </div>

      <div className="flex-1 overflow-y-auto p-4 space-y-4 max-w-4xl w-full mx-auto">
        {/* Repo & Snapshot config card */}
        <div className="bg-[#0f172a]/60 border border-slate-800/80 rounded-xl p-5 space-y-4 backdrop-blur-sm shadow-xl">
          <div className="text-sm font-semibold text-slate-200 flex items-center gap-2 border-b border-slate-850 pb-2">
            <FolderOpen className="w-4 h-4 text-indigo-400" />
            <span>Repository Context</span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <FormGroup label="Repository">
              <Select value={repoId} onChange={(e) => handleRepoChange(e.target.value)}>
                {repos.map((r) => (
                  <option key={r.id} value={r.id}>
                    {r.name}
                  </option>
                ))}
              </Select>
            </FormGroup>
            <FormGroup label="Snapshot">
              <Select value={snapshotId} onChange={(e) => handleSnapshotChange(e.target.value)}>
                {snapshots.map((s) => (
                  <option key={s.id} value={s.id}>
                    {(s.branch ?? 'HEAD')} · {(s.commit_hash ?? 'pending').slice(0, 10)} · {s.status}
                  </option>
                ))}
              </Select>
            </FormGroup>
          </div>

          {repos.length === 0 && (
            <div className="flex items-center gap-2 bg-indigo-950/30 border border-indigo-800/40 rounded-lg px-3 py-2.5 text-xs text-indigo-300">
              <Info className="w-3.5 h-3.5 shrink-0" />
              No AEH repositories added. Add a repository in the{' '}
              <span
                className="underline cursor-pointer"
                onClick={() => navigate('/aeh/repositories')}
              >
                AEH Repositories
              </span>{' '}
              screen first.
            </div>
          )}
        </div>

        {/* CA Sibling report banner */}
        {selectedRepo && (
          <div className={`flex items-center justify-between gap-3 px-4 py-3 rounded-xl border text-xs backdrop-blur-sm ${
            reportExists
              ? 'bg-emerald-950/20 border-emerald-800/40 text-emerald-300'
              : siblingCARepo
              ? 'bg-amber-950/20 border-amber-800/40 text-amber-300'
              : 'bg-slate-900/40 border-slate-800/60 text-slate-400'
          }`}>
            <div className="flex items-center gap-2">
              <Info className="w-3.5 h-3.5 shrink-0" />
              {reportExists
                ? 'Code Analysis report found for this snapshot — will be used as context during discovery.'
                : siblingCARepo
                ? 'No CA report found for this snapshot. Discovery works without it, but a CA report improves accuracy.'
                : 'No CA-mode sibling import found. Import this repo in Code Analysis mode to enable report context.'}
            </div>
            {siblingCARepo && !reportExists && (
              <Button
                variant="ghost"
                className="text-[10px] px-2 py-1 shrink-0 flex items-center gap-1"
                onClick={() => navigate('/ca/analysis')}
              >
                <span>{matchingCASnapshot ? 'Run CA Report' : 'Import in CA'}</span>
                <ArrowRight className="w-3.5 h-3.5" />
              </Button>
            )}
            {reportExists && matchingCAReport && (
              <Button
                variant="ghost"
                className="text-[10px] px-2 py-1 shrink-0 flex items-center gap-1"
                onClick={() => navigate(`/ca/reports?reportId=${encodeURIComponent(matchingCAReport.id)}`)}
              >
                <span>View Report</span>
                <ArrowRight className="w-3.5 h-3.5" />
              </Button>
            )}
          </div>
        )}

        {/* Index Status section */}
        <div className="bg-[#0f172a]/60 border border-slate-800/80 rounded-xl p-5 space-y-4 backdrop-blur-sm shadow-xl">
          <div className="text-sm font-semibold text-slate-200 border-b border-slate-850 pb-2">
            Index Quality & Status
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {/* Code Manifest Card */}
            <div className="bg-slate-900/40 border border-slate-800 rounded-lg p-4 space-y-3 flex flex-col justify-between">
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2 text-xs font-semibold text-slate-300">
                    <FileText className="w-4 h-4 text-indigo-400" />
                    <span>Code Manifest</span>
                  </div>
                  {renderStatusBadge(manifestStatus)}
                </div>
                <p className="text-[11px] text-slate-400 leading-relaxed">
                  Lists files, extensions, and directory structure. Built automatically during snapshot sync.
                </p>
              </div>
            </div>

            {/* Structural Graph Card */}
            <div className="bg-slate-900/40 border border-slate-800 rounded-lg p-4 space-y-3 flex flex-col justify-between">
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2 text-xs font-semibold text-slate-300">
                    <Network className="w-4 h-4 text-indigo-400" />
                    <span>Structural Graph</span>
                  </div>
                  {renderStatusBadge(graphStatus)}
                </div>
                <p className="text-[11px] text-slate-400 leading-relaxed">
                  Maps class/function references and import cycles to determine architectural centrality.
                </p>
              </div>
              {graphStatus === 'built' && graphSummary && (
                <div className="text-[10px] text-slate-500 font-mono pt-1 border-t border-slate-800">
                  Nodes: {graphSummary.total_nodes} · Edges: {graphSummary.total_edges}
                </div>
              )}
            </div>

            {/* Retrieval Index Card */}
            <div className="bg-slate-900/40 border border-slate-800 rounded-lg p-4 space-y-3 flex flex-col justify-between">
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2 text-xs font-semibold text-slate-300">
                    <Search className="w-4 h-4 text-indigo-400" />
                    <span>Retrieval Index</span>
                  </div>
                  {renderStatusBadge(retrievalStatus)}
                </div>
                <p className="text-[11px] text-slate-400 leading-relaxed">
                  AST semantic chunks and BM25 lexicons used for similarity queries during discovery search.
                </p>
              </div>
              {retrievalStatus === 'built' && repomapSummary && (
                <div className="text-[10px] text-slate-500 font-mono pt-1 border-t border-slate-800">
                  Files: {repomapSummary.files_indexed} · Symbols: {repomapSummary.total_symbols}
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Pipeline Stage Strip */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          <div
            className={`border rounded-xl p-4 flex items-center justify-between gap-3 transition-colors ${
              repoId && snapshotId
                ? 'bg-[#0f172a]/60 border-indigo-500/40 hover:border-indigo-400 cursor-pointer shadow-lg'
                : 'bg-[#0f172a]/20 border-slate-900 text-slate-500 cursor-not-allowed opacity-55'
            }`}
            onClick={handleStage1Click}
          >
            <div className="flex items-center gap-3 min-w-0">
              <div className="w-7 h-7 rounded-full bg-indigo-950/50 border border-indigo-500/40 flex items-center justify-center text-indigo-300 text-xs font-bold shrink-0">1</div>
              <div className="min-w-0">
                <div className="text-xs font-semibold text-slate-200">Stage 1: Discovery</div>
                <div className="text-[10px] text-slate-400 truncate">
                  {discoverySession?.status === 'completed'
                    ? `${candidates.length} candidate${candidates.length !== 1 ? 's' : ''} found`
                    : discoverySession?.status === 'running'
                    ? 'Running…'
                    : 'Fingerprint scan + graph clustering'}
                </div>
              </div>
            </div>
            {repoId && snapshotId && <ArrowRight className="w-4 h-4 text-indigo-400 shrink-0" />}
          </div>
          <div
            className={`border rounded-xl p-4 flex items-center justify-between gap-3 transition-colors ${
              repoId && snapshotId && candidates.some((c) => c.verdict === 'confirmed')
                ? 'bg-[#0f172a]/60 border-indigo-500/40 hover:border-indigo-400 cursor-pointer shadow-lg'
                : 'bg-[#0f172a]/20 border-slate-900 text-slate-500 cursor-not-allowed opacity-55'
            }`}
            onClick={() => {
              if (repoId && snapshotId && candidates.some((c) => c.verdict === 'confirmed')) {
                navigate(`stage2?repoId=${repoId}&snapshotId=${snapshotId}`)
              }
            }}
          >
            <div className="flex items-center gap-3 min-w-0">
              <div className="w-7 h-7 rounded-full bg-indigo-950/50 border border-indigo-500/40 flex items-center justify-center text-indigo-300 text-xs font-bold shrink-0">2</div>
              <div className="min-w-0">
                <div className="text-xs font-semibold text-slate-200">Stage 2: Expand to System Map</div>
                <div className="text-[10px] text-slate-400 truncate">
                  {candidates.some((c) => c.verdict === 'confirmed')
                    ? 'Ready to expand confirmed component(s)'
                    : 'Requires at least one confirmed candidate'}
                </div>
              </div>
            </div>
            {repoId && snapshotId && candidates.some((c) => c.verdict === 'confirmed') && <ArrowRight className="w-4 h-4 text-indigo-400 shrink-0" />}
          </div>
          <div
            className="bg-[#0f172a]/40 border border-slate-800/60 rounded-xl p-4 flex items-center gap-3 opacity-60 cursor-not-allowed"
            title="Coming soon (CS-273) — proposes a metric suite per component for review"
          >
            <div className="w-7 h-7 rounded-full bg-slate-900 border border-slate-700 flex items-center justify-center text-slate-500 text-xs font-bold shrink-0">3</div>
            <div className="min-w-0">
              <div className="text-xs font-semibold text-slate-400">Stage 3: Build Evaluation</div>
              <div className="text-[10px] text-slate-600 truncate">Not yet implemented — CS-273</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default function AEHAnalysisScreen(): React.ReactElement {
  return (
    <Routes>
      <Route path="/" element={<AnalysisOverview />} />
      <Route path="stage1" element={<Stage1GraphScreen />} />
      <Route path="stage2" element={<Stage2Screen />} />
    </Routes>
  )
}

import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
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
  Play,
  FolderOpen,
  FileText,
  Search,
  Network,
  Info,
  ArrowRight,
  Loader2,
  ThumbsUp,
  ThumbsDown,
  ChevronDown,
  ChevronRight,
  Cpu,
  Zap,
} from 'lucide-react'
import type { RepoSnapshot, LocalRepo, AEHDiscoverySession, AEHDiscoveryCandidate } from '../../types/electron'

export default function AEHAnalysisScreen(): React.ReactElement {
  const navigate = useNavigate()
  const activeWorkspaceId = useWorkspaceStore((s) => s.activeWorkspaceId)
  const { repos, load: loadRepos } = useLocalRepoStore()
  // AEH's backend process is lazy-started (CS-260) — unlike AEHReportsScreen,
  // which blocks its whole render behind startAEH(), this screen's repo/index
  // status doesn't need the AEH backend at all (that's all CodeSpectra's own
  // backend). Only the discovery-session calls below need it, so each of them
  // awaits startAEH() individually right before calling — startAEH() itself
  // is a no-op once already running, and de-dupes concurrent in-flight starts.
  const { startAEH } = useAEHStore()

  const [persistedConfig, setPersistedConfig] = useState<any>(() => {
    try {
      return JSON.parse(localStorage.getItem('analysis.runConfig.v1') ?? '{}')
    } catch {
      return {}
    }
  })

  const [repoId, setRepoId] = useState(persistedConfig.repoId ?? '')
  const [snapshotId, setSnapshotId] = useState(persistedConfig.snapshotId ?? '')
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

  // Discovery state
  const [discoverySession, setDiscoverySession] = useState<AEHDiscoverySession | null>(null)
  const [candidates, setCandidates] = useState<AEHDiscoveryCandidate[]>([])
  const [runningDiscovery, setRunningDiscovery] = useState(false)
  const [discoveryError, setDiscoveryError] = useState<string | null>(null)
  const [expandedEvidence, setExpandedEvidence] = useState<Set<string>>(new Set())
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)

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

      // 1. Manifest status: if snapshot is ready, manifest is built
      const selectedSnap = snapshots.find((s) => s.id === snapshotId)
      if (selectedSnap && selectedSnap.status === 'ready') {
        setManifestStatus('built')
      } else {
        setManifestStatus('missing')
      }

      // 2. Repomap & Retrieval index status
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

      // 3. Graph status
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
    // The AEH backend is lazy-started — ensure it's up before the very first
    // discovery-related call this screen makes (fixes "AEH server is not
    // running" when a user lands on /aeh/analysis without visiting
    // /aeh/reports first, which is what actually starts it there).
    startAEH()
      .then(() => window.api.aeh.listDiscoverySessions(repoRef))
      .then((sessions) => {
        if (cancelled) return []
        // Find latest session for this snapshot
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

  // Poll discovery session when running
  const startPolling = useCallback((sessionId: string) => {
    if (pollRef.current) clearInterval(pollRef.current)
    pollRef.current = setInterval(async () => {
      try {
        const session = await window.api.aeh.getDiscoverySession(sessionId)
        setDiscoverySession(session)
        if (session.status !== 'running') {
          if (pollRef.current) clearInterval(pollRef.current)
          setRunningDiscovery(false)
          if (session.status === 'completed') {
            const cands = await window.api.aeh.listDiscoveryCandidates(sessionId)
            setCandidates(cands)
          } else if (session.status === 'failed') {
            setDiscoveryError(session.error ?? 'Discovery failed')
          }
        }
      } catch (err) {
        if (pollRef.current) clearInterval(pollRef.current)
        setRunningDiscovery(false)
      }
    }, 2500)
  }, [])

  useEffect(() => () => { if (pollRef.current) clearInterval(pollRef.current) }, [])

  const updatePersistedConfig = (updates: Partial<typeof persistedConfig>) => {
    const nextConfig = { ...persistedConfig, ...updates }
    setPersistedConfig(nextConfig)
    localStorage.setItem('analysis.runConfig.v1', JSON.stringify(nextConfig))
  }

  // Switching repo/snapshot mid-discovery must stop the in-flight poll — otherwise
  // it keeps firing for the OLD session and overwrites whatever the newly-selected
  // repo/snapshot's own state loads next (a real, confusing UI bug, not just stale data).
  const stopPolling = () => {
    if (pollRef.current) {
      clearInterval(pollRef.current)
      pollRef.current = null
    }
    setRunningDiscovery(false)
  }

  const handleRepoChange = (newRepoId: string) => {
    stopPolling()
    setRepoId(newRepoId)
    setSnapshotId('') // let effect select default
    setDiscoverySession(null)
    setCandidates([])
    setDiscoveryError(null)
  }

  const handleSnapshotChange = (newSnapshotId: string) => {
    stopPolling()
    setSnapshotId(newSnapshotId)
    updatePersistedConfig({ snapshotId: newSnapshotId })
    setDiscoverySession(null)
    setCandidates([])
    setDiscoveryError(null)
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

  const reportExists = useMemo(() => {
    if (!matchingCASnapshot) return false
    return reports.some((r) => r.snapshot_id === matchingCASnapshot.id)
  }, [reports, matchingCASnapshot])

  const indexesReady = manifestStatus === 'built' && graphStatus === 'built' && retrievalStatus === 'built'

  const handleRunDiscovery = async () => {
    if (!snapshotId || runningDiscovery) return
    setRunningDiscovery(true)
    setDiscoveryError(null)
    setCandidates([])
    setDiscoverySession(null)

    try {
      await startAEH()
      const repoRef = selectedRepo?.path ?? snapshotId
      const result = await window.api.aeh.startDiscovery({
        repo_ref: repoRef,
        snapshot_id: snapshotId,
      })
      const session = await window.api.aeh.getDiscoverySession(result.session_id)
      setDiscoverySession(session)
      startPolling(result.session_id)
    } catch (err: any) {
      setRunningDiscovery(false)
      setDiscoveryError(err?.message ?? 'Failed to start discovery')
    }
  }

  const handleVerdict = async (candidateId: string, verdict: 'confirmed' | 'rejected') => {
    try {
      await window.api.aeh.updateDiscoveryCandidateVerdict(candidateId, verdict)
      setCandidates((prev) =>
        prev.map((c) => c.id === candidateId ? { ...c, verdict } : c)
      )
    } catch (err) {
      console.error('Failed to update verdict', err)
    }
  }

  const toggleEvidence = (candidateId: string) => {
    setExpandedEvidence((prev) => {
      const next = new Set(prev)
      if (next.has(candidateId)) next.delete(candidateId)
      else next.add(candidateId)
      return next
    })
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

  const confidenceColor = (confidence: string) => {
    if (confidence === 'high') return 'text-emerald-400 bg-emerald-950/40 border-emerald-800/60'
    if (confidence === 'medium') return 'text-amber-400 bg-amber-950/40 border-amber-800/60'
    return 'text-slate-400 bg-slate-900/40 border-slate-700/60'
  }

  const verdictColor = (verdict: string) => {
    if (verdict === 'confirmed') return 'text-emerald-400'
    if (verdict === 'rejected') return 'text-red-400 line-through opacity-60'
    return 'text-slate-300'
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
                onClick={() => navigate('/analysis')}
              >
                <span>{matchingCASnapshot ? 'Run CA Report' : 'Import in CA'}</span>
                <ArrowRight className="w-3.5 h-3.5" />
              </Button>
            )}
            {reportExists && (
              <Button
                variant="ghost"
                className="text-[10px] px-2 py-1 shrink-0 flex items-center gap-1"
                onClick={() => navigate('/analysis')}
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
            {/* Manifest Status Card */}
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

            {/* Graph Status Card */}
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

            {/* Retrieval Status Card */}
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

        {/* Discovery Action Card */}
        <div className="bg-[#0f172a]/60 border border-slate-800/80 rounded-xl p-5 space-y-4 backdrop-blur-sm shadow-xl">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-9 h-9 rounded-full bg-indigo-950/40 border border-indigo-500/30 flex items-center justify-center text-indigo-400">
                <Cpu className="w-4 h-4" />
              </div>
              <div>
                <h3 className="text-sm font-semibold text-slate-200">Run Discovery Agent</h3>
                <p className="text-xs text-slate-400 mt-0.5">
                  Scan for agent frameworks, cluster by graph communities, and synthesize candidates via LLM.
                </p>
              </div>
            </div>
            <Button
              variant="primary"
              disabled={!snapshotId || !indexesReady || runningDiscovery}
              className="text-xs px-5 py-2 flex items-center gap-2 shrink-0"
              onClick={handleRunDiscovery}
            >
              {runningDiscovery ? (
                <>
                  <Loader2 className="w-3.5 h-3.5 animate-spin" />
                  <span>Discovering...</span>
                </>
              ) : (
                <>
                  <Play className="w-3.5 h-3.5 ml-0.5" />
                  <span>Run Discovery</span>
                </>
              )}
            </Button>
          </div>

          {!indexesReady && snapshotId && !runningDiscovery && (
            <div className="flex items-center gap-2 bg-amber-950/20 border border-amber-800/40 rounded-lg px-3 py-2 text-xs text-amber-300">
              <AlertCircle className="w-3.5 h-3.5 shrink-0" />
              Build all three indexes above before running discovery.
            </div>
          )}

          {/* Progress indicator while running */}
          {runningDiscovery && (
            <div className="space-y-2">
              {['Pass A — Framework Fingerprinting', 'Pass B — Graph Community Clustering', 'Pass C — LLM Candidate Synthesis'].map((step, i) => (
                <div key={i} className="flex items-center gap-3 text-xs text-slate-400">
                  <Loader2 className="w-3 h-3 animate-spin text-indigo-400 shrink-0" />
                  <span>{step}</span>
                </div>
              ))}
              <p className="text-[10px] text-slate-500 italic pt-1">
                This may take 30–120 seconds depending on repository size…
              </p>
            </div>
          )}

          {/* Discovery error */}
          {discoveryError && (
            <div className="flex items-start gap-2 bg-red-950/20 border border-red-800/40 rounded-lg px-3 py-2.5 text-xs text-red-300">
              <AlertCircle className="w-3.5 h-3.5 shrink-0 mt-0.5" />
              <span>{discoveryError}</span>
            </div>
          )}

          {/* Session status pill (completed/failed) */}
          {discoverySession && !runningDiscovery && (
            <div className="flex items-center gap-2 text-xs">
              {discoverySession.status === 'completed' ? (
                <span className="flex items-center gap-1.5 text-emerald-400">
                  <CheckCircle2 className="w-3.5 h-3.5" />
                  Discovery completed · {candidates.length} candidate{candidates.length !== 1 ? 's' : ''} found
                </span>
              ) : discoverySession.status === 'failed' ? (
                <span className="flex items-center gap-1.5 text-red-400">
                  <AlertCircle className="w-3.5 h-3.5" />
                  Discovery failed
                </span>
              ) : null}
              <span className="text-slate-600">·</span>
              <span className="text-slate-500 font-mono">{discoverySession.id.slice(0, 8)}</span>
            </div>
          )}
        </div>

        {/* Candidate Cards */}
        {candidates.length > 0 && (
          <div className="space-y-3">
            <div className="text-sm font-semibold text-slate-200 px-1 flex items-center gap-2">
              <Zap className="w-4 h-4 text-indigo-400" />
              <span>Discovered Candidates</span>
              <Badge variant="neutral" size="sm">{candidates.length}</Badge>
            </div>

            {candidates.map((cand) => (
              <div
                key={cand.id}
                className={`bg-[#0f172a]/60 border rounded-xl p-5 backdrop-blur-sm shadow-lg transition-all ${
                  cand.verdict === 'rejected'
                    ? 'border-slate-800/40 opacity-50'
                    : cand.verdict === 'confirmed'
                    ? 'border-emerald-700/40'
                    : 'border-slate-800/80'
                }`}
              >
                <div className="flex items-start justify-between gap-4">
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 flex-wrap">
                      <h4 className={`text-sm font-semibold ${verdictColor(cand.verdict)}`}>
                        {cand.name}
                      </h4>
                      <span className={`text-[10px] font-mono px-1.5 py-0.5 rounded border ${confidenceColor(cand.confidence)}`}>
                        {cand.confidence} confidence
                      </span>
                      {cand.verdict !== 'proposed' && (
                        <span className={`text-[10px] px-1.5 py-0.5 rounded ${
                          cand.verdict === 'confirmed'
                            ? 'bg-emerald-950/40 text-emerald-400 border border-emerald-800/40'
                            : 'bg-red-950/40 text-red-400 border border-red-800/40'
                        }`}>
                          {cand.verdict}
                        </span>
                      )}
                    </div>

                    {cand.frameworks.length > 0 && (
                      <div className="flex flex-wrap gap-1.5 mt-2">
                        {cand.frameworks.map((fw) => (
                          <span key={fw} className="text-[10px] px-1.5 py-0.5 rounded bg-indigo-950/40 border border-indigo-800/30 text-indigo-300 font-mono">
                            {fw}
                          </span>
                        ))}
                      </div>
                    )}

                    {cand.entry_points.length > 0 && (
                      <div className="mt-2 text-[11px] text-slate-400">
                        <span className="text-slate-500">Entry points: </span>
                        {cand.entry_points.join(', ')}
                      </div>
                    )}
                  </div>

                  {/* Confirm / Reject buttons */}
                  <div className="flex items-center gap-2 shrink-0">
                    <button
                      title="Confirm candidate"
                      onClick={() => handleVerdict(cand.id, 'confirmed')}
                      className={`p-1.5 rounded-lg border transition-colors ${
                        cand.verdict === 'confirmed'
                          ? 'bg-emerald-900/40 border-emerald-700/60 text-emerald-400'
                          : 'border-slate-700/60 text-slate-500 hover:border-emerald-700/60 hover:text-emerald-400'
                      }`}
                    >
                      <ThumbsUp className="w-3.5 h-3.5" />
                    </button>
                    <button
                      title="Reject candidate"
                      onClick={() => handleVerdict(cand.id, 'rejected')}
                      className={`p-1.5 rounded-lg border transition-colors ${
                        cand.verdict === 'rejected'
                          ? 'bg-red-900/40 border-red-700/60 text-red-400'
                          : 'border-slate-700/60 text-slate-500 hover:border-red-700/60 hover:text-red-400'
                      }`}
                    >
                      <ThumbsDown className="w-3.5 h-3.5" />
                    </button>
                  </div>
                </div>

                {/* Evidence drill-down */}
                {cand.evidence.length > 0 && (
                  <div className="mt-3 pt-3 border-t border-slate-800">
                    <button
                      onClick={() => toggleEvidence(cand.id)}
                      className="flex items-center gap-1.5 text-[11px] text-slate-400 hover:text-slate-200 transition-colors"
                    >
                      {expandedEvidence.has(cand.id)
                        ? <ChevronDown className="w-3 h-3" />
                        : <ChevronRight className="w-3 h-3" />}
                      {cand.evidence.length} evidence snippet{cand.evidence.length !== 1 ? 's' : ''}
                    </button>

                    {expandedEvidence.has(cand.id) && (
                      <div className="mt-2 space-y-2 max-h-56 overflow-y-auto pr-1">
                        {cand.evidence.slice(0, 10).map((ev: any, i: number) => (
                          <div key={i} className="bg-slate-950/60 border border-slate-800 rounded-lg p-3 text-[10px] font-mono space-y-1">
                            <div className="text-indigo-400 truncate">{ev.file}</div>
                            {ev.symbol && <div className="text-slate-400">symbol: {ev.symbol}</div>}
                            <div className="text-slate-300 whitespace-pre-wrap break-all line-clamp-3">{ev.snippet}</div>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                )}
              </div>
            ))}
          </div>
        )}

        {/* Empty state when discovery completes with no candidates */}
        {discoverySession?.status === 'completed' && candidates.length === 0 && !runningDiscovery && (
          <div className="bg-[#0f172a]/60 border border-slate-800/80 rounded-xl p-8 text-center text-slate-400 text-sm">
            <Search className="w-8 h-8 mx-auto mb-3 text-slate-600" />
            <p className="font-medium text-slate-300 mb-1">No candidates found</p>
            <p className="text-xs text-slate-500">
              No agentic system fingerprints were detected in this snapshot. Check that the codebase uses a supported framework.
            </p>
          </div>
        )}
      </div>
    </div>
  )
}

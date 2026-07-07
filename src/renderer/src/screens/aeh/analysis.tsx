import React, { useEffect, useMemo, useState } from 'react'
import { useNavigate, Routes, Route } from 'react-router-dom'
import { useWorkspaceStore } from '../../store/workspace.store'
import { useLocalRepoStore } from '../../store/local-repo.store'
import { useAEHStore } from '../../store/aeh.store'
import { useProviderStore } from '../../store/provider.store'
import {
  Button,
  FormGroup,
  Select,
  Badge,
  useToastStore,
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
  Play,
  Settings,
  Loader2,
} from 'lucide-react'
import type { RepoSnapshot, LocalRepo } from '../../types/electron'
// AEHDiscoverySession/AEHDiscoveryCandidate are global ambient types.
import Stage1GraphScreen from './Stage1GraphScreen'
import Stage2Screen from './Stage2Screen'
import Stage3Screen from './Stage3Screen'
import LLMConfigModal from './LLMConfigModal'

function AnalysisOverview(): React.ReactElement {
  const navigate = useNavigate()
  const activeWorkspaceId = useWorkspaceStore((s) => s.activeWorkspaceId)
  const { repos, load: loadRepos } = useLocalRepoStore()
  const { startAEH } = useAEHStore()
  const toast = useToastStore()
  const { providers, load: loadProviders } = useProviderStore()

  const [selectedProviderId, setSelectedProviderId] = useState('')
  const [selectedModelId, setSelectedModelId] = useState('')
  const [llmConfigOpen, setLlmConfigOpen] = useState(false)

  const [pollingSessionId, setPollingSessionId] = useState<string | null>(null)
  const [runningAll, setRunningAll] = useState(false)

  // Load providers on mount
  useEffect(() => {
    loadProviders()
  }, [loadProviders])

  // Set default provider/model
  useEffect(() => {
    if (providers.length > 0 && !selectedProviderId) {
      const defaultProv = providers[0]
      setSelectedProviderId(defaultProv.id)
      setSelectedModelId(defaultProv.model_id || '')
    }
  }, [providers, selectedProviderId])

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
  const [reports, setReports] = useState<any[]>([])

  // Statuses for indices
  const [manifestStatus, setManifestStatus] = useState<'loading' | 'built' | 'missing'>('loading')
  const [graphStatus, setGraphStatus] = useState<'loading' | 'built' | 'missing'>('loading')
  const [retrievalStatus, setRetrievalStatus] = useState<'loading' | 'built' | 'missing'>('loading')

  const [repomapSummary, setRepomapSummary] = useState<any | null>(null)
  const [graphSummary, setGraphSummary] = useState<any | null>(null)

  const [discoverySession, setDiscoverySession] = useState<AEHDiscoverySession | null>(null)
  const [candidates, setCandidates] = useState<AEHDiscoveryCandidate[]>([])
  const [completedExpansionsCount, setCompletedExpansionsCount] = useState(0)
  const [hasPlanCount, setHasPlanCount] = useState(0)

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
    window.api.analysis.listReports(siblingCARepo.id)
      .then((list) => setReports(list))
      .catch((err) => console.error('Failed to list CA reports', err))
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
      } catch (e) {
        setRepomapSummary(null)
      }

      try {
        const retSummary = await window.api.retrieval.summary(snapshotId)
        setRetrievalStatus(retSummary && retSummary.built ? 'built' : 'missing')
      } catch (e) {
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
          if (['completed', 'failed', 'paused_rate_limit'].includes(latest.status) || latest.pipeline_stage !== 'fingerprinting') {
            return window.api.aeh.listDiscoveryCandidates(latest.id)
          }
        }
        return []
      })
      .then((cands) => { if (!cancelled) setCandidates(cands) })
      .catch(() => {})
    return () => { cancelled = true }
  }, [snapshotId, selectedRepo?.path, startAEH])
  // Load expansion stats (completed and planned sessions) across confirmed candidates
  useEffect(() => {
    const confirmedCands = candidates.filter((c) => c.verdict === 'confirmed')
    if (confirmedCands.length === 0) {
      setCompletedExpansionsCount(0)
      setHasPlanCount(0)
      return
    }

    let cancelled = false
    const getStats = async () => {
      try {
        let completed = 0
        let planned = 0
        for (const cand of confirmedCands) {
          const sessions = await window.api.aeh.listExpansionSessions(cand.id)
          const completedSess = sessions.filter((s) => s.status === 'completed')
          if (completedSess.length > 0) {
            completed++
            if (completedSess.some((s) => s.plan_path)) {
              planned++
            }
          }
        }
        if (!cancelled) {
          setCompletedExpansionsCount(completed)
          setHasPlanCount(planned)
        }
      } catch (e) {
        console.error('Failed to load expansion stats', e)
      }
    }

    getStats()
    return () => {
      cancelled = true
    }
  }, [candidates])

  const totalConfirmedCandidates = useMemo(() => {
    return candidates.filter((c) => c.verdict === 'confirmed').length
  }, [candidates])

  const effectiveStage = useMemo(() => {
    if (!discoverySession) return 'fingerprinting'
    if (discoverySession.status === 'running') return 'fingerprinting'

    const confirmedCands = candidates.filter((c) => c.verdict === 'confirmed')
    if (confirmedCands.length === 0) {
      return 'awaiting_candidate_review'
    }

    if (completedExpansionsCount === 0) {
      return 'expanding'
    }

    if (hasPlanCount > 0) {
      if (discoverySession.pipeline_stage === 'done') {
        return 'done'
      }
      return 'awaiting_plan_review'
    }

    if (['planning', 'awaiting_plan_review'].includes(discoverySession.pipeline_stage)) {
      return discoverySession.pipeline_stage
    }

    return 'awaiting_map_review'
  }, [discoverySession, candidates, completedExpansionsCount, hasPlanCount])


  // Polling active discovery session
  useEffect(() => {
    if (!pollingSessionId) return
    let cancelled = false
    const interval = setInterval(async () => {
      try {
        const sess = await window.api.aeh.getDiscoverySession(pollingSessionId)
        if (cancelled) return
        setDiscoverySession(sess)
        
        // Also fetch candidates if not fingerprinting
        if (sess.pipeline_stage !== 'fingerprinting') {
          const cands = await window.api.aeh.listDiscoveryCandidates(pollingSessionId)
          if (!cancelled) setCandidates(cands)
        }

        // Stop polling if session is done or failed
        const isRunning = sess.status === 'running' || ['fingerprinting', 'expanding', 'planning'].includes(sess.pipeline_stage)
        if (!isRunning) {
          setPollingSessionId(null)
        }
      } catch (err) {
        console.error('Failed to poll discovery session', err)
      }
    }, 2000)

    return () => {
      cancelled = true
      clearInterval(interval)
    }
  }, [pollingSessionId])

  // Trigger polling whenever a running discovery session is loaded
  useEffect(() => {
    if (discoverySession) {
      const isRunning = discoverySession.status === 'running' || ['fingerprinting', 'expanding', 'planning'].includes(discoverySession.pipeline_stage)
      if (isRunning && pollingSessionId !== discoverySession.id) {
        setPollingSessionId(discoverySession.id)
      }
    } else {
      setPollingSessionId(null)
    }
  }, [discoverySession, pollingSessionId])

  // Confirm and advance handlers
  const handleConfirmCandidates = async () => {
    if (!discoverySession) return
    try {
      const confirmedIds = candidates.filter((c) => c.verdict === 'confirmed').map((c) => c.id)
      if (confirmedIds.length === 0) {
        toast.error('At least one candidate must be confirmed to proceed.')
        return
      }
      await window.api.aeh.advanceSession(discoverySession.id, { confirmed_candidates: confirmedIds })
      toast.success('Candidates confirmed. Advancing pipeline to Stage 2: Expansion.')
      
      const updated = await window.api.aeh.getDiscoverySession(discoverySession.id)
      setDiscoverySession(updated)

      // Start expansions automatically
      await handleRunStage2Expansions(confirmedIds)
    } catch (err: any) {
      toast.error(err?.message ?? 'Failed to advance session.')
    }
  }

  const handleRunStage2Expansions = async (confirmedIds: string[]) => {
    if (!discoverySession) return
    try {
      const provId = selectedProviderId || (providers[0]?.id) || null
      const modId = selectedModelId || (providers[0]?.model_id) || null
      for (const candId of confirmedIds) {
        await window.api.aeh.startExpansion(candId, {
          provider_id: provId,
          model_id: modId,
          node_budget: 100,
          hop_cap: 3,
        })
      }
      toast.success('Triggered Stage 2 iterative expansions.')
      setPollingSessionId(discoverySession.id)
    } catch (err: any) {
      toast.error('Failed to start expansions: ' + err.message)
    }
  }

  const handleConfirmMap = async () => {
    if (!discoverySession) return
    try {
      const confirmedCands = candidates.filter((c) => c.verdict === 'confirmed')
      let mapSessionId: string | null = null
      if (confirmedCands.length > 0) {
        for (const cand of confirmedCands) {
          const sessList = await window.api.aeh.listExpansionSessions(cand.id)
          const completedSess = sessList.find((s) => s.status === 'completed')
          if (completedSess) {
            mapSessionId = completedSess.id
            break
          }
        }
      }
      if (!mapSessionId) {
        toast.error('No completed expansion session found. Run Stage 2 first.')
        return
      }
      await window.api.aeh.advanceSession(discoverySession.id, { confirmed_map_session_id: mapSessionId })
      toast.success('System Map blueprint confirmed. Advancing to Stage 3: Planning.')
      
      const updated = await window.api.aeh.getDiscoverySession(discoverySession.id)
      setDiscoverySession(updated)

      // Start planning automatically
      await handleRunStage3Planning(mapSessionId)
    } catch (err: any) {
      toast.error(err?.message ?? 'Failed to advance session.')
    }
  }

  const handleRunStage3Planning = async (mapSessionId: string) => {
    if (!discoverySession) return
    try {
      const provId = selectedProviderId || (providers[0]?.id) || null
      const modId = selectedModelId || (providers[0]?.model_id) || null
      await window.api.aeh.generatePlan(mapSessionId, {
        provider_id: provId,
        model_id: modId,
      })
      toast.success('Triggered Stage 3 build evaluation planning.')
      setPollingSessionId(discoverySession.id)
    } catch (err: any) {
      toast.error('Failed to start planning: ' + err.message)
    }
  }

  const handleConfirmPlan = async () => {
    if (!discoverySession) return
    try {
      await window.api.aeh.advanceSession(discoverySession.id, { confirmed_plan: true })
      toast.success('Evaluation plan suite finalized successfully!')
      const updated = await window.api.aeh.getDiscoverySession(discoverySession.id)
      setDiscoverySession(updated)
    } catch (err: any) {
      toast.error(err?.message ?? 'Failed to finalize plan.')
    }
  }

  const handleRunAll = async () => {
    if (!snapshotId) return
    setRunningAll(true)
    try {
      let currentSession = discoverySession
      const provId = selectedProviderId || (providers[0]?.id) || null
      const modId = selectedModelId || (providers[0]?.model_id) || null

      if (!currentSession) {
        await startAEH()
        const result = await window.api.aeh.startDiscovery({
          repo_ref: snapshotId,
          snapshot_id: snapshotId,
          provider_id: provId,
          model_id: modId,
        })
        const session = await window.api.aeh.getDiscoverySession(result.session_id)
        setDiscoverySession(session)
        currentSession = session
      }

      setPollingSessionId(currentSession.id)
      toast.success('Pipeline orchestrator started Run All sequence.')
    } catch (err: any) {
      toast.error('Orchestration failed: ' + err.message)
    } finally {
      setRunningAll(false)
    }
  }

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

  const renderPipelineBadge = (stageNum: number) => {
    const stage = effectiveStage

    if (!discoverySession) {
      if (stageNum === 1) return <Badge variant="neutral" size="sm" className="bg-slate-900 border border-slate-800 text-slate-400 font-normal">Not Started</Badge>
      return <Badge variant="neutral" size="sm" className="bg-slate-900 border border-slate-800 text-slate-400 font-normal">Pending</Badge>
    }

    if (stageNum === 1) {
      if (stage === 'fingerprinting') {
        return (
          <Badge variant="neutral" size="sm" className="bg-slate-900 border border-slate-800 text-slate-400 font-normal flex items-center gap-1.5 animate-pulse">
            <Loader2 className="animate-spin w-3 h-3 text-indigo-400" />
            <span>Running</span>
          </Badge>
        )
      }
      if (stage === 'awaiting_candidate_review') {
        return <Badge variant="neutral" size="sm" className="bg-amber-950/30 border border-amber-800/40 text-amber-400 font-normal">Needs Review</Badge>
      }
      return <Badge variant="neutral" size="sm" className="bg-emerald-950/30 border border-emerald-800/40 text-emerald-400 font-normal">Complete</Badge>
    }

    if (stageNum === 2) {
      if (['fingerprinting', 'awaiting_candidate_review'].includes(stage)) {
        return <Badge variant="neutral" size="sm" className="bg-slate-900 border border-slate-800 text-slate-400 font-normal">Pending</Badge>
      }
      if (stage === 'expanding') {
        const progText = totalConfirmedCandidates > 0 ? ` (${completedExpansionsCount}/${totalConfirmedCandidates})` : ''
        return (
          <Badge variant="neutral" size="sm" className="bg-indigo-950/30 border border-indigo-800/40 text-indigo-400 font-normal flex items-center gap-1.5">
            <Loader2 className="animate-spin w-3 h-3 text-indigo-400 animate-spin" />
            <span>Expanding{progText}</span>
          </Badge>
        )
      }
      if (stage === 'awaiting_map_review') {
        return <Badge variant="neutral" size="sm" className="bg-amber-950/30 border border-amber-800/40 text-amber-400 font-normal">Needs Review</Badge>
      }
      return <Badge variant="neutral" size="sm" className="bg-emerald-950/30 border border-emerald-800/40 text-emerald-400 font-normal">Complete</Badge>
    }

    if (stageNum === 3) {
      if (['fingerprinting', 'awaiting_candidate_review', 'expanding', 'awaiting_map_review'].includes(stage)) {
        return <Badge variant="neutral" size="sm" className="bg-slate-900 border border-slate-800 text-slate-400 font-normal">Pending</Badge>
      }
      if (stage === 'planning') {
        return (
          <Badge variant="neutral" size="sm" className="bg-indigo-950/30 border border-indigo-800/40 text-indigo-400 font-normal flex items-center gap-1.5">
            <Loader2 className="animate-spin w-3 h-3 text-indigo-400 animate-spin" />
            <span>Planning</span>
          </Badge>
        )
      }
      if (stage === 'awaiting_plan_review') {
        return <Badge variant="neutral" size="sm" className="bg-amber-950/30 border border-amber-800/40 text-amber-400 font-normal">Needs Review</Badge>
      }
      return <Badge variant="neutral" size="sm" className="bg-emerald-950/30 border border-emerald-800/40 text-emerald-400 font-normal">Complete</Badge>
    }

    return <Badge variant="neutral" size="sm" className="bg-slate-900 border border-slate-800 text-slate-400 font-normal">Pending</Badge>
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

      {/* Orchestrator Control Panel */}
      {repoId && snapshotId && (
        <div className="bg-[#0f172a]/60 border border-slate-800/80 rounded-xl p-5 space-y-4 backdrop-blur-sm shadow-xl flex items-center justify-between">
          <div className="space-y-1">
            <h3 className="text-sm font-semibold text-slate-200">Pipeline Orchestrator</h3>
            <p className="text-[11px] text-slate-500">Trigger the full evaluation harness generation pipeline sequentially.</p>
          </div>
          <div className="flex items-center gap-3">
            <Button
              variant="ghost"
              onClick={() => setLlmConfigOpen(true)}
              className="w-9 h-9 p-0 flex items-center justify-center border border-slate-800 bg-slate-950 text-slate-400 hover:text-slate-200"
              title="Configure pipeline models"
            >
              <Settings size={15} />
            </Button>
            <Button
              variant="primary"
              onClick={handleRunAll}
              loading={runningAll || !!pollingSessionId}
              className="text-xs h-9 px-4 flex items-center gap-1.5 bg-indigo-600 hover:bg-indigo-500 font-semibold"
            >
              <Play size={13} />
              <span>Run All Pipeline</span>
            </Button>
          </div>
        </div>
      )}

      {/* Pipeline Stage Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 items-start">
        {/* Stage 1 Card */}
        <div className={`border rounded-xl p-5 space-y-4 backdrop-blur-sm transition-all ${
          effectiveStage === 'fingerprinting' || effectiveStage === 'awaiting_candidate_review'
            ? 'bg-[#0f172a]/70 border-amber-500/50 shadow-amber-500/5 shadow-lg'
            : 'bg-[#0f172a]/40 border-slate-800/80'
        }`}>
          <div className="flex flex-col gap-3">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 rounded-full bg-slate-900 border border-slate-800 flex items-center justify-center text-slate-300 text-xs font-bold shrink-0">1</div>
              <div className="min-w-0">
                <h4 className="text-xs font-semibold text-slate-200">Stage 1: Candidate Discovery</h4>
                <p className="text-[10px] text-slate-500">Scan code repository structure for component candidates.</p>
              </div>
            </div>

            <div className="flex items-center justify-between gap-2">
              {renderPipelineBadge(1)}
              <Button
                variant="ghost"
                onClick={handleStage1Click}
                disabled={!repoId || !snapshotId}
                className="text-[10px] h-7 px-3 border border-slate-800 bg-slate-950 hover:bg-slate-900 text-slate-300"
              >
                <span>Open Screen</span>
                <ArrowRight className="w-3 h-3 ml-1" />
              </Button>
            </div>
          </div>

          {effectiveStage === 'awaiting_candidate_review' && (
            <div className="border-t border-slate-850 pt-4 space-y-3">
              <div className="text-xs text-amber-300 flex items-center gap-1.5 font-semibold">
                <AlertCircle size={14} className="text-amber-400" />
                <span>Human Triage Required</span>
              </div>
              <p className="text-[11px] text-slate-400">
                Found {candidates.length} candidate(s). Review and confirm component candidates on the Stage 1 interactive screen, then click Confirm & Continue below.
              </p>
              <div className="flex items-center gap-2.5 text-[10px] font-mono text-slate-500">
                <span>Confirmed: {candidates.filter(c => c.verdict === 'confirmed').length}</span>
                <span>•</span>
                <span>Rejected: {candidates.filter(c => c.verdict === 'rejected').length}</span>
                <span>•</span>
                <span>Proposed: {candidates.filter(c => c.verdict === 'proposed').length}</span>
              </div>
              <div className="flex flex-wrap justify-end gap-2">
                <Button
                  variant="ghost"
                  onClick={() => navigate(`stage1?repoId=${repoId}&snapshotId=${snapshotId}`)}
                  className="text-[10px] h-7 px-3 border border-slate-800 bg-slate-950 text-slate-300"
                >
                  Go to Graph Review
                </Button>
                <Button
                  variant="primary"
                  onClick={handleConfirmCandidates}
                  disabled={candidates.filter(c => c.verdict === 'confirmed').length === 0}
                  className="text-[10px] h-7 px-4 bg-emerald-600 hover:bg-emerald-500 text-white font-semibold"
                >
                  Confirm & Continue
                </Button>
              </div>
            </div>
          )}
        </div>

        {/* Stage 2 Card */}
        <div className={`border rounded-xl p-5 space-y-4 backdrop-blur-sm transition-all ${
          effectiveStage === 'expanding' || effectiveStage === 'awaiting_map_review'
            ? 'bg-[#0f172a]/70 border-amber-500/50 shadow-amber-500/5 shadow-lg'
            : 'bg-[#0f172a]/40 border-slate-800/80'
        }`}>
          <div className="flex flex-col gap-3">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 rounded-full bg-slate-900 border border-slate-800 flex items-center justify-center text-slate-300 text-xs font-bold shrink-0">2</div>
              <div className="min-w-0">
                <h4 className="text-xs font-semibold text-slate-200">Stage 2: Iterative Expansion to System Map</h4>
                <p className="text-[10px] text-slate-500">Perform BFS neighborhood search to compile the full component map.</p>
              </div>
            </div>

            <div className="flex items-center justify-between gap-2">
              {renderPipelineBadge(2)}
              <Button
                variant="ghost"
                onClick={() => {
                  if (repoId && snapshotId && candidates.some((c) => c.verdict === 'confirmed')) {
                    navigate(`stage2?repoId=${repoId}&snapshotId=${snapshotId}`)
                  }
                }}
                disabled={!repoId || !snapshotId || !candidates.some((c) => c.verdict === 'confirmed')}
                className="text-[10px] h-7 px-3 border border-slate-800 bg-slate-950 hover:bg-slate-900 text-slate-300 disabled:opacity-40"
              >
                <span>Open Screen</span>
                <ArrowRight className="w-3 h-3 ml-1" />
              </Button>
            </div>
          </div>

          {effectiveStage === 'awaiting_map_review' && (
            <div className="border-t border-slate-850 pt-4 space-y-3">
              <div className="text-xs text-amber-300 flex items-center gap-1.5 font-semibold">
                <AlertCircle size={14} className="text-amber-400" />
                <span>Human Triage Required</span>
              </div>
              <p className="text-[11px] text-slate-400">
                Neighborhood expansion complete. Review system map connections and framework constraints on the Stage 2 interactive screen, then click Confirm & Continue below.
              </p>
              <div className="flex flex-wrap justify-end gap-2">
                <Button
                  variant="ghost"
                  onClick={() => navigate(`stage2?repoId=${repoId}&snapshotId=${snapshotId}`)}
                  className="text-[10px] h-7 px-3 border border-slate-800 bg-slate-950 text-slate-300"
                >
                  Go to Blueprint Review
                </Button>
                <Button
                  variant="primary"
                  onClick={handleConfirmMap}
                  className="text-[10px] h-7 px-4 bg-emerald-600 hover:bg-emerald-500 text-white font-semibold"
                >
                  Confirm & Continue
                </Button>
              </div>
            </div>
          )}
        </div>

        {/* Stage 3 Card */}
        <div className={`border rounded-xl p-5 space-y-4 backdrop-blur-sm transition-all ${
          effectiveStage === 'planning' || effectiveStage === 'awaiting_plan_review'
            ? 'bg-[#0f172a]/70 border-amber-500/50 shadow-amber-500/5 shadow-lg'
            : 'bg-[#0f172a]/40 border-slate-800/80'
        }`}>
          <div className="flex flex-col gap-3">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 rounded-full bg-slate-900 border border-slate-800 flex items-center justify-center text-slate-300 text-xs font-bold shrink-0">3</div>
              <div className="min-w-0">
                <h4 className="text-xs font-semibold text-slate-200">Stage 3: Build Evaluation Plan</h4>
                <p className="text-[10px] text-slate-500">Propose custom metric assertions, judges, and classifier test suites.</p>
              </div>
            </div>

            <div className="flex items-center justify-between gap-2">
              {renderPipelineBadge(3)}
              <Button
                variant="ghost"
                onClick={() => {
                  const stage = effectiveStage
                  if (repoId && snapshotId && !['fingerprinting', 'awaiting_candidate_review', 'expanding', 'awaiting_map_review'].includes(stage)) {
                    navigate(`stage3?repoId=${repoId}&snapshotId=${snapshotId}`)
                  }
                }}
                disabled={!repoId || !snapshotId || ['fingerprinting', 'awaiting_candidate_review', 'expanding', 'awaiting_map_review'].includes(effectiveStage)}
                className="text-[10px] h-7 px-3 border border-slate-800 bg-slate-950 hover:bg-slate-900 text-slate-300 disabled:opacity-40"
              >
                <span>Open Screen</span>
                <ArrowRight className="w-3 h-3 ml-1" />
              </Button>
            </div>
          </div>

          {effectiveStage === 'awaiting_plan_review' && (
            <div className="border-t border-slate-850 pt-4 space-y-3">
              <div className="text-xs text-amber-300 flex items-center gap-1.5 font-semibold">
                <AlertCircle size={14} className="text-amber-400" />
                <span>Human Triage Required</span>
              </div>
              <p className="text-[11px] text-slate-400">
                Metric plan entries successfully tailored. Review individual assertions on the Stage 3 plan screen, customize as needed, and click Confirm & Continue to finalize.
              </p>
              <div className="flex flex-wrap justify-end gap-2">
                <Button
                  variant="ghost"
                  onClick={() => navigate(`stage3?repoId=${repoId}&snapshotId=${snapshotId}`)}
                  className="text-[10px] h-7 px-3 border border-slate-800 bg-slate-950 text-slate-300"
                >
                  Go to Plan Review
                </Button>
                <Button
                  variant="primary"
                  onClick={handleConfirmPlan}
                  className="text-[10px] h-7 px-4 bg-emerald-600 hover:bg-emerald-500 text-white font-semibold"
                >
                  Confirm & Continue
                </Button>
              </div>
            </div>
          )}
        </div>
      </div>

      <LLMConfigModal
        isOpen={llmConfigOpen}
        onClose={() => setLlmConfigOpen(false)}
        providerId={selectedProviderId}
        modelId={selectedModelId}
        onChange={(prov, model) => {
          setSelectedProviderId(prov)
          setSelectedModelId(model)
        }}
        title="Planning LLM Model (CS-274)"
      />
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
      <Route path="stage3" element={<Stage3Screen />} />
    </Routes>
  )
}

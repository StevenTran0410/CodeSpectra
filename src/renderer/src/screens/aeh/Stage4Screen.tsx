import React, { useEffect, useState } from 'react'
import { useSearchParams, useNavigate } from 'react-router-dom'
import {
  AlertCircle,
  ArrowLeft,
  CheckCircle2,
  Copy,
  FolderOpen,
  RefreshCw,
  Trash2,
} from 'lucide-react'
import { Button, useToastStore } from '../../components/ui'
import SystemSelector from './SystemSelector'

export default function Stage4Screen(): React.ReactElement {
  const [searchParams] = useSearchParams()
  const navigate = useNavigate()
  const toast = useToastStore()
  const sessionId = searchParams.get('sessionId') || ''

  const [activeSessionId, setActiveSessionId] = useState(sessionId)

  const [error, setError] = useState<string | null>(null)
  const [planPath, setPlanPath] = useState<string | null>(null)
  const [planDir, setPlanDir] = useState<string | null>(null)
  const [planFiles, setPlanFiles] = useState<string[]>([])
  const [baseRef, setBaseRef] = useState('main')
  const [planCreating, setPlanCreating] = useState(false)
  const [planDeleting, setPlanDeleting] = useState(false)

  // Sync activeSessionId if sessionId URL parameter changes
  useEffect(() => {
    setActiveSessionId(sessionId)
  }, [sessionId])

  // Surface an already-rendered plan on activeSessionId change, so a stale one is never invisible.
  useEffect(() => {
    if (!activeSessionId) return
    let cancelled = false
    window.api.aeh
      .getEvalPlan(activeSessionId)
      .then((data) => {
        if (cancelled) return
        if (data?.plan_path) {
          setPlanPath(data.plan_path)
          setPlanDir(data.plan_dir ?? null)
          setPlanFiles(data.files ?? [])
        } else {
          setPlanPath(null)
          setPlanDir(null)
          setPlanFiles([])
        }
      })
      .catch(() => {
        if (cancelled) return
        setPlanPath(null)
        setPlanDir(null)
        setPlanFiles([])
      })
    return () => {
      cancelled = true
    }
  }, [activeSessionId])

  function handleSelectSystem(newId: string) {
    setActiveSessionId(newId)
    setPlanPath(null)
    setPlanDir(null)
    setPlanFiles([])
    setError(null)
  }

  async function handleDeletePlan() {
    setPlanDeleting(true)
    setError(null)
    try {
      await window.api.aeh.deleteEvalPlan(activeSessionId)
      setPlanPath(null)
      setPlanDir(null)
      setPlanFiles([])
      toast.success('Old plan deleted')
    } catch (err: any) {
      setError(err?.message ?? 'Failed to delete plan')
    } finally {
      setPlanDeleting(false)
    }
  }

  async function handleRegeneratePlan() {
    setPlanCreating(true)
    setError(null)
    try {
      await window.api.aeh.deleteEvalPlan(activeSessionId)
    } catch {
      // A missing plan is fine — regenerate is also the "render it fresh" path.
    }
    await handleRegeneratePlanFresh()
  }

  async function handleRegeneratePlanFresh() {
    setPlanCreating(true)
    setError(null)
    try {
      const data = await window.api.aeh.createEvalPlan(activeSessionId, baseRef)
      setPlanPath(data.plan_path)
      setPlanDir(data.plan_dir ?? null)
      setPlanFiles(data.files ?? [])
      toast.success(`Eval plan created — ${data.files?.length ?? 0} files. Start with AGENTS.md.`)
    } catch (err: any) {
      setError(err?.message ?? 'Failed to create eval plan')
    } finally {
      setPlanCreating(false)
    }
  }

  async function handleCreatePlan() {
    setPlanCreating(true)
    setError(null)
    try {
      const data = await window.api.aeh.createEvalPlan(activeSessionId, baseRef)
      setPlanPath(data.plan_path)
      setPlanDir(data.plan_dir ?? null)
      setPlanFiles(data.files ?? [])
      toast.success(`Eval plan created — ${data.files?.length ?? 0} files. Start with AGENTS.md.`)
    } catch (err: any) {
      setError(err?.message ?? 'Failed to create eval plan')
    } finally {
      setPlanCreating(false)
    }
  }

  async function handleCopyPath() {
    if (!planDir) return
    await window.api.app.copyToClipboard(planDir)
    toast.success('Folder path copied')
  }

  async function handleOpenFolder() {
    if (!planPath) return
    await window.api.app.showInFolder(planPath)
  }

  if (!sessionId) {
    return (
      <div className="flex flex-col h-full bg-[#090d16] text-slate-100 p-6">
        <div className="flex items-center gap-2 mb-6">
          <Button
            variant="ghost"
            className="w-8 h-8 p-0 rounded-full flex items-center justify-center border border-slate-800 bg-slate-900/40 text-slate-400 hover:text-slate-200"
            onClick={() => navigate(-1)}
          >
            <ArrowLeft size={16} />
          </Button>
          <h1 className="text-lg font-semibold">Stage 4: Eval Plan</h1>
        </div>
        <div className="text-sm text-amber-400">No session ID provided.</div>
      </div>
    )
  }

  return (
    <div className="flex flex-col h-full bg-[#090d16] text-slate-100">
      <div className="screen-header shrink-0 flex items-center justify-between px-6 py-4 border-b border-slate-850">
        <div className="flex items-center gap-3">
          <Button
            variant="ghost"
            className="w-8 h-8 p-0 rounded-full flex items-center justify-center border border-slate-800 bg-slate-900/40 text-slate-400 hover:text-slate-200"
            onClick={() => navigate(-1)}
          >
            <ArrowLeft size={16} />
          </Button>
          <div>
            <h1 className="screen-title">Stage 4: Eval Plan</h1>
            <p className="screen-subtitle">Generate the handoff plan for your coding agent</p>
          </div>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto p-6">
        <div className="space-y-6 max-w-2xl">
          {error && (
            <div className="bg-rose-950/20 border border-rose-800/40 rounded-xl p-4 text-xs text-rose-300 space-y-1">
              <div className="font-semibold flex items-center gap-1.5">
                <AlertCircle size={14} className="text-rose-400" />
                <span>Error Occurred</span>
              </div>
              <p className="text-[11px] leading-relaxed text-rose-300/80">{error}</p>
            </div>
          )}

          <div className="bg-slate-950/40 border border-slate-850 rounded-xl p-6 space-y-4">
            <div>
              <h2 className="text-sm font-semibold text-slate-200 mb-3">Create Eval Plan</h2>
              <p className="text-xs text-slate-400 mb-4">
                Writes 4 self-contained files — AGENTS.md (rules), TASKS.md (checklist),
                REFERENCE.md (per-agent detail), CODE.md (generated code) — with everything
                your coding agent needs, including creating the eval branch itself. AEH doesn't
                touch git for this; the plan tells the coding agent to do it as its own
                first step.
              </p>

              <div className="space-y-3">
                <SystemSelector
                  sessionId={sessionId}
                  activeSessionId={activeSessionId}
                  onSelect={handleSelectSystem}
                  disabled={planCreating || planDeleting}
                />

                <div>
                  <label className="block text-xs font-medium text-slate-400 mb-1">Base Branch (default: main)</label>
                  <input
                    type="text"
                    value={baseRef}
                    onChange={(e) => setBaseRef(e.target.value)}
                    disabled={!!planPath || planCreating}
                    className="w-full text-xs px-3 py-2 bg-slate-900 border border-slate-800 rounded text-slate-200 disabled:opacity-50"
                    placeholder="main"
                  />
                  <p className="text-[10px] text-slate-600 mt-1">
                    The branch your coding agent's eval branch will be created from
                  </p>
                </div>

                {planPath ? (
                  <div className="space-y-2">
                    <div className="p-3 bg-emerald-950/20 border border-emerald-800/40 rounded-lg space-y-2">
                      <div className="flex items-center gap-2">
                        <CheckCircle2 size={16} className="text-emerald-400 shrink-0" />
                        <div className="text-xs text-emerald-300 break-all">
                          {planDir ?? planPath}
                        </div>
                      </div>
                      {planFiles.length > 0 && (
                        <div className="flex flex-wrap gap-1.5 pl-6">
                          {planFiles.map((f) => (
                            <span
                              key={f}
                              className="text-[10px] px-1.5 py-0.5 rounded bg-emerald-900/30 border border-emerald-800/40 text-emerald-300"
                            >
                              {f}
                            </span>
                          ))}
                        </div>
                      )}
                      <div className="flex gap-2 pl-6">
                        <Button
                          onClick={handleCopyPath}
                          disabled={!planDir}
                          className="text-[11px] h-7 px-2 bg-slate-800 hover:bg-slate-700 text-slate-200"
                        >
                          <Copy size={12} className="mr-1" />
                          Copy path
                        </Button>
                        <Button
                          onClick={handleOpenFolder}
                          className="text-[11px] h-7 px-2 bg-slate-800 hover:bg-slate-700 text-slate-200"
                        >
                          <FolderOpen size={12} className="mr-1" />
                          Open folder
                        </Button>
                        <Button
                          onClick={handleRegeneratePlan}
                          loading={planCreating}
                          disabled={planCreating || planDeleting}
                          className="text-[11px] h-7 px-2 bg-indigo-700 hover:bg-indigo-600 text-white"
                        >
                          <RefreshCw size={12} className="mr-1" />
                          Regenerate
                        </Button>
                        <Button
                          onClick={handleDeletePlan}
                          loading={planDeleting}
                          disabled={planCreating || planDeleting}
                          className="text-[11px] h-7 px-2 bg-rose-900/60 hover:bg-rose-800 text-rose-100"
                        >
                          <Trash2 size={12} className="mr-1" />
                          Delete
                        </Button>
                      </div>
                      <p className="text-[10px] text-slate-500 pl-6">
                        Regenerate after any Stage 2/3 change — an existing plan is served
                        as-is and can go stale.
                      </p>
                    </div>
                    <p className="text-[11px] text-slate-400">
                      Copy this folder to your target repo's machine and point your coding agent at
                      it — <span className="text-slate-300">AGENTS.md</span> first, then work through{' '}
                      <span className="text-slate-300">TASKS.md</span>.
                    </p>
                  </div>
                ) : (
                  <Button
                    variant="primary"
                    onClick={handleCreatePlan}
                    loading={planCreating}
                    disabled={planCreating}
                    className="w-full text-xs h-8 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white font-medium"
                  >
                    {planCreating ? 'Creating plan...' : 'Create Eval Plan'}
                  </Button>
                )}
              </div>
            </div>
          </div>

        </div>
      </div>
    </div>
  )
}

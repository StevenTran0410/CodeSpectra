import React, { useState } from 'react'
import { useSearchParams, useNavigate } from 'react-router-dom'
import {
  Loader2,
  AlertCircle,
  ArrowLeft,
  CheckCircle2,
} from 'lucide-react'
import { Button, useToastStore } from '../../components/ui'

export default function Stage4Screen(): React.ReactElement {
  const [searchParams] = useSearchParams()
  const navigate = useNavigate()
  const toast = useToastStore()
  const sessionId = searchParams.get('sessionId') || ''

  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [branchInfo, setBranchInfo] = useState<{ branch_name: string; previous_branch: string } | null>(null)
  const [planPath, setPlanPath] = useState<string | null>(null)
  const [confirmOpen, setConfirmOpen] = useState(false)
  const [baseRef, setBaseRef] = useState('main')
  const [planCreating, setPlanCreating] = useState(false)

  async function handleCreateBranch() {
    setLoading(true)
    setError(null)
    try {
      const data = await window.api.aeh.createEvalBranch(sessionId, baseRef)
      setBranchInfo(data)
      toast.success(`Created eval branch: ${data.branch_name}`)
    } catch (err: any) {
      setError(err?.message ?? 'Failed to create eval branch')
    } finally {
      setLoading(false)
      setConfirmOpen(false)
    }
  }

  async function handleCreatePlan() {
    setPlanCreating(true)
    setError(null)
    try {
      const data = await window.api.aeh.createEvalPlan(sessionId)
      setPlanPath(data.plan_path)
      toast.success('Eval plan created. Have your coding agent read AEH_EVAL_PLAN.md.')
    } catch (err: any) {
      setError(err?.message ?? 'Failed to create eval plan')
    } finally {
      setPlanCreating(false)
    }
  }

  if (!sessionId) {
    return (
      <div className="flex flex-col h-full bg-[#090d16] text-slate-100 p-6">
        <div className="flex items-center gap-2 mb-6">
          <Button
            variant="ghost"
            className="w-8 h-8 p-0 rounded-full flex items-center justify-center border border-slate-800 bg-slate-900/40 text-slate-400 hover:text-slate-200"
            onClick={() => navigate('/aeh/analysis')}
          >
            <ArrowLeft size={16} />
          </Button>
          <h1 className="text-lg font-semibold">Stage 4: Eval Branch</h1>
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
            onClick={() => navigate('/aeh/analysis')}
          >
            <ArrowLeft size={16} />
          </Button>
          <div>
            <h1 className="screen-title">Stage 4: Eval Branch</h1>
            <p className="screen-subtitle">Create an isolated git branch for instrumentation</p>
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
              <h2 className="text-sm font-semibold text-slate-200 mb-3">Step 1: Create Eval Branch</h2>
              <p className="text-xs text-slate-400 mb-4">
                Creates an isolated branch (aeh/eval-{sessionId}) in your working directory.
              </p>

              <div className="space-y-3">
                <div>
                  <label className="block text-xs font-medium text-slate-400 mb-1">Base Branch (default: main)</label>
                  <input
                    type="text"
                    value={baseRef}
                    onChange={(e) => setBaseRef(e.target.value)}
                    disabled={!!branchInfo || loading}
                    className="w-full text-xs px-3 py-2 bg-slate-900 border border-slate-800 rounded text-slate-200 disabled:opacity-50"
                    placeholder="main"
                  />
                  <p className="text-[10px] text-slate-600 mt-1">
                    Use if your repo defaults to "master" instead of "main"
                  </p>
                </div>

                {!branchInfo ? (
                  <Button
                    variant="primary"
                    onClick={() => setConfirmOpen(true)}
                    loading={loading}
                    disabled={loading}
                    className="w-full text-xs h-8 bg-indigo-600 hover:bg-indigo-500 text-white font-medium"
                  >
                    {loading ? 'Creating...' : 'Create Eval Branch'}
                  </Button>
                ) : (
                  <div className="flex items-center gap-2 p-3 bg-emerald-950/20 border border-emerald-800/40 rounded-lg">
                    <CheckCircle2 size={16} className="text-emerald-400 shrink-0" />
                    <div className="text-xs">
                      <div className="text-emerald-300 font-medium">{branchInfo.branch_name}</div>
                      <div className="text-emerald-300/70">Previous: {branchInfo.previous_branch}</div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>

          <div className="bg-slate-950/40 border border-slate-850 rounded-xl p-6 space-y-4">
            <div>
              <h2 className="text-sm font-semibold text-slate-200 mb-3">Step 2: Create Eval Plan</h2>
              <p className="text-xs text-slate-400 mb-4">
                Writes AEH_EVAL_PLAN.md with all instrumentation files for your coding agent.
              </p>

              {planPath ? (
                <div className="space-y-2">
                  <div className="flex items-center gap-2 p-3 bg-emerald-950/20 border border-emerald-800/40 rounded-lg">
                    <CheckCircle2 size={16} className="text-emerald-400 shrink-0" />
                    <div className="text-xs text-emerald-300">{planPath}</div>
                  </div>
                  <p className="text-[11px] text-slate-400">
                    Have your coding agent read the plan and implement the instrumentation.
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

      {confirmOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/80 backdrop-blur-sm p-4">
          <div className="glass rounded-2xl w-full max-w-sm border-slate-850 shadow-2xl p-5 space-y-4 text-slate-100">
            <div className="flex items-center gap-3 text-amber-400">
              <AlertCircle className="shrink-0 w-6 h-6" />
              <h3 className="text-sm font-semibold text-slate-200">Create Eval Branch?</h3>
            </div>
            <p className="text-[11px] text-slate-450 leading-relaxed">
              This will switch the CodeSpectra working directory to branch aeh/eval-{sessionId}.
              <br /><br />
              Requirements:
              <br />• The working tree must be clean (no uncommitted changes)
              <br />• Restart the CodeSpectra backend after switching branches
            </p>
            <div className="flex justify-end gap-2.5 pt-2">
              <Button
                variant="ghost"
                onClick={() => setConfirmOpen(false)}
                disabled={loading}
                className="text-xs px-4 py-2"
              >
                Cancel
              </Button>
              <Button
                variant="primary"
                onClick={handleCreateBranch}
                loading={loading}
                className="text-xs px-4 py-2 bg-indigo-600 hover:bg-indigo-500"
              >
                Continue
              </Button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

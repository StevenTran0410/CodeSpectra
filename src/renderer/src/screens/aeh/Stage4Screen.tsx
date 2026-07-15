import React, { useState } from 'react'
import { useSearchParams, useNavigate } from 'react-router-dom'
import {
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

  const [error, setError] = useState<string | null>(null)
  const [planPath, setPlanPath] = useState<string | null>(null)
  const [baseRef, setBaseRef] = useState('main')
  const [planCreating, setPlanCreating] = useState(false)

  async function handleCreatePlan() {
    setPlanCreating(true)
    setError(null)
    try {
      const data = await window.api.aeh.createEvalPlan(sessionId, baseRef)
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
            onClick={() => navigate('/aeh/analysis')}
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
                Writes AEH_EVAL_PLAN.md with all instrumentation files and instructions for
                your coding agent — including creating the eval branch itself. AEH doesn't
                touch git for this; the plan tells the coding agent to do it as its own
                first step.
              </p>

              <div className="space-y-3">
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
      </div>
    </div>
  )
}

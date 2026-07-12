import React, { useState } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'
import {
  Loader2,
  AlertCircle,
  ArrowLeft,
  CheckCircle2,
} from 'lucide-react'
import { Button, useToastStore } from '../../components/ui'

export default function Stage5Screen(): React.ReactElement {
  const [searchParams] = useSearchParams()
  const navigate = useNavigate()
  const toast = useToastStore()
  const sessionId = searchParams.get('sessionId') || ''

  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<{ run_id: string; status: string } | null>(null)
  const [error, setError] = useState<string | null>(null)

  async function handleLoadResults() {
    setLoading(true)
    setError(null)
    try {
      const data = await window.api.aeh.loadEvalResults(sessionId)
      setResult(data)
      toast.success('Eval results loaded successfully')
    } catch (err: any) {
      if (err?.message?.includes('manifest not found')) {
        setError(
          'No results yet. Start your CodeSpectra backend on the eval branch, ' +
          'run POST /aeh/run-eval from /docs, then click Load Results again.'
        )
        return
      }
      setError(err?.message ?? 'Failed to load eval results')
    } finally {
      setLoading(false)
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
          <h1 className="text-lg font-semibold">Stage 5: Load Results</h1>
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
            <h1 className="screen-title">Stage 5: Load Results</h1>
            <p className="screen-subtitle">Ingest evaluation results from the instrumented backend</p>
          </div>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto p-6">
        <div className="space-y-6 max-w-2xl">
          {error && (
            <div className="bg-rose-950/20 border border-rose-800/40 rounded-xl p-4 text-xs text-rose-300 space-y-1">
              <div className="font-semibold flex items-center gap-1.5">
                <AlertCircle size={14} className="text-rose-400" />
                <span>Note</span>
              </div>
              <p className="text-[11px] leading-relaxed text-rose-300/80">{error}</p>
            </div>
          )}

          <div className="bg-slate-950/40 border border-slate-850 rounded-xl p-6 space-y-4">
            <div>
              <h2 className="text-sm font-semibold text-slate-200 mb-3">Load Evaluation Results</h2>
              <p className="text-xs text-slate-400 mb-6">
                Once you have run the eval driver on your instrumented backend (by calling POST /aeh/run-eval),
                click below to ingest the results into AEH.
              </p>

              {result ? (
                <div className="space-y-4">
                  <div className="flex items-start gap-3 p-4 bg-emerald-950/20 border border-emerald-800/40 rounded-lg">
                    <CheckCircle2 size={20} className="text-emerald-400 shrink-0 mt-0.5" />
                    <div>
                      <div className="text-sm font-semibold text-emerald-300 mb-1">Results Loaded</div>
                      <div className="text-xs text-emerald-300/70 space-y-1">
                        <div>Run ID: <span className="font-mono">{result.run_id}</span></div>
                        <div>Status: <span className="font-mono">{result.status}</span></div>
                      </div>
                    </div>
                  </div>

                  <Button
                    variant="primary"
                    onClick={() => navigate('/aeh/reports')}
                    className="w-full text-xs h-8 bg-indigo-600 hover:bg-indigo-500 text-white font-medium"
                  >
                    View Results
                  </Button>
                </div>
              ) : (
                <Button
                  variant="primary"
                  onClick={handleLoadResults}
                  loading={loading}
                  disabled={loading}
                  className="w-full text-xs h-9 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white font-medium flex items-center justify-center gap-2"
                >
                  {loading && <Loader2 size={14} className="animate-spin" />}
                  {loading ? 'Loading Results...' : 'Load Results'}
                </Button>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

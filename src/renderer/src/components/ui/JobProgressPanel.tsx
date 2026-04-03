import React from 'react'
import { CheckCircle2, XCircle, Loader2, Clock, Ban, ChevronRight } from 'lucide-react'
import type { Job, StepState, StepStatus } from '../../types/electron'

const STEP_LABELS: Record<string, string> = {
  clone: 'Clone',
  sync: 'Sync',
  manifest: 'File manifest',
  parse: 'Parse symbols',
  graph: 'Build graph',
  embed: 'Embeddings',
  generate: 'Generate report',
  export: 'Export',
}

function StepIcon({ status }: { status: StepStatus }) {
  if (status === 'done')    return <CheckCircle2 size={14} className="text-emerald-400 shrink-0" />
  if (status === 'failed')  return <XCircle size={14} className="text-red-400 shrink-0" />
  if (status === 'running') return <Loader2 size={14} className="animate-spin text-blue-400 shrink-0" />
  if (status === 'skipped') return <ChevronRight size={14} className="text-zinc-600 shrink-0" />
  return <Clock size={14} className="text-zinc-600 shrink-0" />
}

function StepRow({ name, state }: { name: string; state: StepState }) {
  const label = STEP_LABELS[name] ?? name
  const isActive = state.status === 'running'

  return (
    <div className={`flex items-center gap-2.5 py-1.5 ${isActive ? '' : 'opacity-60'}`}>
      <StepIcon status={state.status} />
      <span className="text-xs text-zinc-300 w-28 shrink-0">{label}</span>
      <div className="flex-1 h-1.5 bg-zinc-700 rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-300 ${
            state.status === 'done'    ? 'bg-emerald-500' :
            state.status === 'failed'  ? 'bg-red-500' :
            state.status === 'running' ? 'bg-blue-500' : 'bg-zinc-600'
          }`}
          style={{ width: `${state.progress}%` }}
        />
      </div>
      <span className="text-xs text-zinc-500 w-8 text-right shrink-0">{state.progress}%</span>
    </div>
  )
}

export function JobProgressPanel({
  job,
  onCancel,
}: {
  job: Job
  onCancel?: () => void
}) {
  const isActive = job.status === 'running' || job.status === 'pending'

  const statusColor = {
    pending:   'text-zinc-400',
    running:   'text-blue-400',
    done:      'text-emerald-400',
    failed:    'text-red-400',
    cancelled: 'text-zinc-500',
  }[job.status]

  return (
    <div className="bg-zinc-800/60 border border-zinc-700 rounded-xl p-4 space-y-3">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          {isActive && <Loader2 size={13} className="animate-spin text-blue-400" />}
          {job.status === 'done' && <CheckCircle2 size={13} className="text-emerald-400" />}
          {job.status === 'failed' && <XCircle size={13} className="text-red-400" />}
          {job.status === 'cancelled' && <Ban size={13} className="text-zinc-500" />}
          <span className={`text-xs font-semibold capitalize ${statusColor}`}>
            {job.status}
          </span>
          {job.current_step && isActive && (
            <span className="text-xs text-zinc-500">· {STEP_LABELS[job.current_step] ?? job.current_step}</span>
          )}
        </div>
        {isActive && onCancel && (
          <button
            onClick={onCancel}
            className="text-xs text-zinc-500 hover:text-red-400 transition-colors"
          >
            Cancel
          </button>
        )}
      </div>

      {/* Steps */}
      {Object.keys(job.steps).length > 0 && (
        <div className="space-y-0.5">
          {Object.entries(job.steps).map(([name, state]) => (
            <StepRow key={name} name={name} state={state} />
          ))}
        </div>
      )}

      {/* Error */}
      {job.error && (
        <p className="text-xs text-red-400 bg-red-500/10 border border-red-500/20 rounded-lg px-3 py-2">
          {job.error}
        </p>
      )}
    </div>
  )
}

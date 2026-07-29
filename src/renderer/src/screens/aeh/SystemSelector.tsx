import React, { useEffect, useState } from 'react'

export interface SiblingSystem {
  session_id: string
  name: string
  framework: string | null
  agent_count: number | null
  dataset_count?: number
  runnable_cases_count?: number
  ready: boolean
  is_current: boolean
  created_at?: string
}

interface SystemSelectorProps {
  sessionId: string
  activeSessionId: string
  onSelect: (sessionId: string) => void
  disabled?: boolean
  // When the parent passes a non-empty `systems`, this component renders it directly and does
  // NOT fetch — so it stays instant across unmount/remount (e.g. while the parent shows a loading
  // state) instead of blanking out and refetching. Parents that don't pass it self-fetch.
  systems?: SiblingSystem[]
  onSystemsLoaded?: (systems: SiblingSystem[]) => void
}

export default function SystemSelector({
  sessionId,
  activeSessionId,
  onSelect,
  disabled = false,
  systems,
  onSystemsLoaded,
}: SystemSelectorProps): React.ReactElement | null {
  const [fetched, setFetched] = useState<SiblingSystem[]>([])
  const controlled = !!(systems && systems.length > 0)
  const list = controlled ? (systems as SiblingSystem[]) : fetched

  useEffect(() => {
    if (controlled) return // parent owns the data — don't fetch
    if (!sessionId) return
    let cancelled = false
    window.api.aeh
      .listSiblingSystems(sessionId)
      .then((data) => {
        if (cancelled) return
        const sorted = (data || []) as SiblingSystem[]
        setFetched(sorted)
        onSystemsLoaded?.(sorted)
      })
      .catch(() => {})
    return () => {
      cancelled = true
    }
  }, [sessionId, controlled])

  if (list.length === 0) return null

  return (
    <div className="mb-4">
      <label className="block text-xs font-medium text-slate-400 mb-2">
        Select Agent System
      </label>
      {list.length > 4 ? (
        <select
          value={activeSessionId}
          onChange={(e) => onSelect(e.target.value)}
          disabled={disabled}
          className="w-full text-xs px-3 py-2 bg-slate-900 border border-slate-800 rounded text-slate-200"
        >
          {list.map((sys) => (
            <option key={sys.session_id} value={sys.session_id} disabled={!sys.ready}>
              {sys.name} ({sys.framework ?? 'unknown'}) {sys.agent_count ? `— ${sys.agent_count} agents · ${sys.dataset_count ?? 0} datasets · ${sys.runnable_cases_count ?? 0} cases ${((sys.runnable_cases_count ?? 0) > 0 ? '(runnable)' : '(needs review)')}` : ''} {!sys.ready ? ' (Stage 3 not complete)' : ''}
            </option>
          ))}
        </select>
      ) : (
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
          {list.map((sys) => {
            const isSelected = activeSessionId === sys.session_id
            return (
              <button
                key={sys.session_id}
                disabled={!sys.ready || disabled}
                onClick={() => onSelect(sys.session_id)}
                type="button"
                className={`flex flex-col text-left p-3 rounded-lg border text-xs transition-all ${
                  isSelected
                    ? 'bg-indigo-950/30 border-indigo-500 text-slate-100'
                    : sys.ready
                    ? 'bg-slate-900 border-slate-800 text-slate-300 hover:border-slate-700'
                    : 'bg-slate-900/40 border-slate-850/40 text-slate-500 cursor-not-allowed'
                }`}
              >
                <div className="flex items-center justify-between w-full mb-1">
                  <span className={`font-semibold truncate max-w-[150px] ${
                    isSelected ? 'text-slate-100' : sys.ready ? 'text-slate-200' : 'text-slate-500'
                  }`}>
                    {sys.name}
                  </span>
                  {sys.framework && (
                    <span className={`text-[9px] px-1.5 py-0.5 rounded font-mono uppercase tracking-wider ${
                      isSelected
                        ? 'bg-indigo-900/50 border border-indigo-800 text-indigo-300'
                        : sys.ready
                        ? 'bg-slate-800 text-slate-400 border border-slate-750'
                        : 'bg-slate-900 text-slate-600 border border-slate-850'
                    }`}>
                      {sys.framework}
                    </span>
                  )}
                </div>
                <div className="flex justify-between items-start w-full text-[10px] flex-col gap-0.5 mt-1">
                  <span className={isSelected ? 'text-indigo-200 font-medium' : sys.ready ? 'text-slate-400' : 'text-slate-650'}>
                    {sys.agent_count ?? 0} agents · {sys.dataset_count ?? 0} datasets · {sys.runnable_cases_count ?? 0} cases
                  </span>
                  {sys.ready && (
                    <span className={`text-[9px] font-medium ${
                      (sys.runnable_cases_count ?? 0) > 0
                        ? isSelected ? 'text-emerald-300' : 'text-emerald-400'
                        : isSelected ? 'text-amber-300' : 'text-amber-500'
                    }`}>
                      {(sys.runnable_cases_count ?? 0) > 0 ? 'Runnable' : 'Needs review'}
                    </span>
                  )}
                  {!sys.ready && (
                    <span className="text-amber-600/85 text-[9px] font-medium italic">
                      Stage 3 not complete
                    </span>
                  )}
                </div>
              </button>
            )
          })}
        </div>
      )}
    </div>
  )
}

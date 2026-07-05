import React, { useState, useEffect } from 'react'
import { AlertCircle, RefreshCw } from 'lucide-react'
import { useAEHReportsStore } from '../../store/aeh-reports.store'
import type { AEHRunDetailResponse } from '../../types/electron'

interface RerunModalProps {
  run: AEHRunDetailResponse
  isOpen: boolean
  onClose: () => void
  onSuccess: () => void
}

export default function RerunModal({ run, isOpen, onClose, onSuccess }: RerunModalProps): React.ReactElement | null {
  const { availableProviders, fetchProviders, triggerRerun } = useAEHReportsStore()

  const [rerunActiveDefects, setRerunActiveDefects] = useState<string[]>([])
  const [rerunModelOverrides, setRerunModelOverrides] = useState<Record<string, string>>({})
  const [customOverrideComponents, setCustomOverrideComponents] = useState<Set<string>>(new Set())
  const [submitLoading, setSubmitLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Initialize
  useEffect(() => {
    setRerunActiveDefects([...run.active_defects])
    const initialOverrides: Record<string, string> = {}
    if (run.model_overrides) {
      Object.entries(run.model_overrides).forEach(([compId, modelId]) => {
        initialOverrides[compId] = modelId
      })
    }
    setRerunModelOverrides(initialOverrides)
    setCustomOverrideComponents(new Set())
    setError(null)
    fetchProviders().catch(() => {})
  }, [run])

  const handleSubmit = async () => {
    setSubmitLoading(true)
    setError(null)
    try {
      // Filter overrides: if Custom was selected but left empty, remove it
      const cleanOverrides: Record<string, string> = {}
      Object.entries(rerunModelOverrides).forEach(([compId, val]) => {
        if (customOverrideComponents.has(compId) && !val.trim()) {
          // Skip if empty custom
          return
        }
        cleanOverrides[compId] = val
      })

      await triggerRerun(run.id, {
        model_overrides: cleanOverrides,
        active_defects: rerunActiveDefects,
      })
      onSuccess()
      onClose()
    } catch (err: any) {
      setError(err.message || 'Failed to trigger rerun')
    } finally {
      setSubmitLoading(false)
    }
  }

  // Early return AFTER all hooks (Rules of Hooks) — this component is always
  // mounted by RunDetailScreen with `isOpen` toggling as a prop, so returning
  // null before the hooks above would change how many hooks are called between
  // renders of the same instance and corrupt React's internal hook bookkeeping.
  if (!isOpen) return null

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/80 backdrop-blur-sm p-4 overflow-y-auto">
      <div className="glass rounded-2xl w-full max-w-2xl border-slate-800 shadow-2xl flex flex-col max-h-[90vh] text-slate-100">
        {/* Modal Header */}
        <div className="px-6 py-4 border-b border-surface-border flex items-center justify-between">
          <div>
            <h3 className="text-lg font-bold text-gray-100 flex items-center gap-2">
              <RefreshCw className="h-5 w-5 text-indigo-400" />
              <span>Configure Rerun: {run.target_system_id}</span>
            </h3>
            <p className="text-xs text-gray-400 mt-1">
              Customize LLM stages and plant regression gauntlet defects.
            </p>
          </div>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-200 text-lg font-bold px-2"
          >
            &times;
          </button>
        </div>

        {/* Modal Content */}
        <div className="p-6 space-y-6 overflow-y-auto flex-1">
          {error && (
            <div className="p-4 bg-red-950/20 border border-red-900/30 text-red-400 rounded-xl text-xs flex items-center gap-2 font-semibold">
              <AlertCircle className="h-4 w-4 flex-shrink-0" />
              <span>{error}</span>
            </div>
          )}

          {/* Model Overrides Section */}
          <div className="space-y-3">
            <h4 className="text-xs text-gray-400 font-semibold uppercase tracking-wider">
              Component Model Overrides
            </h4>
            <div className="border border-surface-border bg-surface-overlay/50 rounded-xl overflow-hidden">
              <table className="w-full text-left border-collapse text-xs">
                <thead>
                  <tr className="border-b border-surface-border bg-surface-overlay text-gray-300 font-semibold">
                    <th className="py-2.5 px-4">Component</th>
                    <th className="py-2.5 px-4">Role</th>
                    <th className="py-2.5 px-4 w-1/2">Override Model</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-surface-border">
                  {run.system_map.components.map((comp) => {
                    const currentVal = rerunModelOverrides[comp.id] || ''
                    const matchesKnownProvider = availableProviders.some(
                      (p) => `${p.provider_id}:${p.model_id}` === currentVal
                    )
                    const isCustomMode =
                      customOverrideComponents.has(comp.id) ||
                      (currentVal !== '' && !matchesKnownProvider)

                    return (
                      <tr key={comp.id} className="text-gray-300">
                        <td className="py-3 px-4 font-mono font-bold text-gray-100">{comp.id}</td>
                        <td className="py-3 px-4 uppercase text-[10px] text-gray-500 font-semibold">
                          {comp.role}
                        </td>
                        <td className="py-3 px-4">
                          {availableProviders.length === 0 ? (
                            <input
                              type="text"
                              placeholder="Override model ID (e.g. gpt-4)"
                              value={currentVal}
                              onChange={(e) =>
                                setRerunModelOverrides({
                                  ...rerunModelOverrides,
                                  [comp.id]: e.target.value,
                                })
                              }
                              className="w-full bg-slate-900 border border-slate-700/80 text-xs text-white rounded-lg px-2.5 py-1.5 focus:border-indigo-500 focus:outline-none"
                            />
                          ) : (
                            <div className="space-y-1.5">
                              <select
                                value={isCustomMode ? 'custom' : currentVal}
                                onChange={(e) => {
                                  const val = e.target.value
                                  setCustomOverrideComponents((prev) => {
                                    const next = new Set(prev)
                                    if (val === 'custom') next.add(comp.id)
                                    else next.delete(comp.id)
                                    return next
                                  })
                                  setRerunModelOverrides({
                                    ...rerunModelOverrides,
                                    [comp.id]: val === 'custom' ? '' : val,
                                  })
                                }}
                                className="w-full bg-slate-900 border border-slate-700/80 text-xs text-white rounded-lg px-2.5 py-1.5 focus:border-indigo-500 focus:outline-none"
                              >
                                <option value="">Default ({comp.model || 'system default'})</option>
                                {availableProviders.map((p) => (
                                  <option
                                    key={`${p.provider_id}:${p.model_id}`}
                                    value={`${p.provider_id}:${p.model_id}`}
                                  >
                                    {p.display_name} ({p.model_id})
                                  </option>
                                ))}
                                <option value="custom">Custom...</option>
                              </select>
                              {isCustomMode && (
                                <input
                                  type="text"
                                  placeholder="Enter custom model ID"
                                  value={currentVal}
                                  onChange={(e) =>
                                    setRerunModelOverrides({
                                      ...rerunModelOverrides,
                                      [comp.id]: e.target.value,
                                    })
                                  }
                                  className="w-full bg-slate-900 border border-slate-700/80 text-xs text-white rounded-lg px-2.5 py-1.5 focus:border-indigo-500 focus:outline-none mt-1"
                                />
                              )}
                            </div>
                          )}
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          </div>

          {/* Planted Defects Section */}
          <div className="space-y-3">
            <h4 className="text-xs text-gray-400 font-semibold uppercase tracking-wider">
              Plant Regression Defects
            </h4>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              {[
                'planner_overpack',
                'guard_leak',
                'wrong_tool',
                'judge_rubber_stamp',
                'writer_hallucinate',
                'no_retry',
              ].map((def) => {
                const isChecked = rerunActiveDefects.includes(def)
                return (
                  <label
                    key={def}
                    className={`p-3 rounded-xl border flex items-center justify-between text-xs cursor-pointer select-none transition-all ${
                      isChecked
                        ? 'bg-red-950/20 border-red-950 text-red-400 font-bold'
                        : 'bg-slate-900/40 border-slate-800 text-gray-400 hover:border-slate-700'
                    }`}
                  >
                    <span>{def.toUpperCase()}</span>
                    <input
                      type="checkbox"
                      checked={isChecked}
                      onChange={() => {
                        if (isChecked) {
                          setRerunActiveDefects(rerunActiveDefects.filter((d) => d !== def))
                        } else {
                          setRerunActiveDefects([...rerunActiveDefects, def])
                        }
                      }}
                      className="h-4 w-4 rounded border-slate-700 text-indigo-600 focus:ring-indigo-500"
                    />
                  </label>
                )
              })}
            </div>
          </div>
        </div>

        {/* Modal Footer */}
        <div className="px-6 py-4 border-t border-surface-border flex items-center justify-end gap-3 bg-surface-overlay/20">
          <button
            onClick={onClose}
            className="px-4 py-2 border border-surface-border hover:border-gray-600 text-gray-400 hover:text-gray-200 rounded-xl text-xs font-bold transition-all"
          >
            Cancel
          </button>
          <button
            onClick={handleSubmit}
            disabled={submitLoading}
            className="inline-flex items-center gap-2 bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-500 hover:to-purple-500 disabled:from-indigo-800 disabled:to-purple-800 text-white px-4 py-2 rounded-xl text-xs font-bold shadow-lg shadow-indigo-500/20 disabled:shadow-none transition-all active:scale-[0.98] border border-indigo-500/30"
          >
            {submitLoading ? (
              <>
                <RefreshCw className="h-3 w-3 animate-spin" />
                <span>Launching Rerun...</span>
              </>
            ) : (
              <>
                <RefreshCw className="h-3 w-3" />
                <span>Run Evaluation</span>
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  )
}

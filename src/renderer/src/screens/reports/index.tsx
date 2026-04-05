import React, { useEffect, useMemo, useState } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'
import type { AnalysisReport, AnalysisReportSummary } from '../../types/electron'
import { ErrorBanner } from '../../components/ui/ErrorBanner'
import { toErrorMessage } from '../../lib/errors'

const SECTION_LABELS: Record<string, string> = {
  'A/B/C/G/H': 'Structure, Architecture, Important Files',
  'D/E': 'Conventions and Negative Rules',
  'F/I/J': 'Feature Map, Domain Terms, Risks',
}

const SECTION_AGENT: Record<string, string> = {
  'A/B/C/G/H': 'Structure Intelligence Agent',
  'D/E': 'Convention Intelligence Agent',
  'F/I/J': 'Domain & Risk Intelligence Agent',
}

export default function ReportViewerScreen(): React.ReactElement {
  const navigate = useNavigate()
  const [params] = useSearchParams()
  const repoId = params.get('repoId') ?? undefined
  const reportIdInUrl = params.get('reportId') ?? ''
  const isDetailMode = reportIdInUrl.trim().length > 0

  const [reports, setReports] = useState<AnalysisReportSummary[]>([])
  const [selectedReportId, setSelectedReportId] = useState('')
  const [report, setReport] = useState<AnalysisReport | null>(null)
  const [loading, setLoading] = useState(false)
  const [deleting, setDeleting] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [confirmDelete, setConfirmDelete] = useState(false)
  const [hideDeleteWarning, setHideDeleteWarning] = useState(
    localStorage.getItem('reports.deleteWarningHidden') === '1'
  )

  const refreshList = async (preferredReportId?: string) => {
    setLoading(true)
    setError(null)
    try {
      const list = await window.api.analysis.listReports(repoId, 50)
      setReports(list)
      if (list.length === 0) {
        setSelectedReportId('')
        setReport(null)
        return
      }
      const preferred = preferredReportId || reportIdInUrl || selectedReportId
      const stillExists = preferred ? list.some((r) => r.id === preferred) : false
      setSelectedReportId(stillExists ? (preferred as string) : list[0].id)
    } catch (err) {
      setError(toErrorMessage(err))
    } finally {
      setLoading(false)
    }
  }

  const deleteSelectedReport = async () => {
    if (!report) return
    setDeleting(true)
    setError(null)
    try {
      const deletedId = report.summary.id
      await window.api.analysis.deleteReport(deletedId)
      setConfirmDelete(false)
      const next = reports.find((r) => r.id !== deletedId)?.id
      await refreshList(next)
      navigate('/reports')
    } catch (err) {
      setError(toErrorMessage(err))
    } finally {
      setDeleting(false)
    }
  }

  useEffect(() => {
    void refreshList()
  }, [repoId, reportIdInUrl])

  useEffect(() => {
    const run = async () => {
      if (!isDetailMode) {
        setReport(null)
        return
      }
      const targetId = reportIdInUrl || selectedReportId
      if (!targetId) {
        setReport(null)
        return
      }
      try {
        const out = await window.api.analysis.getReport(targetId)
        setReport(out)
      } catch (err) {
        setError(toErrorMessage(err))
      }
    }
    run()
  }, [isDetailMode, selectedReportId, reportIdInUrl])

  const sections = useMemo(() => report?.report.sections ?? [], [report])
  const confidence = report?.report.confidence_summary

  return (
    <>
      <div className="screen-header">
        <h1 className="screen-title">Reports</h1>
        <p className="screen-subtitle">View generated analysis artifacts</p>
      </div>
      <div className="h-[calc(100vh-10rem)] overflow-y-auto p-4 space-y-3">
        {error && <ErrorBanner message={error} onDismiss={() => setError(null)} />}

        {!isDetailMode ? (
          <div className="bg-zinc-900/60 border border-zinc-700 rounded-xl p-3 space-y-2">
            <div className="text-xs font-semibold text-zinc-100">Selection Only</div>
            {loading ? (
              <div className="text-xs text-zinc-500">Loading...</div>
            ) : reports.length === 0 ? (
              <div className="text-xs text-zinc-500">No report yet. Run analysis first.</div>
            ) : (
              <>
                <div className="space-y-1.5 max-h-[64vh] overflow-auto">
                  {reports.map((r) => (
                    <button
                      key={r.id}
                      onClick={() => setSelectedReportId(r.id)}
                      onDoubleClick={() => navigate(`/reports?reportId=${encodeURIComponent(r.id)}`)}
                      className={`w-full text-left rounded-md border px-2 py-1.5 text-[11px] ${
                        r.id === selectedReportId
                          ? 'border-indigo-600 bg-indigo-950/40 text-indigo-200'
                          : 'border-zinc-700 bg-zinc-950 text-zinc-300 hover:border-zinc-600'
                      }`}
                    >
                      <div className="font-mono truncate">{r.id.slice(0, 10)}</div>
                      <div className="grid grid-cols-[54px,1fr] gap-x-2 gap-y-0.5 mt-1 text-[11px]">
                        <div className="text-zinc-500">repo</div>
                        <div className="text-zinc-300 truncate">{r.repo_name || r.repo_id}</div>
                        <div className="text-zinc-500">branch</div>
                        <div className="text-zinc-300 truncate">{r.branch || 'unknown'}</div>
                        <div className="text-zinc-500">model</div>
                        <div className="text-zinc-300 truncate">{r.model_id}</div>
                      </div>
                      <div className="text-zinc-500 truncate mt-1">{r.created_at}</div>
                    </button>
                  ))}
                </div>
                <div className="pt-1">
                  <button
                    onClick={() => selectedReportId && navigate(`/reports?reportId=${encodeURIComponent(selectedReportId)}`)}
                    disabled={!selectedReportId}
                    className="px-3 py-1.5 text-xs bg-indigo-600 hover:bg-indigo-500 text-white rounded-md disabled:opacity-50"
                  >
                    Select
                  </button>
                </div>
              </>
            )}
          </div>
        ) : (
          <div className="bg-zinc-900/60 border border-zinc-700 rounded-xl p-3 space-y-3">
            {report ? (
              <>
                <div className="flex items-center justify-between gap-2">
                  <div className="text-xs text-zinc-400">
                    repo: <span className="text-zinc-200 font-mono">{report.summary.repo_name || report.summary.repo_id}</span>
                    <span className="mx-2 text-zinc-700">|</span>
                    branch: <span className="text-zinc-200 font-mono">{report.summary.branch || 'unknown'}</span>
                    <span className="mx-2 text-zinc-700">|</span>
                    model: <span className="text-zinc-200 font-mono">{report.summary.model_id}</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <button
                      onClick={() => navigate('/reports')}
                      className="px-2 py-1 text-[11px] border border-zinc-700 text-zinc-300 rounded hover:border-zinc-600"
                    >
                      Back
                    </button>
                    <button
                      onClick={() => {
                        if (hideDeleteWarning) {
                          void deleteSelectedReport()
                          return
                        }
                        setConfirmDelete(true)
                      }}
                      disabled={deleting}
                      className="px-2 py-1 text-[11px] border border-rose-700 text-rose-300 rounded hover:border-rose-500 disabled:opacity-50"
                    >
                      {deleting ? 'Deleting...' : 'Delete report'}
                    </button>
                  </div>
                </div>
                {confidence && (
                  <div className="text-xs text-zinc-400">
                    confidence - high: <span className="text-zinc-200">{confidence.high}</span>
                    <span className="mx-2 text-zinc-700">|</span>
                    medium: <span className="text-zinc-200">{confidence.medium}</span>
                    <span className="mx-2 text-zinc-700">|</span>
                    low: <span className="text-zinc-200">{confidence.low}</span>
                  </div>
                )}
                <div className="space-y-2 max-h-[64vh] overflow-auto pr-1">
                  {sections.map((s, i) => (
                    <div key={`${s.section}-${i}`} className="rounded-md border border-zinc-800 bg-zinc-950 p-3 space-y-2">
                      <div className="text-sm text-zinc-100">
                        {SECTION_LABELS[s.section] ?? s.section}
                        <span className="ml-2 text-xs text-zinc-500">({s.section})</span>
                        <span className="ml-2 text-xs text-zinc-500">· {s.confidence}</span>
                      </div>
                      <div className="text-xs text-emerald-400">{SECTION_AGENT[s.section] ?? 'Evidence Auditor & Composer Agent'}</div>
                      <div className="text-sm text-zinc-300 whitespace-pre-wrap leading-6">{s.content}</div>
                      {s.evidence_files.length > 0 && (
                        <div className="space-y-1">
                          <div className="text-xs text-zinc-500">Evidence files</div>
                          <div className="space-y-1">
                            {s.evidence_files.slice(0, 8).map((f) => (
                              <div key={f} className="text-xs font-mono text-zinc-400 break-all">{f}</div>
                            ))}
                          </div>
                        </div>
                      )}
                      {s.blind_spots.length > 0 && (
                        <div className="text-xs text-amber-400">
                          Blind spots: {s.blind_spots.join(', ')}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </>
            ) : (
              <div className="text-xs text-zinc-500">Loading report...</div>
            )}
          </div>
        )}
      </div>
      {confirmDelete && report && (
        <div className="fixed inset-0 z-50 bg-black/55 flex items-center justify-center p-4">
          <div className="w-full max-w-md rounded-lg border border-zinc-700 bg-zinc-900 p-4 space-y-3">
            <div className="text-sm font-semibold text-zinc-100">Delete this analysis report?</div>
            <div className="text-xs text-zinc-400">
              This removes the generated report artifact from local database.
            </div>
            {!hideDeleteWarning && (
              <label className="inline-flex items-center gap-2 text-xs text-zinc-300">
                <input
                  type="checkbox"
                  onChange={(e) => {
                    const v = e.target.checked
                    setHideDeleteWarning(v)
                    localStorage.setItem('reports.deleteWarningHidden', v ? '1' : '0')
                  }}
                />
                Do not show this again
              </label>
            )}
            <div className="flex items-center justify-end gap-2">
              <button
                onClick={() => setConfirmDelete(false)}
                disabled={deleting}
                className="px-3 py-1.5 text-xs border border-zinc-700 rounded-md text-zinc-300 hover:border-zinc-600 disabled:opacity-50"
              >
                Cancel
              </button>
              <button
                onClick={() => { void deleteSelectedReport() }}
                disabled={deleting}
                className="px-3 py-1.5 text-xs bg-rose-600 hover:bg-rose-500 rounded-md text-white disabled:opacity-50"
              >
                {deleting ? 'Deleting...' : 'Delete'}
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  )
}

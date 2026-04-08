import React, { useEffect, useMemo, useState } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'
import type {
  AnalysisReport,
  AnalysisReportSummary,
  ReportDiffResult,
} from '../../types/electron'
import {
  type SectionA,
  type SectionB,
  type SectionC,
  type SectionD,
  type SectionE,
  type SectionF,
  type SectionG,
  type SectionH,
  type SectionI,
  type SectionJ,
  type SectionK,
  type SectionL,
} from '../../types/analysis'
import { ErrorBanner } from '../../components/ui/ErrorBanner'
import { toErrorMessage } from '../../lib/errors'
import SectionCardA from './components/SectionCardA'
import SectionCardB from './components/SectionCardB'
import SectionCardC from './components/SectionCardC'
import SectionCardD from './components/SectionCardD'
import SectionCardE from './components/SectionCardE'
import SectionCardF from './components/SectionCardF'
import SectionCardG from './components/SectionCardG'
import SectionCardH from './components/SectionCardH'
import SectionCardI from './components/SectionCardI'
import SectionCardJ from './components/SectionCardJ'
import SectionCardK from './components/SectionCardK'
import SectionCardL from './components/SectionCardL'

function getReportSections(raw: unknown): Record<string, unknown> | null {
  if (!raw || typeof raw !== 'object') return null
  const r = raw as Record<string, unknown>
  const blob =
    r.version === 2 || r.version === 3 ? r.sections : r.sections_v2
  return blob && typeof blob === 'object' ? (blob as Record<string, unknown>) : null
}

const REPORT_SECTION_ORDER = [
  'L',
  'A',
  'B',
  'C',
  'D',
  'E',
  'F',
  'G',
  'H',
  'I',
  'J',
  'K',
] as const

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
  const [exportingMd, setExportingMd] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [success, setSuccess] = useState<string | null>(null)
  const [confirmDelete, setConfirmDelete] = useState(false)
  const [hideDeleteWarning, setHideDeleteWarning] = useState(
    localStorage.getItem('reports.deleteWarningHidden') === '1'
  )
  const [rerunLetter, setRerunLetter] = useState<string | null>(null)
  const [exportAuditBusy, setExportAuditBusy] = useState(false)
  const [repoReportsForCompare, setRepoReportsForCompare] = useState<AnalysisReportSummary[]>([])
  const [compareOpen, setCompareOpen] = useState(false)
  const [compareOtherId, setCompareOtherId] = useState('')
  const [compareBusy, setCompareBusy] = useState(false)
  const [diffResult, setDiffResult] = useState<ReportDiffResult | null>(null)

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

  const rerunSection = async (letter: string) => {
    if (!report) return
    setRerunLetter(letter)
    setError(null)
    try {
      await window.api.analysis.rerunSection({
        report_id: report.summary.id,
        section: letter,
        provider_id: report.summary.provider_id,
        model_id: report.summary.model_id,
      })
      const fresh = await window.api.analysis.getReport(report.summary.id)
      setReport(fresh)
    } catch (err) {
      setError(toErrorMessage(err))
    } finally {
      setRerunLetter(null)
    }
  }

  const exportAudit = async () => {
    if (!report) return
    setExportAuditBusy(true)
    setError(null)
    setSuccess(null)
    try {
      const out = await window.api.analysis.exportAuditSection(report.summary.id)
      if (out.saved && out.file_path) {
        setSuccess(`Audit markdown saved: ${out.file_path}`)
      }
    } catch (err) {
      setError(toErrorMessage(err))
    } finally {
      setExportAuditBusy(false)
    }
  }

  const runCompare = async () => {
    if (!report || !compareOtherId.trim()) return
    setCompareBusy(true)
    setError(null)
    try {
      const res = await window.api.analysis.compareReports({
        report_id_a: report.summary.id,
        report_id_b: compareOtherId.trim(),
      })
      setDiffResult(res)
      setCompareOpen(false)
    } catch (err) {
      setError(toErrorMessage(err))
    } finally {
      setCompareBusy(false)
    }
  }

  const exportMarkdown = async () => {
    if (!report) return
    setExportingMd(true)
    setError(null)
    setSuccess(null)
    try {
      const out = await window.api.analysis.exportReportMarkdown(report.summary.id)
      if (out.saved && out.file_path) {
        setSuccess(`Markdown saved: ${out.file_path}`)
      }
    } catch (err) {
      setError(toErrorMessage(err))
    } finally {
      setExportingMd(false)
    }
  }

  useEffect(() => {
    void refreshList()
  }, [repoId, reportIdInUrl])

  useEffect(() => {
    if (!isDetailMode || !report) {
      setRepoReportsForCompare([])
      return
    }
    const rid = report.summary.repo_id
    void window.api.analysis.listReports(rid, 50).then(setRepoReportsForCompare)
  }, [isDetailMode, report?.summary.repo_id, report?.summary.id])

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

  const sectionsV2 = useMemo(() => {
    if (!report?.report) return null
    return getReportSections(report.report as unknown)
  }, [report])

  const v2Ok = (letter: string): boolean => {
    if (!sectionsV2)
      return false
    const block = sectionsV2[letter]
    if (!block || typeof block !== 'object')
      return false
    return !('error' in block)
  }

  return (
    <>
      <div className="screen-header">
        <h1 className="screen-title">Reports</h1>
        <p className="screen-subtitle">View generated analysis artifacts</p>
      </div>
      <div className="h-[calc(100vh-10rem)] overflow-y-auto p-4 space-y-3">
        {error && <ErrorBanner message={error} onDismiss={() => setError(null)} />}
        {success && (
          <div className="rounded-md border border-emerald-800/60 bg-emerald-950/30 px-3 py-2 text-xs text-emerald-300">
            {success}
          </div>
        )}

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
                    {repoReportsForCompare.length >= 2 && (
                      <button
                        type="button"
                        onClick={() => {
                          setCompareOtherId(
                            repoReportsForCompare.find((r) => r.id !== report.summary.id)?.id ?? ''
                          )
                          setCompareOpen(true)
                        }}
                        className="px-2 py-1 text-[11px] border border-sky-800 text-sky-300 rounded hover:border-sky-600"
                      >
                        Compare with…
                      </button>
                    )}
                    <button
                      onClick={() => { void exportMarkdown() }}
                      disabled={exportingMd}
                      className="px-2 py-1 text-[11px] border border-indigo-700 text-indigo-300 rounded hover:border-indigo-500 disabled:opacity-50"
                    >
                      {exportingMd ? 'Exporting...' : 'Export .md'}
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
                {diffResult && (
                  <div className="rounded-lg border border-zinc-600/80 bg-zinc-950/60 p-3 space-y-2">
                    <div className="flex flex-wrap items-center gap-2 justify-between">
                      <div className="flex flex-wrap items-center gap-2">
                        <span
                          className={`text-[10px] uppercase tracking-wide px-2 py-0.5 rounded border ${
                            diffResult.quality_trend === 'improving'
                              ? 'border-emerald-800 text-emerald-300 bg-emerald-950/30'
                              : diffResult.quality_trend === 'degrading'
                                ? 'border-rose-800 text-rose-300 bg-rose-950/30'
                                : 'border-zinc-600 text-zinc-400 bg-zinc-900/40'
                          }`}
                        >
                          {diffResult.quality_trend}
                        </span>
                        <span className="text-[11px] text-zinc-400">
                          {diffResult.identical
                            ? 'No section differences'
                            : `${diffResult.sections_changed} section(s) changed`}
                        </span>
                      </div>
                      <button
                        type="button"
                        onClick={() => setDiffResult(null)}
                        className="text-[11px] text-zinc-500 hover:text-zinc-300"
                      >
                        Dismiss
                      </button>
                    </div>
                    <div className="max-h-56 overflow-y-auto space-y-1">
                      {Object.entries(diffResult.section_diffs)
                        .filter(([, d]) => d.changed)
                        .map(([letter, d]) => (
                          <div
                            key={letter}
                            className={`text-[11px] rounded border px-2 py-1.5 ${
                              d.improvement === true
                                ? 'border-emerald-800/70 bg-emerald-950/15'
                                : d.improvement === false
                                  ? 'border-rose-800/70 bg-rose-950/15'
                                  : 'border-zinc-700 bg-zinc-900/30'
                            }`}
                          >
                            <span className="font-mono font-semibold text-zinc-200">{letter}</span>
                            {d.confidence_delta && (
                              <span className="text-zinc-400"> · {d.confidence_delta}</span>
                            )}
                            {d.list_added.length > 0 && (
                              <span className="text-emerald-400/90"> · +{d.list_added.length}</span>
                            )}
                            {d.list_removed.length > 0 && (
                              <span className="text-rose-400/90"> · −{d.list_removed.length}</span>
                            )}
                          </div>
                        ))}
                    </div>
                  </div>
                )}
                {sectionsV2 ? (
                  <div className="space-y-2">
                    {REPORT_SECTION_ORDER.map((letter) => {
                      if (!v2Ok(letter)) return null
                      switch (letter) {
                        case 'L':
                          return (
                            <SectionCardL
                              key={letter}
                              data={sectionsV2.L as SectionL}
                              onRerun={() => void rerunSection('L')}
                              rerunBusy={rerunLetter === 'L'}
                            />
                          )
                        case 'A':
                          return (
                            <SectionCardA
                              key={letter}
                              data={sectionsV2.A as SectionA}
                              onRerun={() => void rerunSection('A')}
                              rerunBusy={rerunLetter === 'A'}
                            />
                          )
                        case 'B':
                          return (
                            <SectionCardB
                              key={letter}
                              data={sectionsV2.B as SectionB}
                              onRerun={() => void rerunSection('B')}
                              rerunBusy={rerunLetter === 'B'}
                            />
                          )
                        case 'C':
                          return (
                            <SectionCardC
                              key={letter}
                              data={sectionsV2.C as SectionC}
                              onRerun={() => void rerunSection('C')}
                              rerunBusy={rerunLetter === 'C'}
                            />
                          )
                        case 'D':
                          return (
                            <SectionCardD
                              key={letter}
                              data={sectionsV2.D as SectionD}
                              onRerun={() => void rerunSection('D')}
                              rerunBusy={rerunLetter === 'D'}
                            />
                          )
                        case 'E':
                          return (
                            <SectionCardE
                              key={letter}
                              data={sectionsV2.E as SectionE}
                              onRerun={() => void rerunSection('E')}
                              rerunBusy={rerunLetter === 'E'}
                            />
                          )
                        case 'F':
                          return (
                            <SectionCardF
                              key={letter}
                              data={sectionsV2.F as SectionF}
                              onRerun={() => void rerunSection('F')}
                              rerunBusy={rerunLetter === 'F'}
                            />
                          )
                        case 'G':
                          return (
                            <SectionCardG
                              key={letter}
                              data={sectionsV2.G as SectionG}
                              onRerun={() => void rerunSection('G')}
                              rerunBusy={rerunLetter === 'G'}
                            />
                          )
                        case 'H':
                          return (
                            <SectionCardH
                              key={letter}
                              data={sectionsV2.H as SectionH}
                              onRerun={() => void rerunSection('H')}
                              rerunBusy={rerunLetter === 'H'}
                            />
                          )
                        case 'I':
                          return (
                            <SectionCardI
                              key={letter}
                              data={sectionsV2.I as SectionI}
                              onRerun={() => void rerunSection('I')}
                              rerunBusy={rerunLetter === 'I'}
                            />
                          )
                        case 'J':
                          return (
                            <SectionCardJ
                              key={letter}
                              data={sectionsV2.J as SectionJ}
                              onRerun={() => void rerunSection('J')}
                              rerunBusy={rerunLetter === 'J'}
                            />
                          )
                        case 'K':
                          return (
                            <SectionCardK
                              key={letter}
                              data={sectionsV2.K as SectionK}
                              onRerun={() => void rerunSection('K')}
                              rerunBusy={rerunLetter === 'K'}
                              onExportAudit={() => void exportAudit()}
                              exportAuditBusy={exportAuditBusy}
                            />
                          )
                        default:
                          return null
                      }
                    })}
                  </div>
                ) : (
                  <div className="text-xs text-zinc-500 py-4 text-center">
                    No analysis sections found. Re-run analysis to generate section data.
                  </div>
                )}
              </>
            ) : (
              <div className="text-xs text-zinc-500">Loading report...</div>
            )}
          </div>
        )}
      </div>
      {compareOpen && report && (
        <div className="fixed inset-0 z-50 bg-black/55 flex items-center justify-center p-4">
          <div className="w-full max-w-md rounded-lg border border-zinc-700 bg-zinc-900 p-4 space-y-3">
            <div className="text-sm font-semibold text-zinc-100">Compare reports</div>
            <div className="text-xs text-zinc-400">
              Select another run from the same repository to diff against this report.
            </div>
            <select
              value={compareOtherId}
              onChange={(e) => setCompareOtherId(e.target.value)}
              className="w-full rounded border border-zinc-700 bg-zinc-950 text-xs text-zinc-200 px-2 py-2"
            >
              {repoReportsForCompare
                .filter((r) => r.id !== report.summary.id)
                .map((r) => (
                  <option key={r.id} value={r.id}>
                    {r.id.slice(0, 10)} — {r.model_id} — {r.created_at}
                  </option>
                ))}
            </select>
            <div className="flex items-center justify-end gap-2">
              <button
                type="button"
                onClick={() => setCompareOpen(false)}
                disabled={compareBusy}
                className="px-3 py-1.5 text-xs border border-zinc-700 rounded-md text-zinc-300 hover:border-zinc-600 disabled:opacity-50"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={() => void runCompare()}
                disabled={compareBusy || !compareOtherId}
                className="px-3 py-1.5 text-xs bg-sky-700 hover:bg-sky-600 rounded-md text-white disabled:opacity-50"
              >
                {compareBusy ? 'Comparing…' : 'Compare'}
              </button>
            </div>
          </div>
        </div>
      )}
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
